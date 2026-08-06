r"""Independent forward-chaining entailment and scoring for C18.

The oracle derives a label only from an instance's public ``question`` and
``query`` text. It does not import the vendored generator or inspect its proof
objects, generated label, or random state. Pool generation requires this
surface-text derivation to agree with every generated label.

Task shape (PrOntoQA ModusPonens, fictional ontology). Each instance is:

* **facts** -- a single individual's memberships/properties, e.g.
  ``"Sally is a brimpus."``;
* **rules** -- universally quantified subsumptions over nonce kinds,
  either ``"Every brimpus is a numpus."`` (kind -> kind membership,
  always positive) or ``"Every zumpus is not floral."`` (kind ->
  possibly-negated property);
* a **query** ``"True or false: Sally is not sour."`` -- ``True`` iff the
  queried (property, polarity) is forward-chaining-derivable, else
  ``False``.

The algorithm is a textbook forward-chaining fixpoint:

1. Parse the ``question`` into ground facts about the individual and a
   rule set over kinds.
2. Seed the derived set with the individual's asserted kinds/properties.
3. Repeatedly fire every rule whose antecedent kind the individual is
   known to hold, adding the consequent, until no new fact appears (the
   least fixpoint). Chains are positive kind-subsumptions terminating in
   one possibly-negated property, so a single polarity bit per property
   suffices -- no negation-as-failure search is needed.
4. Resolve the query against the closure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from whetstone_envs.probes import normalize
from whetstone_envs.scoring import exact_match

# The nine proper-noun individuals the fictional ontology draws from
# (upstream ``available_entity_names``). A sentence whose subject is one
# of these is a ground fact about that individual; every other sentence
# is a universally quantified rule over kinds. Pinned here so the oracle
# classifies facts vs rules structurally, from the public text alone.
ENTITY_NAMES: frozenset[str] = frozenset(
    {"Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"},
)


class OracleError(ValueError):
    """Raised when the public question/query text cannot be parsed."""


def _singular(noun: str) -> str:
    """Normalize a kind noun to a canonical singular key.

    Every nonce kind is an ``...us`` singular pluralized by appending
    ``es`` -- ``brimpus`` -> ``brimpuses``, ``wumpus`` -> ``wumpuses``
    (the sibilant ``-s`` plural rule). Canonicalization therefore strips
    exactly the ``es`` plural suffix after a sibilant, collapsing the
    plural form back onto its ``...us`` singular; a bare singular already
    ends in ``us`` and is left untouched (its lone trailing ``s`` is the
    stem, not a plural marker, so it must *not* be stripped -- doing so
    would split ``jompus`` and ``jompuses`` onto different keys). Applied
    uniformly to every kind token, this is a pure lower-cased
    canonicalization, not a re-derivation of generator state.
    """
    key = noun.strip().lower()
    if key.endswith(("ses", "xes", "zes", "ches", "shes")):
        return key[:-2]
    return key


@dataclass(frozen=True, slots=True)
class _Consequent:
    """A rule/fact consequent: a kind membership or a signed property."""

    is_kind: bool
    name: str
    negated: bool = False


@dataclass(frozen=True, slots=True)
class _Rule:
    """``antecedent`` kind implies ``consequent`` (kind or signed prop)."""

    antecedent: str
    consequent: _Consequent


@dataclass(frozen=True, slots=True)
class _Parsed:
    """The parsed theory: the individual plus its facts and the rules."""

    entity: str
    kinds: frozenset[str]
    props: frozenset[tuple[str, bool]]
    rules: tuple[_Rule, ...]


# --- Surface grammar (matches the upstream ``inflect`` output exactly) -----
# A predicate phrase is: optional "a"/"an" article + optional "not" + the
# target noun/adjective. Article present => the target is a kind; article
# absent => the target is a property (adjective). Negation ("not") only
# ever attaches to a property in this ontology (verified: kind membership
# is never negated), but the parser records polarity uniformly.
_PRED = r"(?:(?P<article>an?)\s+)?(?P<neg>not\s+)?(?P<target>[A-Za-z][\w-]*)"
_FACT_RE = re.compile(rf"^(?P<subj>[A-Z][a-z]+)\s+is\s+{_PRED}$")
_RULE_EACH_RE = re.compile(
    rf"^(?:Every|Each|All)\s+(?P<subj>[A-Za-z][\w-]*)\s+is\s+{_PRED}$",
)
_RULE_PLURAL_RE = re.compile(
    rf"^(?P<subj>[A-Za-z][\w-]*)\s+are\s+{_PRED}$",
)
_QUERY_RE = re.compile(
    rf"^True or false:\s+(?P<subj>[A-Z][a-z]+)\s+is\s+{_PRED}\.?$",
)


def _is_plural_kind(target: str) -> bool:
    """True if ``target`` is an article-less plural kind (``...uses``).

    A membership rule with a plural subject drops the article --
    ``"Tumpuses are wumpuses."`` -- so the consequent kind carries no
    ``a``/``an`` marker. Every nonce kind is an ``...us`` word whose plural
    is ``...uses``; no property adjective ends in ``uses``. That makes the
    ``uses`` suffix an exact discriminator between an article-less kind
    membership and an article-less property assignment.
    """
    return target.lower().endswith("uses")


def _consequent_from_match(m: re.Match[str]) -> _Consequent:
    """Build a consequent from a matched predicate phrase.

    A consequent is a **kind membership** when it carries an ``a``/``an``
    article (a singular kind: ``"is a numpus"``) or is an article-less
    ``...uses`` plural (``"are wumpuses"``); otherwise it is a **property**
    adjective (``"is sour"`` / ``"are wumpuses"`` vs ``"are small"``). A
    kind is canonicalized to its singular key so rule antecedents and fact
    kinds share one namespace.
    """
    negated = m.group("neg") is not None
    target = m.group("target")
    if m.group("article") is not None or _is_plural_kind(target):
        return _Consequent(
            is_kind=True,
            name=_singular(target),
            negated=negated,
        )
    return _Consequent(is_kind=False, name=target.lower(), negated=negated)


def _split_sentences(question: str) -> list[str]:
    """Split the question paragraph into individual sentences.

    Sentences are period-terminated; the upstream surface never embeds a
    period inside a sentence, so a simple split on ``.`` boundaries is
    exact. Empty fragments (trailing period) are dropped.
    """
    return [s.strip() for s in question.split(".") if s.strip()]


def _parse(question: str, query: str) -> tuple[_Parsed, _Consequent]:
    """Parse the public question + query into a theory and a query goal.

    Returns the parsed theory (the individual's asserted kinds/properties
    plus the rule set) and the query's target consequent. Raises
    :class:`OracleError` if the query is malformed or a sentence matches
    no known surface form -- a loud signal rather than a silent drop that
    could mask a parse gap as a ``False`` label.
    """
    qm = _QUERY_RE.match(query.strip())
    if qm is None:
        msg = f"unparsable query: {query!r}"
        raise OracleError(msg)
    entity = qm.group("subj")
    goal = _consequent_from_match(qm)

    kinds: set[str] = set()
    props: set[tuple[str, bool]] = set()
    rules: list[_Rule] = []
    for sentence in _split_sentences(question):
        fact_m = _FACT_RE.match(sentence)
        if fact_m is not None and fact_m.group("subj") in ENTITY_NAMES:
            if fact_m.group("subj") != entity:
                continue
            cons = _consequent_from_match(fact_m)
            if cons.is_kind:
                kinds.add(cons.name)
            else:
                props.add((cons.name, cons.negated))
            continue
        rule_m = _RULE_EACH_RE.match(sentence) or _RULE_PLURAL_RE.match(
            sentence,
        )
        if rule_m is not None:
            antecedent = _singular(rule_m.group("subj"))
            rules.append(
                _Rule(antecedent, _consequent_from_match(rule_m)),
            )
            continue
        msg = f"unparsable sentence: {sentence!r}"
        raise OracleError(msg)
    return (
        _Parsed(
            entity=entity,
            kinds=frozenset(kinds),
            props=frozenset(props),
            rules=tuple(rules),
        ),
        goal,
    )


def _forward_chain(parsed: _Parsed) -> set[tuple[str, bool]]:
    """Run the forward-chaining fixpoint; return the derived property set.

    Seeds the closure with the individual's asserted kinds/properties,
    then fires every applicable rule until no new kind or property is
    added (the least fixpoint). Positive kind-subsumptions grow the kind
    set; a property consequent (possibly negated) is recorded in the
    returned ``{(property, negated)}`` set the query is resolved against.
    """
    kinds = set(parsed.kinds)
    props = set(parsed.props)
    changed = True
    while changed:
        changed = False
        for rule in parsed.rules:
            if rule.antecedent not in kinds:
                continue
            cons = rule.consequent
            if cons.is_kind:
                if cons.name not in kinds:
                    kinds.add(cons.name)
                    changed = True
            else:
                key = (cons.name, cons.negated)
                if key not in props:
                    props.add(key)
                    changed = True
    return props


def entailment_label(question: str, query: str) -> str:
    """Return ``"True"`` or ``"False"`` for the query under the theory.

    The independent re-derivation the generator's stored ``answer`` is
    checked against: parse the public text, forward-chain to a fixpoint,
    and return ``"True"`` iff the query's exact (property, polarity) pair
    is in the closure, else ``"False"``. A property is queried (never a
    kind) in this ontology; a kind-valued query is rejected as malformed.
    """
    parsed, goal = _parse(question, query)
    if goal.is_kind:
        msg = f"query targets a kind, not a property: {query!r}"
        raise OracleError(msg)
    closure = _forward_chain(parsed)
    return "True" if (goal.name, goal.negated) in closure else "False"


# The two C18 probes request either a bare verdict or a reasoned response with
# a terminal verdict. Extraction accepts those two shapes without scanning a
# rationale for a later boolean word that could overwrite the actual verdict.
_VERDICT_RE = re.compile(r"(true|false)\b", re.IGNORECASE)
_FINAL_VERDICT_RE = re.compile(r"(true|false)\.?", re.IGNORECASE)


def extract_verdict(text: str) -> str:
    """Extract the final True/False verdict from a normalized reply.

    Returns the canonical ``"True"`` / ``"False"`` for a verdict-only final
    non-empty line, optionally followed by one terminal period (the ceiling
    response shape), or a verdict at the start of the reply followed by a
    rationale (the naive response shape). Text matching neither protocol is
    returned unchanged so it still fails exact match rather than being
    silently coerced to a label.
    """
    non_empty_lines = [
        line.strip() for line in text.splitlines() if line.strip()
    ]
    if non_empty_lines:
        final_match = _FINAL_VERDICT_RE.fullmatch(non_empty_lines[-1])
        if final_match is not None:
            return final_match.group(1).capitalize()
    leading_match = _VERDICT_RE.match(text.lstrip())
    if leading_match is not None:
        return leading_match.group(1).capitalize()
    return text


def _score_prediction(prediction: str, gold: str) -> int:
    return exact_match(extract_verdict(normalize(prediction)), gold)


def score(prediction: str, question: str, query: str) -> int:
    """Return 1 iff ``prediction`` matches the re-derived label, else 0.

    Shared normalization removes surrounding whitespace or one complete code
    fence. Verdict extraction then canonicalizes a protocol-compliant answer
    before shared exact-match scoring. Unparseable public input scores zero.
    """
    try:
        gold = entailment_label(question, query)
    except OracleError:
        return 0
    return _score_prediction(prediction, gold)


def score_gold(prediction: str, gold: str) -> int:
    """Return the 0/1 score of ``prediction`` against a frozen ``gold``.

    This is the pool-facing scoring path. Use :func:`score` when the caller
    needs to re-derive gold from public question and query text.
    """
    return _score_prediction(prediction, gold)
