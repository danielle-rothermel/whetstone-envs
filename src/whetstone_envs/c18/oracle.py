r"""The from-scratch forward-chaining fixpoint ORACLE for c18.

A pure function of an instance's **public** fields -- the concatenated
``question`` (facts + if-then rules) and the ``query`` statement -- and
nothing else. It never consults the vendored generator's internal
``Theory`` / proof object or its RNG, so it cannot silently become a
re-derivation of how the generator built the instance (rubric criteria 2,
8, 11). It re-derives the entailment label from the surface text alone.

The independence matters here more than for any other candidate: the
generator's stored ``answer`` is *definitional* -- it comes from a 50 %
negation flag, not from an independent prover (the repos review's red
flag #5). A generation-soundness bug would be invisible in ``answer``
alone. This oracle is the independent verdict that catches it, so the
generator asserts oracle==answer at construction (see
:mod:`whetstone_envs.c18.generate`).

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
from dataclasses import dataclass, field

from whetstone_envs.core.probes import normalize

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


@dataclass
class _Parsed:
    """The parsed theory: the individual plus its facts and the rules."""

    entity: str
    kinds: set[str] = field(default_factory=set)
    props: set[tuple[str, bool]] = field(default_factory=set)
    rules: list[_Rule] = field(default_factory=list)


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

    parsed = _Parsed(entity=entity)
    for sentence in _split_sentences(question):
        fact_m = _FACT_RE.match(sentence)
        if fact_m is not None and fact_m.group("subj") in ENTITY_NAMES:
            if fact_m.group("subj") != entity:
                continue
            cons = _consequent_from_match(fact_m)
            if cons.is_kind:
                parsed.kinds.add(cons.name)
            else:
                parsed.props.add((cons.name, cons.negated))
            continue
        rule_m = _RULE_EACH_RE.match(sentence) or _RULE_PLURAL_RE.match(
            sentence,
        )
        if rule_m is not None:
            antecedent = _singular(rule_m.group("subj"))
            parsed.rules.append(
                _Rule(antecedent, _consequent_from_match(rule_m)),
            )
            continue
        msg = f"unparsable sentence: {sentence!r}"
        raise OracleError(msg)
    return parsed, goal


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


# --- Verdict extraction (baseline spec Section 3 decision rule) ------------
# The spec's decision rule scores a single True/False verdict by exact
# match, and the ceiling probe (Section 2.2) instructs the model to "end
# your reply with exactly one word on its own final line: either True or
# False". A chain-of-thought reply therefore ends with the verdict but is
# many lines long, and even the naive probe may append a trailing rationale
# clause. Matching the *whole* reply against the gold token scores every
# such well-formed reply 0 (observed live: a CoT reply ending
# "...not entailed.\n\nFalse" with gold "False" scored 0). This extractor
# therefore accepts the two response shapes requested by the probes: an
# exact verdict on the final non-empty line (ceiling) or a leading verdict
# followed by a rationale (naive). It never scans rationale text for a
# later boolean word that could overwrite the response's actual verdict.
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


def score(prediction: str, question: str, query: str) -> int:
    """Return 1 iff ``prediction`` matches the re-derived label, else 0.

    The model's ``prediction`` is passed through the shared
    :func:`whetstone_envs.core.probes.normalize` (strip surrounding
    whitespace / one code fence) and then :func:`extract_verdict` (the
    spec Section 3 verdict extraction: recover the protocol verdict from
    a chain-of-thought or rationale-trailing reply) before a
    case-insensitive exact-match compare against the freshly re-derived
    gold -- no partial credit (rubric criterion 2). A question/query the
    oracle cannot parse scores ``0`` rather than raising: a model response
    is graded, not trusted.
    """
    try:
        gold = entailment_label(question, query)
    except OracleError:
        return 0
    predicted = extract_verdict(normalize(prediction))
    return int(predicted.lower() == normalize(gold).lower())


def score_gold(prediction: str, gold: str) -> int:
    """Return the 0/1 score of ``prediction`` against a frozen ``gold``.

    The pool-facing entry point mirroring the other candidates'
    ``score_gold``: given an instance's already-derived ``gold`` label and
    a model response, return 0 or 1. The prediction is shared-normalized
    and then run through :func:`extract_verdict` (spec Section 3) so the
    protocol verdict in a chain-of-thought or rationale-trailing reply is
    scored instead of its whole text. Use :func:`score` instead to re-derive
    straight from the public question + query.
    """
    predicted = extract_verdict(normalize(prediction))
    return int(predicted.lower() == normalize(gold).lower())
