from __future__ import annotations

import re
from dataclasses import dataclass

from whetstone_envs.probes import normalize
from whetstone_envs.scoring import exact_match

# Pinned upstream names distinguish ground facts from universal rules.
ENTITY_NAMES: frozenset[str] = frozenset(
    {"Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"},
)


class OracleError(ValueError):
    """Raised when the public question/query text cannot be parsed."""


def _singular(noun: str) -> str:
    """Map PrOntoQA's ``...uses`` plurals to ``...us`` keys."""
    key = noun.strip().lower()
    if key.endswith(("ses", "xes", "zes", "ches", "shes")):
        return key[:-2]
    return key


@dataclass(frozen=True, slots=True)
class _Consequent:
    is_kind: bool
    name: str
    negated: bool = False


@dataclass(frozen=True, slots=True)
class _Rule:
    antecedent: str
    consequent: _Consequent


@dataclass(frozen=True, slots=True)
class _Parsed:
    entity: str
    kinds: frozenset[str]
    props: frozenset[tuple[str, bool]]
    rules: tuple[_Rule, ...]


# An article marks a kind; an article-free adjective marks a property.
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
    """Recognize PrOntoQA's article-free plural kind form."""
    return target.lower().endswith("uses")


def _consequent_from_match(m: re.Match[str]) -> _Consequent:
    """Build a kind membership or signed property from a surface match."""
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
    return [s.strip() for s in question.split(".") if s.strip()]


def _parse(question: str, query: str) -> tuple[_Parsed, _Consequent]:
    """Parse public text, rejecting every unknown surface form."""
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
    if not kinds and not props:
        msg = f"question has no ground fact for query entity {entity!r}"
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
    """Return signed properties in the least forward-chaining fixpoint."""
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
    """Derive the query label independently from public text."""
    parsed, goal = _parse(question, query)
    if goal.is_kind:
        msg = f"query targets a kind, not a property: {query!r}"
        raise OracleError(msg)
    closure = _forward_chain(parsed)
    return "True" if (goal.name, goal.negated) in closure else "False"


# The two C18 probes request either a bare verdict or a reasoned response with
# a terminal verdict. Extraction accepts only those unambiguous shapes.
_FINAL_VERDICT_RE = re.compile(r"(true|false)\.?", re.IGNORECASE)


def extract_verdict(text: str) -> str:
    """Canonicalize a protocol verdict, leaving ambiguous text unchanged."""
    non_empty_lines = [
        line.strip() for line in text.splitlines() if line.strip()
    ]
    if non_empty_lines:
        final_match = _FINAL_VERDICT_RE.fullmatch(non_empty_lines[-1])
        if final_match is not None:
            return final_match.group(1).capitalize()
    return text


def _score_prediction(prediction: str, gold: str) -> int:
    return exact_match(extract_verdict(normalize(prediction)), gold)


def score(prediction: str, question: str, query: str) -> int:
    """Score against public-text gold; unparsable input scores zero."""
    try:
        gold = entailment_label(question, query)
    except OracleError:
        return 0
    return _score_prediction(prediction, gold)


def score_gold(prediction: str, gold: str) -> int:
    """Score a prediction against frozen pool gold."""
    return _score_prediction(prediction, gold)
