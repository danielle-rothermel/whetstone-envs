from __future__ import annotations

import pytest

from whetstone_envs.c22 import score
from whetstone_envs.c22.constraints import (
    Constraint,
    ConstraintStack,
    EndPhrase,
    ExactWordCount,
    ForbiddenLetter,
    ForbiddenWord,
    HighlightedSections,
    NoComma,
    Placeholders,
    Postscript,
    Quotation,
    RequiredKeyword,
    Title,
)
from whetstone_envs.c22.oracle import evaluate

_FIXTURES: tuple[tuple[Constraint, str, str], ...] = (
    (
        RequiredKeyword(keyword="zylthorn"),
        "the answer mentions zylthorn",
        "the answer omits it",
    ),
    (EndPhrase(phrase="DONE"), "some words DONE", "some words"),
    (Title(), "here is <<A Real Title>>", "a plain answer"),
    (Quotation(), '"the whole response"', "not quoted"),
    (NoComma(), "no commas here", "this one, has a comma"),
    (Placeholders(count=2), "fill [name] and [date]", "fill [name]"),
    (Postscript(marker="P.S."), "main text P.S. after", "main text"),
    (
        HighlightedSections(count=1),
        "an *emphasized* fragment",
        "plain text",
    ),
    (ExactWordCount(count=4), "exactly four words here", "only three words"),
    (
        ForbiddenLetter(letter="z"),
        "the letter is absent",
        "a buzzing haze",
    ),
    (
        ForbiddenWord(word="quarnex"),
        "a clean response",
        "this mentions quarnex",
    ),
)


@pytest.mark.parametrize(
    ("constraint", "passing", "failing"),
    _FIXTURES,
    ids=[constraint.kind for constraint, _, _ in _FIXTURES],
)
def test_every_supported_constraint_has_an_independent_pass_fail_fixture(
    constraint: Constraint,
    passing: str,
    failing: str,
) -> None:
    stack = ConstraintStack(constraints=(constraint,))
    assert evaluate(stack, passing).score == 1
    assert evaluate(stack, failing).score == 0


def test_strict_all_pass_and_diagnostics_follow_stack_order() -> None:
    stack = ConstraintStack(
        constraints=(
            RequiredKeyword(keyword="zylthorn"),
            NoComma(),
            EndPhrase(phrase="DONE"),
        )
    )
    passing = evaluate(stack, "zylthorn blue DONE")
    assert passing.follow_all
    assert passing.per_constraint == (
        ("required_keyword", True),
        ("no_comma", True),
        ("end_phrase", True),
    )

    failing = evaluate(stack, "zylthorn blue")
    assert not failing.follow_all
    assert failing.per_constraint[-1] == ("end_phrase", False)


def test_score_parses_gold_and_normalizes_outer_fences() -> None:
    stack = ConstraintStack(constraints=(EndPhrase(phrase="END"),))
    assert score(stack.to_gold(), "```\nfinal words END\n```") == 1


def test_exact_word_count_rejects_both_adjacent_counts() -> None:
    stack = ConstraintStack(constraints=(ExactWordCount(count=4),))
    assert evaluate(stack, "one two three four").score == 1
    assert evaluate(stack, "one two three").score == 0
    assert evaluate(stack, "one two three four five").score == 0


def test_evaluate_rejects_non_string_responses() -> None:
    stack = ConstraintStack(constraints=(NoComma(),))
    with pytest.raises(TypeError, match="response"):
        evaluate(stack, None)  # ty: ignore[invalid-argument-type]
