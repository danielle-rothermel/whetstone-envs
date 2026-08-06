from __future__ import annotations

import json

import pytest
from dr_serialize import DuplicateJsonKeyError
from pydantic import ValidationError

from whetstone_envs.c22.constraints import (
    ConstraintStack,
    EndPhrase,
    ExactWordCount,
    ForbiddenLetter,
    ForbiddenWord,
    NoComma,
    RequiredKeyword,
    Title,
)


def test_gold_round_trips_as_the_closed_composition() -> None:
    stack = ConstraintStack(
        constraints=(RequiredKeyword(keyword="café"), NoComma())
    )
    gold = stack.to_gold()
    assert ConstraintStack.from_gold(gold) == stack
    assert json.loads(gold) == stack.model_dump(mode="json")


@pytest.mark.parametrize(
    "constraints",
    [
        (),
        (Title(), Title()),
        (
            RequiredKeyword(keyword="quarnex"),
            ForbiddenWord(word="quarnex"),
        ),
        (EndPhrase(phrase="DONE,"), NoComma()),
        (
            RequiredKeyword(keyword="zylthorn"),
            ForbiddenLetter(letter="z"),
        ),
        (ExactWordCount(count=1), EndPhrase(phrase="END OF ANSWER")),
    ],
)
def test_contradictory_or_empty_stacks_are_rejected(
    constraints: tuple[object, ...],
) -> None:
    with pytest.raises(ValidationError):
        ConstraintStack(constraints=constraints)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("word", ["", " word", "word ", "a,", "a-"])
def test_keywords_are_full_tokens(word: str) -> None:
    with pytest.raises(ValidationError):
        RequiredKeyword(keyword=word)
    with pytest.raises(ValidationError):
        ForbiddenWord(word=word)


@pytest.mark.parametrize("letter", ["", "ab", "\u017f", "猫", "1"])
def test_forbidden_letters_are_ascii(letter: str) -> None:
    with pytest.raises(ValidationError):
        ForbiddenLetter(letter=letter)


@pytest.mark.parametrize(
    ("gold", "error"),
    [
        (
            '{"constraints":[{"kind":"no_comma","kind":"no_comma"}]}',
            DuplicateJsonKeyError,
        ),
        (
            '{"constraints":[{"kind":"no_comma","extra":true}]}',
            ValidationError,
        ),
        (
            '{"constraints":[{"kind":"semantic_task"}]}',
            ValidationError,
        ),
    ],
)
def test_gold_rejects_nonclosed_shapes(
    gold: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ConstraintStack.from_gold(gold)
