from __future__ import annotations

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


def test_gold_is_canonical_closed_and_composed() -> None:
    stack = ConstraintStack(
        constraints=(RequiredKeyword(keyword="café"), NoComma())
    )
    gold = stack.to_gold()
    assert gold == (
        '{"constraints":[{"keyword":"caf\\u00e9",'
        '"kind":"required_keyword"},{"kind":"no_comma"}]}'
    )
    assert ConstraintStack.from_gold(gold) == stack
    assert "description" not in gold
    assert "kwargs" not in gold


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


def test_gold_rejects_duplicate_keys_and_unknown_fields() -> None:
    duplicate = '{"constraints":[{"kind":"no_comma","kind":"no_comma"}]}'
    with pytest.raises(DuplicateJsonKeyError):
        ConstraintStack.from_gold(duplicate)

    unknown = '{"constraints":[{"kind":"no_comma","extra":true}]}'
    with pytest.raises(ValidationError):
        ConstraintStack.from_gold(unknown)


def test_gold_rejects_unknown_constraint_kind() -> None:
    with pytest.raises(ValidationError):
        ConstraintStack.from_gold('{"constraints":[{"kind":"semantic_task"}]}')
