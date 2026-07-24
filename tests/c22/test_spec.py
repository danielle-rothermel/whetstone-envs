"""Strict validation checks for C22's serialized gold boundary."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from whetstone_envs.c22.spec import ConstraintSpec

_VALID_DATA: dict[str, object] = {
    "base_task": "Name a color.",
    "constraint_descriptions": ["Use exactly four words."],
    "instruction_id_list": ["length_constraints:number_words"],
    "kwargs_list": [{"num_words": 4, "relation": "exactly"}],
}


def _gold(data: dict[str, object]) -> str:
    return json.dumps(data)


@pytest.mark.parametrize(
    "gold",
    [
        "{",
        _gold({**_VALID_DATA, "unknown": True}),
        _gold({**_VALID_DATA, "base_task": 7}),
        _gold({**_VALID_DATA, "base_task": ""}),
        _gold({**_VALID_DATA, "constraint_descriptions": []}),
        _gold(
            {
                **_VALID_DATA,
                "instruction_id_list": ["unknown:atom"],
            },
        ),
        _gold(
            {
                **_VALID_DATA,
                "constraint_descriptions": ["one", "two"],
            },
        ),
        _gold(
            {
                **_VALID_DATA,
                "kwargs_list": [{"num_words": "4", "relation": "exactly"}],
            },
        ),
        _gold(
            {
                **_VALID_DATA,
                "kwargs_list": [
                    {
                        "num_words": 4,
                        "relation": "exactly",
                        "unexpected": 1,
                    },
                ],
            },
        ),
        _gold(
            {
                **_VALID_DATA,
                "kwargs_list": [{"num_words": 4, "relation": "at least"}],
            },
        ),
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": ["required", "forbidden"],
                "instruction_id_list": [
                    "keywords:existence",
                    "keywords:forbidden_words",
                ],
                "kwargs_list": [
                    {"keywords": ["zylthorn"]},
                    {"forbidden_words": ["zylthorn"]},
                ],
            },
        ),
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": ["required", "letter"],
                "instruction_id_list": [
                    "startend:end_checker",
                    "keywords:letter_frequency",
                ],
                "kwargs_list": [
                    {"end_phrase": "QUIET END"},
                    {
                        "letter": "q",
                        "let_frequency": 1,
                        "let_relation": "less than",
                    },
                ],
            },
        ),
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": ["title", "quotation"],
                "instruction_id_list": [
                    "detectable_format:title",
                    "startend:quotation",
                ],
                "kwargs_list": [{}, {}],
            },
        ),
    ],
    ids=[
        "malformed-json",
        "unknown-field",
        "wrong-type",
        "empty-base-task",
        "empty-stack",
        "unknown-id",
        "mismatched-stack",
        "coercible-kwarg",
        "unexpected-kwarg",
        "non-c22-word-relation",
        "required-and-forbidden-keyword",
        "required-literal-forbidden-letter",
        "registry-conflict",
    ],
)
def test_invalid_gold_fails_with_pydantic_validation(gold: str) -> None:
    with pytest.raises(ValidationError):
        ConstraintSpec.from_gold(gold)


def test_non_json_compatible_direct_kwarg_fails_validation() -> None:
    with pytest.raises(ValidationError, match="non-JSON-compatible"):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=("required",),
            instruction_id_list=("keywords:existence",),
            kwargs_list=({"keywords": [{"not-json"}]},),
        )


def test_gold_serialization_is_canonical_and_deterministic() -> None:
    spec = ConstraintSpec.from_gold(_gold(_VALID_DATA))
    expected = (
        '{"base_task":"Name a color.",'
        '"constraint_descriptions":["Use exactly four words."],'
        '"instruction_id_list":["length_constraints:number_words"],'
        '"kwargs_list":[{"num_words":4,"relation":"exactly"}]}'
    )
    assert spec.to_gold() == expected
    assert ConstraintSpec.from_gold(spec.to_gold()).to_gold() == expected
