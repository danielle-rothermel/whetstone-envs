"""Strict validation checks for C22's serialized gold boundary."""

from __future__ import annotations

import copy
import json
import pickle

import pytest
from pydantic import ValidationError

from whetstone_envs.c22 import oracle
from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.spec import ConstraintSpec

_VALID_DATA: dict[str, object] = {
    "base_task": "Name a color.",
    "constraint_descriptions": ["Answer with exactly 4 words."],
    "instruction_id_list": ["length_constraints:number_words"],
    "kwargs_list": [{"num_words": 4, "relation": "exactly"}],
}


def _gold(data: dict[str, object]) -> str:
    return json.dumps(data)


def _canonical_descriptions(
    instruction_ids: list[str],
    kwargs_list: list[dict[str, object]],
) -> list[str]:
    descriptions: list[str] = []
    for instruction_id, kwargs in zip(
        instruction_ids,
        kwargs_list,
        strict=True,
    ):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](
            instruction_id,
        )
        descriptions.append(instruction.build_description(**kwargs))
    return descriptions


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
        '"constraint_descriptions":["Answer with exactly 4 words."],'
        '"instruction_id_list":["length_constraints:number_words"],'
        '"kwargs_list":[{"num_words":4,"relation":"exactly"}]}'
    )
    assert spec.to_gold() == expected
    assert ConstraintSpec.from_gold(spec.to_gold()).to_gold() == expected


def test_direct_kwargs_mapping_cannot_be_mutated_after_construction() -> None:
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=("Answer with exactly 1 words.",),
        instruction_id_list=("length_constraints:number_words",),
        kwargs_list=({"num_words": 1, "relation": "exactly"},),
    )
    original_gold = spec.to_gold()

    with pytest.raises(TypeError, match="item assignment"):
        spec.kwargs_list[0]["relation"] = "at least"  # ty: ignore[invalid-assignment]

    assert spec.to_gold() == original_gold
    assert oracle.check(spec, "blue").score == 1
    assert oracle.check(spec, "light blue").score == 0


def test_nested_kwargs_collection_cannot_be_mutated_after_gold_parse() -> None:
    spec = ConstraintSpec.from_gold(
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": [
                    "Include keywords ['ok'] in the response.",
                ],
                "instruction_id_list": ["keywords:existence"],
                "kwargs_list": [{"keywords": ["ok"]}],
            },
        ),
    )
    original_gold = spec.to_gold()
    keywords = spec.kwargs_list[0]["keywords"]
    assert isinstance(keywords, tuple)

    with pytest.raises(TypeError, match="item assignment"):
        keywords[0] = "["  # ty: ignore[invalid-assignment]

    assert spec.to_gold() == original_gold
    assert oracle.check(spec, "ok").score == 1


def test_construction_detaches_nested_kwargs_from_mutable_input() -> None:
    kwargs: dict[str, object] = {"keywords": ["ok"]}
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=(
            "Include keywords ['ok'] in the response.",
        ),
        instruction_id_list=("keywords:existence",),
        kwargs_list=(kwargs,),
    )
    mutable_keywords = kwargs["keywords"]
    assert isinstance(mutable_keywords, list)
    mutable_keywords[0] = "["

    assert spec.to_gold().endswith('"kwargs_list":[{"keywords":["ok"]}]}')
    assert oracle.check(spec, "ok").score == 1


def test_deep_model_copy_preserves_immutable_validated_gold() -> None:
    spec = ConstraintSpec.from_gold(
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": [
                    "Include keywords ['ok'] in the response.",
                ],
                "instruction_id_list": ["keywords:existence"],
                "kwargs_list": [{"keywords": ["ok"]}],
            },
        ),
    )
    copied = spec.model_copy(deep=True)
    assert copied is not spec
    assert copied.to_gold() == spec.to_gold()

    copied_keywords = copied.kwargs_list[0]["keywords"]
    assert isinstance(copied_keywords, tuple)
    with pytest.raises(TypeError, match="item assignment"):
        copied_keywords[0] = "["  # ty: ignore[invalid-assignment]

    assert copied.to_gold() == spec.to_gold()
    assert oracle.check(copied, "ok").score == 1


def test_pickle_round_trip_preserves_immutable_validated_gold() -> None:
    spec = ConstraintSpec.from_gold(
        _gold(
            {
                "base_task": "Name a color.",
                "constraint_descriptions": [
                    "Include keywords ['ok'] in the response.",
                ],
                "instruction_id_list": ["keywords:existence"],
                "kwargs_list": [{"keywords": ["ok"]}],
            },
        ),
    )
    restored = pickle.loads(  # noqa: S301 - trusted in-memory fixture
        pickle.dumps(spec),
    )
    assert isinstance(restored, ConstraintSpec)
    assert restored.to_gold() == spec.to_gold()

    restored_keywords = restored.kwargs_list[0]["keywords"]
    assert isinstance(restored_keywords, tuple)
    with pytest.raises(TypeError, match="item assignment"):
        restored_keywords[0] = "["

    assert restored.to_gold() == spec.to_gold()
    assert oracle.check(restored, "ok").score == 1


def test_standard_deepcopy_preserves_immutable_validated_gold() -> None:
    spec = ConstraintSpec.from_gold(_gold(_VALID_DATA))
    copied = copy.deepcopy(spec)
    assert copied is not spec
    assert copied.to_gold() == spec.to_gold()


def test_prompt_description_must_match_scored_kwargs() -> None:
    data = dict(_VALID_DATA)
    data["constraint_descriptions"] = ["Answer with exactly 100 words."]
    with pytest.raises(
        ValidationError,
        match="canonical vendored description",
    ):
        ConstraintSpec.from_gold(_gold(data))


@pytest.mark.parametrize(
    ("instruction_id", "kwargs"),
    [
        ("keywords:existence", {"keywords": ["["]}),
        ("keywords:forbidden_words", {"forbidden_words": ["("]}),
        ("detectable_content:postscript", {"postscript_marker": "["}),
    ],
)
def test_regex_bearing_kwargs_reject_unsafe_patterns(
    instruction_id: str,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        ValidationError,
        match="without regex metacharacters",
    ):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=("invalid regex",),
            instruction_id_list=(instruction_id,),
            kwargs_list=(kwargs,),
        )


def test_score_gold_rejects_invalid_regex_before_scoring() -> None:
    invalid_gold = _gold(
        {
            "base_task": "Name a color.",
            "constraint_descriptions": [
                "Include keywords ['['] in the response.",
            ],
            "instruction_id_list": ["keywords:existence"],
            "kwargs_list": [{"keywords": ["["]}],
        },
    )
    with pytest.raises(ValidationError, match="regex metacharacters"):
        oracle.score_gold(invalid_gold, "response")


@pytest.mark.parametrize(
    ("required_id", "required_kwargs"),
    [
        ("startend:end_checker", {"end_phrase": "BLOCKED"}),
        (
            "detectable_content:postscript",
            {"postscript_marker": "BLOCKED"},
        ),
    ],
)
def test_required_literal_cannot_also_be_a_forbidden_word(
    required_id: str,
    required_kwargs: dict[str, object],
) -> None:
    instruction_ids = [required_id, "keywords:forbidden_words"]
    kwargs_list: list[dict[str, object]] = [
        required_kwargs,
        {"forbidden_words": ["BLOCKED"]},
    ]
    with pytest.raises(ValidationError, match="contains forbidden word"):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=tuple(
                _canonical_descriptions(instruction_ids, kwargs_list),
            ),
            instruction_id_list=tuple(instruction_ids),
            kwargs_list=tuple(kwargs_list),
        )


def test_direct_required_end_cannot_conflict_with_no_comma() -> None:
    instruction_ids = ["startend:end_checker", "punctuation:no_comma"]
    kwargs_list: list[dict[str, object]] = [
        {"end_phrase": "DONE, NOW"},
        {},
    ]
    with pytest.raises(ValidationError, match=r"comma.*no_comma"):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=tuple(
                _canonical_descriptions(instruction_ids, kwargs_list),
            ),
            instruction_id_list=tuple(instruction_ids),
            kwargs_list=tuple(kwargs_list),
        )


@pytest.mark.parametrize(
    ("required_id", "required_kwargs"),
    [
        (
            "detectable_content:postscript",
            {"postscript_marker": "POST,SCRIPT"},
        ),
    ],
)
def test_gold_required_literal_cannot_conflict_with_no_comma(
    required_id: str,
    required_kwargs: dict[str, object],
) -> None:
    instruction_ids = [required_id, "punctuation:no_comma"]
    kwargs_list: list[dict[str, object]] = [required_kwargs, {}]
    data: dict[str, object] = {
        "base_task": "Name a color.",
        "constraint_descriptions": _canonical_descriptions(
            instruction_ids,
            kwargs_list,
        ),
        "instruction_id_list": instruction_ids,
        "kwargs_list": kwargs_list,
    }
    with pytest.raises(ValidationError, match=r"comma.*no_comma"):
        ConstraintSpec.from_gold(_gold(data))


def test_multi_keyword_exact_budget_is_outside_c22_contract() -> None:
    instruction_ids = [
        "length_constraints:number_words",
        "keywords:existence",
    ]
    kwargs_list: list[dict[str, object]] = [
        {"num_words": 2, "relation": "exactly"},
        {"keywords": ["a b", "c d"]},
    ]
    with pytest.raises(
        ValidationError,
        match="exactly one single-token literal",
    ):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=tuple(
                _canonical_descriptions(instruction_ids, kwargs_list),
            ),
            instruction_id_list=tuple(instruction_ids),
            kwargs_list=tuple(kwargs_list),
        )


@pytest.mark.parametrize(
    ("keyword", "response"),
    [("café", "CAFÉ"), ("猫", "猫")],
)
def test_safe_unicode_single_token_keyword_remains_supported(
    keyword: str,
    response: str,
) -> None:
    kwargs: dict[str, object] = {"keywords": [keyword]}
    instruction_ids = ["keywords:existence"]
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=tuple(
            _canonical_descriptions(instruction_ids, [kwargs]),
        ),
        instruction_id_list=tuple(instruction_ids),
        kwargs_list=(kwargs,),
    )
    assert oracle.check(spec, response).score == 1


@pytest.mark.parametrize("forbidden_word", ["a,", "-a"])
def test_serialized_forbidden_word_rejects_edge_punctuation(
    forbidden_word: str,
) -> None:
    instruction_ids = ["keywords:forbidden_words"]
    kwargs_list: list[dict[str, object]] = [
        {"forbidden_words": [forbidden_word]},
    ]
    data: dict[str, object] = {
        "base_task": "Name a color.",
        "constraint_descriptions": _canonical_descriptions(
            instruction_ids,
            kwargs_list,
        ),
        "instruction_id_list": instruction_ids,
        "kwargs_list": kwargs_list,
    }

    with pytest.raises(ValidationError, match="full Unicode word token"):
        ConstraintSpec.from_gold(_gold(data))


def test_serialized_required_keyword_rejects_outer_whitespace() -> None:
    instruction_ids = [
        "length_constraints:number_words",
        "keywords:existence",
    ]
    kwargs_list: list[dict[str, object]] = [
        {"num_words": 1, "relation": "exactly"},
        {"keywords": [" a "]},
    ]
    data: dict[str, object] = {
        "base_task": "Name a color.",
        "constraint_descriptions": _canonical_descriptions(
            instruction_ids,
            kwargs_list,
        ),
        "instruction_id_list": instruction_ids,
        "kwargs_list": kwargs_list,
    }

    with pytest.raises(ValidationError, match="full Unicode word token"):
        ConstraintSpec.from_gold(_gold(data))


def test_valid_forbidden_word_is_detected_by_oracle() -> None:
    instruction_ids = ["keywords:forbidden_words"]
    kwargs: dict[str, object] = {"forbidden_words": ["blocked"]}
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=tuple(
            _canonical_descriptions(instruction_ids, [kwargs]),
        ),
        instruction_id_list=tuple(instruction_ids),
        kwargs_list=(kwargs,),
    )

    assert oracle.check(spec, "This is BLOCKED here.").score == 0
    assert oracle.check(spec, "This is allowed.").score == 1


@pytest.mark.parametrize(
    ("marker", "end_phrase", "keyword"),
    [
        ("P.P.S", "END OF ANSWER", None),
        ("P.S.", "END OF ANSWER", "wexcorb"),
    ],
)
def test_exact_budget_rejects_proven_impossible_required_literals(
    marker: str,
    end_phrase: str,
    keyword: str | None,
) -> None:
    instruction_ids = [
        "length_constraints:number_words",
        "detectable_content:postscript",
        "startend:end_checker",
    ]
    kwargs_list: list[dict[str, object]] = [
        {"num_words": 4, "relation": "exactly"},
        {"postscript_marker": marker},
        {"end_phrase": end_phrase},
    ]
    if keyword is not None:
        instruction_ids.append("keywords:existence")
        kwargs_list.append({"keywords": [keyword]})

    with pytest.raises(
        ValidationError,
        match="mandatory required-literal lower bound 5",
    ):
        ConstraintSpec(
            base_task="Name a color.",
            constraint_descriptions=tuple(
                _canonical_descriptions(instruction_ids, kwargs_list),
            ),
            instruction_id_list=tuple(instruction_ids),
            kwargs_list=tuple(kwargs_list),
        )


def test_exact_budget_allows_checker_valid_literal_overlap() -> None:
    instruction_ids = [
        "length_constraints:number_words",
        "detectable_content:postscript",
        "startend:end_checker",
        "keywords:existence",
    ]
    kwargs_list: list[dict[str, object]] = [
        {"num_words": 4, "relation": "exactly"},
        {"postscript_marker": "P.S."},
        {"end_phrase": "SIGNED OFF"},
        {"keywords": ["jaxbryn"]},
    ]
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=tuple(
            _canonical_descriptions(instruction_ids, kwargs_list),
        ),
        instruction_id_list=tuple(instruction_ids),
        kwargs_list=tuple(kwargs_list),
    )
    assert oracle.check(spec, "P.S.jaxbrynSIGNED OFF").score == 1
