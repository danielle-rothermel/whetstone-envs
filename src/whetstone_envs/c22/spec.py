"""Validated serialized constraint stacks shared by generation and scoring."""

from __future__ import annotations

import json
import math
import string
from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.atoms import all_atom_ids

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ALLOWED_ATOM_IDS = frozenset(all_atom_ids())
_NO_KWARGS = frozenset(
    {
        "detectable_format:title",
        "startend:quotation",
        "punctuation:no_comma",
    },
)


def _require_exact_keys(
    instruction_id: str,
    kwargs: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(kwargs)
    if actual != expected:
        msg = (
            f"{instruction_id} kwargs must contain exactly "
            f"{sorted(expected)!r}; got {sorted(actual)!r}"
        )
        raise ValueError(msg)


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise ValueError(msg)
    return value


def _require_positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)
    return value


def _require_nonempty_string_list(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    if not isinstance(value, list) or not value:
        msg = f"{field_name} must be a nonempty JSON array of strings"
        raise ValueError(msg)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        msg = f"{field_name} must contain only nonempty strings"
        raise ValueError(msg)
    return cast("list[str]", value)


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        msg = f"{path} contains a non-finite float"
        raise ValueError(msg)
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path} contains a non-string object key"
                raise ValueError(msg)  # noqa: TRY004 - Pydantic wraps it
            _validate_json_value(item, path=f"{path}.{key}")
        return
    msg = f"{path} contains non-JSON-compatible value {type(value).__name__}"
    raise ValueError(msg)


def _validate_atom_kwargs(  # noqa: PLR0912
    instruction_id: str,
    kwargs: Mapping[str, object],
) -> None:
    if instruction_id in _NO_KWARGS:
        _require_exact_keys(instruction_id, kwargs, set())
        return

    if instruction_id == "keywords:existence":
        _require_exact_keys(instruction_id, kwargs, {"keywords"})
        _require_nonempty_string_list(
            kwargs["keywords"],
            field_name=f"{instruction_id}.keywords",
        )
    elif instruction_id == "keywords:forbidden_words":
        _require_exact_keys(instruction_id, kwargs, {"forbidden_words"})
        _require_nonempty_string_list(
            kwargs["forbidden_words"],
            field_name=f"{instruction_id}.forbidden_words",
        )
    elif instruction_id == "startend:end_checker":
        _require_exact_keys(instruction_id, kwargs, {"end_phrase"})
        _require_nonempty_string(
            kwargs["end_phrase"],
            field_name=f"{instruction_id}.end_phrase",
        )
    elif instruction_id == "detectable_content:postscript":
        _require_exact_keys(instruction_id, kwargs, {"postscript_marker"})
        _require_nonempty_string(
            kwargs["postscript_marker"],
            field_name=f"{instruction_id}.postscript_marker",
        )
    elif instruction_id == "detectable_content:number_placeholders":
        _require_exact_keys(instruction_id, kwargs, {"num_placeholders"})
        _require_positive_integer(
            kwargs["num_placeholders"],
            field_name=f"{instruction_id}.num_placeholders",
        )
    elif instruction_id == "detectable_format:number_highlighted_sections":
        _require_exact_keys(instruction_id, kwargs, {"num_highlights"})
        _require_positive_integer(
            kwargs["num_highlights"],
            field_name=f"{instruction_id}.num_highlights",
        )
    elif instruction_id == "length_constraints:number_words":
        _require_exact_keys(
            instruction_id,
            kwargs,
            {"num_words", "relation"},
        )
        _require_positive_integer(
            kwargs["num_words"],
            field_name=f"{instruction_id}.num_words",
        )
        if kwargs["relation"] != "exactly":
            msg = (
                "length_constraints:number_words.relation must be "
                "'exactly' for C22"
            )
            raise ValueError(msg)
    elif instruction_id == "keywords:letter_frequency":
        _require_exact_keys(
            instruction_id,
            kwargs,
            {"letter", "let_frequency", "let_relation"},
        )
        letter = kwargs["letter"]
        if (
            not isinstance(letter, str)
            or len(letter) != 1
            or letter.casefold() not in string.ascii_lowercase
        ):
            msg = (
                "keywords:letter_frequency.letter must be one ASCII letter"
            )
            raise ValueError(msg)
        if kwargs["let_frequency"] != 1 or isinstance(
            kwargs["let_frequency"],
            bool,
        ):
            msg = "C22 letter frequency must be the forbidden-letter form"
            raise ValueError(msg)
        if kwargs["let_relation"] != "less than":
            msg = "C22 letter relation must be 'less than'"
            raise ValueError(msg)
    else:  # pragma: no cover - instruction ids are checked first
        msg = f"unsupported C22 instruction id {instruction_id!r}"
        raise ValueError(msg)


def compatibility_error(  # noqa: PLR0912
    instruction_ids: Sequence[str],
    kwargs_list: Sequence[Mapping[str, object]],
) -> str | None:
    """Return why a resolved stack is contradictory, if it is."""
    conflicts = instructions_registry.INSTRUCTION_CONFLICTS
    for index, instruction_id in enumerate(instruction_ids):
        for other in instruction_ids[index + 1 :]:
            if (
                other in conflicts.get(instruction_id, set())
                or instruction_id in conflicts.get(other, set())
            ):
                return f"{instruction_id!r} conflicts with {other!r}"

    required_keywords: list[str] = []
    required_literals: list[str] = []
    forbidden_words: list[str] = []
    forbidden_letters: list[str] = []
    for instruction_id, kwargs in zip(
        instruction_ids,
        kwargs_list,
        strict=True,
    ):
        if instruction_id == "keywords:existence":
            keywords = cast("list[str]", kwargs["keywords"])
            required_keywords.extend(keywords)
            required_literals.extend(keywords)
        elif instruction_id == "startend:end_checker":
            end_phrase = kwargs["end_phrase"]
            assert isinstance(end_phrase, str)
            required_literals.append(end_phrase)
        elif instruction_id == "detectable_content:postscript":
            marker = kwargs["postscript_marker"]
            assert isinstance(marker, str)
            required_literals.append(marker)
        elif instruction_id == "keywords:forbidden_words":
            words = cast("list[str]", kwargs["forbidden_words"])
            forbidden_words.extend(words)
        elif instruction_id == "keywords:letter_frequency":
            letter = kwargs["letter"]
            assert isinstance(letter, str)
            forbidden_letters.append(letter)

    forbidden_words_folded = {word.casefold() for word in forbidden_words}
    for keyword in required_keywords:
        if keyword.casefold() in forbidden_words_folded:
            return f"required keyword {keyword!r} is also forbidden"

    for literal in required_literals:
        for letter in forbidden_letters:
            if letter.casefold() in literal.casefold():
                return (
                    f"required literal {literal!r} contains forbidden "
                    f"letter {letter!r}"
                )
    return None


class ConstraintSpec(BaseModel):
    """The strict persistence boundary for one C22 constraint stack."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    base_task: str
    constraint_descriptions: tuple[str, ...]
    instruction_id_list: tuple[str, ...]
    kwargs_list: tuple[dict[str, object], ...]

    @field_validator("base_task")
    @classmethod
    def _validate_base_task(cls, value: str) -> str:
        return _require_nonempty_string(value, field_name="base_task")

    @field_validator("constraint_descriptions")
    @classmethod
    def _validate_descriptions(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            msg = "constraint_descriptions must not be empty"
            raise ValueError(msg)
        for description in value:
            _require_nonempty_string(
                description,
                field_name="constraint_descriptions item",
            )
        return value

    @field_validator("instruction_id_list")
    @classmethod
    def _validate_instruction_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            msg = "instruction_id_list must not be empty"
            raise ValueError(msg)
        unknown = set(value) - _ALLOWED_ATOM_IDS
        if unknown:
            msg = f"unknown C22 instruction ids: {sorted(unknown)!r}"
            raise ValueError(msg)
        if len(value) != len(set(value)):
            msg = "instruction_id_list must not contain duplicates"
            raise ValueError(msg)
        return value

    @field_validator("kwargs_list")
    @classmethod
    def _validate_json_kwargs(
        cls,
        value: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        if not value:
            msg = "kwargs_list must not be empty"
            raise ValueError(msg)
        for index, kwargs in enumerate(value):
            _validate_json_value(kwargs, path=f"kwargs_list[{index}]")
        return value

    @model_validator(mode="after")
    def _validate_stack(self) -> Self:
        lengths = {
            len(self.constraint_descriptions),
            len(self.instruction_id_list),
            len(self.kwargs_list),
        }
        if len(lengths) != 1:
            msg = (
                "constraint_descriptions, instruction_id_list, and "
                "kwargs_list must have equal nonzero lengths"
            )
            raise ValueError(msg)

        for instruction_id, kwargs in zip(
            self.instruction_id_list,
            self.kwargs_list,
            strict=True,
        ):
            _validate_atom_kwargs(instruction_id, kwargs)

        error = compatibility_error(
            self.instruction_id_list,
            self.kwargs_list,
        )
        if error is not None:
            msg = f"contradictory C22 constraint stack: {error}"
            raise ValueError(msg)
        return self

    def constraints_block(self) -> str:
        """Render the base task plus the model-visible constraint lines."""
        lines = [self.base_task, ""]
        lines.extend(
            f"{index}. {description}"
            for index, description in enumerate(
                self.constraint_descriptions,
                start=1,
            )
        )
        return "\n".join(lines)

    def to_gold(self) -> str:
        """Serialize with deterministic key order and no incidental space."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_gold(cls, gold: str) -> ConstraintSpec:
        """Validate a serialized gold document without type coercion."""
        return cls.model_validate_json(gold, strict=True)
