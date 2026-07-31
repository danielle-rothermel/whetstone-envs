"""Validated serialized constraint stacks shared by generation and scoring."""

from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Mapping
from typing import TYPE_CHECKING, Self, cast

from immutabledict import immutabledict
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
    instructions_util,
)
from whetstone_envs.c22.atoms import all_atom_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

_ALLOWED_ATOM_IDS = frozenset(all_atom_ids())
_NO_KWARGS = frozenset(
    {
        "detectable_format:title",
        "startend:quotation",
        "punctuation:no_comma",
    },
)
_REGEX_META_CHARACTERS = frozenset(r".^$*+?{}[]\|()")


def _freeze_json_value(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        frozen = {
            key: _freeze_json_value(item) for key, item in mapping.items()
        }
        return immutabledict(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


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


def _require_single_token_literal_list(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    literals = _require_nonempty_string_list(value, field_name=field_name)
    if len(literals) != 1:
        msg = f"{field_name} must contain exactly one single-token literal"
        raise ValueError(msg)
    _require_safe_regex_literal(
        literals[0],
        field_name=f"{field_name} item",
    )
    if re.fullmatch(r"\w+", literals[0]) is None:
        msg = f"{field_name} must contain one full Unicode word token"
        raise ValueError(msg)
    return literals


def _require_safe_regex_literal(value: str, *, field_name: str) -> None:
    if any(character in _REGEX_META_CHARACTERS for character in value):
        msg = (
            f"{field_name} must be a literal without regex metacharacters"
        )
        raise ValueError(msg)


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
        _require_single_token_literal_list(
            kwargs["keywords"],
            field_name=f"{instruction_id}.keywords",
        )
    elif instruction_id == "keywords:forbidden_words":
        _require_exact_keys(instruction_id, kwargs, {"forbidden_words"})
        _require_single_token_literal_list(
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
        marker = _require_nonempty_string(
            kwargs["postscript_marker"],
            field_name=f"{instruction_id}.postscript_marker",
        )
        if marker not in {"P.S.", "P.P.S"}:
            _require_safe_regex_literal(
                marker,
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
            or letter not in string.ascii_letters
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


def _canonical_description(
    instruction_id: str,
    kwargs: Mapping[str, object],
) -> str:
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    return cast("str", instruction.build_description(**dict(kwargs)))


def _ordered_literal_merge(first: str, final: str) -> str:
    """Shortest exact merge containing ``first`` and ending in ``final``."""
    first_folded = first.casefold()
    final_folded = final.casefold()
    if first_folded in final_folded:
        return final
    max_overlap = min(len(first_folded), len(final_folded))
    for width in range(max_overlap, 0, -1):
        if first_folded[-width:] == final_folded[:width]:
            return first + final[width:]
    return first + final


def _required_word_lower_bound(
    *,
    required_keywords: Sequence[str],
    end_phrase: str | None,
    postscript_marker: str | None,
) -> int:
    """Sound token lower bound for C22's coexisting required literals."""
    if postscript_marker is not None and end_phrase is not None:
        structural_literal = _ordered_literal_merge(
            postscript_marker,
            end_phrase,
        )
    else:
        structural_literal = postscript_marker or end_phrase or ""

    literal_counts = [
        instructions_util.count_words(structural_literal),
        *(
            instructions_util.count_words(keyword)
            for keyword in required_keywords
        ),
    ]
    return max(literal_counts)


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
    end_phrase: str | None = None
    postscript_marker: str | None = None
    exact_word_budget: int | None = None
    forbids_commas = False
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
            raw_end_phrase = kwargs["end_phrase"]
            assert isinstance(raw_end_phrase, str)
            end_phrase = raw_end_phrase.strip()
            required_literals.append(end_phrase)
        elif instruction_id == "detectable_content:postscript":
            raw_marker = kwargs["postscript_marker"]
            assert isinstance(raw_marker, str)
            postscript_marker = raw_marker.strip()
            required_literals.append(postscript_marker)
        elif instruction_id == "keywords:forbidden_words":
            words = cast("list[str]", kwargs["forbidden_words"])
            forbidden_words.extend(words)
        elif instruction_id == "keywords:letter_frequency":
            letter = kwargs["letter"]
            assert isinstance(letter, str)
            forbidden_letters.append(letter)
        elif instruction_id == "length_constraints:number_words":
            budget = kwargs["num_words"]
            assert isinstance(budget, int)
            exact_word_budget = budget
        elif instruction_id == "punctuation:no_comma":
            forbids_commas = True

    if forbids_commas:
        for literal in required_literals:
            if "," in literal:
                return (
                    f"required literal {literal!r} contains a comma "
                    "forbidden by 'punctuation:no_comma'"
                )

    for forbidden_word in forbidden_words:
        pattern = re.compile(
            rf"\b{forbidden_word}\b",
            flags=re.IGNORECASE,
        )
        for literal in required_literals:
            if pattern.search(literal):
                return (
                    f"required literal {literal!r} contains forbidden "
                    f"word {forbidden_word!r}"
                )

    for literal in required_literals:
        for letter in forbidden_letters:
            if letter.casefold() in literal.casefold():
                return (
                    f"required literal {literal!r} contains forbidden "
                    f"letter {letter!r}"
                )

    if exact_word_budget is not None:
        lower_bound = _required_word_lower_bound(
            required_keywords=required_keywords,
            end_phrase=end_phrase,
            postscript_marker=postscript_marker,
        )
        if exact_word_budget < lower_bound:
            return (
                f"exact word budget {exact_word_budget} is below the "
                f"mandatory required-literal lower bound {lower_bound}"
            )
    return None


class ConstraintSpec(BaseModel):
    """The strict persistence boundary for one C22 constraint stack."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    base_task: str
    constraint_descriptions: tuple[str, ...]
    instruction_id_list: tuple[str, ...]
    kwargs_list: tuple[Mapping[str, object], ...]

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
        value: tuple[Mapping[str, object], ...],
    ) -> tuple[Mapping[str, object], ...]:
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

        for instruction_id, kwargs, description in zip(
            self.instruction_id_list,
            self.kwargs_list,
            self.constraint_descriptions,
            strict=True,
        ):
            canonical = _canonical_description(instruction_id, kwargs)
            if description != canonical:
                msg = (
                    f"description for {instruction_id!r} must equal its "
                    f"canonical vendored description {canonical!r}"
                )
                raise ValueError(msg)

        error = compatibility_error(
            self.instruction_id_list,
            self.kwargs_list,
        )
        if error is not None:
            msg = f"contradictory C22 constraint stack: {error}"
            raise ValueError(msg)
        frozen_kwargs = tuple(
            cast(
                "immutabledict[str, object]",
                _freeze_json_value(dict(kwargs)),
            )
            for kwargs in self.kwargs_list
        )
        object.__setattr__(self, "kwargs_list", frozen_kwargs)
        return self

    @field_serializer("kwargs_list")
    def _serialize_kwargs(
        self,
        value: tuple[Mapping[str, object], ...],
    ) -> list[dict[str, object]]:
        return [
            cast("dict[str, object]", _thaw_json_value(kwargs))
            for kwargs in value
        ]

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
