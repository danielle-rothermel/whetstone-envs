from __future__ import annotations

import re
import string
from typing import Annotated, Literal, Self, cast

from dr_serialize import (
    CANONICAL_JSON_MAX_CONTAINER_DEPTH,
    Jsonable,
    canonical_json,
    decode_strict_json_bytes,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_GOLD_MAX_BYTES = 1 << 16


class _ConstraintModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_word(value: str, *, field_name: str) -> str:
    if value != value.strip() or re.fullmatch(r"\w+", value) is None:
        msg = f"{field_name} must be one full Unicode word token"
        raise ValueError(msg)
    return value


class RequiredKeyword(_ConstraintModel):
    kind: Literal["required_keyword"] = "required_keyword"
    keyword: str

    @field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, value: str) -> str:
        return _validate_word(value, field_name="keyword")


class ForbiddenWord(_ConstraintModel):
    kind: Literal["forbidden_word"] = "forbidden_word"
    word: str

    @field_validator("word")
    @classmethod
    def _validate_forbidden_word(cls, value: str) -> str:
        return _validate_word(value, field_name="word")


class EndPhrase(_ConstraintModel):
    kind: Literal["end_phrase"] = "end_phrase"
    phrase: str

    @field_validator("phrase")
    @classmethod
    def _validate_phrase(cls, value: str) -> str:
        if not value or value != value.strip():
            msg = "phrase must be nonempty without outer whitespace"
            raise ValueError(msg)
        return value


class Title(_ConstraintModel):
    kind: Literal["title"] = "title"


class Quotation(_ConstraintModel):
    kind: Literal["quotation"] = "quotation"


class NoComma(_ConstraintModel):
    kind: Literal["no_comma"] = "no_comma"


class Placeholders(_ConstraintModel):
    kind: Literal["placeholders"] = "placeholders"
    count: int = Field(gt=0)


class Postscript(_ConstraintModel):
    kind: Literal["postscript"] = "postscript"
    marker: Literal["P.S.", "P.P.S"]


class HighlightedSections(_ConstraintModel):
    kind: Literal["highlighted_sections"] = "highlighted_sections"
    count: int = Field(gt=0)


class ExactWordCount(_ConstraintModel):
    kind: Literal["exact_word_count"] = "exact_word_count"
    count: int = Field(gt=0)


class ForbiddenLetter(_ConstraintModel):
    kind: Literal["forbidden_letter"] = "forbidden_letter"
    letter: str

    @field_validator("letter")
    @classmethod
    def _validate_letter(cls, value: str) -> str:
        if len(value) != 1 or value not in string.ascii_letters:
            msg = "letter must be one ASCII letter"
            raise ValueError(msg)
        return value


type Constraint = Annotated[
    RequiredKeyword
    | ForbiddenWord
    | EndPhrase
    | Title
    | Quotation
    | NoComma
    | Placeholders
    | Postscript
    | HighlightedSections
    | ExactWordCount
    | ForbiddenLetter,
    Field(discriminator="kind"),
]


def _ordered_literal_merge(first: str, final: str) -> str:
    first_folded = first.casefold()
    final_folded = final.casefold()
    if first_folded in final_folded:
        return final
    for width in range(min(len(first_folded), len(final_folded)), 0, -1):
        if first_folded[-width:] == final_folded[:width]:
            return first + final[width:]
    return first + final


def _word_count(value: str) -> int:
    return len(re.findall(r"\w+", value))


def _required_literals(
    by_kind: dict[str, Constraint],
) -> tuple[list[str], list[str], str | None, str | None]:
    required_literals: list[str] = []
    required_keywords: list[str] = []
    end_phrase: str | None = None
    postscript_marker: str | None = None

    required_keyword = by_kind.get("required_keyword")
    if isinstance(required_keyword, RequiredKeyword):
        required_keywords.append(required_keyword.keyword)
        required_literals.append(required_keyword.keyword)

    end = by_kind.get("end_phrase")
    if isinstance(end, EndPhrase):
        end_phrase = end.phrase
        required_literals.append(end.phrase)

    postscript = by_kind.get("postscript")
    if isinstance(postscript, Postscript):
        postscript_marker = postscript.marker
        required_literals.append(postscript.marker)
    return (
        required_literals,
        required_keywords,
        end_phrase,
        postscript_marker,
    )


def _literal_compatibility_error(
    by_kind: dict[str, Constraint],
    required_literals: list[str],
) -> str | None:
    if "no_comma" in by_kind:
        for literal in required_literals:
            if "," in literal:
                return f"required literal {literal!r} contains a comma"

    forbidden_word = by_kind.get("forbidden_word")
    if isinstance(forbidden_word, ForbiddenWord):
        pattern = re.compile(
            rf"\b{re.escape(forbidden_word.word)}\b",
            flags=re.IGNORECASE,
        )
        for literal in required_literals:
            if pattern.search(literal):
                return (
                    f"required literal {literal!r} contains forbidden word "
                    f"{forbidden_word.word!r}"
                )

    forbidden_letter = by_kind.get("forbidden_letter")
    if isinstance(forbidden_letter, ForbiddenLetter):
        for literal in required_literals:
            if forbidden_letter.letter.lower() in literal.lower():
                return (
                    f"required literal {literal!r} contains forbidden letter "
                    f"{forbidden_letter.letter!r}"
                )
    return None


def _word_budget_error(
    by_kind: dict[str, Constraint],
    *,
    required_keywords: list[str],
    end_phrase: str | None,
    postscript_marker: str | None,
) -> str | None:
    exact_words = by_kind.get("exact_word_count")
    if isinstance(exact_words, ExactWordCount):
        structural_literal = postscript_marker or end_phrase or ""
        if postscript_marker is not None and end_phrase is not None:
            structural_literal = _ordered_literal_merge(
                postscript_marker,
                end_phrase,
            )
        required_word_counts = [_word_count(structural_literal)]
        required_word_counts.extend(
            _word_count(word) for word in required_keywords
        )
        lower_bound = max(required_word_counts)
        if exact_words.count < lower_bound:
            return (
                f"exact word count {exact_words.count} is below mandatory "
                f"literal lower bound {lower_bound}"
            )
    return None


def _compatibility_error(constraints: tuple[Constraint, ...]) -> str | None:
    by_kind: dict[str, Constraint] = {}
    for constraint in constraints:
        by_kind[constraint.kind] = constraint
    (
        required_literals,
        required_keywords,
        end_phrase,
        postscript_marker,
    ) = _required_literals(by_kind)
    literal_error = _literal_compatibility_error(
        by_kind,
        required_literals,
    )
    if literal_error is not None:
        return literal_error
    return _word_budget_error(
        by_kind,
        required_keywords=required_keywords,
        end_phrase=end_phrase,
        postscript_marker=postscript_marker,
    )


class ConstraintStack(BaseModel):
    """Closed serialized gold for one C22 task."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    constraints: tuple[Constraint, ...]

    @model_validator(mode="after")
    def _validate_stack(self) -> Self:
        if not self.constraints:
            msg = "constraints must not be empty"
            raise ValueError(msg)
        kinds = tuple(constraint.kind for constraint in self.constraints)
        if len(kinds) != len(set(kinds)):
            msg = "constraint kinds must not repeat"
            raise ValueError(msg)
        error = _compatibility_error(self.constraints)
        if error is not None:
            msg = f"contradictory C22 constraint stack: {error}"
            raise ValueError(msg)
        return self

    def to_gold(self) -> str:
        payload = cast("Jsonable", self.model_dump(mode="json"))
        return canonical_json(payload)

    @classmethod
    def from_gold(cls, gold: str) -> ConstraintStack:
        if not isinstance(gold, str):
            msg = "gold must be a string"
            raise TypeError(msg)
        payload = decode_strict_json_bytes(
            gold.encode(),
            max_bytes=_GOLD_MAX_BYTES,
            max_depth=CANONICAL_JSON_MAX_CONTAINER_DEPTH,
        )
        return cls.model_validate_json(canonical_json(payload), strict=True)


HARD_CONSTRAINT_KINDS = frozenset(
    {"exact_word_count", "forbidden_letter", "forbidden_word"}
)


__all__ = [
    "Constraint",
    "ConstraintStack",
    "EndPhrase",
    "ExactWordCount",
    "ForbiddenLetter",
    "ForbiddenWord",
    "HighlightedSections",
    "NoComma",
    "Placeholders",
    "Postscript",
    "Quotation",
    "RequiredKeyword",
    "Title",
]
