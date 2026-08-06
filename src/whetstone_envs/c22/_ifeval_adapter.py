from __future__ import annotations

from typing import cast

from whetstone_envs.c22._vendor.instruction_following_eval import (
    evaluation_lib,
    instructions_registry,
)
from whetstone_envs.c22.constraints import (
    Constraint,
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

_RESPONSE_KEY = "c22"


def _instruction_arguments(
    constraint: Constraint,
) -> tuple[str, dict[str, object]]:
    text = _text_instruction_arguments(constraint)
    if text is not None:
        return text
    format_constraint = _format_instruction_arguments(constraint)
    if format_constraint is not None:
        return format_constraint
    numeric = _numeric_instruction_arguments(constraint)
    if numeric is not None:
        return numeric
    msg = f"unsupported constraint: {type(constraint).__name__}"
    raise AssertionError(msg)


def _text_instruction_arguments(
    constraint: Constraint,
) -> tuple[str, dict[str, object]] | None:
    if isinstance(constraint, RequiredKeyword):
        return "keywords:existence", {"keywords": [constraint.keyword]}
    if isinstance(constraint, ForbiddenWord):
        return "keywords:forbidden_words", {
            "forbidden_words": [constraint.word]
        }
    if isinstance(constraint, EndPhrase):
        return "startend:end_checker", {"end_phrase": constraint.phrase}
    if isinstance(constraint, Title):
        return "detectable_format:title", {}
    return None


def _format_instruction_arguments(
    constraint: Constraint,
) -> tuple[str, dict[str, object]] | None:
    if isinstance(constraint, Quotation):
        return "startend:quotation", {}
    if isinstance(constraint, NoComma):
        return "punctuation:no_comma", {}
    if isinstance(constraint, Placeholders):
        return "detectable_content:number_placeholders", {
            "num_placeholders": constraint.count
        }
    if isinstance(constraint, Postscript):
        return "detectable_content:postscript", {
            "postscript_marker": constraint.marker
        }
    if isinstance(constraint, HighlightedSections):
        return "detectable_format:number_highlighted_sections", {
            "num_highlights": constraint.count
        }
    return None


def _numeric_instruction_arguments(
    constraint: Constraint,
) -> tuple[str, dict[str, object]] | None:
    if isinstance(constraint, ExactWordCount):
        return "length_constraints:number_words", {
            "num_words": constraint.count,
            "relation": "exactly",
        }
    if isinstance(constraint, ForbiddenLetter):
        return "keywords:letter_frequency", {
            "letter": constraint.letter,
            "let_frequency": 1,
            "let_relation": "less than",
        }
    return None


def describe(constraint: Constraint) -> str:
    instruction_id, arguments = _instruction_arguments(constraint)
    instruction_type = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_type(instruction_id)
    return cast("str", instruction.build_description(**arguments))


def render_constraint_block(constraints: tuple[Constraint, ...]) -> str:
    return "\n".join(
        f"{index}. {describe(constraint)}"
        for index, constraint in enumerate(constraints, start=1)
    )


def check(
    constraints: tuple[Constraint, ...],
    response: str,
) -> tuple[tuple[str, bool], ...]:
    instructions = tuple(
        _instruction_arguments(constraint) for constraint in constraints
    )
    # Upstream's annotation excludes the sequence-valued arguments its own
    # keyword checkers accept. Keep that mismatch at the vendor boundary.
    arguments = cast(
        "list[dict[str, str | int | None]]",
        [values for _, values in instructions],
    )
    example = evaluation_lib.InputExample(
        key=0,
        instruction_id_list=[name for name, _ in instructions],
        prompt=_RESPONSE_KEY,
        kwargs=arguments,
    )
    output = evaluation_lib.test_instruction_following_strict(
        example,
        {_RESPONSE_KEY: response},
    )
    return tuple(
        (constraint.kind, verdict)
        for constraint, verdict in zip(
            constraints,
            output.follow_instruction_list,
            strict=True,
        )
    )
