from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.probes import ProbePair, render_with_prompt_inputs

if TYPE_CHECKING:
    from whetstone_envs.c23._domain import Demonstration

NAIVE_TEMPLATE = """{demos_block}

{query} -> """

CEILING_TEMPLATE = """You are solving a hidden-rule string-transformation
puzzle.
One deterministic rule maps each input to its output. The rule examines
individual characters in a bounded adjacent context. Its context comes from
the input, the left-to-right partial output, or the right-to-left partial
output. Infer the one rule that fits every demonstration, then apply it to the
query. Do not explain your reasoning.

Demonstrations:
{demos_block}

Query:
{query}

Return only the transformed string on one line prefixed with "Output:"."""

PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=render_with_prompt_inputs,
)


def render_demonstrations(
    demonstrations: tuple[Demonstration, ...],
) -> str:
    return "\n".join(
        f"{example.input} -> {example.output}" for example in demonstrations
    )
