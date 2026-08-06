from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.probes import ProbePair, render_with_prompt_inputs

if TYPE_CHECKING:
    from whetstone_envs.c23._domain import Demonstration

NAIVE_TEMPLATE = """{demos_block}

{query} -> """

CEILING_TEMPLATE = """You are solving a hidden-rule string-transformation
puzzle.
One deterministic rule maps each input to its output. The hidden rule is
exactly one of these four forms over a, b, c, and d: ISL k=2, ISL k=3,
left-OSL k=2, or right-OSL k=2. ISL reads context from the input; OSL reads
context from the partial output in the named direction. When the full context
matches, its final character is deleted or replaced by a different character.
Infer the query output that is consistent with every demonstration. More than
one rule may fit the demonstrations, but they agree on the query output.
Do not explain your reasoning.

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
