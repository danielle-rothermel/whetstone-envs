# Modified from InductionBench e0b8392; see attribution/PROVENANCE.md.

from __future__ import annotations

from whetstone_envs.c23._domain import Hypothesis, RuleFamily


def apply_reference(hypothesis: Hypothesis, value: str) -> str:
    family = hypothesis.configuration.family
    if family is RuleFamily.ISL:
        return _apply_isl(hypothesis, value)
    if family is RuleFamily.L_OSL:
        return _apply_l_osl(hypothesis, value)
    if family is RuleFamily.R_OSL:
        return _apply_r_osl(hypothesis, value)
    raise AssertionError(f"unhandled rule family: {family!r}")


def _apply_isl(hypothesis: Hypothesis, value: str) -> str:
    output: list[str] = []
    context_length = hypothesis.configuration.context_length
    for index, symbol in enumerate(value):
        context = value[: index + 1][-context_length:]
        output.append(
            hypothesis.replacement
            if context == hypothesis.context
            else symbol,
        )
    return "".join(output)


def _apply_l_osl(hypothesis: Hypothesis, value: str) -> str:
    output = ""
    context_length = hypothesis.configuration.context_length
    for symbol in value:
        output += symbol
        if output[-context_length:] == hypothesis.context:
            output = output[:-1] + hypothesis.replacement
    return output


def _apply_r_osl(hypothesis: Hypothesis, value: str) -> str:
    return _apply_l_osl(hypothesis, value[::-1])[::-1]
