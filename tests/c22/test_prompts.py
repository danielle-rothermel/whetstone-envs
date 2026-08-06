from __future__ import annotations

import string

from whetstone_envs.c22.constraints import ConstraintStack, EndPhrase, NoComma
from whetstone_envs.c22.prompts import CEILING_TEMPLATE, NAIVE_TEMPLATE, PROBES
from whetstone_envs.instances import make_instance

_BLOCK = (
    "1. In your entire response, refrain from the use of any commas.\n"
    "2. Finish your response with this exact phrase DONE. No other words "
    "should follow this phrase."
)
_STACK = ConstraintStack(constraints=(NoComma(), EndPhrase(phrase="DONE")))
_INSTANCE = make_instance(
    id="c22-fixture",
    seed=1_000_000,
    strata="n2_easy",
    prompt_inputs={"constraints": _BLOCK},
    gold=_STACK.to_gold(),
)


def test_naive_prompt_renders_only_the_constraint_request() -> None:
    assert PROBES.render_naive(_INSTANCE) == (
        f"Satisfy every constraint below:\n{_BLOCK}\n\nAnswer:"
    )


def test_ceiling_prompt_adds_guidance_without_private_gold() -> None:
    naive = PROBES.render_naive(_INSTANCE)
    ceiling = PROBES.render_ceiling(_INSTANCE)
    assert _BLOCK in ceiling
    assert len(ceiling) > len(naive)
    assert "no_comma" not in ceiling
    assert "end_phrase" not in ceiling


def test_templates_reference_only_public_constraint_text() -> None:
    for template in (NAIVE_TEMPLATE, CEILING_TEMPLATE):
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
        assert fields == {"constraints"}
