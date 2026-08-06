from __future__ import annotations

import string
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c22.constraints import ConstraintStack, EndPhrase, NoComma
from whetstone_envs.c22.prompts import CEILING_TEMPLATE, NAIVE_TEMPLATE, PROBES
from whetstone_envs.instances import make_instance

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance

_BLOCK = "Two model-visible constraint descriptions"
_STACK = ConstraintStack(constraints=(NoComma(), EndPhrase(phrase="DONE")))
_INSTANCE = make_instance(
    id="c22-fixture",
    seed=1_000_000,
    strata="n2_easy",
    prompt_inputs={"constraints": _BLOCK},
    gold=_STACK.to_gold(),
)


@pytest.mark.parametrize(
    "render", [PROBES.render_naive, PROBES.render_ceiling]
)
def test_prompts_render_public_constraints_without_private_gold(
    render: Callable[[Instance], str],
) -> None:
    prompt = render(_INSTANCE)
    assert _BLOCK in prompt
    assert _INSTANCE.gold not in prompt


def test_templates_reference_only_public_constraint_text() -> None:
    for template in (NAIVE_TEMPLATE, CEILING_TEMPLATE):
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
        assert fields == {"constraints"}
