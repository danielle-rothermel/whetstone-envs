from string import Formatter

import pytest

from whetstone_envs.c19.probes import CEILING_TEMPLATE, NAIVE_TEMPLATE


@pytest.mark.parametrize("template", [NAIVE_TEMPLATE, CEILING_TEMPLATE])
def test_templates_reference_only_public_c19_inputs(template: str) -> None:
    referenced = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }

    assert referenced == {"grid", "command", "question"}
