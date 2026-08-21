from __future__ import annotations

import pytest

pytest.importorskip("dr_providers")

from dr_providers import RequestControl

from whetstone_envs.optim.provider import openrouter_seeded_call_config


def test_openrouter_preset_advertises_seed() -> None:
    config = openrouter_seeded_call_config(model="openai/gpt-4.1-nano")
    assert config.definition.constraints.supports(RequestControl.SEED)
