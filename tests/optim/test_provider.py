from __future__ import annotations

import pytest

pytest.importorskip("dr_providers")

from dr_providers import RequestControl

from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    openrouter_seeded_call_config,
)


def test_openrouter_preset_advertises_seed() -> None:
    config = openrouter_seeded_call_config(model="openai/gpt-4.1-nano")
    assert config.definition.constraints.supports(RequestControl.SEED)


def test_bind_openrouter_transport_reuses_one_client() -> None:
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    transport, factory = bind_openrouter_transport(policy)
    assert factory(policy) is transport
