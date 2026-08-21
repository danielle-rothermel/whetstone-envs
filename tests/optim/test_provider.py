from __future__ import annotations

import pytest

pytest.importorskip("dr_providers")

from dr_providers import RequestControl

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    build_c19_experiment,
    c19_render_contract,
)
from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    c19_fake_task_reply,
    openrouter_seeded_call_config,
)


def test_openrouter_preset_advertises_seed() -> None:
    config = openrouter_seeded_call_config(model="openai/gpt-4.1-nano")
    assert config.definition.constraints.supports(RequestControl.SEED)


def test_fake_transport_emits_gold_for_ceiling_prompt() -> None:
    pytest.importorskip("whetstone.experiment.env")
    experiment = build_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(2, 2, 0),
        num_seeds=1,
    )
    task = experiment.eval_configs.internal.tasks[0]
    gold = getattr(task, "gold", None)
    inputs = getattr(task, "prompt_inputs", None)
    assert isinstance(gold, str)
    assert isinstance(inputs, dict)
    contract = c19_render_contract()
    ceiling = contract.render(PROBES.ceiling_template, inputs)
    naive = contract.render(PROBES.naive_template, inputs)
    gold_by_prompt = {ceiling: gold}

    assert c19_fake_task_reply(ceiling, gold_by_prompt) == gold
    assert c19_fake_task_reply(naive, gold_by_prompt) == f"generated: {naive}"


def test_bind_openrouter_transport_reuses_one_client() -> None:
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    transport, factory = bind_openrouter_transport(policy)
    assert factory(policy) is transport
