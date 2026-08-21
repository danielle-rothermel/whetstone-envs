from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_PROMPT_FIELDS,
    build_c19_experiment,
    c19_render_contract,
    probe_candidates_from_templates,
    reward_policy_for_exact_match,
)
from whetstone_envs.optim.rows import task_rows_from_instances


def _small_pool():
    return generate_pool(n_per_stratum=2, seed_start=765_432)


def test_probe_pair_maps_to_initial_and_ceiling_candidates() -> None:
    initial, ceiling = probe_candidates_from_templates(
        naive_template=PROBES.naive_template,
        ceiling_template=PROBES.ceiling_template,
    )
    assert initial.payload[C19_MUTATION_FIELD] == PROBES.naive_template
    assert ceiling.payload[C19_MUTATION_FIELD] == PROBES.ceiling_template


def test_render_contract_requires_c19_prompt_fields() -> None:
    contract = c19_render_contract()
    assert contract.available_fields == C19_PROMPT_FIELDS
    observed = contract.validate_template(PROBES.naive_template)
    assert set(observed) == set(C19_PROMPT_FIELDS)


def test_reward_policy_is_single_exact_match_term() -> None:
    policy = reward_policy_for_exact_match()
    assert len(policy.terms) == 1
    assert policy.terms[0].name == "score"
    assert policy.terms[0].weight == 1.0


def test_build_c19_experiment_maps_split_to_eval_rows() -> None:
    pool = _small_pool()
    experiment = build_c19_experiment(pool, split_sizes=(2, 2, 0), num_seeds=1)
    split = pool.split(2, 2, 0)
    internal_ids = {
        row.task_id for row in experiment.eval_configs.internal.tasks
    }
    official_ids = {
        row.task_id for row in experiment.eval_configs.official.tasks
    }
    assert internal_ids == {instance.id for instance in split.internal_eval}
    assert official_ids == {instance.id for instance in split.official}
    assert experiment.reward_policy.terms[0].name == "score"
    rows = task_rows_from_instances(split.internal_eval)
    assert rows[0].gold == split.internal_eval[0].gold
