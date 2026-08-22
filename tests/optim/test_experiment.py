from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.experiment.sampling import HELD_OUT, INTERNAL_EVAL, OFFICIAL

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_PROMPT_FIELDS,
    c19_render_contract,
    prepare_c19_experiment,
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


def test_prepare_c19_experiment_maps_split_to_eval_rows() -> None:
    pool = _small_pool()
    prepared = prepare_c19_experiment(pool, split_sizes=(2, 2, 0), num_seeds=1)
    experiment = prepared.experiment
    split = pool.split(2, 2, 0)
    assert prepared.split == split
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
    assert experiment.eval_configs.held_out_task_hashes == ()


def test_prepare_c19_experiment_records_held_out_hashes() -> None:
    pool = _small_pool()
    experiment = prepare_c19_experiment(
        pool, split_sizes=(1, 1, 1), num_seeds=1
    ).experiment
    split = pool.split(1, 1, 1)
    assert experiment.eval_configs.held_out_task_hashes == tuple(
        row.task_hash for row in task_rows_from_instances(split.held_out)
    )


def test_held_out_is_a_derived_split_under_its_own_role() -> None:
    """Held-out rows become a full EvalSplit, not a bare hash tuple."""
    pool = _small_pool()
    configs = prepare_c19_experiment(
        pool, split_sizes=(1, 1, 1), num_seeds=1
    ).experiment.eval_configs
    held_out = configs.held_out
    assert held_out is not None
    assert held_out.split_role == HELD_OUT
    assert configs.split_for(HELD_OUT) is held_out
    assert set(configs.splits()) == {INTERNAL_EVAL, OFFICIAL, HELD_OUT}
    # Construction enforces disjointness, so the roles cannot share a task.
    covered = [
        set(split.task_set.task_hashes) for split in configs.splits().values()
    ]
    assert not covered[0] & covered[1]
    assert not covered[0] & covered[2]
    assert not covered[1] & covered[2]


def test_experiment_without_held_out_rows_leaves_the_split_absent() -> None:
    configs = prepare_c19_experiment(
        _small_pool(), split_sizes=(2, 2, 0), num_seeds=1
    ).experiment.eval_configs
    assert configs.held_out is None
    assert configs.held_out_task_hashes == ()
    assert set(configs.splits()) == {INTERNAL_EVAL, OFFICIAL}
