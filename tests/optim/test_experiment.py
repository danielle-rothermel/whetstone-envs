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


def test_a_rare_failed_row_reduces_completeness_instead_of_voiding() -> None:
    """351 good rows are evidence; one 429 must not erase them.

    This is the exact shape that aborted the live paid Stage 0: the naive
    anchor evaluated 352 rows, one failed with an unretried 429, and under
    ``missing_data="propagate"`` the aggregate went to ``None``. A ``None``
    aggregate makes the Reward Policy's ``score`` term missing, and its
    FAIL missing-data policy then raised -- voiding the evaluation, the
    stage, and the money already spent on the other 351 rows.

    Fails-before: with ``propagate`` this same input aggregates to
    ``status=missing_data, value=None``.

    Binding ``skip`` is what the protocol already pre-registered for this
    problem -- §8's O7 recommends per-task weighting with a hard backstop
    at 90%, the analysis already weights by ``per_task_counts``, and
    ``COMPLETENESS_BACKSTOP`` is already 0.90. Propagating was the piece
    inconsistent with that rule.
    """
    pytest.importorskip("whetstone.eval.aggregation")
    from whetstone.eval.aggregation import AggregationInput, aggregate

    from whetstone_envs.optim.experiment import _reference_aggregation

    config = _reference_aggregation("whetstone_envs.c19")
    inputs = (
        *(AggregationInput(value=1.0) for _ in range(351)),
        AggregationInput(value=None),
    )
    output = aggregate(config, inputs)
    assert output.status.value == "ok"
    assert output.value == 1.0
    # The shortfall is not hidden: it is reported as reduced completeness,
    # which is what the analysis weights by.
    assert output.count_present == 351
    assert output.count_applicable == 352


def test_losing_more_than_the_tolerance_still_refuses_a_number() -> None:
    """The tolerance is a floor, not permission to average a biased subset."""
    pytest.importorskip("whetstone.eval.aggregate")
    from whetstone.eval.aggregate import _policy_from_aggregation_config

    from whetstone_envs.optim.experiment import (
        MAX_SKIP_FRACTION,
        _reference_aggregation,
    )

    policy = _policy_from_aggregation_config(
        _reference_aggregation("whetstone_envs.c19")
    )
    assert policy.row_policy.value == "skip"
    assert policy.max_skip_fraction == MAX_SKIP_FRACTION
    # One row in 352 is tolerated; a tenth of the matrix is not.
    assert policy.within_tolerance(skipped=1, planned=352)
    assert not policy.within_tolerance(skipped=36, planned=352)


def test_the_skip_tolerance_matches_the_pre_registered_backstop() -> None:
    """One rule, not two: the aggregate and the report agree on 90%."""
    from whetstone_envs.optim.experiment import MAX_SKIP_FRACTION
    from whetstone_envs.optim.study.manifest import COMPLETENESS_BACKSTOP

    assert pytest.approx(1.0 - COMPLETENESS_BACKSTOP) == MAX_SKIP_FRACTION
