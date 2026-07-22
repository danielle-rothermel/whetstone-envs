"""Tests for exact-match scoring and the aggregation ladder.

The failed/missing cases here are the rubric-criterion-13 guard: a
non-scored observation must make every aggregate that depends on it
visibly incomplete (``mean is None``, non-zero failed/missing counts),
never a silent zero.
"""

from __future__ import annotations

import pytest

from whetstone_envs.core.scoring import (
    Observation,
    Outcome,
    aggregate,
    aggregate_overall,
    aggregate_stratum,
    aggregate_task,
    exact_match,
    failed,
    missing,
    scored,
)


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("yes", "yes", 1),
        ("  yes  ", "yes", 1),
        ("```\nyes\n```", "yes", 1),
        ("yes", "no", 0),
        ("Yes", "yes", 0),
    ],
)
def test_exact_match(pred: str, gold: str, expected: int) -> None:
    result = exact_match(pred, gold)
    assert result == expected
    assert result in (0, 1)


def test_scored_requires_binary_score() -> None:
    with pytest.raises(ValueError, match="score 0 or 1"):
        scored("t", 0, 2)


def test_non_scored_must_not_carry_score() -> None:
    with pytest.raises(ValueError, match="must not"):
        Observation("t", 0, Outcome.FAILED, 1)


# --- repeat -> task level ------------------------------------------------


def test_task_mean_over_repeats() -> None:
    obs = [scored("t", 0, 1), scored("t", 1, 0), scored("t", 2, 1)]
    agg = aggregate_task(obs)
    assert agg.mean == pytest.approx(2 / 3)
    assert agg.usable == 3
    assert agg.complete is True


def test_failed_repeat_makes_task_incomplete_not_zero() -> None:
    obs = [scored("t", 0, 1), failed("t", 1), scored("t", 2, 1)]
    agg = aggregate_task(obs)
    # Two scored 1s would naively average to 1.0; the failure must not
    # silently drag it toward zero -- it makes the mean unavailable.
    assert agg.mean is None
    assert agg.complete is False
    assert agg.failed_count == 1
    assert agg.usable == 2


def test_missing_repeat_makes_task_incomplete() -> None:
    agg = aggregate_task([scored("t", 0, 1), missing("t", 1)])
    assert agg.mean is None
    assert agg.missing_count == 1
    assert agg.complete is False


# --- task -> stratum -> overall -----------------------------------------


def test_full_ladder_all_complete() -> None:
    observations = [
        scored("easy-0", 0, 1),
        scored("easy-1", 0, 0),
        scored("hard-0", 0, 1),
        scored("hard-1", 0, 1),
    ]
    task_strata = {
        "easy-0": "easy",
        "easy-1": "easy",
        "hard-0": "hard",
        "hard-1": "hard",
    }
    root = aggregate(observations, task_strata)
    # easy stratum mean = (1 + 0)/2 = 0.5; hard = (1 + 1)/2 = 1.0;
    # overall = mean of stratum means = 0.75. Crucially this is the mean
    # of stratum means, not of raw scores -- aggregation crosses strata.
    assert root.mean == pytest.approx(0.75)
    assert root.complete is True
    child_means = sorted(c.mean for c in root.children if c.mean is not None)
    assert child_means == pytest.approx([0.5, 1.0])


def test_aggregation_crosses_strata_not_raw_mean() -> None:
    # An imbalanced pool: 3 easy tasks all correct, 1 hard task wrong.
    # Raw pooled mean would be 3/4 = 0.75; the strata-crossing mean is
    # mean(1.0, 0.0) = 0.5. Confirm we get the strata-crossing value.
    observations = [
        scored("easy-0", 0, 1),
        scored("easy-1", 0, 1),
        scored("easy-2", 0, 1),
        scored("hard-0", 0, 0),
    ]
    task_strata = {
        "easy-0": "easy",
        "easy-1": "easy",
        "easy-2": "easy",
        "hard-0": "hard",
    }
    root = aggregate(observations, task_strata)
    assert root.mean == pytest.approx(0.5)


def test_failed_observation_propagates_to_overall() -> None:
    observations = [
        scored("easy-0", 0, 1),
        failed("hard-0", 0),
    ]
    task_strata = {"easy-0": "easy", "hard-0": "hard"}
    root = aggregate(observations, task_strata)
    # The whole run's overall aggregate is visibly incomplete even
    # though one stratum was fully scored.
    assert root.mean is None
    assert root.complete is False
    assert root.failed_count == 1
    assert root.total == 2


def test_missing_task_stratum_raises() -> None:
    with pytest.raises(KeyError, match="no stratum"):
        aggregate([scored("t", 0, 1)], {})


def test_stratum_and_overall_helpers_compose() -> None:
    t0 = aggregate_task([scored("a", 0, 1), scored("a", 1, 1)])
    t1 = aggregate_task([scored("b", 0, 0)])
    stratum = aggregate_stratum([t0, t1])
    assert stratum.mean == pytest.approx(0.5)
    overall = aggregate_overall([stratum])
    assert overall.mean == pytest.approx(0.5)


def test_empty_stratum_is_incomplete_not_zero() -> None:
    # No children means nothing to average; mean is None, not 0.0, and
    # the aggregate must be visibly incomplete -- a zero-observation
    # aggregate is not vacuously complete (rubric 13).
    agg = aggregate_stratum([])
    assert agg.mean is None
    assert agg.total == 0
    assert agg.complete is False


def test_empty_task_is_incomplete_not_zero() -> None:
    agg = aggregate_task([])
    assert agg.mean is None
    assert agg.total == 0
    assert agg.complete is False


def test_empty_overall_is_incomplete_not_zero() -> None:
    agg = aggregate_overall([aggregate_stratum([])])
    assert agg.mean is None
    assert agg.complete is False


def test_empty_task_mixed_with_scored_does_not_silently_vanish() -> None:
    # An empty task composed alongside a scored one must not disappear
    # from the parent mean: the empty child forces incompleteness rather
    # than the parent averaging only the scored child to 1.0.
    empty = aggregate_task([])
    scored_task = aggregate_task([scored("a", 0, 1)])
    stratum = aggregate_stratum([empty, scored_task])
    assert stratum.mean is None
    assert stratum.complete is False


def test_empty_stratum_mixed_with_scored_does_not_silently_vanish() -> None:
    empty = aggregate_stratum([])
    scored_stratum = aggregate_stratum([aggregate_task([scored("a", 0, 1)])])
    overall = aggregate_overall([empty, scored_stratum])
    assert overall.mean is None
    assert overall.complete is False
