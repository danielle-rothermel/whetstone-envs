from __future__ import annotations

import pytest

from whetstone_envs.scoring import (
    aggregate,
    aggregate_overall,
    aggregate_stratum,
    aggregate_task,
    failed,
    missing,
    scored,
)


def test_task_mean_over_repeats() -> None:
    obs = [scored("t", 0, 1), scored("t", 1, 0), scored("t", 2, 1)]
    agg = aggregate_task(obs)
    assert agg.mean == pytest.approx(2 / 3)
    assert agg.usable == 3
    assert agg.complete is True


def test_failed_repeat_makes_task_incomplete_not_zero() -> None:
    obs = [scored("t", 0, 1), failed("t", 1), scored("t", 2, 1)]
    agg = aggregate_task(obs)
    assert agg.mean is None
    assert agg.complete is False
    assert agg.failed_count == 1
    assert agg.usable == 2


def test_missing_repeat_makes_task_incomplete() -> None:
    agg = aggregate_task([scored("t", 0, 1), missing("t", 1)])
    assert agg.mean is None
    assert agg.missing_count == 1
    assert agg.complete is False


def test_aggregate_task_rejects_mixed_task_ids() -> None:
    with pytest.raises(ValueError, match="single task"):
        aggregate_task([scored("a", 0, 1), scored("b", 0, 1)])


def test_aggregate_task_rejects_duplicate_repeat_ids() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*duplicate)(?=.*task)(?=.*repeat(?:_id)?\D*7)",
    ):
        aggregate_task([scored("task", 7, 1), scored("task", 7, 0)])


def test_aggregate_task_rejects_corrupted_outcome() -> None:
    observation = scored("task", 0, 1)
    object.__setattr__(observation, "outcome", "corrupted")

    with pytest.raises(TypeError, match="Outcome"):
        aggregate_task([observation])


def test_full_ladder_all_complete() -> None:
    observations = [
        scored("easy-0", 0, 1),
        scored("easy-1", 0, 0),
        scored("hard-0", 0, 1),
        scored("hard-1", 0, 1),
    ]
    task_strata = {
        "easy-0": ("easy",),
        "easy-1": ("easy",),
        "hard-0": ("hard",),
        "hard-1": ("hard",),
    }
    root = aggregate(
        observations,
        task_strata,
        expected_repeat_ids=(0,),
    )
    assert root.mean == pytest.approx(0.75)
    assert root.complete is True
    strata = {child.label: child for child in root.children}
    assert set(strata) == {"easy", "hard"}
    assert strata["easy"].mean == pytest.approx(0.5)
    assert strata["hard"].mean == pytest.approx(1.0)
    assert {child.label for child in strata["easy"].children} == {
        "easy-0",
        "easy-1",
    }
    assert {child.label for child in strata["hard"].children} == {
        "hard-0",
        "hard-1",
    }


def test_aggregation_crosses_strata_not_raw_mean() -> None:
    # Equal-weighted stratum means yield 0.5, not the raw pooled mean of 0.75.
    observations = [
        scored("easy-0", 0, 1),
        scored("easy-1", 0, 1),
        scored("easy-2", 0, 1),
        scored("hard-0", 0, 0),
    ]
    task_strata = {
        "easy-0": ("easy",),
        "easy-1": ("easy",),
        "easy-2": ("easy",),
        "hard-0": ("hard",),
    }
    root = aggregate(
        observations,
        task_strata,
        expected_repeat_ids=(0,),
    )
    assert root.mean == pytest.approx(0.5)


def test_planned_matrix_marks_unobserved_repeat_missing() -> None:
    root = aggregate(
        [scored("task", 0, 1)],
        {"task": ("easy",)},
        expected_repeat_ids=(0, 1),
    )
    task = root.children[0].children[0]

    assert task.label == "task"
    assert task.total == 2
    assert task.usable == 1
    assert task.missing_count == 1
    assert task.mean is None
    assert root.complete is False


def test_planned_matrix_includes_fully_absent_expected_task() -> None:
    root = aggregate(
        [scored("observed", 0, 1)],
        {"observed": ("easy",), "absent": ("hard",)},
        expected_repeat_ids=(0,),
    )
    strata = {child.label: child for child in root.children}
    absent_task = strata["hard"].children[0]

    assert absent_task.label == "absent"
    assert absent_task.total == 1
    assert absent_task.usable == 0
    assert absent_task.missing_count == 1
    assert absent_task.mean is None
    assert root.missing_count == 1
    assert root.complete is False


def test_planned_matrix_rejects_duplicate_task_repeat_pair() -> None:
    observations = [scored("task-a", 7, 1), scored("task-a", 7, 1)]
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*duplicate)(?=.*task-a)(?=.*repeat(?:_id)?\D*7)",
    ):
        aggregate(
            observations,
            {"task-a": ("easy",)},
            expected_repeat_ids=(7,),
        )


def test_planned_matrix_rejects_unexpected_repeat() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*unexpected)(?=.*repeat(?:_id)?\D*3)",
    ):
        aggregate(
            [scored("task", 3, 1)],
            {"task": ("easy",)},
            expected_repeat_ids=(0,),
        )


def test_planned_matrix_rejects_duplicate_expected_repeat_ids() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*duplicate)(?=.*expected)(?=.*repeat_id\D*0)",
    ):
        aggregate(
            [],
            {"task": ("easy",)},
            expected_repeat_ids=(0, 0),
        )


def test_multi_stratum_task_contributes_once_to_each_stratum() -> None:
    root = aggregate(
        [scored("shared", 0, 1), scored("shared", 1, 0)],
        {"shared": ("easy", "hard")},
        expected_repeat_ids=(0, 1),
    )
    strata = {child.label: child for child in root.children}

    assert [child.label for child in root.children] == ["easy", "hard"]
    assert set(strata) == {"easy", "hard"}
    for stratum in strata.values():
        assert len(stratum.children) == 1
        task = stratum.children[0]
        assert task.label == "shared"
        assert task.total == 2
        assert task.usable == 2
        assert task.mean == pytest.approx(0.5)
    assert root.total == 4
    assert root.mean == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("strata", "match"),
    [
        ((), "at least one stratum"),
        ((" ",), "nonblank"),
        (("easy", "easy"), "duplicate stratum"),
    ],
    ids=["empty", "blank", "duplicate"],
)
def test_task_strata_require_canonical_label_tuples(
    strata: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate(
            [],
            {"task": strata},
            expected_repeat_ids=(0,),
        )


def test_task_strata_require_tuple() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        aggregate(
            [],
            {"task": ["easy"]},  # ty: ignore[invalid-argument-type]
            expected_repeat_ids=(0,),
        )


def test_task_strata_require_string_labels() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        aggregate(
            [],
            {"task": (1,)},  # ty: ignore[invalid-argument-type]
            expected_repeat_ids=(0,),
        )


def test_failed_observation_propagates_to_overall() -> None:
    observations = [
        scored("easy-0", 0, 1),
        failed("hard-0", 0),
    ]
    task_strata = {"easy-0": ("easy",), "hard-0": ("hard",)}
    root = aggregate(
        observations,
        task_strata,
        expected_repeat_ids=(0,),
    )
    assert root.mean is None
    assert root.complete is False
    assert root.failed_count == 1
    assert root.total == 2


def test_missing_task_stratum_raises() -> None:
    with pytest.raises(KeyError, match="no stratum"):
        aggregate(
            [scored("t", 0, 1)],
            {},
            expected_repeat_ids=(0,),
        )


def test_stratum_and_overall_helpers_compose() -> None:
    t0 = aggregate_task([scored("a", 0, 1), scored("a", 1, 1)])
    t1 = aggregate_task([scored("b", 0, 0)])
    stratum = aggregate_stratum([t0, t1])
    assert stratum.mean == pytest.approx(0.5)
    overall = aggregate_overall([stratum])
    assert overall.mean == pytest.approx(0.5)


def test_empty_stratum_is_incomplete_not_zero() -> None:
    agg = aggregate_stratum([])
    assert agg.mean is None
    assert agg.total == 0
    assert agg.complete is False


def test_empty_task_is_incomplete_not_zero() -> None:
    agg = aggregate_task([])
    assert agg.mean is None
    assert agg.total == 0
    assert agg.complete is False


def test_overall_with_empty_stratum_is_incomplete_not_zero() -> None:
    agg = aggregate_overall([aggregate_stratum([])])
    assert agg.mean is None
    assert agg.complete is False


def test_empty_overall_is_incomplete_not_zero() -> None:
    agg = aggregate_overall([])
    assert agg.mean is None
    assert agg.total == 0
    assert agg.complete is False


def test_empty_task_mixed_with_scored_does_not_silently_vanish() -> None:
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
