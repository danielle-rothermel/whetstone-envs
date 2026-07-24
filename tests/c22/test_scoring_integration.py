"""c22 oracle wired into the shared strata-crossing aggregation ladder.

Checklist A's aggregation-crosses-strata check is exercised on the core
harness in ``tests/core/test_scoring.py``; this file confirms the c22
oracle's 0/1 results feed that ladder correctly, including the
rubric-13 exhausted-observation case (a failed observation must make the
aggregate visibly incomplete, never silently zero).
"""

from __future__ import annotations

from whetstone_envs.c22 import oracle
from whetstone_envs.c22.spec import ConstraintSpec
from whetstone_envs.core.scoring import (
    aggregate,
    failed,
    scored,
)


def _spec(end_phrase: str) -> ConstraintSpec:
    return ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=("end with a phrase",),
        instruction_id_list=("startend:end_checker",),
        kwargs_list=({"end_phrase": end_phrase},),
    )


def test_oracle_scores_feed_the_aggregation_ladder() -> None:
    spec = _spec("DONE")
    # Two tasks in one stratum, two in another; hand-picked responses.
    observations = [
        scored("task-a", 0, oracle.check(spec, "blue DONE").score),  # 1
        scored("task-a", 1, oracle.check(spec, "blue DONE").score),  # 1
        scored("task-b", 0, oracle.check(spec, "blue").score),  # 0
        scored("task-b", 1, oracle.check(spec, "blue").score),  # 0
        scored("task-c", 0, oracle.check(spec, "red DONE").score),  # 1
        scored("task-c", 1, oracle.check(spec, "red DONE").score),  # 1
        scored("task-d", 0, oracle.check(spec, "red").score),  # 0
        scored("task-d", 1, oracle.check(spec, "red").score),  # 0
    ]
    task_strata = {
        "task-a": ("n3_easy",),
        "task-b": ("n3_easy",),
        "task-c": ("n3_mixed",),
        "task-d": ("n3_mixed",),
    }
    root = aggregate(observations, task_strata, expected_repeat_ids=(0, 1))
    # n3_easy: mean(1, 0) = 0.5 ; n3_mixed: mean(1, 0) = 0.5 ; overall 0.5
    assert root.complete
    assert root.mean == 0.5
    assert len(root.children) == 2


def test_failed_observation_makes_aggregate_visibly_incomplete() -> None:
    spec = _spec("DONE")
    observations = [
        scored("task-a", 0, oracle.check(spec, "blue DONE").score),  # 1
        failed("task-b", 0),  # infra failure, no score
    ]
    task_strata = {"task-a": ("n3_easy",), "task-b": ("n3_easy",)}
    root = aggregate(observations, task_strata, expected_repeat_ids=(0,))
    # rubric 13: a failed observation must not silently score zero; the
    # aggregate is incomplete and names the shortfall.
    assert root.mean is None
    assert not root.complete
    assert root.failed_count == 1
