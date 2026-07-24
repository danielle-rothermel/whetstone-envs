"""c11 oracle wired into the shared strata-crossing aggregation ladder.

Checklist A's aggregation-crosses-strata check is exercised on the core
harness in ``tests/core/test_scoring.py``; this file confirms the c11
oracle's 0/1 results feed that ladder correctly, including the rubric-13
exhausted-observation case (a failed observation must make the aggregate
visibly incomplete, never silently zero).
"""

from __future__ import annotations

from whetstone_envs.c11 import oracle
from whetstone_envs.core.scoring import aggregate, failed, scored

# One hand-built instance: messy input and its canonical gold.
_MESSY = '{"b": 2, "a": 1}'
_GOLD = '{"a":1,"b":2}'


def test_oracle_scores_feed_the_aggregation_ladder() -> None:
    # Two tasks per stratum, hand-picked predictions (correct/incorrect).
    observations = [
        scored("task-a", 0, oracle.score(_GOLD, _MESSY)),  # 1 (canonical)
        scored("task-a", 1, oracle.score(_GOLD, _MESSY)),  # 1
        scored("task-b", 0, oracle.score(_MESSY, _MESSY)),  # 0 (still messy)
        scored("task-b", 1, oracle.score(_MESSY, _MESSY)),  # 0
        scored("task-c", 0, oracle.score(_GOLD, _MESSY)),  # 1
        scored("task-c", 1, oracle.score(_GOLD, _MESSY)),  # 1
        scored("task-d", 0, oracle.score("garbage", _MESSY)),  # 0
        scored("task-d", 1, oracle.score("garbage", _MESSY)),  # 0
    ]
    task_strata = {
        "task-a": ("S1_flat",),
        "task-b": ("S1_flat",),
        "task-c": ("S2_keysort",),
        "task-d": ("S2_keysort",),
    }
    root = aggregate(observations, task_strata, expected_repeat_ids=(0, 1))
    # S1_flat: mean(1, 0) = 0.5 ; S2_keysort: mean(1, 0) = 0.5 ; overall 0.5
    assert root.complete
    assert root.mean == 0.5
    assert len(root.children) == 2


def test_failed_observation_makes_aggregate_visibly_incomplete() -> None:
    observations = [
        scored("task-a", 0, oracle.score(_GOLD, _MESSY)),  # 1
        failed("task-b", 0),  # infra failure, no score
    ]
    task_strata = {"task-a": ("S1_flat",), "task-b": ("S1_flat",)}
    root = aggregate(observations, task_strata, expected_repeat_ids=(0,))
    # rubric 13: a failed observation must not silently score zero.
    assert root.mean is None
    assert not root.complete
    assert root.failed_count == 1
