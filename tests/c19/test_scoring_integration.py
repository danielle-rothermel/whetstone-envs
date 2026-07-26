"""c19 oracle wired into the shared strata-crossing aggregation ladder.

Checklist A's aggregation-crosses-strata check is exercised on the core
harness in ``tests/core/test_scoring.py``; this file confirms the c19
oracle's 0/1 results feed that ladder correctly, including the rubric-13
exhausted-observation case (a failed observation must make the aggregate
visibly incomplete, never silently zero).
"""

from __future__ import annotations

from whetstone_envs.c19 import oracle
from whetstone_envs.core.scoring import aggregate, failed, scored

# Two hand-built grids with hand-traced golds (see test_oracle.py tracing).
_GRID = "\n".join(
    ["WGWGWGWGWG", "WG>>    WG", "WG      WG", "WG      WG", "WGWGWGWGWG"],
)
_CMD = "FF"  # east from (1,1) -> (1,3)
_GOLD = "1,3"


def test_oracle_scores_feed_the_aggregation_ladder() -> None:
    observations = [
        scored("task-a", 0, oracle.score(_GOLD, _GRID, _CMD, "coordinate")),
        scored("task-a", 1, oracle.score(_GOLD, _GRID, _CMD, "coordinate")),
        scored("task-b", 0, oracle.score("9,9", _GRID, _CMD, "coordinate")),
        scored("task-b", 1, oracle.score("9,9", _GRID, _CMD, "coordinate")),
        scored("task-c", 0, oracle.score("E", _GRID, _CMD, "heading")),
        scored("task-d", 0, oracle.score("garbage", _GRID, _CMD, "heading")),
    ]
    task_strata = {
        "task-a": "Empty|coordinate",
        "task-b": "Empty|coordinate",
        "task-c": "Empty|heading",
        "task-d": "Empty|heading",
    }
    root = aggregate(observations, task_strata)
    # coordinate: mean(1, 0) = 0.5 ; heading: mean(1, 0) = 0.5 ; overall 0.5
    assert root.complete
    assert root.mean == 0.5
    assert len(root.children) == 2


def test_failed_observation_makes_aggregate_visibly_incomplete() -> None:
    observations = [
        scored("task-a", 0, oracle.score(_GOLD, _GRID, _CMD, "coordinate")),
        failed("task-b", 0),
    ]
    task_strata = {"task-a": "Empty|coordinate", "task-b": "Empty|coordinate"}
    root = aggregate(observations, task_strata)
    # rubric 13: a failed observation must not silently score zero.
    assert root.mean is None
    assert not root.complete
    assert root.failed_count == 1
