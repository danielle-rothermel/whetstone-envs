"""c23 oracle wired into the shared strata-crossing aggregation ladder.

Checklist A's aggregation-crosses-strata check is exercised on the core
harness in ``tests/core/test_scoring.py``; this file confirms the c23
oracle's 0/1 results feed that ladder correctly across ISL/L-OSL/R-OSL
strata, including the rubric-13 exhausted-observation case (a failed
observation must make the aggregate visibly incomplete, never silently
zero).
"""

from __future__ import annotations

from whetstone_envs.c23 import oracle, upstream
from whetstone_envs.core.scoring import aggregate, failed, scored

# Hand-traced fixtures (see test_oracle.py): one per stratum family.
_ISL = (upstream.ISL, 2, {"cb": "d"}, "acb", "acd")  # gold 'acd'
_LOSL = (upstream.L_OSL, 2, {"ab": "c"}, "abab", "acac")  # gold 'acac'
_ROSL = (upstream.R_OSL, 2, {"ab": "d"}, "abab", "adab")  # gold 'adab'


def _score(fixture: tuple, prediction: str) -> int:
    rule_type, k, rule, query, _gold = fixture
    return oracle.score(prediction, rule_type, k, rule, query)


def test_oracle_scores_feed_the_aggregation_ladder() -> None:
    observations = [
        # S1 (ISL): one correct, one wrong.
        scored("s1-a", 0, _score(_ISL, _ISL[4])),
        scored("s1-a", 1, _score(_ISL, _ISL[4])),
        scored("s1-b", 0, _score(_ISL, "acb")),  # wrong (identity)
        scored("s1-b", 1, _score(_ISL, "acb")),
        # S2 (L-OSL): one correct, one wrong.
        scored("s2-a", 0, _score(_LOSL, _LOSL[4])),
        scored("s2-a", 1, _score(_LOSL, _LOSL[4])),
        scored("s2-b", 0, _score(_LOSL, "abab")),  # wrong
        scored("s2-b", 1, _score(_LOSL, "abab")),
        # S3 (R-OSL): one correct.
        scored("s3-a", 0, _score(_ROSL, _ROSL[4])),
        scored("s3-a", 1, _score(_ROSL, _ROSL[4])),
    ]
    task_strata = {
        "s1-a": ("S1",),
        "s1-b": ("S1",),
        "s2-a": ("S2",),
        "s2-b": ("S2",),
        "s3-a": ("S3",),
    }
    root = aggregate(observations, task_strata, expected_repeat_ids=(0, 1))
    # S1: mean(1, 0) = 0.5 ; S2: mean(1, 0) = 0.5 ; S3: mean(1) = 1.0
    # overall = mean(0.5, 0.5, 1.0) = 2/3.
    assert root.complete
    assert root.mean is not None
    assert abs(root.mean - (2 / 3)) < 1e-9
    assert len(root.children) == 3


def test_failed_observation_makes_aggregate_visibly_incomplete() -> None:
    observations = [
        scored("s1-a", 0, _score(_ISL, _ISL[4])),
        failed("s2-a", 0),
    ]
    task_strata = {"s1-a": ("S1",), "s2-a": ("S2",)}
    root = aggregate(observations, task_strata, expected_repeat_ids=(0,))
    # rubric 13: a failed observation must not silently score zero.
    assert root.mean is None
    assert not root.complete
    assert root.failed_count == 1
