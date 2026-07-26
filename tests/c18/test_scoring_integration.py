"""c18 oracle wired into the shared strata-crossing aggregation ladder.

Checklist A's aggregation-crosses-strata check is exercised on the core
harness in ``tests/core/test_scoring.py``; this file confirms the c18
forward-chaining oracle's 0/1 results feed that ladder correctly across
depth strata, including the rubric-13 exhausted-observation case (a failed
observation must make the aggregate visibly incomplete, never silently
zero).
"""

from __future__ import annotations

from whetstone_envs.c18 import oracle
from whetstone_envs.core.scoring import aggregate, failed, scored

# Hand-traced theories (see test_oracle.py): D1 one-hop, D2 two-hop.
_D1 = "Sally is a brimpus. Every brimpus is sour."
_D1_QUERY = "True or false: Sally is sour."  # gold True
_D2 = "Stella is a lempus. Lempuses are zumpuses. Every zumpus is not floral."
_D2_QUERY = "True or false: Stella is not floral."  # gold True


def test_oracle_scores_feed_the_aggregation_ladder() -> None:
    observations = [
        # D1 stratum: one correct, one wrong prediction.
        scored("d1-a", 0, oracle.score("True", _D1, _D1_QUERY)),
        scored("d1-a", 1, oracle.score("True", _D1, _D1_QUERY)),
        scored("d1-b", 0, oracle.score("False", _D1, _D1_QUERY)),
        scored("d1-b", 1, oracle.score("False", _D1, _D1_QUERY)),
        # D2 stratum: one correct, one wrong.
        scored("d2-a", 0, oracle.score("True", _D2, _D2_QUERY)),
        scored("d2-b", 0, oracle.score("False", _D2, _D2_QUERY)),
    ]
    task_strata = {
        "d1-a": "D1",
        "d1-b": "D1",
        "d2-a": "D2",
        "d2-b": "D2",
    }
    root = aggregate(observations, task_strata)
    # D1: mean(1, 0) = 0.5 ; D2: mean(1, 0) = 0.5 ; overall 0.5.
    assert root.complete
    assert root.mean == 0.5
    assert len(root.children) == 2


def test_failed_observation_makes_aggregate_visibly_incomplete() -> None:
    observations = [
        scored("d1-a", 0, oracle.score("True", _D1, _D1_QUERY)),
        failed("d2-a", 0),
    ]
    task_strata = {"d1-a": "D1", "d2-a": "D2"}
    root = aggregate(observations, task_strata)
    # rubric 13: a failed observation must not silently score zero.
    assert root.mean is None
    assert not root.complete
    assert root.failed_count == 1
