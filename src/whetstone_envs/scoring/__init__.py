"""Exact-match scoring, observations, and aggregate reductions."""

from whetstone_envs.scoring.aggregation import (
    Aggregate,
    aggregate,
    aggregate_overall,
    aggregate_stratum,
    aggregate_task,
)
from whetstone_envs.scoring.exact_match import exact_match
from whetstone_envs.scoring.observations import (
    Observation,
    Outcome,
    failed,
    missing,
    scored,
)

__all__ = [
    "Aggregate",
    "Observation",
    "Outcome",
    "aggregate",
    "aggregate_overall",
    "aggregate_stratum",
    "aggregate_task",
    "exact_match",
    "failed",
    "missing",
    "scored",
]
