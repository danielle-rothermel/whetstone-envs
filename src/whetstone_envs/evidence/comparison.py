from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify

from whetstone_envs.evidence.evaluation import (
    EvalReport,
    Observation,
    ObservationState,
)


@verify(UNIQUE)
class ComparisonOutcome(StrEnum):
    BOTH_CORRECT = auto()
    BOTH_WRONG = auto()
    LEFT_ONLY = auto()
    RIGHT_ONLY = auto()
    EXECUTION_MISMATCH = auto()


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    outcome: ComparisonOutcome
    task_id: str
    seed_index: int
    candidate_a: Observation | None
    candidate_b: Observation | None


def classify_pair(
    left: Observation | None, right: Observation | None
) -> ComparisonOutcome:
    if (
        left is None
        or right is None
        or left.state is not ObservationState.SCORED
        or right.state is not ObservationState.SCORED
    ):
        return ComparisonOutcome.EXECUTION_MISMATCH
    if left.score == 1.0 and right.score == 1.0:
        return ComparisonOutcome.BOTH_CORRECT
    if left.score == 0.0 and right.score == 0.0:
        return ComparisonOutcome.BOTH_WRONG
    return (
        ComparisonOutcome.LEFT_ONLY
        if left.score == 1.0
        else ComparisonOutcome.RIGHT_ONLY
    )


def compare_candidates(
    report: EvalReport, candidate_a: str, candidate_b: str
) -> tuple[ComparisonRow, ...]:
    by_coordinate = {
        (row.candidate_name, row.task_id, row.seed_index): row
        for row in report.observations
    }
    return tuple(
        _compare_coordinate(
            task.task_id,
            seed_index,
            by_coordinate.get((candidate_a, task.task_id, seed_index)),
            by_coordinate.get((candidate_b, task.task_id, seed_index)),
        )
        for task in report.tasks
        for seed_index in range(report.run.repeats)
    )


def _compare_coordinate(
    task_id: str,
    seed_index: int,
    left: Observation | None,
    right: Observation | None,
) -> ComparisonRow:
    return ComparisonRow(
        outcome=classify_pair(left, right),
        task_id=task_id,
        seed_index=seed_index,
        candidate_a=left,
        candidate_b=right,
    )
