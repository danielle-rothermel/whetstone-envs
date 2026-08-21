from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.reporting.schema import (
    EvalReport,
    Observation,
    ObservationState,
)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    bucket: str
    task_id: str
    seed_index: int
    candidate_a: Observation | None
    candidate_b: Observation | None


def compare_buckets(
    report: EvalReport, candidate_a: str, candidate_b: str
) -> tuple[ComparisonRow, ...]:
    """Classify paired observations without presentation dependencies."""
    by_coordinate = {
        (row.candidate_name, row.task_id, row.seed_index): row
        for row in report.observations
    }
    rows: list[ComparisonRow] = []
    for task in report.tasks:
        for seed_index in range(report.run.repeats):
            a = by_coordinate.get((candidate_a, task.task_id, seed_index))
            b = by_coordinate.get((candidate_b, task.task_id, seed_index))
            if (
                a is None
                or b is None
                or a.state != b.state
                or a.state is not ObservationState.SCORED
            ):
                bucket = "execution mismatch"
            elif a.score == 1.0 and b.score == 1.0:
                bucket = "both correct"
            elif a.score == 0.0 and b.score == 0.0:
                bucket = "both wrong"
            elif a.score == 1.0:
                bucket = f"{candidate_a} only"
            else:
                bucket = f"{candidate_b} only"
            rows.append(
                ComparisonRow(
                    bucket=bucket,
                    task_id=task.task_id,
                    seed_index=seed_index,
                    candidate_a=a,
                    candidate_b=b,
                )
            )
    return tuple(rows)


__all__ = ["ComparisonRow", "compare_buckets"]
