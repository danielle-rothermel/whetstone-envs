from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Outcome(Enum):
    SCORED = "scored"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class Observation:
    """One repeat result: binary score, failed result, or absent expected
    result.
    """

    task_id: str
    repeat_id: int
    outcome: Outcome = Outcome.SCORED
    score: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            msg = f"outcome must be an Outcome, got {self.outcome!r}"
            raise TypeError(msg)
        if self.outcome is Outcome.SCORED:
            if self.score not in (0, 1):
                msg = (
                    f"scored observation for task {self.task_id!r} "
                    f"repeat {self.repeat_id} must have score 0 or 1, "
                    f"got {self.score!r}"
                )
                raise ValueError(msg)
        elif self.score is not None:
            msg = (
                f"{self.outcome.value} observation for task "
                f"{self.task_id!r} repeat {self.repeat_id} must not "
                f"carry a score, got {self.score!r}"
            )
            raise ValueError(msg)


def scored(task_id: str, repeat_id: int, score: int) -> Observation:
    return Observation(task_id, repeat_id, Outcome.SCORED, score)


def failed(task_id: str, repeat_id: int) -> Observation:
    return Observation(task_id, repeat_id, Outcome.FAILED, None)


def missing(task_id: str, repeat_id: int) -> Observation:
    return Observation(task_id, repeat_id, Outcome.MISSING, None)
