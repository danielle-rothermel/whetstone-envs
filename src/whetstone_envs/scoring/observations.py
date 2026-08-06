"""Repeat-level binary scores and explicit failure states.

An :class:`Observation` is either a scored 0/1 or a ``failed`` / ``missing``
marker. Failed results are never silently coerced to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Outcome(Enum):
    """Why an :class:`Observation` does or does not carry a score."""

    SCORED = "scored"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class Observation:
    """One repeat's result for one task.

    A ``SCORED`` observation carries ``score`` in ``{0, 1}``. A ``FAILED``
    observation stands for an exhausted infrastructure failure; a ``MISSING``
    one for a repeat that was expected but never produced a result. Neither
    carries a score, and both force any aggregate over them to be visibly
    incomplete.
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
    """Build a ``SCORED`` observation."""
    return Observation(task_id, repeat_id, Outcome.SCORED, score)


def failed(task_id: str, repeat_id: int) -> Observation:
    """Build a ``FAILED`` observation (exhausted infra failure)."""
    return Observation(task_id, repeat_id, Outcome.FAILED, None)


def missing(task_id: str, repeat_id: int) -> Observation:
    """Build a ``MISSING`` observation (expected result never produced)."""
    return Observation(task_id, repeat_id, Outcome.MISSING, None)
