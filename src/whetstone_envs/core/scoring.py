"""Exact-match scoring and the strata-crossing aggregation ladder.

Two pieces live here, shared by every candidate:

* :func:`exact_match` -- the deterministic 0/1 check. Correctness is a
  string equality after :func:`whetstone_envs.core.probes.normalize`;
  there is no per-instance partial credit (rubric criteria 2 and the
  aggregation-crosses-strata callout).
* the aggregation ladder -- reduce repeat -> task -> stratum -> overall,
  in that order, so improvements only ever come from solving more
  complete instances across strata, never from partial credit inside
  one.

Rubric criterion 13 is enforced structurally: an :class:`Observation`
is either a scored 0/1 *or* a ``failed`` / ``missing`` marker. A failed
or missing observation makes every aggregate that depends on it
**visibly incomplete** -- the returned :class:`Aggregate` reports the
count of unusable observations and its ``mean`` is ``None`` unless every
contributing observation resolved. Failed results are never silently
coerced to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from whetstone_envs.core.probes import normalize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if ``prediction`` equals ``gold`` after normalization.

    Both sides are passed through
    :func:`whetstone_envs.core.probes.normalize` so fence/whitespace
    handling is identical everywhere. The result is exactly ``0`` or
    ``1`` -- no partial credit.
    """
    return int(normalize(prediction) == normalize(gold))


class Outcome(Enum):
    """Why an :class:`Observation` does or does not carry a score."""

    SCORED = "scored"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class Observation:
    """One repeat's result for one task.

    A ``SCORED`` observation carries ``score`` in ``{0, 1}``. A
    ``FAILED`` observation stands for an exhausted infrastructure
    failure; a ``MISSING`` one for a repeat that was expected but never
    produced a result. Neither carries a score, and both force any
    aggregate over them to be visibly incomplete.
    """

    task_id: str
    repeat_id: int
    outcome: Outcome = Outcome.SCORED
    score: int | None = None

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A mean over lower-level results, plus its completeness accounting.

    ``mean`` is ``None`` whenever any contributing observation was
    ``FAILED`` or ``MISSING`` (or, at higher levels, whenever any
    contributing sub-aggregate was itself incomplete). ``complete`` is
    the boolean form of the same fact. ``usable`` / ``failed_count`` /
    ``missing_count`` expose *why* an aggregate is incomplete so the gap
    is visible rather than silently averaged away.
    """

    mean: float | None
    usable: int
    failed_count: int
    missing_count: int
    children: tuple[Aggregate, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Number of observations/children this aggregate spans."""
        return self.usable + self.failed_count + self.missing_count

    @property
    def complete(self) -> bool:
        """True only if every contributing result resolved to a score."""
        return self.failed_count == 0 and self.missing_count == 0


def _mean_of_observations(obs: Iterable[Observation]) -> Aggregate:
    """Aggregate raw observations (the repeat level).

    Any non-``SCORED`` observation makes the result incomplete: ``mean``
    is ``None`` and the failed/missing counts record the shortfall.
    """
    scores: list[int] = []
    failed_count = 0
    missing_count = 0
    for ob in obs:
        if ob.outcome is Outcome.SCORED:
            assert ob.score is not None
            scores.append(ob.score)
        elif ob.outcome is Outcome.FAILED:
            failed_count += 1
        else:
            missing_count += 1
    complete = failed_count == 0 and missing_count == 0
    mean = (sum(scores) / len(scores)) if (scores and complete) else None
    return Aggregate(
        mean=mean,
        usable=len(scores),
        failed_count=failed_count,
        missing_count=missing_count,
    )


def _mean_of_children(children: list[Aggregate]) -> Aggregate:
    """Aggregate one level up from a list of sub-aggregates.

    A child that is itself incomplete propagates incompleteness upward:
    the parent's ``mean`` is ``None`` unless every child is complete.
    Failed/missing counts are summed so the root aggregate still names
    the total shortfall.
    """
    usable = sum(c.usable for c in children)
    failed_count = sum(c.failed_count for c in children)
    missing_count = sum(c.missing_count for c in children)
    all_complete = all(c.complete for c in children) and bool(children)
    means = [c.mean for c in children if c.mean is not None]
    mean = (sum(means) / len(means)) if (all_complete and means) else None
    return Aggregate(
        mean=mean,
        usable=usable,
        failed_count=failed_count,
        missing_count=missing_count,
        children=tuple(children),
    )


def aggregate_task(observations: Iterable[Observation]) -> Aggregate:
    """Mean the repeats within a single task."""
    return _mean_of_observations(observations)


def aggregate_stratum(task_aggregates: list[Aggregate]) -> Aggregate:
    """Mean the per-task aggregates within a single stratum."""
    return _mean_of_children(task_aggregates)


def aggregate_overall(stratum_aggregates: list[Aggregate]) -> Aggregate:
    """Mean the per-stratum aggregates across all strata."""
    return _mean_of_children(stratum_aggregates)


def aggregate(
    observations: Iterable[Observation],
    task_strata: Mapping[str, str],
) -> Aggregate:
    """Run the full repeat -> task -> stratum -> overall ladder.

    ``observations`` are grouped by ``task_id`` (repeats collapse first),
    each task is mapped to its stratum via ``task_strata``, tasks are
    meaned within stratum, and strata are meaned into the overall
    aggregate. The returned root :class:`Aggregate` exposes the stratum
    aggregates as ``children`` (each of which exposes its task
    aggregates), so the whole reduction is inspectable.

    Every task that appears in an observation must have an entry in
    ``task_strata``; a missing mapping raises rather than silently
    dropping the task from the aggregate.
    """
    by_task: dict[str, list[Observation]] = {}
    for ob in observations:
        by_task.setdefault(ob.task_id, []).append(ob)

    by_stratum: dict[str, list[Aggregate]] = {}
    for task_id, obs in by_task.items():
        if task_id not in task_strata:
            msg = f"task {task_id!r} has no stratum in task_strata"
            raise KeyError(msg)
        stratum = task_strata[task_id]
        by_stratum.setdefault(stratum, []).append(aggregate_task(obs))

    stratum_aggregates = [
        aggregate_stratum(tasks) for tasks in by_stratum.values()
    ]
    return aggregate_overall(stratum_aggregates)
