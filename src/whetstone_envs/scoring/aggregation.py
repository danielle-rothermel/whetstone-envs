from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from whetstone_envs.scoring.observations import (
    Observation,
    Outcome,
    missing,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Aggregate:
    """An equal-weighted mean with leaf/path contribution counts.

    Aggregation helpers return ``mean=None`` for empty or incomplete
    contributions. ``label`` identifies task or stratum nodes returned by
    :func:`aggregate`. Counters record leaf/path contributions, so a
    task/repeat cell in multiple strata counts once per stratum at the root.
    """

    mean: float | None
    usable: int
    failed_count: int
    missing_count: int
    label: str | None = None
    children: tuple[Aggregate, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Return the summed leaf/path contribution count."""
        return self.usable + self.failed_count + self.missing_count

    @property
    def complete(self) -> bool:
        """Reports whether mean is available."""
        return self.mean is not None


def _mean_of_observations(
    observations: Iterable[Observation],
    *,
    label: str | None = None,
) -> Aggregate:
    scores: list[int] = []
    failed_count = 0
    missing_count = 0
    for observation in observations:
        if observation.outcome is Outcome.SCORED:
            assert observation.score is not None
            scores.append(observation.score)
        elif observation.outcome is Outcome.FAILED:
            failed_count += 1
        elif observation.outcome is Outcome.MISSING:
            missing_count += 1
        else:
            msg = (
                "observation outcome must be an Outcome, "
                f"got {observation.outcome!r}"
            )
            raise TypeError(msg)
    complete = failed_count == 0 and missing_count == 0
    mean = (sum(scores) / len(scores)) if (scores and complete) else None
    return Aggregate(
        mean=mean,
        usable=len(scores),
        failed_count=failed_count,
        missing_count=missing_count,
        label=label,
    )


def _mean_of_children(
    children: list[Aggregate],
    *,
    label: str | None = None,
) -> Aggregate:
    """Average equally only when all children are complete.

    Always sum counts.
    """
    usable = sum(child.usable for child in children)
    failed_count = sum(child.failed_count for child in children)
    missing_count = sum(child.missing_count for child in children)
    all_complete = all(child.complete for child in children) and bool(children)
    means = [child.mean for child in children if child.mean is not None]
    mean = (sum(means) / len(means)) if (all_complete and means) else None
    return Aggregate(
        mean=mean,
        usable=usable,
        failed_count=failed_count,
        missing_count=missing_count,
        label=label,
        children=tuple(children),
    )


def aggregate_task(observations: Iterable[Observation]) -> Aggregate:
    """Aggregate unique repeat IDs for one task."""
    materialized = list(observations)
    task_ids = {observation.task_id for observation in materialized}
    if len(task_ids) > 1:
        msg = "aggregate_task requires observations from a single task"
        raise ValueError(msg)
    label = next(iter(task_ids), None)
    repeat_ids: set[int] = set()
    for observation in materialized:
        if observation.repeat_id in repeat_ids:
            msg = (
                f"duplicate observation for task {observation.task_id!r} "
                f"repeat_id {observation.repeat_id}"
            )
            raise ValueError(msg)
        repeat_ids.add(observation.repeat_id)
    return _mean_of_observations(materialized, label=label)


def aggregate_stratum(task_aggregates: list[Aggregate]) -> Aggregate:
    return _mean_of_children(task_aggregates)


def aggregate_overall(stratum_aggregates: list[Aggregate]) -> Aggregate:
    return _mean_of_children(stratum_aggregates)


def _validate_task_strata(
    task_strata: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate and materialize task-to-strata assignments."""
    planned_tasks = tuple(task_strata.items())
    for task_id, strata in planned_tasks:
        if not isinstance(strata, tuple):
            msg = (
                f"task_strata for task {task_id!r} must be a tuple of "
                "stratum labels"
            )
            raise TypeError(msg)
        if not strata:
            msg = f"task {task_id!r} must declare at least one stratum"
            raise ValueError(msg)
        seen_strata: set[str] = set()
        for stratum in strata:
            if not isinstance(stratum, str):
                msg = f"stratum labels for task {task_id!r} must be strings"
                raise TypeError(msg)
            if not stratum.strip():
                msg = f"stratum labels for task {task_id!r} must be nonblank"
                raise ValueError(msg)
            if stratum in seen_strata:
                msg = (
                    f"duplicate stratum label {stratum!r} for task "
                    f"{task_id!r}; labels must be deduplicated"
                )
                raise ValueError(msg)
            seen_strata.add(stratum)
    return planned_tasks


def aggregate(
    observations: Iterable[Observation],
    task_strata: Mapping[str, tuple[str, ...]],
    *,
    expected_repeat_ids: Iterable[int],
) -> Aggregate:
    """Aggregate the complete planned task/repeat matrix.

    Missing cells become ``MISSING``. Repeats within tasks, tasks within
    strata, and strata into the root are equally weighted. Unknown task or
    repeat IDs and duplicate expected or observed cells raise.
    """
    expected_repeats = tuple(expected_repeat_ids)
    expected_repeat_set: set[int] = set()
    for repeat_id in expected_repeats:
        if repeat_id in expected_repeat_set:
            msg = f"duplicate expected repeat_id {repeat_id}"
            raise ValueError(msg)
        expected_repeat_set.add(repeat_id)

    planned_tasks = _validate_task_strata(task_strata)
    expected_task_ids = {task_id for task_id, _ in planned_tasks}

    observations_by_cell: dict[tuple[str, int], Observation] = {}
    for observation in observations:
        if observation.task_id not in expected_task_ids:
            msg = f"task {observation.task_id!r} has no stratum in task_strata"
            raise KeyError(msg)
        if observation.repeat_id not in expected_repeat_set:
            msg = (
                f"unexpected repeat_id {observation.repeat_id} "
                f"for task {observation.task_id!r}"
            )
            raise ValueError(msg)
        cell = (observation.task_id, observation.repeat_id)
        if cell in observations_by_cell:
            msg = (
                f"duplicate observation for task {observation.task_id!r} "
                f"repeat_id {observation.repeat_id}"
            )
            raise ValueError(msg)
        observations_by_cell[cell] = observation

    by_stratum: dict[str, list[Aggregate]] = {}
    for task_id, strata in planned_tasks:
        task_observations = [
            observations_by_cell[(task_id, repeat_id)]
            if (task_id, repeat_id) in observations_by_cell
            else missing(task_id, repeat_id)
            for repeat_id in expected_repeats
        ]
        task_aggregate = _mean_of_observations(
            task_observations,
            label=task_id,
        )
        for stratum in strata:
            by_stratum.setdefault(stratum, []).append(task_aggregate)

    stratum_aggregates = [
        _mean_of_children(tasks, label=stratum)
        for stratum, tasks in by_stratum.items()
    ]
    return aggregate_overall(stratum_aggregates)
