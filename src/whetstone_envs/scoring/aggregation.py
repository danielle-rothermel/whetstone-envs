"""Complete repeat-to-task-to-stratum-to-overall aggregation.

The reduction operates over the complete planned task/repeat matrix, so absent
results remain visible and improvements only come from solving more complete
instances across strata, never from partial credit inside one. A failed or
missing observation makes every dependent aggregate visibly incomplete.
"""

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
    """A mean over lower-level results, plus completeness accounting.

    ``mean`` is ``None`` whenever any contributing observation was ``FAILED``
    or ``MISSING`` or any contributing sub-aggregate was incomplete.
    ``usable``, ``failed_count``, and ``missing_count`` expose why an aggregate
    is incomplete. ``label`` identifies task and stratum children returned by
    :func:`aggregate`; helper and root aggregates may leave it unset.
    """

    mean: float | None
    usable: int
    failed_count: int
    missing_count: int
    label: str | None = None
    children: tuple[Aggregate, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Return the number of observations this aggregate spans."""
        return self.usable + self.failed_count + self.missing_count

    @property
    def complete(self) -> bool:
        """Return whether every contributing result resolved to a score.

        Equivalent to ``mean is not None``: a resolved mean is produced only
        when there is at least one usable observation, no failed or missing
        observation, and every contributing child is itself complete. A
        zero-observation aggregate therefore reports ``complete=False``, and
        an incomplete child forces its parent incomplete rather than silently
        vanishing from the mean.
        """
        return self.mean is not None


def _mean_of_observations(
    observations: Iterable[Observation],
    *,
    label: str | None = None,
) -> Aggregate:
    """Aggregate raw observations at the repeat level.

    Any non-``SCORED`` observation makes the result incomplete: ``mean`` is
    ``None`` and the failed or missing counts record the shortfall.
    """
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
    """Aggregate one level up from lower-level aggregates.

    A child that is itself incomplete propagates incompleteness upward: the
    parent's ``mean`` is ``None`` unless every child is complete. Failed and
    missing counts are summed so the root aggregate still names the shortfall.
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
    """Mean unique repeats within one task."""
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
    """Mean the per-task aggregates within one stratum."""
    return _mean_of_children(task_aggregates)


def aggregate_overall(stratum_aggregates: list[Aggregate]) -> Aggregate:
    """Mean the per-stratum aggregates across all strata."""
    return _mean_of_children(stratum_aggregates)


def _validate_task_strata(
    task_strata: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Materialize canonical task-to-strata assignments."""
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
    """Run the complete repeat-to-task-to-stratum-to-overall reduction.

    Every key in ``task_strata`` and every unique ID in
    ``expected_repeat_ids`` defines one planned task/repeat cell. Each task
    maps to a non-empty, ordered tuple of unique, nonblank stratum labels and
    contributes its complete repeat aggregate once to every named stratum.
    Absent cells are materialized as ``MISSING`` observations before repeats
    collapse into tasks, tasks into strata, and strata into the overall
    aggregate. The returned root :class:`Aggregate` exposes labeled stratum
    children, each of which exposes labeled task children, so the whole
    reduction is inspectable.

    Every observed task must have an entry in ``task_strata``. Duplicate
    expected repeat IDs, unexpected observed repeats, and duplicate observed
    task/repeat cells raise rather than changing or silently shrinking the
    planned reduction.
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
