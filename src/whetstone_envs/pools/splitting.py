"""Deterministic, destination-balanced task-pool splitting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from whetstone_envs.instances import Instance

if TYPE_CHECKING:
    from whetstone_envs.pools.pool import TaskPool


def _validate_split_sizes(
    destination_sizes: tuple[int, int, int],
    *,
    pool_size: int,
) -> int:
    """Validate destination capacities and return their total."""
    for name, size in zip(
        ("internal_eval_n", "official_n", "held_out_n"),
        destination_sizes,
        strict=True,
    ):
        if size < 0:
            msg = f"{name} must be non-negative, got {size}"
            raise ValueError(msg)
    total = sum(destination_sizes)
    if total > pool_size:
        msg = (
            f"split sizes sum to {total} but pool has only "
            f"{pool_size} instances"
        )
        raise ValueError(msg)
    return total


@dataclass(frozen=True, slots=True)
class PoolSplit:
    """Three disjoint instance subsets carved from a :class:`TaskPool`.

    The constructor asserts the subsets share no instance ``id`` -- the
    guarantee the optimizer relies on to keep held-out tasks unseen.
    """

    internal_eval: tuple[Instance, ...]
    official: tuple[Instance, ...]
    held_out: tuple[Instance, ...]

    def __post_init__(self) -> None:
        groups = {
            "internal_eval": self.internal_eval,
            "official": self.official,
            "held_out": self.held_out,
        }
        seen: dict[str, str] = {}
        for name, subset in groups.items():
            for inst in subset:
                if inst.id in seen:
                    msg = (
                        f"split is not disjoint: instance {inst.id!r} "
                        f"appears in both {seen[inst.id]!r} and "
                        f"{name!r}"
                    )
                    raise AssertionError(msg)
                seen[inst.id] = name


def split_pool(
    pool: TaskPool,
    internal_eval_n: int,
    official_n: int,
    held_out_n: int,
) -> PoolSplit:
    """Carve ``pool`` into three disjoint stratified subsets.

    This function implements the algorithm documented by
    :meth:`whetstone_envs.pools.TaskPool.split`; callers normally use that
    method rather than invoking this internal implementation directly.
    """
    destination_sizes = (
        internal_eval_n,
        official_n,
        held_out_n,
    )
    total = _validate_split_sizes(
        destination_sizes,
        pool_size=len(pool.instances),
    )

    by_combination: dict[tuple[str, ...], deque[Instance]] = {}
    for inst in pool.instances:
        by_combination.setdefault(inst.strata, deque()).append(inst)

    selected_by_combination = dict.fromkeys(by_combination, 0)
    active = deque(by_combination)
    selected = 0
    while selected < total:
        combination = active.popleft()
        selected_by_combination[combination] += 1
        selected += 1
        if selected_by_combination[combination] < len(
            by_combination[combination]
        ):
            active.append(combination)

    destination_assigned = [0, 0, 0]
    destination_quotas = [
        dict.fromkeys(by_combination, 0) for _ in destination_sizes
    ]
    destinations_by_combination: dict[tuple[str, ...], list[int]] = {
        combination: [] for combination in by_combination
    }
    coverage_targets = [
        min(size, len(by_combination)) for size in destination_sizes
    ]

    for combination in by_combination:
        combination_quota = selected_by_combination[combination]
        for _ in range(min(combination_quota, len(destination_sizes))):
            eligible = [
                index
                for index in range(len(destination_sizes))
                if destination_quotas[index][combination] == 0
                and destination_assigned[index] < coverage_targets[index]
            ]
            if not eligible:
                break
            destination = min(
                eligible,
                key=lambda index: (
                    -(coverage_targets[index] - destination_assigned[index]),
                    -destination_sizes[index],
                    index,
                ),
            )
            destination_quotas[destination][combination] += 1
            destination_assigned[destination] += 1
            destinations_by_combination[combination].append(destination)

    combination_order = {
        combination: index for index, combination in enumerate(by_combination)
    }
    for destination, size in enumerate(destination_sizes):
        while destination_assigned[destination] < size:
            eligible = [
                combination
                for combination in by_combination
                if len(destinations_by_combination[combination])
                < selected_by_combination[combination]
            ]
            combination = min(
                eligible,
                key=lambda candidate: (
                    destination_quotas[destination][candidate],
                    -(
                        selected_by_combination[candidate]
                        - len(destinations_by_combination[candidate])
                    ),
                    combination_order[candidate],
                ),
            )
            destination_quotas[destination][combination] += 1
            destination_assigned[destination] += 1
            destinations_by_combination[combination].append(destination)

    assignment_by_id: dict[str, int] = {}
    for combination, combination_instances in by_combination.items():
        destinations = destinations_by_combination[combination]
        for instance, destination in zip(
            tuple(combination_instances)[: len(destinations)],
            destinations,
            strict=True,
        ):
            assignment_by_id[instance.id] = destination

    destination_instances = tuple(
        tuple(
            instance
            for instance in pool.instances
            if assignment_by_id.get(instance.id) == destination
        )
        for destination in range(len(destination_sizes))
    )
    return PoolSplit(
        internal_eval=destination_instances[0],
        official=destination_instances[1],
        held_out=destination_instances[2],
    )
