from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dr_graph.flow.transport import (
    TransportCell,
    TransportProblem,
    solve_separable_transport,
)

from whetstone_envs.models import InstanceGroup, PoolSplit

if TYPE_CHECKING:
    from whetstone_envs.models import Instance, TaskPool


def _destination_cell_marginal_costs(
    *,
    combination_supply: int,
    destination_demand: int,
    total_flow: int,
) -> tuple[int, ...]:
    # One extra coverage miss must cost more than all squared-spread costs.
    coverage_penalty = total_flow * total_flow + 1
    return tuple(
        2 * unit - 1 + (coverage_penalty if unit > 1 else 0)
        for unit in range(
            1,
            min(combination_supply, destination_demand) + 1,
        )
    )


def _allocate_destination_quotas(
    selected_by_combination: dict[tuple[str, ...], int],
    destination_sizes: tuple[int, ...],
) -> list[dict[tuple[str, ...], int]]:
    combinations = tuple(selected_by_combination)
    supplies = tuple(selected_by_combination.values())
    total = sum(destination_sizes)
    cells: list[TransportCell] = []
    for combination_index, supply in enumerate(supplies):
        for destination_index, demand in enumerate(destination_sizes):
            marginal_costs = _destination_cell_marginal_costs(
                combination_supply=supply,
                destination_demand=demand,
                total_flow=total,
            )
            if marginal_costs:
                cells.append(
                    TransportCell(
                        source_index=combination_index,
                        destination_index=destination_index,
                        marginal_costs=marginal_costs,
                    )
                )
    solution = solve_separable_transport(
        TransportProblem(
            supplies=supplies,
            demands=destination_sizes,
            cells=tuple(cells),
        )
    )
    return [
        {
            combination: solution.allocations[combination_index][destination]
            for combination_index, combination in enumerate(combinations)
        }
        for destination in range(len(destination_sizes))
    ]


def _validate_split_sizes(
    sizes: Mapping[str, int],
    *,
    pool_size: int,
) -> dict[str, int]:
    if not isinstance(sizes, Mapping):
        raise TypeError("split sizes must be a mapping of names to integers")
    requested_sizes = dict(sizes)
    for name, size in requested_sizes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("split group names must be nonblank strings")
        if type(size) is not int:
            msg = f"{name} must be an int, got {type(size).__name__}"
            raise TypeError(msg)
        if size < 0:
            msg = f"{name} must be non-negative, got {size}"
            raise ValueError(msg)
    total = sum(requested_sizes.values())
    if total > pool_size:
        msg = (
            f"split sizes sum to {total} but pool has only "
            f"{pool_size} instances"
        )
        raise ValueError(msg)
    return requested_sizes


def split_pool(
    pool: TaskPool,
    sizes: Mapping[str, int],
) -> PoolSplit:
    requested_sizes = _validate_split_sizes(
        sizes,
        pool_size=len(pool.instance_group.instances),
    )
    destination_sizes = tuple(requested_sizes.values())
    total = sum(destination_sizes)

    by_combination: dict[tuple[str, ...], deque[Instance]] = {}
    for inst in pool.instance_group.instances:
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

    destination_quotas = _allocate_destination_quotas(
        selected_by_combination,
        destination_sizes,
    )
    combinations = tuple(by_combination)
    destinations_by_combination = {
        combination: [
            destination
            for destination, quotas in enumerate(destination_quotas)
            for _ in range(quotas[combination])
        ]
        for combination in combinations
    }

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
            for instance in pool.instance_group.instances
            if assignment_by_id.get(instance.id) == destination
        )
        for destination in range(len(destination_sizes))
    )
    return PoolSplit(
        instance_groups=tuple(
            InstanceGroup(name=name, instances=instances)
            for name, instances in zip(
                requested_sizes,
                destination_instances,
                strict=True,
            )
        ),
    )
