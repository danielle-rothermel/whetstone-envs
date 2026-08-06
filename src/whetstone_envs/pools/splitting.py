from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_graph.flow.transport import (
    TransportCell,
    TransportProblem,
    solve_separable_transport,
)

from whetstone_envs.instances import Instance

if TYPE_CHECKING:
    from whetstone_envs.pools.pool import TaskPool


def _destination_cell_marginal_costs(
    *,
    combination_supply: int,
    destination_demand: int,
    total_flow: int,
) -> tuple[int, ...]:
    """Return Whetstone's coverage-first, squared-spread cell costs."""
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
    destination_sizes: tuple[int, int, int],
) -> list[dict[tuple[str, ...], int]]:
    """Set global quotas maximizing coverage before squared-count balance."""
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
    destination_sizes: tuple[int, int, int],
    *,
    pool_size: int,
) -> int:
    for name, size in zip(
        ("internal_eval_n", "official_n", "held_out_n"),
        destination_sizes,
        strict=True,
    ):
        if type(size) is not int:
            msg = f"{name} must be an int, got {type(size).__name__}"
            raise TypeError(msg)
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
    """Three subsets whose instance IDs must be pairwise disjoint."""

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
