"""Deterministic, destination-balanced task-pool splitting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import TYPE_CHECKING

from whetstone_envs.instances import Instance

if TYPE_CHECKING:
    from whetstone_envs.pools.pool import TaskPool


@dataclass(slots=True)
class _FlowEdge:
    destination: int
    reverse: int
    capacity: int
    cost: int


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    destination: int,
    capacity: int,
    cost: int,
) -> _FlowEdge:
    """Add a residual edge pair and return the forward edge."""
    forward = _FlowEdge(destination, len(graph[destination]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[destination].append(reverse)
    return forward


def _flow_step(
    previous: list[tuple[int, int] | None],
    node: int,
) -> tuple[int, int]:
    """Return one predecessor step from a complete residual path."""
    step = previous[node]
    if step is None:
        msg = "destination quota path is incomplete"
        raise AssertionError(msg)
    return step


def _send_min_cost_flow(
    graph: list[list[_FlowEdge]],
    source: int,
    sink: int,
    required_flow: int,
) -> None:
    """Send exact flow using successive shortest residual paths."""
    node_count = len(graph)
    potentials = [0] * node_count
    sent = 0

    while sent < required_flow:
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        queue = [(0, source)]

        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity == 0:
                    continue
                candidate = (
                    distance
                    + edge.cost
                    + potentials[node]
                    - potentials[edge.destination]
                )
                known = distances[edge.destination]
                if known is not None and candidate >= known:
                    continue
                distances[edge.destination] = candidate
                previous[edge.destination] = (node, edge_index)
                heappush(queue, (candidate, edge.destination))

        if distances[sink] is None:
            msg = "destination quota flow is infeasible"
            raise AssertionError(msg)
        for node, distance in enumerate(distances):
            if distance is not None:
                potentials[node] += distance

        amount = required_flow - sent
        node = sink
        while node != source:
            previous_node, edge_index = _flow_step(previous, node)
            amount = min(amount, graph[previous_node][edge_index].capacity)
            node = previous_node

        node = sink
        while node != source:
            previous_node, edge_index = _flow_step(previous, node)
            edge = graph[previous_node][edge_index]
            edge.capacity -= amount
            graph[node][edge.reverse].capacity += amount
            node = previous_node
        sent += amount


def _allocate_destination_quotas(
    selected_by_combination: dict[tuple[str, ...], int],
    destination_sizes: tuple[int, int, int],
) -> list[dict[tuple[str, ...], int]]:
    """Globally optimize coverage, then squared quota spread."""
    combinations = tuple(selected_by_combination)
    combination_count = len(combinations)
    destination_count = len(destination_sizes)
    source = 0
    first_combination = 1
    first_destination = first_combination + combination_count
    sink = first_destination + destination_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    total = sum(destination_sizes)
    # With fixed total flow, non-first cell units equal total - coverage. The
    # penalty exceeds the full possible range of the squared-count objective,
    # so coverage is optimized first without needing a separate greedy pass.
    coverage_penalty = total * total + 1
    cell_edges: list[list[list[_FlowEdge]]] = [
        [[] for _ in destination_sizes] for _ in combinations
    ]

    for combination_index, combination in enumerate(combinations):
        quota = selected_by_combination[combination]
        combination_node = first_combination + combination_index
        _add_flow_edge(graph, source, combination_node, quota, 0)
        for destination, size in enumerate(destination_sizes):
            destination_node = first_destination + destination
            for unit in range(1, min(quota, size) + 1):
                marginal_square_cost = 2 * unit - 1
                edge = _add_flow_edge(
                    graph,
                    combination_node,
                    destination_node,
                    1,
                    marginal_square_cost
                    + (coverage_penalty if unit > 1 else 0),
                )
                cell_edges[combination_index][destination].append(edge)

    for destination, size in enumerate(destination_sizes):
        _add_flow_edge(
            graph,
            first_destination + destination,
            sink,
            size,
            0,
        )

    _send_min_cost_flow(graph, source, sink, total)
    return [
        {
            combination: sum(
                edge.capacity == 0
                for edge in cell_edges[combination_index][destination]
            )
            for combination_index, combination in enumerate(combinations)
        }
        for destination in range(destination_count)
    ]


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
