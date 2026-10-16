from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

from dr_graph.flow.transport import (
    TransportCell,
    TransportProblem,
    solve_separable_transport,
)
from pydantic import BaseModel, ConfigDict, model_validator

from whetstone_envs.instances.instance_group import InstanceGroup

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.instances.instance import Instance


class InstancePool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_group: InstanceGroup

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self._by_category)

    @cached_property
    def _by_category(self) -> Mapping[str, tuple[Instance, ...]]:
        by_category: dict[str, list[Instance]] = {}
        for inst in self.instance_group.instances:
            for label in inst.categories:
                by_category.setdefault(label, []).append(inst)
        return MappingProxyType(
            {key: tuple(value) for key, value in by_category.items()},
        )

    @model_validator(mode="after")
    def _validate_pool(self) -> InstancePool:
        seen_public_identities: dict[
            tuple[tuple[str, str], ...],
            str,
        ] = {}
        for inst in self.instance_group.instances:
            public_identity = inst.public_prompt_identity
            if public_identity in seen_public_identities:
                first_id = seen_public_identities[public_identity]
                msg = (
                    "duplicate public prompt identity in pool: "
                    f"instances {first_id!r} and {inst.id!r} have the "
                    "same prompt_inputs"
                )
                raise ValueError(msg)
            seen_public_identities[public_identity] = inst.id
        return self

    def __len__(self) -> int:
        return len(self.instance_group.instances)

    def category_counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._by_category.items()}

    def in_category(self, label: str) -> tuple[Instance, ...]:
        return self._by_category.get(label, ())

    def split(
        self,
        sizes: Mapping[str, int],
    ) -> InstancePoolSplit:
        instances = self.instance_group.instances
        requested_sizes = _validate_split_sizes(
            sizes,
            pool_size=len(instances),
        )
        instances_by_combination = _group_by_category_combination(instances)
        selected_by_combination = _select_balanced_combination_counts(
            instances_by_combination,
            total=sum(requested_sizes.values()),
        )
        destination_counts = _allocate_combination_counts(
            selected_by_combination,
            destination_sizes=tuple(requested_sizes.values()),
        )
        assignment_by_id = _assign_instances_to_destinations(
            instances_by_combination,
            destination_counts,
        )
        return InstancePoolSplit(
            pool=self,
            split_groups=_build_destination_groups(
                instances,
                destination_names=tuple(requested_sizes),
                assignment_by_id=assignment_by_id,
            ),
        )

    def as_sequence(self) -> Sequence[Instance]:
        return self.instance_group.instances


class InstancePoolSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pool: InstancePool
    split_groups: tuple[InstanceGroup, ...]

    @cached_property
    def splits(self) -> Mapping[str, InstanceGroup]:
        return MappingProxyType(
            {group.name: group for group in self.split_groups},
        )

    @model_validator(mode="after")
    def _validate_split_groups(self) -> InstancePoolSplit:
        source_by_id = {
            instance.id: instance
            for instance in self.pool.instance_group.instances
        }
        seen_names: set[str] = set()
        group_by_instance_id: dict[str, str] = {}
        for group in self.split_groups:
            if group.name in seen_names:
                raise ValueError(f"duplicate split group name: {group.name!r}")
            seen_names.add(group.name)
            for instance in group.instances:
                if instance.id not in source_by_id:
                    raise ValueError(
                        f"split instance {instance.id!r} "
                        "is not in source pool",
                    )
                if instance != source_by_id[instance.id]:
                    raise ValueError(
                        f"split instance {instance.id!r} does not match "
                        "source pool instance",
                    )
                if instance.id in group_by_instance_id:
                    previous_group = group_by_instance_id[instance.id]
                    raise ValueError(
                        f"split is not disjoint: instance {instance.id!r} "
                        f"appears in both {previous_group!r} and "
                        f"{group.name!r}",
                    )
                group_by_instance_id[instance.id] = group.name
        return self


def _group_by_category_combination(
    instances: tuple[Instance, ...],
) -> dict[tuple[str, ...], list[Instance]]:
    instances_by_combination: dict[tuple[str, ...], list[Instance]] = {}
    for instance in instances:
        instances_by_combination.setdefault(instance.categories, []).append(
            instance,
        )
    return instances_by_combination


def _select_balanced_combination_counts(
    instances_by_combination: Mapping[tuple[str, ...], list[Instance]],
    *,
    total: int,
) -> dict[tuple[str, ...], int]:
    selected_counts = dict.fromkeys(instances_by_combination, 0)
    active = deque(instances_by_combination)
    for _ in range(total):
        combination = active.popleft()
        selected_counts[combination] += 1
        if selected_counts[combination] < len(
            instances_by_combination[combination],
        ):
            active.append(combination)
    return selected_counts


def _allocate_combination_counts(
    selected_by_combination: dict[tuple[str, ...], int],
    *,
    destination_sizes: tuple[int, ...],
) -> tuple[dict[tuple[str, ...], int], ...]:
    combinations = tuple(selected_by_combination)
    supplies = tuple(selected_by_combination.values())
    problem = TransportProblem(
        supplies=supplies,
        demands=destination_sizes,
        cells=_build_coverage_first_transport_cells(
            supplies,
            destination_sizes,
        ),
    )
    solution = solve_separable_transport(problem)
    return tuple(
        {
            combination: solution.allocations[combination_index][destination]
            for combination_index, combination in enumerate(combinations)
        }
        for destination in range(len(destination_sizes))
    )


def _build_coverage_first_transport_cells(
    combination_supplies: tuple[int, ...],
    destination_sizes: tuple[int, ...],
) -> tuple[TransportCell, ...]:
    total = sum(destination_sizes)
    cells: list[TransportCell] = []
    for combination_index, supply in enumerate(combination_supplies):
        for destination_index, demand in enumerate(destination_sizes):
            marginal_costs = _coverage_first_marginal_costs(
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
    return tuple(cells)


def _coverage_first_marginal_costs(
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


def _assign_instances_to_destinations(
    instances_by_combination: Mapping[tuple[str, ...], list[Instance]],
    destination_counts: tuple[dict[tuple[str, ...], int], ...],
) -> dict[str, int]:
    assignment_by_id: dict[str, int] = {}
    for combination, instances in instances_by_combination.items():
        destinations = [
            destination
            for destination, counts in enumerate(destination_counts)
            for _ in range(counts[combination])
        ]
        for instance, destination in zip(
            instances[: len(destinations)],
            destinations,
            strict=True,
        ):
            assignment_by_id[instance.id] = destination
    return assignment_by_id


def _build_destination_groups(
    instances: tuple[Instance, ...],
    *,
    destination_names: tuple[str, ...],
    assignment_by_id: Mapping[str, int],
) -> tuple[InstanceGroup, ...]:
    return tuple(
        InstanceGroup(
            name=name,
            instances=tuple(
                instance
                for instance in instances
                if assignment_by_id.get(instance.id) == destination
            ),
        )
        for destination, name in enumerate(destination_names)
    )


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
