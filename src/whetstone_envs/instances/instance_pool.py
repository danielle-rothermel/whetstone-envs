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
from whetstone_envs.instances.instance_pool_split import InstancePoolSplit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.instances.instance import Instance


class InstancePool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_group: InstanceGroup

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(self._by_stratum)

    @cached_property
    def _by_stratum(self) -> Mapping[str, tuple[Instance, ...]]:
        by_stratum: dict[str, list[Instance]] = {}
        for inst in self.instance_group.instances:
            for label in inst.strata:
                by_stratum.setdefault(label, []).append(inst)
        return MappingProxyType(
            {key: tuple(value) for key, value in by_stratum.items()},
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

    def stratum_counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._by_stratum.items()}

    def in_stratum(self, label: str) -> tuple[Instance, ...]:
        return self._by_stratum.get(label, ())

    def split(
        self,
        sizes: Mapping[str, int],
    ) -> InstancePoolSplit:
        requested_sizes = _validate_split_sizes(
            sizes,
            pool_size=len(self.instance_group.instances),
        )
        destination_sizes = tuple(requested_sizes.values())
        total = sum(destination_sizes)

        by_combination: dict[tuple[str, ...], deque[Instance]] = {}
        for inst in self.instance_group.instances:
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
                for instance in self.instance_group.instances
                if assignment_by_id.get(instance.id) == destination
            )
            for destination in range(len(destination_sizes))
        )
        return InstancePoolSplit(
            instance_groups=tuple(
                InstanceGroup(name=name, instances=instances)
                for name, instances in zip(
                    requested_sizes,
                    destination_instances,
                    strict=True,
                )
            ),
        )

    def as_sequence(self) -> Sequence[Instance]:
        return self.instance_group.instances


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
