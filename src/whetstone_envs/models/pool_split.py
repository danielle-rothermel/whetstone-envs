from __future__ import annotations

from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from whetstone_envs.models.instance_group import InstanceGroup  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping


class PoolSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_groups: tuple[InstanceGroup, ...]

    @field_validator("instance_groups")
    @classmethod
    def _validate_instance_groups(
        cls,
        value: tuple[InstanceGroup, ...],
    ) -> tuple[InstanceGroup, ...]:
        seen_names: set[str] = set()
        group_by_instance_id: dict[str, str] = {}
        for group in value:
            if group.name in seen_names:
                raise ValueError(f"duplicate split group name: {group.name!r}")
            seen_names.add(group.name)
            for instance in group.instances:
                if instance.id in group_by_instance_id:
                    previous_group = group_by_instance_id[instance.id]
                    raise ValueError(
                        f"split is not disjoint: instance {instance.id!r} "
                        f"appears in both {previous_group!r} and "
                        f"{group.name!r}",
                    )
                group_by_instance_id[instance.id] = group.name
        return value

    @cached_property
    def _by_name(self) -> Mapping[str, InstanceGroup]:
        return MappingProxyType(
            {group.name: group for group in self.instance_groups},
        )

    def __getitem__(self, name: str) -> InstanceGroup:
        return self._by_name[name]
