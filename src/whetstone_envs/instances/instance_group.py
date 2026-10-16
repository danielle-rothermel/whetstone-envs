from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from whetstone_envs.instances.instance import Instance  # noqa: TC001


class InstanceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    instances: tuple[Instance, ...]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instance group name must be nonblank")
        return value

    @field_validator("instances")
    @classmethod
    def _validate_instances(
        cls,
        value: tuple[Instance, ...],
    ) -> tuple[Instance, ...]:
        seen_ids: set[str] = set()
        for instance in value:
            if instance.id in seen_ids:
                raise ValueError(
                    f"duplicate instance id in group: {instance.id!r}",
                )
            seen_ids.add(instance.id)
        return value
