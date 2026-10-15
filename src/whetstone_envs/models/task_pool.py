from __future__ import annotations

from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator

from whetstone_envs.models.instance_group import InstanceGroup  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from whetstone_envs.models.instance import Instance
    from whetstone_envs.pools.splitting import PoolSplit


class TaskPool(BaseModel):
    """An ordered pool with unique instance IDs and prompt-input identities.

    Rendered-prompt uniqueness depends on each renderer and template.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_group: InstanceGroup

    @model_validator(mode="after")
    def _validate_pool(self) -> TaskPool:
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

    @cached_property
    def _by_stratum(self) -> Mapping[str, tuple[Instance, ...]]:
        by_stratum: dict[str, list[Instance]] = {}
        for inst in self.instance_group.instances:
            for label in inst.strata:
                by_stratum.setdefault(label, []).append(inst)
        return MappingProxyType(
            {key: tuple(value) for key, value in by_stratum.items()},
        )

    def __len__(self) -> int:
        return len(self.instance_group.instances)

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(self._by_stratum)

    def stratum_counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._by_stratum.items()}

    def in_stratum(self, label: str) -> tuple[Instance, ...]:
        return self._by_stratum.get(label, ())

    def split(
        self,
        internal_eval_n: int,
        official_n: int,
        held_out_n: int,
    ) -> PoolSplit:
        """Split by complete strata tuples without oversubscription.

        First-seen combinations are selected round-robin in pool order. Global
        destination allocation maximizes combination coverage, then minimizes
        squared counts. Selected instances retain pool order, and an unused
        pool tail may remain unassigned.
        """
        # Keep the allocation solver outside the models import path.
        from whetstone_envs.pools.splitting import split_pool  # noqa: PLC0415

        return split_pool(
            self,
            internal_eval_n,
            official_n,
            held_out_n,
        )

    def as_sequence(self) -> Sequence[Instance]:
        return self.instance_group.instances
