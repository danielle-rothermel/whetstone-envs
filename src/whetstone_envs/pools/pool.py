from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from whetstone_envs.instances import Instance, public_prompt_identity
from whetstone_envs.pools.splitting import PoolSplit, split_pool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TaskPool:
    """An ordered pool with unique instance IDs and prompt-input identities.

    Rendered-prompt uniqueness depends on each renderer and template.
    """

    instances: tuple[Instance, ...]
    _by_stratum: dict[str, tuple[Instance, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, instances: Iterable[Instance]) -> None:
        materialized = tuple(instances)
        seen_ids: set[str] = set()
        seen_public_identities: dict[
            tuple[tuple[str, str], ...],
            str,
        ] = {}
        for inst in materialized:
            if inst.id in seen_ids:
                msg = f"duplicate instance id in pool: {inst.id!r}"
                raise ValueError(msg)
            seen_ids.add(inst.id)

            public_identity = public_prompt_identity(inst)
            if public_identity in seen_public_identities:
                first_id = seen_public_identities[public_identity]
                msg = (
                    "duplicate public prompt identity in pool: "
                    f"instances {first_id!r} and {inst.id!r} have the "
                    "same prompt_inputs"
                )
                raise ValueError(msg)
            seen_public_identities[public_identity] = inst.id
        object.__setattr__(self, "instances", materialized)

        by_stratum: dict[str, list[Instance]] = {}
        for inst in materialized:
            for label in inst.strata:
                by_stratum.setdefault(label, []).append(inst)
        object.__setattr__(
            self,
            "_by_stratum",
            {key: tuple(value) for key, value in by_stratum.items()},
        )

    def __len__(self) -> int:
        return len(self.instances)

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
        return split_pool(
            self,
            internal_eval_n,
            official_n,
            held_out_n,
        )

    def as_sequence(self) -> Sequence[Instance]:
        return self.instances
