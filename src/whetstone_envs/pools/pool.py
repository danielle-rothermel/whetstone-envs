"""Validated task-pool membership and stratum indexes.

A task generator returns one :class:`TaskPool`: the pinned list of instances
plus derived per-stratum membership. :meth:`TaskPool.split` carves the pool
into the internal-eval, official, and held-out subsets the optimizer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from whetstone_envs.instances import Instance, public_prompt_identity
from whetstone_envs.pools.splitting import PoolSplit, split_pool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TaskPool:
    """A pinned pool of instances with derived stratum membership.

    Parameters
    ----------
    instances:
        The pinned instances, in generation order. Instance ``id`` must be
        unique within the pool. The canonical public task identity, defined
        solely by sorted ``prompt_inputs``, must also be unique. Either kind
        of duplicate raises at construction.
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
        """Return the stratum labels present in first-seen order."""
        return tuple(self._by_stratum)

    def stratum_counts(self) -> dict[str, int]:
        """Map each stratum label to its instance count.

        An instance in multiple strata counts once per label, so these counts
        sum to at least ``len(self)`` and match the manifest's declared
        per-stratum counts.
        """
        return {key: len(value) for key, value in self._by_stratum.items()}

    def in_stratum(self, label: str) -> tuple[Instance, ...]:
        """Return the instances carrying ``label`` or an empty tuple."""
        return self._by_stratum.get(label, ())

    def split(
        self,
        internal_eval_n: int,
        official_n: int,
        held_out_n: int,
    ) -> PoolSplit:
        """Carve the pool into three disjoint stratified subsets.

        Instances are grouped by their complete ``strata`` tuple.
        Round-robin selection first determines a balanced quota for each
        combination. Those quotas are then distributed across the three
        destinations. The allocation first maximizes distinct combination
        coverage, then fills each destination from its least-represented
        combinations with remaining quota. This keeps scarce combinations
        from being consumed before coverage is established and balances
        composition within each destination.

        Sizes are supplied by the caller rather than hardcoded here. Their
        sum must not exceed the pool size. Selected instances retain pool
        order within each destination, unused per-combination tails remain
        unassigned, and the result asserts the subsets are disjoint.
        """
        return split_pool(
            self,
            internal_eval_n,
            official_n,
            held_out_n,
        )

    def as_sequence(self) -> Sequence[Instance]:
        """Return the instances as a read-only sequence."""
        return self.instances
