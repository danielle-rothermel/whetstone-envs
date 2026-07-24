"""The :class:`TaskPool` container and its disjoint ``.split``.

A candidate generator returns one :class:`TaskPool`: the pinned list of
instances plus the derived per-stratum membership. ``.split`` carves the
pool into the internal-eval / official / held-out subsets the optimizer
machinery consumes, asserting the three are disjoint so a task can never
leak from held-out into an eval the optimizer can see.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from whetstone_envs.core.instance import Instance

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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


@dataclass(frozen=True, slots=True)
class TaskPool:
    """A pinned pool of instances with derived stratum membership.

    Parameters
    ----------
    instances:
        The pinned instances, in generation order. Instance ``id`` must
        be unique within the pool; a duplicate raises at construction.
    """

    instances: tuple[Instance, ...]
    _by_stratum: dict[str, tuple[Instance, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, instances: Iterable[Instance]) -> None:
        materialized = tuple(instances)
        seen: set[str] = set()
        for inst in materialized:
            if inst.id in seen:
                msg = f"duplicate instance id in pool: {inst.id!r}"
                raise ValueError(msg)
            seen.add(inst.id)
        object.__setattr__(self, "instances", materialized)

        by_stratum: dict[str, list[Instance]] = {}
        for inst in materialized:
            for label in inst.strata:
                by_stratum.setdefault(label, []).append(inst)
        object.__setattr__(
            self,
            "_by_stratum",
            {k: tuple(v) for k, v in by_stratum.items()},
        )

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def strata(self) -> tuple[str, ...]:
        """The stratum labels present in the pool, in first-seen order."""
        return tuple(self._by_stratum)

    def stratum_counts(self) -> dict[str, int]:
        """Map each stratum label to its instance count.

        An instance in multiple strata counts once per label, so these
        counts sum to at least ``len(self)`` and match the manifest's
        declared per-stratum counts.
        """
        return {k: len(v) for k, v in self._by_stratum.items()}

    def in_stratum(self, label: str) -> tuple[Instance, ...]:
        """Return the instances carrying ``label`` (empty if none)."""
        return self._by_stratum.get(label, ())

    def split(
        self,
        internal_eval_n: int,
        official_n: int,
        held_out_n: int,
    ) -> PoolSplit:
        """Carve the pool into three disjoint stratified subsets.

        Instances are grouped by their complete ``strata`` tuple, then
        drawn round-robin in first-seen combination order. This makes
        membership independent of whether generation order blocks or
        interleaves the combinations, while preserving the original
        order within each combination. Sizes are the per-spec proposed
        numbers, passed in by the caller rather than hardcoded here.
        Their sum must not exceed the pool size; any unused tail remains
        unassigned, and the result asserts the three subsets are
        disjoint.
        """
        for name, size in (
            ("internal_eval_n", internal_eval_n),
            ("official_n", official_n),
            ("held_out_n", held_out_n),
        ):
            if size < 0:
                msg = f"{name} must be non-negative, got {size}"
                raise ValueError(msg)
        total = internal_eval_n + official_n + held_out_n
        if total > len(self.instances):
            msg = (
                f"split sizes sum to {total} but pool has only "
                f"{len(self.instances)} instances"
            )
            raise ValueError(msg)

        by_combination: dict[tuple[str, ...], deque[Instance]] = {}
        for inst in self.instances:
            by_combination.setdefault(inst.strata, deque()).append(inst)

        active = deque(by_combination.values())
        stratified: list[Instance] = []
        while len(stratified) < total:
            combination = active.popleft()
            stratified.append(combination.popleft())
            if combination:
                active.append(combination)

        cut1 = internal_eval_n
        cut2 = cut1 + official_n
        cut3 = cut2 + held_out_n
        return PoolSplit(
            internal_eval=tuple(stratified[:cut1]),
            official=tuple(stratified[cut1:cut2]),
            held_out=tuple(stratified[cut2:cut3]),
        )

    def as_sequence(self) -> Sequence[Instance]:
        """Return the instances as a read-only sequence."""
        return self.instances
