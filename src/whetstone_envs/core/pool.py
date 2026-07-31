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
from fractions import Fraction
from typing import TYPE_CHECKING

from whetstone_envs.core.instance import Instance

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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


@dataclass(frozen=True, slots=True)
class TaskPool:
    """A pinned pool of instances with derived stratum membership.

    Parameters
    ----------
    instances:
        The pinned instances, in generation order. Instance ``id`` must
        be unique within the pool. The canonical public task identity,
        defined solely by sorted ``prompt_inputs``, must also be unique.
        Either kind of duplicate raises at construction.
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

            public_identity = tuple(sorted(inst.prompt_inputs.items()))
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

        Instances are grouped by their complete ``strata`` tuple.
        Round-robin selection first determines a balanced quota for each
        combination. Those quotas are then distributed across the three
        destinations, preferring combination coverage, lower relative
        fill, and destinations with more remaining capacity. This keeps
        scarce combinations from being consumed by the first
        destination when another destination can retain coverage.

        Sizes are the per-spec proposed numbers, passed in by the caller
        rather than hardcoded here. Their sum must not exceed the pool
        size. Selected instances retain pool order within each
        destination, unused per-combination tails remain unassigned, and
        the result asserts the three subsets are disjoint.
        """
        destination_sizes = (
            internal_eval_n,
            official_n,
            held_out_n,
        )
        total = _validate_split_sizes(
            destination_sizes,
            pool_size=len(self.instances),
        )

        by_combination: dict[tuple[str, ...], deque[Instance]] = {}
        for inst in self.instances:
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

        destination_assigned = [0, 0, 0]
        destination_quotas = [
            dict.fromkeys(by_combination, 0) for _ in destination_sizes
        ]
        destinations_by_combination: dict[tuple[str, ...], list[int]] = {
            combination: [] for combination in by_combination
        }
        coverage_targets = [
            min(size, len(by_combination)) for size in destination_sizes
        ]

        for combination in by_combination:
            combination_quota = selected_by_combination[combination]
            for _ in range(min(combination_quota, len(destination_sizes))):
                eligible = [
                    index
                    for index in range(len(destination_sizes))
                    if destination_quotas[index][combination] == 0
                    and destination_assigned[index] < coverage_targets[index]
                ]
                if not eligible:
                    break
                destination = min(
                    eligible,
                    key=lambda index: (
                        -(
                            coverage_targets[index]
                            - destination_assigned[index]
                        ),
                        -destination_sizes[index],
                        index,
                    ),
                )
                destination_quotas[destination][combination] += 1
                destination_assigned[destination] += 1
                destinations_by_combination[combination].append(destination)

        for combination in by_combination:
            combination_quota = selected_by_combination[combination]
            remaining = combination_quota - len(
                destinations_by_combination[combination]
            )
            for _ in range(remaining):
                eligible = [
                    index
                    for index, size in enumerate(destination_sizes)
                    if destination_assigned[index] < size
                ]
                destination = min(
                    eligible,
                    key=lambda index: (
                        destination_quotas[index][combination],
                        Fraction(
                            destination_assigned[index],
                            destination_sizes[index],
                        ),
                        -(
                            destination_sizes[index]
                            - destination_assigned[index]
                        ),
                        index,
                    ),
                )
                destination_quotas[destination][combination] += 1
                destination_assigned[destination] += 1
                destinations_by_combination[combination].append(destination)

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
                for instance in self.instances
                if assignment_by_id.get(instance.id) == destination
            )
            for destination in range(len(destination_sizes))
        )
        return PoolSplit(
            internal_eval=destination_instances[0],
            official=destination_instances[1],
            held_out=destination_instances[2],
        )

    def as_sequence(self) -> Sequence[Instance]:
        """Return the instances as a read-only sequence."""
        return self.instances
