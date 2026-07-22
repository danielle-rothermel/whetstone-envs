"""The frozen :class:`Instance` type shared by every candidate.

Every candidate generator produces a ``list[Instance]``: a minimal,
immutable record carrying a stable task identity (``id`` / ``seed``),
one or more stratum labels, the rendered prompt inputs a probe template
consumes, and the gold/oracle-checkable state used for exact-match
scoring. It has no model-call dependency and no candidate-specific
logic, so it lives in :mod:`whetstone_envs.core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def _freeze_inputs(
    inputs: Mapping[str, str],
) -> MappingProxyType[str, str]:
    """Return a read-only, order-preserving view of ``inputs``.

    Copying into a fresh ``dict`` first detaches the view from the
    caller's mutable mapping, so a frozen instance cannot be mutated
    through the reference the caller still holds.
    """
    return MappingProxyType(dict(inputs))


@dataclass(frozen=True, slots=True)
class Instance:
    """One pinned, oracle-checkable task instance.

    Parameters
    ----------
    id:
        Stable task identity, unique within a pool. Used as the task
        key when aggregating scores.
    seed:
        The generator seed that produced this instance. Recorded for
        determinism and contamination auditing; distinct from ``id`` so
        a generator may derive several instances from one seed if a
        candidate ever needs to.
    strata:
        One or more stratum labels this instance belongs to (the
        latent-rule / difficulty strata from a spec's Section 1). Kept
        as an ordered, deduplicated tuple so membership is hashable and
        stable.
    prompt_inputs:
        The rendered fields a probe template consumes. Read-only; never
        includes gold/oracle-only state.
    gold:
        The exact string the oracle expects an extracted prediction to
        equal for a score of 1.
    """

    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: MappingProxyType[str, str] = field(
        default_factory=lambda: _freeze_inputs({}),
    )
    gold: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            msg = "Instance.id must be a non-empty string"
            raise ValueError(msg)
        if not self.strata:
            msg = f"Instance {self.id!r} must declare at least one stratum"
            raise ValueError(msg)
        if not isinstance(self.prompt_inputs, MappingProxyType):
            object.__setattr__(
                self,
                "prompt_inputs",
                _freeze_inputs(self.prompt_inputs),
            )


def make_instance(
    *,
    id: str,  # noqa: A002 - matches the frozen field name intentionally
    seed: int,
    strata: tuple[str, ...] | str,
    prompt_inputs: Mapping[str, str] | None = None,
    gold: str = "",
) -> Instance:
    """Construct an :class:`Instance`, normalizing convenience inputs.

    A single stratum may be passed as a bare string; ``prompt_inputs``
    defaults to an empty read-only mapping. This is the constructor
    candidate generators should call so the freezing convention stays
    in one place.
    """
    labels = (strata,) if isinstance(strata, str) else tuple(strata)
    return Instance(
        id=id,
        seed=seed,
        strata=labels,
        prompt_inputs=_freeze_inputs(prompt_inputs or {}),
        gold=gold,
    )
