"""The frozen :class:`Instance` type shared by every task environment.

Every task generator produces a ``list[Instance]``: a minimal, immutable
record carrying a stable task identity (``id`` / ``seed``), one or more
stratum labels, the rendered prompt inputs a probe template consumes, and the
gold/oracle-checkable state used for exact-match scoring. It has no model-call
dependency and no task-family-specific logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _freeze_inputs(
    inputs: Mapping[str, str],
) -> MappingProxyType[str, str]:
    """Validate and return a detached, read-only copy of ``inputs``.

    Copying into a fresh ``dict`` first detaches the view from the caller's
    mutable mapping, so a frozen instance cannot be mutated through the
    reference the caller still holds.
    """
    detached = dict(inputs)
    for key, value in detached.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = "Instance.prompt_inputs keys and values must be strings"
            raise TypeError(msg)
    return MappingProxyType(detached)


@dataclass(frozen=True, slots=True)
class Instance:
    """One pinned, oracle-checkable task instance.

    Parameters
    ----------
    id:
        Stable task identity, unique within a pool. Used as the task key when
        aggregating scores.
    seed:
        The generator seed that produced this instance. Recorded for
        determinism and contamination auditing; distinct from ``id`` so a
        generator may derive several instances from one seed if a task family
        ever needs to.
    strata:
        One or more stratum labels this instance belongs to. Kept as an
        ordered, deduplicated tuple so membership is hashable and stable.
    prompt_inputs:
        The rendered fields a probe template consumes. Read-only; never
        includes gold/oracle-only state.
    gold:
        The exact string the oracle expects an extracted prediction to equal
        for a score of 1.
    """

    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: Mapping[str, str] = field(
        default_factory=lambda: _freeze_inputs({}),
    )
    gold: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            msg = "Instance.id must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.strata, tuple):
            msg = "Instance.strata must be a tuple of strings"
            raise TypeError(msg)
        labels = self.strata
        if not labels:
            msg = f"Instance {self.id!r} must declare at least one stratum"
            raise ValueError(msg)
        if any(not isinstance(label, str) for label in labels):
            msg = f"Instance {self.id!r} stratum labels must be strings"
            raise TypeError(msg)
        if any(not label.strip() for label in labels):
            msg = (
                f"Instance {self.id!r} stratum labels must be non-empty "
                "and not blank"
            )
            raise ValueError(msg)
        object.__setattr__(self, "strata", tuple(dict.fromkeys(labels)))
        object.__setattr__(
            self,
            "prompt_inputs",
            _freeze_inputs(self.prompt_inputs),
        )

    def __hash__(self) -> int:
        # The dataclass-generated __hash__ would try to hash the unhashable
        # MappingProxyType; hash its sorted items instead so value-equal
        # instances still hash equal.
        return hash(
            (
                self.id,
                self.seed,
                self.strata,
                tuple(sorted(self.prompt_inputs.items())),
                self.gold,
            ),
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
    defaults to an empty read-only mapping. Task generators should use this
    constructor so the freezing convention stays in one place.
    """
    labels = (strata,) if isinstance(strata, str) else tuple(strata)
    return Instance(
        id=id,
        seed=seed,
        strata=labels,
        prompt_inputs={} if prompt_inputs is None else prompt_inputs,
        gold=gold,
    )
