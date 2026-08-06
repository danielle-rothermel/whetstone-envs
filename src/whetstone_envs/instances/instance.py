from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _freeze_inputs(
    inputs: Mapping[str, str],
) -> MappingProxyType[str, str]:
    detached = dict(inputs)
    for key, value in detached.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = "Instance.prompt_inputs keys and values must be strings"
            raise TypeError(msg)
    return MappingProxyType(detached)


@dataclass(frozen=True, slots=True)
class Instance:
    """An immutable task instance.

    ``strata`` is deduplicated in first-seen order. ``prompt_inputs`` is
    detached from caller-owned mappings and exposed read-only.
    """

    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: Mapping[str, str] = field(
        default_factory=lambda: _freeze_inputs({}),
    )
    gold: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            msg = "Instance.id must be a string"
            raise TypeError(msg)
        if not self.id:
            msg = "Instance.id must be a non-empty string"
            raise ValueError(msg)
        if type(self.seed) is not int:
            msg = "Instance.seed must be an int"
            raise TypeError(msg)
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
        if not isinstance(self.gold, str):
            msg = "Instance.gold must be a string"
            raise TypeError(msg)
        object.__setattr__(self, "strata", tuple(dict.fromkeys(labels)))
        object.__setattr__(
            self,
            "prompt_inputs",
            _freeze_inputs(self.prompt_inputs),
        )

    def __hash__(self) -> int:
        # MappingProxyType is unhashable, so hash a sorted item tuple instead.
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
    """Accept one stratum string and omitted prompt inputs as conveniences."""
    if isinstance(strata, str):
        labels = (strata,)
    elif isinstance(strata, tuple):
        labels = strata
    else:
        msg = "make_instance strata must be a string or tuple of strings"
        raise TypeError(msg)
    return Instance(
        id=id,
        seed=seed,
        strata=labels,
        prompt_inputs={} if prompt_inputs is None else prompt_inputs,
        gold=gold,
    )
