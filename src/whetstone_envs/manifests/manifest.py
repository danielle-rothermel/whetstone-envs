"""Validated, serialized manifest model for one generated task pool.

Every task generator writes a :class:`Manifest` alongside its pool. The
manifest pins the generator version and seed range, then records per-stratum
counts and a content hash so regenerated content can be compared with a frozen
pool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from whetstone_envs.manifests.hashing import content_hash
from whetstone_envs.manifests.validation import (
    _PERSISTED_FIELDS,
    _SEED_RANGE_LEN,
    MANIFEST_SCHEMA_VERSION,
    validate_content_hash,
    validate_generator_version,
    validate_schema_version,
    validate_seed_range,
    validate_stratum_counts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.pools import TaskPool


def _retained_seeds_within_range(
    pool: TaskPool,
    seed_range: tuple[int, int],
) -> bool:
    """Return whether every retained instance seed is in ``seed_range``."""
    start, end = seed_range
    return all(start <= instance.seed < end for instance in pool.instances)


@dataclass(frozen=True, slots=True)
class Manifest:
    """A pinned, diffable description of one generated pool.

    Parameters
    ----------
    generator_version:
        The task generator's version or commit identity. Bumping this is the
        license to change generated content.
    seed_range:
        The ``(start, end)`` half-open range containing every retained
        instance seed. It describes retained pool content, not generator
        attempts that produced no retained instance.
    stratum_counts:
        Declared per-stratum instance counts, matched against a pool's own
        :meth:`~whetstone_envs.pools.TaskPool.stratum_counts`. Copied into a
        read-only mapping at construction.
    content_hash:
        SHA-256 of the pool's instances (see
        :func:`whetstone_envs.manifests.content_hash`).
    schema_version:
        Current manifest format identifier. Other versions are rejected.
    """

    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: Mapping[str, int]
    content_hash: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and canonicalize every construction path."""
        object.__setattr__(
            self,
            "schema_version",
            validate_schema_version(self.schema_version),
        )
        object.__setattr__(
            self,
            "generator_version",
            validate_generator_version(self.generator_version),
        )
        object.__setattr__(
            self,
            "seed_range",
            validate_seed_range(self.seed_range),
        )
        object.__setattr__(
            self,
            "stratum_counts",
            validate_stratum_counts(self.stratum_counts),
        )
        object.__setattr__(
            self,
            "content_hash",
            validate_content_hash(self.content_hash),
        )

    @classmethod
    def from_pool(
        cls,
        pool: TaskPool,
        *,
        generator_version: str,
        seed_range: tuple[int, int],
    ) -> Manifest:
        """Derive a manifest from a pool and its generation inputs.

        Per-stratum counts and the content hash are read from the pool, and
        every retained instance seed must lie in the declared half-open range.
        The manifest does not reconstruct or attest to generator attempts that
        produced no retained instance.
        """
        validated_seed_range = validate_seed_range(seed_range)
        if not _retained_seeds_within_range(pool, validated_seed_range):
            start, end = validated_seed_range
            msg = (
                "pool contains a retained instance seed outside declared "
                f"seed_range [{start}, {end})"
            )
            raise ValueError(msg)
        return cls(
            generator_version=generator_version,
            seed_range=validated_seed_range,
            stratum_counts=pool.stratum_counts(),
            content_hash=content_hash(pool),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-ready mapping form of the manifest."""
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed_range": list(self.seed_range),
            "stratum_counts": dict(sorted(self.stratum_counts.items())),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: object) -> Manifest:
        """Reconstruct a manifest from its :meth:`to_dict` form."""
        if not isinstance(data, dict):
            msg = "manifest must be a JSON object"
            raise TypeError(msg)
        payload = cast("dict[object, object]", data)

        missing_fields = _PERSISTED_FIELDS.difference(payload)
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            msg = f"manifest required fields missing: {names}"
            raise ValueError(msg)

        unknown_fields = set(payload).difference(_PERSISTED_FIELDS)
        if unknown_fields:
            names = ", ".join(sorted(map(str, unknown_fields)))
            msg = f"manifest contains unknown fields: {names}"
            raise ValueError(msg)

        seed_range = payload["seed_range"]
        if (
            not isinstance(seed_range, list)
            or len(seed_range) != _SEED_RANGE_LEN
        ):
            msg = "manifest seed_range must be a two-element list"
            raise TypeError(msg)
        raw_counts = payload["stratum_counts"]
        if not isinstance(raw_counts, dict):
            msg = "manifest stratum_counts must be an object"
            raise TypeError(msg)
        return cls(
            schema_version=cast("int", payload["schema_version"]),
            generator_version=cast(
                "str",
                payload["generator_version"],
            ),
            seed_range=cast("tuple[int, int]", tuple(seed_range)),
            stratum_counts=cast("Mapping[str, int]", raw_counts),
            content_hash=cast("str", payload["content_hash"]),
        )

    def to_json(self) -> str:
        """Serialize to a stable, human-diffable JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    def write(self, path: Path) -> None:
        """Write the manifest JSON to ``path`` with a trailing newline."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        """Read a manifest previously written by :meth:`write`."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def matches_pool(self, pool: TaskPool) -> bool:
        """Return whether ``pool`` matches this manifest.

        This is the regeneration diff check: build a fresh pool, then assert
        the frozen manifest still describes it. An out-of-range retained seed,
        a drift in any stratum count, or a content-hash change returns
        ``False``.
        """
        return (
            _retained_seeds_within_range(pool, self.seed_range)
            and self.stratum_counts == pool.stratum_counts()
            and self.content_hash == content_hash(pool)
        )
