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
    start, end = seed_range
    return all(start <= instance.seed < end for instance in pool.instances)


@dataclass(frozen=True, slots=True)
class Manifest:
    """A validated, read-only description of a generated pool.

    ``seed_range`` covers retained instances only, and ``stratum_counts`` is
    detached and frozen. Matching retained seeds, counts, and the content hash
    to a pool is explicit via :meth:`matches_pool`.
    """

    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: Mapping[str, int]
    content_hash: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
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

        Retained seeds must lie in the declared half-open range. The manifest
        says nothing about generator attempts producing no retained instance.
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
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed_range": list(self.seed_range),
            "stratum_counts": dict(sorted(self.stratum_counts.items())),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: object) -> Manifest:
        """Parse and validate the closed persisted schema."""
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
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def matches_pool(self, pool: TaskPool) -> bool:
        """Check retained seeds, stratum counts, and content hash."""
        return (
            _retained_seeds_within_range(pool, self.seed_range)
            and self.stratum_counts == pool.stratum_counts()
            and self.content_hash == content_hash(pool)
        )
