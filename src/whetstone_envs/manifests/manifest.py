from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from typing import TYPE_CHECKING, cast

from dr_serialize import (
    CANONICAL_JSON_MAX_CONTAINER_DEPTH,
    Jsonable,
    Sha256Digest,
    canonical_json,
    canonical_json_bytes,
    decode_strict_json_bytes,
    validate_strict_json,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from whetstone_envs.manifests.hashing import content_hash
from whetstone_envs.manifests.validation import (
    MANIFEST_SCHEMA_VERSION,
    validate_generator_version,
    validate_schema_version,
    validate_seed_range,
    validate_stratum_counts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.pools import TaskPool

_MANIFEST_MAX_BYTES = 1 << 20
_PERSISTED_FIELDS = frozenset(
    {
        "schema_version",
        "generator_version",
        "seed_range",
        "stratum_counts",
        "content_hash",
    }
)


def _retained_seeds_within_range(
    pool: TaskPool,
    seed_range: tuple[int, int],
) -> bool:
    start, end = seed_range
    return all(start <= instance.seed < end for instance in pool.instances)


class Manifest(BaseModel):
    """A validated, read-only description of a generated pool.

    ``seed_range`` covers retained instances only, and ``stratum_counts`` is
    detached and frozen. Matching retained seeds, counts, and the content hash
    to a pool is explicit via :meth:`matches_pool`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: Mapping[str, int]
    content_hash: Sha256Digest
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        return validate_schema_version(value)

    @field_validator("generator_version")
    @classmethod
    def _validate_generator_version(cls, value: str) -> str:
        return validate_generator_version(value)

    @field_validator("seed_range", mode="before")
    @classmethod
    def _materialize_json_seed_range(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if info.mode == "json" and isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("seed_range")
    @classmethod
    def _validate_seed_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        return validate_seed_range(value)

    @field_validator("stratum_counts")
    @classmethod
    def _validate_stratum_counts(
        cls,
        value: Mapping[str, int],
    ) -> Mapping[str, int]:
        return validate_stratum_counts(value)

    @field_serializer("stratum_counts")
    def _serialize_stratum_counts(
        self,
        value: Mapping[str, int],
    ) -> dict[str, int]:
        return dict(sorted(value.items()))

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

    def to_dict(self) -> dict[str, Jsonable]:
        return cast("dict[str, Jsonable]", self.model_dump(mode="json"))

    @classmethod
    def from_dict(cls, data: object) -> Manifest:
        """Parse and validate the closed persisted schema."""
        if not isinstance(data, dict):
            msg = "manifest must be a JSON object"
            raise TypeError(msg)
        missing_fields = _PERSISTED_FIELDS.difference(data)
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            msg = f"manifest required fields missing: {names}"
            raise ValueError(msg)
        unknown_fields = set(data).difference(_PERSISTED_FIELDS)
        if unknown_fields:
            names = ", ".join(sorted(map(str, unknown_fields)))
            msg = f"manifest contains unknown fields: {names}"
            raise ValueError(msg)
        payload = validate_strict_json(data)
        return cls.model_validate_json(canonical_json(payload), strict=True)

    def to_json(self) -> str:
        """Serialize to the exact Canonical JSON Text representation."""
        return canonical_json(self.to_dict())

    def write(self, path: Path) -> None:
        """Write canonical bytes without atomicity or durability guarantees."""
        path.write_bytes(canonical_json_bytes(self.to_dict()))

    @classmethod
    def read(cls, path: Path) -> Manifest:
        with path.open("rb") as manifest_file:
            raw = manifest_file.read(_MANIFEST_MAX_BYTES + 1)
        payload = decode_strict_json_bytes(
            raw,
            max_bytes=_MANIFEST_MAX_BYTES,
            max_depth=CANONICAL_JSON_MAX_CONTAINER_DEPTH,
        )
        if canonical_json_bytes(payload) != raw:
            msg = "manifest file must contain exact Canonical JSON Text"
            raise ValueError(msg)
        return cls.from_dict(payload)

    def matches_pool(self, pool: TaskPool) -> bool:
        """Check retained seeds, stratum counts, and content hash."""
        return (
            _retained_seeds_within_range(pool, self.seed_range)
            and self.stratum_counts == pool.stratum_counts()
            and self.content_hash == content_hash(pool)
        )
