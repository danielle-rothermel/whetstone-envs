from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from dr_serialize import (
    CANONICAL_JSON_MAX_CONTAINER_DEPTH,
    Jsonable,
    Sha256Digest,
    canonical_json,
    validate_strict_json,
)
from dr_store import CanonicalJsonFile
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from whetstone_envs.manifests.hashing import content_hash

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.instances import InstancePool

MANIFEST_SCHEMA_VERSION = 2
_SEED_RANGE_LEN = 2


def _require_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"manifest {field_name} must be an integer"
        raise TypeError(msg)
    return value


def _validate_schema_version(value: object) -> int:
    schema_version = _require_integer(
        value,
        field_name="schema_version",
    )
    if schema_version != MANIFEST_SCHEMA_VERSION:
        msg = (
            f"manifest schema_version {schema_version} is unsupported; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return schema_version


def _validate_generator_version(value: object) -> str:
    if not isinstance(value, str):
        msg = "manifest generator_version must be a string"
        raise TypeError(msg)
    if not value:
        msg = "manifest generator_version must be a nonempty string"
        raise ValueError(msg)
    return value


def _validate_seed_range(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != _SEED_RANGE_LEN:
        msg = "manifest seed_range must be a two-element tuple"
        raise TypeError(msg)
    start = _require_integer(value[0], field_name="seed_range")
    end = _require_integer(value[1], field_name="seed_range")
    if start >= end:
        msg = "manifest seed_range must be ordered with start less than end"
        raise ValueError(msg)
    return (start, end)


def _validate_category_counts(
    value: object,
) -> MappingProxyType[str, int]:
    """Validate positive counts and return a detached, read-only mapping."""
    if not isinstance(value, Mapping):
        msg = "manifest category_counts must be an object"
        raise TypeError(msg)

    counts: dict[str, int] = {}
    for name, raw_count in value.items():
        if not isinstance(name, str):
            msg = "manifest category_counts names must be strings"
            raise TypeError(msg)
        if not name.strip():
            msg = "manifest category_counts names must be nonblank strings"
            raise ValueError(msg)
        count = _require_integer(
            raw_count,
            field_name="category_counts",
        )
        if count < 0:
            msg = "manifest category_counts values must be non-negative"
            raise ValueError(msg)
        if count == 0:
            msg = "manifest category_counts values must be positive"
            raise ValueError(msg)
        counts[name] = count
    if sum(counts.values()) == 0:
        msg = "manifest category_counts must have a positive total"
        raise ValueError(msg)
    return MappingProxyType(counts)


_MANIFEST_MAX_BYTES = 1 << 20
_PERSISTED_FIELDS = frozenset(
    {
        "schema_version",
        "generator_version",
        "seed_range",
        "category_counts",
        "content_hash",
    }
)


def _retained_seeds_within_range(
    pool: InstancePool,
    seed_range: tuple[int, int],
) -> bool:
    start, end = seed_range
    return all(
        start <= instance.seed < end
        for instance in pool.instance_group.instances
    )


def _manifest_document(path: Path) -> CanonicalJsonFile:
    return CanonicalJsonFile(
        path.parent,
        path.name,
        max_bytes=_MANIFEST_MAX_BYTES,
        max_depth=CANONICAL_JSON_MAX_CONTAINER_DEPTH,
    )


class Manifest(BaseModel):
    """A validated, read-only description of a generated pool.

    ``seed_range`` covers retained instances only, and ``category_counts`` is
    detached and frozen. Matching retained seeds, counts, and the content hash
    to a pool is explicit via :meth:`matches_pool`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generator_version: str
    seed_range: tuple[int, int]
    category_counts: Mapping[str, int]
    content_hash: Sha256Digest
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        return _validate_schema_version(value)

    @field_validator("generator_version")
    @classmethod
    def _validate_generator_version(cls, value: str) -> str:
        return _validate_generator_version(value)

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
        return _validate_seed_range(value)

    @field_validator("category_counts")
    @classmethod
    def _validate_category_counts(
        cls,
        value: Mapping[str, int],
    ) -> Mapping[str, int]:
        return _validate_category_counts(value)

    @field_serializer("category_counts")
    def _serialize_category_counts(
        self,
        value: Mapping[str, int],
    ) -> dict[str, int]:
        return dict(sorted(value.items()))

    @classmethod
    def from_pool(
        cls,
        pool: InstancePool,
        *,
        generator_version: str,
        seed_range: tuple[int, int],
    ) -> Manifest:
        """Derive a manifest from a pool and its generation inputs.

        Retained seeds must lie in the declared half-open range. The manifest
        says nothing about generator attempts producing no retained instance.
        """
        validated_seed_range = _validate_seed_range(seed_range)
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
            category_counts=pool.category_counts(),
            content_hash=content_hash(pool),
        )

    def to_dict(self) -> dict[str, Jsonable]:
        return cast("dict[str, Jsonable]", self.model_dump(mode="json"))

    @classmethod
    def from_dict(cls, data: object) -> Manifest:
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
        return canonical_json(self.to_dict())

    def write(self, path: Path) -> None:
        _manifest_document(path).publish(self.to_dict())

    @classmethod
    def read(cls, path: Path) -> Manifest:
        return cls.from_dict(_manifest_document(path).read())

    def matches_pool(self, pool: InstancePool) -> bool:
        return (
            _retained_seeds_within_range(pool, self.seed_range)
            and self.category_counts == pool.category_counts()
            and self.content_hash == content_hash(pool)
        )
