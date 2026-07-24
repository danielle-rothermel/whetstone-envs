"""Pool manifest: a diffable JSON record of a generated pool.

Every candidate's generation script writes a :class:`Manifest`
alongside the pool it produces. The manifest pins the inputs that
determine the pool (generator version/commit, seed range) and records
what came out (per-stratum counts and a content hash of the instances),
so a regenerated pool can be diffed against a frozen one: if the
generator is deterministic, the hashes match, and if a stratum count
drifts the manifest shows exactly which one.

The content hash is computed over a canonical serialization of the
pool's instances (sorted keys, stable field order), so it is
independent of dict iteration order and reproducible across runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.core.instance import Instance
    from whetstone_envs.core.pool import TaskPool

MANIFEST_SCHEMA_VERSION = 1
_SEED_RANGE_LEN = 2
_PERSISTED_FIELDS = frozenset(
    {
        "schema_version",
        "generator_version",
        "seed_range",
        "stratum_counts",
        "content_hash",
    },
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _require_integer(value: object, *, field_name: str) -> int:
    """Return an integer field without coercing JSON scalar values."""
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
    if start > end:
        msg = "manifest seed_range must be ordered from start to end"
        raise ValueError(msg)
    return (start, end)


def _validate_stratum_counts(
    value: object,
) -> MappingProxyType[str, int]:
    if not isinstance(value, Mapping):
        msg = "manifest stratum_counts must be an object"
        raise TypeError(msg)

    counts: dict[str, int] = {}
    for name, raw_count in value.items():
        if not isinstance(name, str):
            msg = "manifest stratum_counts names must be strings"
            raise TypeError(msg)
        if not name.strip():
            msg = "manifest stratum_counts names must be nonblank strings"
            raise ValueError(msg)
        count = _require_integer(
            raw_count,
            field_name="stratum_counts",
        )
        if count < 0:
            msg = "manifest stratum_counts values must be non-negative"
            raise ValueError(msg)
        counts[name] = count
    return MappingProxyType(counts)


def _validate_content_hash(value: object) -> str:
    if not isinstance(value, str):
        msg = "manifest content_hash must be a string"
        raise TypeError(msg)
    if _SHA256_HEX.fullmatch(value) is None:
        msg = (
            "manifest content_hash must be a canonical lowercase "
            "SHA-256 hex digest"
        )
        raise ValueError(msg)
    return value


def _canonical_instance(instance: Instance) -> dict[str, object]:
    """Project an instance onto its content-hashable public fields.

    Prompt inputs are sorted by key so the serialization does not depend
    on insertion order; every field that defines the instance's identity
    for a downstream consumer is included.
    """
    return {
        "id": instance.id,
        "seed": instance.seed,
        "strata": list(instance.strata),
        "prompt_inputs": dict(sorted(instance.prompt_inputs.items())),
        "gold": instance.gold,
    }


def content_hash(pool: TaskPool) -> str:
    """Return a stable SHA-256 hex digest of the pool's instances.

    Instances are serialized in pool order via a canonical JSON encoding
    (sorted keys, no incidental whitespace). Two pools hash equal iff
    their instances are field-for-field identical in the same order.
    """
    payload = [_canonical_instance(i) for i in pool.instances]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Manifest:
    """A pinned, diffable description of one generated pool.

    Parameters
    ----------
    generator_version:
        The candidate generator's version/commit identity. Bumping this
        is the license to change generated content.
    seed_range:
        The ``(start, end)`` seed pins the pool was generated from,
        inclusive of ``start`` and exclusive of ``end`` by convention.
    stratum_counts:
        Declared per-stratum instance counts, matched against a pool's
        own :meth:`~whetstone_envs.core.pool.TaskPool.stratum_counts`.
        Copied into a read-only mapping at construction.
    content_hash:
        SHA-256 of the pool's instances (see :func:`content_hash`).
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
            _validate_schema_version(self.schema_version),
        )
        object.__setattr__(
            self,
            "generator_version",
            _validate_generator_version(self.generator_version),
        )
        object.__setattr__(
            self,
            "seed_range",
            _validate_seed_range(self.seed_range),
        )
        object.__setattr__(
            self,
            "stratum_counts",
            _validate_stratum_counts(self.stratum_counts),
        )
        object.__setattr__(
            self,
            "content_hash",
            _validate_content_hash(self.content_hash),
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

        Per-stratum counts and the content hash are read off the pool,
        so a manifest built this way always describes the pool it was
        built from.
        """
        return cls(
            generator_version=generator_version,
            seed_range=seed_range,
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
        """Write the manifest JSON to ``path`` (trailing newline)."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        """Read a manifest previously written by :meth:`write`."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def matches_pool(self, pool: TaskPool) -> bool:
        """True if ``pool`` matches this manifest's counts and hash.

        This is the regeneration diff check: build a fresh pool, then
        assert the frozen manifest still describes it. A drift in any
        stratum count or in the content hash returns ``False``.
        """
        return (
            self.stratum_counts == pool.stratum_counts()
            and self.content_hash == content_hash(pool)
        )
