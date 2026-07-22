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
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.core.instance import Instance
    from whetstone_envs.core.pool import TaskPool

MANIFEST_SCHEMA_VERSION = 1
_SEED_RANGE_LEN = 2


def _as_int(value: object) -> int:
    """Coerce a JSON scalar to ``int`` or raise ``TypeError``.

    ``json`` decodes numbers as ``int`` or ``float`` and never as
    arbitrary objects, so this both narrows the static type and rejects
    a manifest whose numeric field was corrupted into a non-number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"expected an integer manifest field, got {value!r}"
        raise TypeError(msg)
    return int(value)


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
    content_hash:
        SHA-256 of the pool's instances (see :func:`content_hash`).
    schema_version:
        Version of the manifest format itself, for forward migration.
    """

    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: dict[str, int]
    content_hash: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

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
    def from_dict(cls, data: dict[str, object]) -> Manifest:
        """Reconstruct a manifest from its :meth:`to_dict` form."""
        seed_range = data["seed_range"]
        if (
            not isinstance(seed_range, list)
            or len(seed_range) != _SEED_RANGE_LEN
        ):
            msg = "manifest seed_range must be a two-element list"
            raise TypeError(msg)
        raw_counts = data["stratum_counts"]
        if not isinstance(raw_counts, dict):
            msg = "manifest stratum_counts must be an object"
            raise TypeError(msg)
        return cls(
            generator_version=str(data["generator_version"]),
            seed_range=(_as_int(seed_range[0]), _as_int(seed_range[1])),
            stratum_counts={str(k): _as_int(v) for k, v in raw_counts.items()},
            content_hash=str(data["content_hash"]),
            schema_version=_as_int(
                data.get("schema_version", MANIFEST_SCHEMA_VERSION),
            ),
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
