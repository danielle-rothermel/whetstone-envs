"""Canonical task-pool projection and content hashing.

The hash covers a canonical serialization of the pool's instances, so it is
independent of mapping insertion order and reproducible across runs.
"""

import hashlib
import json

from whetstone_envs.instances import Instance
from whetstone_envs.pools import TaskPool


def _canonical_instance(instance: Instance) -> dict[str, object]:
    """Project an instance onto its content-hashable public fields.

    Prompt inputs are sorted by key so serialization does not depend on
    insertion order; every field that defines the instance's identity for a
    downstream consumer is included.
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

    Instances are serialized in pool order via canonical JSON with sorted
    keys and no incidental whitespace. Two pools hash equal iff their
    instances are field-for-field identical in the same order.
    """
    payload = [_canonical_instance(instance) for instance in pool.instances]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
