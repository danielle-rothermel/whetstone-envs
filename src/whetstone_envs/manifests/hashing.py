import hashlib
import json

from whetstone_envs.instances import Instance
from whetstone_envs.pools import TaskPool


def _canonical_instance(instance: Instance) -> dict[str, object]:
    """Return the canonical JSON-ready instance hash payload."""
    return {
        "id": instance.id,
        "seed": instance.seed,
        "strata": list(instance.strata),
        "prompt_inputs": dict(sorted(instance.prompt_inputs.items())),
        "gold": instance.gold,
    }


def content_hash(pool: TaskPool) -> str:
    """Hash instances in pool order using canonical JSON.

    Prompt-input mapping order does not affect the digest.
    """
    payload = [_canonical_instance(instance) for instance in pool.instances]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
