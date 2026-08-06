from dr_serialize import (
    Jsonable,
    Sha256Digest,
    build_identity_document,
    identity_document_hash,
)

from whetstone_envs.instances import Instance
from whetstone_envs.pools import TaskPool

_POOL_IDENTITY_SCHEMA = "whetstone_envs.task_pool"
_POOL_IDENTITY_SCHEMA_VERSION = 1


def _identity_instance(instance: Instance) -> dict[str, Jsonable]:
    """Return the task-instance facts participating in pool identity."""
    return {
        "id": instance.id,
        "seed": instance.seed,
        "strata": list(instance.strata),
        "prompt_inputs": dict(sorted(instance.prompt_inputs.items())),
        "gold": instance.gold,
    }


def content_hash(pool: TaskPool) -> Sha256Digest:
    """Return the versioned identity hash for the pool's ordered content.

    The identity schema freezes which task-instance facts participate. Pool
    order remains significant; prompt-input mapping order does not.
    """
    document = build_identity_document(
        schema=_POOL_IDENTITY_SCHEMA,
        schema_version=_POOL_IDENTITY_SCHEMA_VERSION,
        payload={
            "instances": [
                _identity_instance(instance) for instance in pool.instances
            ],
        },
    )
    return identity_document_hash(document)
