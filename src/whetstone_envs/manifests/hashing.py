from dr_serialize import (
    Jsonable,
    Sha256Digest,
    build_identity_document,
    identity_document_hash,
)

from whetstone_envs.instances import Instance, InstancePool

_POOL_IDENTITY_SCHEMA = "whetstone_envs.task_pool"
_POOL_IDENTITY_SCHEMA_VERSION = 2


def _identity_instance(instance: Instance) -> dict[str, Jsonable]:
    return {
        "id": instance.id,
        "seed": instance.seed,
        "categories": list(instance.categories),
        "prompt_inputs": dict(sorted(instance.prompt_inputs.items())),
        "gold": instance.gold,
    }


def content_hash(pool: InstancePool) -> Sha256Digest:
    """Return the versioned identity hash for the pool's ordered content.

    The identity schema freezes which task-instance facts participate. Pool
    order remains significant; prompt-input mapping order and group names do
    not participate.
    """
    document = build_identity_document(
        schema=_POOL_IDENTITY_SCHEMA,
        schema_version=_POOL_IDENTITY_SCHEMA_VERSION,
        payload={
            "instances": [
                _identity_instance(instance)
                for instance in pool.instance_group.instances
            ],
        },
    )
    return identity_document_hash(document)
