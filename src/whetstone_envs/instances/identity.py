"""Public identities derived from task instances."""

from whetstone_envs.instances.instance import Instance


def public_prompt_identity(instance: Instance) -> tuple[tuple[str, str], ...]:
    """Return the canonical public identity of an instance's prompt.

    Public prompt identity is defined solely by ``prompt_inputs``, sorted by
    key. It intentionally excludes private generation and evaluation metadata
    such as ``id``, ``seed``, ``strata``, and ``gold``.
    """
    return tuple(sorted(instance.prompt_inputs.items()))
