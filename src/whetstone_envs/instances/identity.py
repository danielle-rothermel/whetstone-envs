from whetstone_envs.instances.instance import Instance


def public_prompt_identity(instance: Instance) -> tuple[tuple[str, str], ...]:
    """Return the canonical identity of sorted ``prompt_inputs``.

    This identity does not necessarily identify rendered text.
    """
    return tuple(sorted(instance.prompt_inputs.items()))
