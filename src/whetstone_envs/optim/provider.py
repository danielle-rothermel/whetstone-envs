from __future__ import annotations

from dr_providers import RequestControl, openrouter_chat_config


def openrouter_seeded_call_config(*, model: str):
    """Return the OpenRouter chat preset, which advertises SEED."""
    config = openrouter_chat_config(model=model)
    if not config.definition.constraints.supports(RequestControl.SEED):
        msg = (
            "OpenRouter preset for "
            f"{model!r} does not advertise RequestControl.SEED"
        )
        raise ValueError(msg)
    return config


__all__ = ["openrouter_seeded_call_config"]
