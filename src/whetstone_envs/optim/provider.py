from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers import HttpProvider, RequestControl, openrouter_chat_config

if TYPE_CHECKING:
    from whetstone.provider.policy import ProviderExecutionPolicy


class OpenRouterTransport:
    """Hold one HttpProvider so eval and proposal share one live client."""

    def __init__(self, policy: ProviderExecutionPolicy) -> None:
        self._provider = HttpProvider(policy=policy.transport_policy)

    def __call__(self, request: object):
        return self._provider.invoke(request)


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


def openrouter_transport_factory(policy: ProviderExecutionPolicy):
    return OpenRouterTransport(policy)


def bind_openrouter_transport(policy: ProviderExecutionPolicy):
    """Return one transport and a factory that always yields it."""
    transport = OpenRouterTransport(policy)

    def factory(_policy: ProviderExecutionPolicy) -> OpenRouterTransport:
        return transport

    return transport, factory


__all__ = [
    "OpenRouterTransport",
    "bind_openrouter_transport",
    "openrouter_seeded_call_config",
    "openrouter_transport_factory",
]
