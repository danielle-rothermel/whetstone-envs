from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers import (
    HttpProvider,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    RequestControl,
    openrouter_chat_config,
)
from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence
from dr_providers.outcomes.models import ProviderTransportResponse

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.experiment import c19_render_contract

if TYPE_CHECKING:
    from collections.abc import Mapping

    from whetstone.experiment.env import Experiment
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


def c19_fake_task_reply(prompt: str, gold_by_prompt: Mapping[str, str]) -> str:
    return gold_by_prompt.get(prompt, f"generated: {prompt}")


def c19_fake_gold_by_prompt(experiment: Experiment) -> dict[str, str]:
    """Map every prepared C19 ceiling prompt to its exact task gold."""
    contract = c19_render_contract()
    gold_by_prompt: dict[str, str] = {}
    for split in (
        experiment.eval_configs.internal,
        experiment.eval_configs.official,
    ):
        for task in split.tasks:
            gold = getattr(task, "gold", None)
            inputs = getattr(task, "prompt_inputs", None)
            if not isinstance(gold, str) or not isinstance(inputs, dict):
                raise TypeError(
                    "C19 task must expose strict prompt_inputs and gold"
                )
            prompt = contract.render(PROBES.ceiling_template, inputs)
            existing = gold_by_prompt.get(prompt)
            if existing is not None and existing != gold:
                raise ValueError(
                    "one C19 ceiling prompt maps to multiple golds"
                )
            gold_by_prompt[prompt] = gold
    return gold_by_prompt


class C19FakeTaskTransport:
    """Emit gold for ceiling-rendered prompts; echo every other prompt."""

    def __init__(
        self,
        policy: ProviderExecutionPolicy,
        gold_by_prompt: Mapping[str, str],
    ) -> None:
        self._policy = policy
        self._gold_by_prompt = dict(gold_by_prompt)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        messages = getattr(
            getattr(request, "transcript", None), "messages", ()
        )
        prompt = messages[-1].content if messages else ""
        if not isinstance(prompt, str):
            prompt = str(prompt or "")
        text = c19_fake_task_reply(prompt, self._gold_by_prompt)
        response = ProviderTransportResponse(text=text, stop_reason="stop")
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self._policy.transport_policy,
            http_request=ProviderHttpRequestEvidence(
                method="POST",
                url="http://whetstone.fake/llm",
                headers={},
                body={},
                body_bytes=0,
            ),
            outcome=response,
        )


def c19_fake_transport_factory(*, gold_by_prompt: Mapping[str, str]):
    replies = dict(gold_by_prompt)

    def factory(policy: ProviderExecutionPolicy) -> C19FakeTaskTransport:
        return C19FakeTaskTransport(policy, replies)

    return factory


__all__ = [
    "C19FakeTaskTransport",
    "OpenRouterTransport",
    "bind_openrouter_transport",
    "c19_fake_gold_by_prompt",
    "c19_fake_task_reply",
    "c19_fake_transport_factory",
    "openrouter_seeded_call_config",
    "openrouter_transport_factory",
]
