from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from dr_providers import (
    GenerationControls,
    HttpProvider,
    ProviderCallConfig,
    ProviderGenerateRequest,
    ProviderInvocationEvidence,
    ReasoningEffort,
    RecoverabilityClass,
    RequestControl,
    openrouter_chat_config,
)
from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence
from dr_providers.outcomes.models import ProviderTransportResponse

from whetstone_envs.evaluation.concurrency import (
    DEFAULT_PROVIDER_CONCURRENCY,
    MAX_UNFORCED_PROVIDER_CONCURRENCY,
    PROVIDER_CONCURRENCY_FLAG,
    PROVIDER_CONCURRENCY_FORCE_FLAG,
    resolve_provider_concurrency,
    validate_provider_concurrency,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from whetstone.provider.driver import TransportCall
    from whetstone.provider.policy import ProviderExecutionPolicy


TASK_CALL_TIMEOUT_SECONDS = 300.0

TASK_CALL_MAX_ATTEMPTS = 5

# The transport owns the whole retry budget; disable the driver retry loop.
DRIVER_MAX_ATTEMPTS = 1

RETRY_BASE_SECONDS = 2.0
RETRY_MULTIPLIER = 2.0
RETRY_MAX_SECONDS = 32.0

RETRY_JITTER_FRACTION = 0.25

MAX_HONOURED_RETRY_AFTER_SECONDS = 120.0


def _retry_after_seconds(evidence: ProviderInvocationEvidence) -> float | None:
    # Honor delta hints only; HTTP dates need a clock-dependent policy.
    hint = evidence.retry_after
    if hint is None or hint.kind != "delta_seconds":
        return None
    try:
        seconds = float(hint.value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return min(seconds, MAX_HONOURED_RETRY_AFTER_SECONDS)


def _is_transient(evidence: ProviderInvocationEvidence) -> bool:
    failure = evidence.failure
    if failure is None:
        return False
    return failure.recoverability in _TRANSIENT_RECOVERABILITY


_TRANSIENT_RECOVERABILITY = frozenset(
    {
        RecoverabilityClass.TRANSIENT,
        RecoverabilityClass.RATE_LIMITED,
        RecoverabilityClass.RESOURCE_EXHAUSTION,
    }
)


class OpenRouterTransport:
    def __init__(self, policy: ProviderExecutionPolicy) -> None:
        self._provider = HttpProvider(policy=policy.transport_policy)

    def __call__(
        self, request: ProviderGenerateRequest
    ) -> ProviderInvocationEvidence:
        return self._provider.invoke(request)


class RetryingTransport:
    def __init__(
        self,
        inner: TransportCall,
        *,
        max_attempts: int = TASK_CALL_MAX_ATTEMPTS,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts is at least 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._sleep = sleep if sleep is not None else time.sleep
        self._rng = rng if rng is not None else random.Random()  # noqa: S311

    def delay_for(self, attempt_number: int) -> float:
        if attempt_number < 1:
            raise ValueError("attempt_number is at least 1")
        raw = RETRY_BASE_SECONDS * RETRY_MULTIPLIER ** (attempt_number - 1)
        capped = min(raw, RETRY_MAX_SECONDS)
        low = capped * (1.0 - RETRY_JITTER_FRACTION)
        return self._rng.uniform(low, capped)

    def __call__(
        self, request: ProviderGenerateRequest
    ) -> ProviderInvocationEvidence:
        evidence = self._inner(request)
        for attempt_number in range(1, self._max_attempts):
            if not _is_transient(evidence):
                return evidence

            requested = _retry_after_seconds(evidence)
            delay = (
                requested
                if requested is not None
                else self.delay_for(attempt_number)
            )
            self._sleep(delay)
            evidence = self._inner(request)
        return evidence


def widened_execution_policy(
    policy: ProviderExecutionPolicy, *, concurrency: int
) -> ProviderExecutionPolicy:
    validate_provider_concurrency(concurrency)
    transport_policy = policy.transport_policy
    if transport_policy.max_connections >= concurrency:
        return policy
    return policy.model_copy(
        update={
            "transport_policy": transport_policy.model_copy(
                update={
                    "max_connections": concurrency,
                    "max_keepalive_connections": concurrency,
                }
            )
        }
    )


def openrouter_seeded_call_config(
    *, model: str, reasoning_effort: ReasoningEffort | None = None
) -> ProviderCallConfig:
    controls = (
        None
        if reasoning_effort is None
        else GenerationControls(reasoning=reasoning_effort)
    )
    config = openrouter_chat_config(model=model, controls=controls)
    if not config.definition.constraints.supports(RequestControl.SEED):
        msg = (
            "OpenRouter preset for "
            f"{model!r} does not advertise RequestControl.SEED"
        )
        raise ValueError(msg)
    return config


def openrouter_transport_factory(
    policy: ProviderExecutionPolicy,
) -> OpenRouterTransport:
    return OpenRouterTransport(policy)


def hardened_execution_policy(
    policy: ProviderExecutionPolicy,
) -> ProviderExecutionPolicy:
    return policy.model_copy(
        update={
            "max_attempts": DRIVER_MAX_ATTEMPTS,
            "transport_policy": policy.transport_policy.model_copy(
                update={
                    "timeout_seconds": TASK_CALL_TIMEOUT_SECONDS,
                    "idle_timeout_seconds": TASK_CALL_TIMEOUT_SECONDS,
                }
            ),
        }
    )


def bind_openrouter_transport(
    policy: ProviderExecutionPolicy,
) -> tuple[
    RetryingTransport, Callable[[ProviderExecutionPolicy], RetryingTransport]
]:
    transport = RetryingTransport(OpenRouterTransport(policy))

    def factory(_policy: ProviderExecutionPolicy) -> RetryingTransport:
        return transport

    return transport, factory


def fake_task_reply(prompt: str, replies: Mapping[str, str]) -> str:
    return replies.get(prompt, f"generated: {prompt}")


class FakeTaskTransport:
    def __init__(
        self,
        policy: ProviderExecutionPolicy,
        replies: Mapping[str, str],
    ) -> None:
        self._policy = policy
        self._replies = dict(replies)

    def __call__(
        self, request: ProviderGenerateRequest
    ) -> ProviderInvocationEvidence:
        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
        text = fake_task_reply(prompt, self._replies)
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


def fake_transport_factory(
    *,
    replies: Mapping[str, str],
) -> Callable[[ProviderExecutionPolicy], FakeTaskTransport]:
    replies = dict(replies)

    def factory(policy: ProviderExecutionPolicy) -> FakeTaskTransport:
        return FakeTaskTransport(policy, replies)

    return factory


__all__ = [
    "DEFAULT_PROVIDER_CONCURRENCY",
    "DRIVER_MAX_ATTEMPTS",
    "MAX_HONOURED_RETRY_AFTER_SECONDS",
    "MAX_UNFORCED_PROVIDER_CONCURRENCY",
    "PROVIDER_CONCURRENCY_FLAG",
    "PROVIDER_CONCURRENCY_FORCE_FLAG",
    "RETRY_BASE_SECONDS",
    "RETRY_JITTER_FRACTION",
    "RETRY_MAX_SECONDS",
    "RETRY_MULTIPLIER",
    "TASK_CALL_MAX_ATTEMPTS",
    "TASK_CALL_TIMEOUT_SECONDS",
    "FakeTaskTransport",
    "OpenRouterTransport",
    "RetryingTransport",
    "bind_openrouter_transport",
    "fake_task_reply",
    "fake_transport_factory",
    "hardened_execution_policy",
    "openrouter_seeded_call_config",
    "openrouter_transport_factory",
    "resolve_provider_concurrency",
    "validate_provider_concurrency",
    "widened_execution_policy",
]
