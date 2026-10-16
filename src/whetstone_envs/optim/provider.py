from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

from dr_providers import (
    GenerationControls,
    HttpProvider,
    ProviderGenerateRequest,
    ProviderInvocationEvidence,
    ReasoningEffort,
    RecoverabilityClass,
    RequestControl,
    openrouter_chat_config,
)
from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence
from dr_providers.outcomes.models import ProviderTransportResponse

from whetstone_envs.optim.concurrency import (
    DEFAULT_PROVIDER_CONCURRENCY,
    MAX_UNFORCED_PROVIDER_CONCURRENCY,
    PROVIDER_CONCURRENCY_FLAG,
    PROVIDER_CONCURRENCY_FORCE_FLAG,
    resolve_provider_concurrency,
    validate_provider_concurrency,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.experiment.env import Experiment
    from whetstone.provider.policy import ProviderExecutionPolicy


TASK_CALL_TIMEOUT_SECONDS = 300.0

#: How many times one provider call may be attempted.
#:
#: Five rather than whetstone's three: a 429 under load is the failure
#: this exists for, and a rate limit that clears needs enough attempts to
#: outlast a burst. Every attempt after the first waits -- see
#: :class:`RetryingTransport` -- so this is bounded in time as well as in
#: count.
#:
#: This is the *whole* budget for one logical call, spent inside
#: :class:`RetryingTransport`. The driver's own loop is disabled -- see
#: :data:`DRIVER_MAX_ATTEMPTS` -- so five means five, not five per driver
#: attempt.
TASK_CALL_MAX_ATTEMPTS = 5

# The transport owns retry delays; disable the driver's second retry loop.
DRIVER_MAX_ATTEMPTS = 1

#: The backoff schedule between attempts: 2 s, 4 s, 8 s, 16 s, capped.
#:
#: Exponential because a rate limit that is still firing should be given
#: geometrically more room rather than be re-probed at a fixed rate.
RETRY_BASE_SECONDS = 2.0
RETRY_MULTIPLIER = 2.0
RETRY_MAX_SECONDS = 32.0

#: The fraction of a computed delay that is randomized.
#:
#: Without jitter, N concurrent workers rate-limited by the same burst all
#: sleep the same duration and retry in lockstep, reproducing the burst
#: that limited them. The wait is drawn from ``[(1-J)*d, d]`` so the
#: retries spread out while the schedule's ceiling still holds.
RETRY_JITTER_FRACTION = 0.25

#: How long a provider's own ``Retry-After`` may hold a worker.
#:
#: The header is honoured because the provider knows better than this
#: schedule when it will accept traffic again, but it is bounded: a
#: mistaken or hostile header naming an hour would otherwise park a
#: worker for an hour.
MAX_HONOURED_RETRY_AFTER_SECONDS = 120.0


def _retry_after_seconds(evidence: object) -> float | None:
    """Seconds the provider asked for, when it named a plain delta.

    Only ``delta_seconds`` is honoured. The HTTP-date form would have to
    be parsed against the provider's clock and this package's, and a
    disagreement between the two is exactly the case where waiting the
    wrong amount is worst; the schedule's own backoff is the safer answer.
    """
    hint = getattr(evidence, "retry_after", None)
    if hint is None or getattr(hint, "kind", None) != "delta_seconds":
        return None
    try:
        seconds = float(hint.value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return min(seconds, MAX_HONOURED_RETRY_AFTER_SECONDS)


def _is_transient(evidence: object) -> bool:
    """Whether this outcome is worth another attempt.

    Read off dr-providers' own ``RecoverabilityClass`` rather than off
    status codes, so the classification stays the transport's to make: a
    rate limit, a transient 5xx, and a contained timeout are all retried,
    and a permanent rejection -- a bad request, a refused key -- is not,
    because re-sending it would spend again to be told the same thing.
    """
    failure = getattr(evidence, "failure", None)
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
    """Hold one HttpProvider so eval and proposal share one live client."""

    def __init__(self, policy: ProviderExecutionPolicy) -> None:
        self._provider = HttpProvider(policy=policy.transport_policy)

    def __call__(self, request: object):
        return self._provider.invoke(request)


class RetryingTransport:
    def __init__(
        self,
        inner: Callable[[Any], Any],
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
        # Jitter spreads retries apart so concurrent workers do not
        # retry in lockstep; it is not a security primitive.
        self._rng = rng if rng is not None else random.Random()  # noqa: S311

    def delay_for(self, attempt_number: int) -> float:
        """The jittered wait before ``attempt_number``'s retry."""
        if attempt_number < 1:
            raise ValueError("attempt_number is at least 1")
        raw = RETRY_BASE_SECONDS * RETRY_MULTIPLIER ** (attempt_number - 1)
        capped = min(raw, RETRY_MAX_SECONDS)
        low = capped * (1.0 - RETRY_JITTER_FRACTION)
        return self._rng.uniform(low, capped)

    def __call__(self, request: object):
        evidence = self._inner(request)
        for attempt_number in range(1, self._max_attempts):
            if not _is_transient(evidence):
                return evidence

            # The provider's own instruction wins over the schedule when
            # it gave one, because it knows when it will accept traffic.
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
):
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


def openrouter_transport_factory(policy: ProviderExecutionPolicy):
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


def bind_openrouter_transport(policy: ProviderExecutionPolicy):
    transport = RetryingTransport(OpenRouterTransport(policy))

    def factory(_policy: ProviderExecutionPolicy) -> RetryingTransport:
        return transport

    return transport, factory


def fake_task_reply(prompt: str, gold_by_prompt: Mapping[str, str]) -> str:
    return gold_by_prompt.get(prompt, f"generated: {prompt}")


def fake_gold_by_prompt(
    experiment: Experiment,
    *,
    render_contract: TemplateRenderContract,
    ceiling_template: str,
) -> dict[str, str]:
    """Map every prepared ceiling prompt to its exact task gold.

    The family supplies its own contract and ceiling template, so one fake
    transport serves every family: the scripted ceiling answer is that
    family's own gold, and any other prompt is echoed.
    """
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
                    "an eval task must expose strict prompt_inputs and gold"
                )
            prompt = render_contract.render(ceiling_template, inputs)
            existing = gold_by_prompt.get(prompt)
            if existing is not None and existing != gold:
                raise ValueError("one ceiling prompt maps to multiple golds")
            gold_by_prompt[prompt] = gold
    return gold_by_prompt


class FakeTaskTransport:
    """Emit gold for ceiling-rendered prompts; echo every other prompt."""

    def __init__(
        self,
        policy: ProviderExecutionPolicy,
        gold_by_prompt: Mapping[str, str],
    ) -> None:
        self._policy = policy
        self._gold_by_prompt = dict(gold_by_prompt)

    def __call__(
        self, request: ProviderGenerateRequest
    ) -> ProviderInvocationEvidence:
        messages = getattr(
            getattr(request, "transcript", None), "messages", ()
        )
        prompt = messages[-1].content if messages else ""
        if not isinstance(prompt, str):
            prompt = str(prompt or "")
        text = fake_task_reply(prompt, self._gold_by_prompt)
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


def fake_transport_factory(*, gold_by_prompt: Mapping[str, str]):
    replies = dict(gold_by_prompt)

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
    "fake_gold_by_prompt",
    "fake_task_reply",
    "fake_transport_factory",
    "hardened_execution_policy",
    "openrouter_seeded_call_config",
    "openrouter_transport_factory",
    "resolve_provider_concurrency",
    "validate_provider_concurrency",
    "widened_execution_policy",
]
