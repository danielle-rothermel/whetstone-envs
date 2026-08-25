from __future__ import annotations

import random
import time
from threading import Lock
from typing import TYPE_CHECKING, Any

from dr_providers import (
    GenerationControls,
    HttpProvider,
    ProviderCallRequest,
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


#: How long a task call may take, in seconds.
#:
#: whetstone's ``default_transport_policy`` allows 30 s, which is a fine
#: bound for a chat completion and far too short for a reasoning model.
#: The live Stage 0 measured a median of 4,466 completion tokens and a
#: maximum of 12,335 on ``gpt-5-nano``; at that size a call routinely runs
#: past 30 s, and a timeout is charged for the tokens already generated
#: before it is retried. This bound is set from that measurement with room
#: above the observed maximum, so the timeout catches a genuinely stuck
#: call rather than a merely slow one.
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

#: How many times whetstone's driver may attempt one call: exactly once.
#:
#: **Retries have exactly one owner.** ``whetstone.provider.driver`` loops
#: over ``policy.max_attempts`` and re-invokes the transport on a
#: retryable failure; :class:`RetryingTransport` *also* loops. Left at
#: five apiece the two compose multiplicatively rather than additively: a
#: row against a rate limit that never clears makes 5x5 = 25 billed
#: invocations, sleeping the wrapper's full schedule five times over
#: (~4 minutes of backoff per driver attempt, and up to ~40 minutes of
#: wall clock for a single row once the provider's own ``Retry-After``
#: hints are honoured).
#:
#: It also corrupts the record. The driver appends one
#: ``ProviderCallAttempt`` per *driver* iteration, so a row that really
#: cost 25 invocations persists five attempts: the ledger under-counts
#: billed calls by exactly the wrapper's factor, which is the number an
#: operator would use to reconcile spend.
#:
#: The wrapper keeps the budget rather than the driver because the
#: wrapper is the layer that actually waits. The driver's own backoff is
#: applied through an injected ``sleep`` that the eval path never
#: supplies (``GraphRolloutEvalDriver`` builds its ``LlmCallContext``
#: without one), so driver-owned retries fire within microseconds and are
#: no retry at all against a live rate limit. Setting this to one makes
#: the driver a pass-through and leaves the single waiting loop in
#: charge.
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
    """Wait between attempts, which whetstone's own retry loop does not.

    **This is why one 429 aborted a paid Stage 0.**
    ``ProviderExecutionPolicy`` already classifies a rate limit as
    retryable and already computes a backoff delay, and
    ``whetstone.provider.driver`` already loops over ``max_attempts``. But
    the delay is applied through an injected ``sleep``, and the eval path
    never injects one: ``GraphRolloutEvalDriver`` builds its
    ``LlmCallContext`` without a ``sleep``, so ``run_provider_call`` falls
    back to ``_no_sleep`` and all three attempts are made within
    microseconds of each other. Against a rate limit that is still
    firing, three instant retries are one retry.

    Waiting here rather than there is deliberate. The sleep seam exists
    upstream but is not reachable from anything this package binds -- the
    driver takes no ``sleep`` argument to forward -- so the wait is
    applied at the transport, which is the seam this package *does* own.

    **This wrapper is therefore the sole owner of the retry budget.**
    Because it owns the waiting it must also own the counting: two
    loops over the same failure multiply rather than compose, since this
    wrapper exhausts its attempts and then *returns* the transient
    failure, which the driver reads as one failed attempt and retries in
    full. The policy the driver runs under is pinned to
    :data:`DRIVER_MAX_ATTEMPTS` (one) by
    :func:`hardened_execution_policy` so that outer loop never turns over
    -- see that constant for what the multiplication costs in spend, wall
    clock, and ledger accuracy.
    """

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
        self._attempts = 0
        self._transient_outcomes: list[str] = []
        self._lock = Lock()

    @property
    def attempts(self) -> int:
        """Provider invocations this wrapper has made, across all calls.

        **Not the same number as ``calls``, and deliberately so.** ``calls``
        counts persisted output rows -- one per logical call, which is what
        the study is billed a completion for and what every cost projection
        re-derives from the store. A retried call still persists one row,
        so a report that showed only ``calls`` said nothing about the 429s
        it survived: the wrapper's retries were invisible in the record
        even though each one was a real request to the provider.

        This is a live counter rather than a projected number because it
        is the one quantity that *cannot* be read back: a transient
        attempt leaves no row to re-derive it from. It is therefore
        reported beside ``calls`` rather than folded into it, and nothing
        here changes what ``calls`` means.
        """
        with self._lock:
            return self._attempts

    @property
    def transient_outcomes(self) -> tuple[str, ...]:
        """Each transient failure this wrapper retried past, in order.

        The recoverability class rather than a status code, because that
        is what the retry decision was actually made on -- see
        :func:`_is_transient` -- so a stage report explains its own
        attempt count in the same terms the wrapper used to produce it.
        """
        with self._lock:
            return tuple(self._transient_outcomes)

    def delay_for(self, attempt_number: int) -> float:
        """The jittered wait before ``attempt_number``'s retry."""
        if attempt_number < 1:
            raise ValueError("attempt_number is at least 1")
        raw = RETRY_BASE_SECONDS * RETRY_MULTIPLIER ** (attempt_number - 1)
        capped = min(raw, RETRY_MAX_SECONDS)
        low = capped * (1.0 - RETRY_JITTER_FRACTION)
        return self._rng.uniform(low, capped)

    def _count(self, evidence: object, *, transient: bool) -> None:
        """Record one provider invocation, and why it was retried.

        Under a lock because one wrapper is shared by every worker in the
        stage -- ``bind_openrouter_transport`` returns a single instance so
        the pool is shared -- and a counter incremented from N concurrent
        evaluations would otherwise lose increments to the read-modify-write
        race, under-reporting exactly the attempt storm it exists to show.
        """
        with self._lock:
            self._attempts += 1
            if transient:
                failure = getattr(evidence, "failure", None)
                recoverability = getattr(failure, "recoverability", None)
                self._transient_outcomes.append(
                    getattr(recoverability, "value", str(recoverability))
                )

    def __call__(self, request: object):
        evidence = self._inner(request)
        for attempt_number in range(1, self._max_attempts):
            if not _is_transient(evidence):
                self._count(evidence, transient=False)
                return evidence
            self._count(evidence, transient=True)
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
        # The budget is spent. This last outcome is returned whatever it
        # is, so it is counted here rather than in the loop -- which would
        # have counted it only on the paths that retried past it.
        self._count(evidence, transient=False)
        return evidence


def widened_execution_policy(
    policy: ProviderExecutionPolicy, *, concurrency: int
) -> ProviderExecutionPolicy:
    """Return ``policy`` with a connection pool that can hold ``concurrency``.

    The engine's worker pool and the HTTP client's connection pool are two
    separate bounds, and the smaller one wins. whetstone's
    ``default_transport_policy`` fixes the client at ten connections, so
    raising the worker count alone would leave workers queued on sockets
    rather than talking to the provider -- the requested width would be
    recorded and not run.

    Keepalive is raised with the ceiling rather than left at its default:
    these are many short requests to one host, so a connection returned to
    the pool and immediately discarded would pay a fresh TLS handshake per
    call, which is the cost the pool exists to avoid.

    This changes the policy's ``identity_hash``, which is correct and
    harmless: that hash identifies the transport configuration a call was
    made under, and this *is* a different transport configuration. It is
    not an input to the pre-registration design hash, which covers the
    design fields alone -- see
    :func:`~whetstone_envs.optim.study.manifest.
    pre_registration_design_hash`.
    """
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
    """Return the OpenRouter chat preset, which advertises SEED.

    ``reasoning_effort`` pins the route's reasoning budget. The OpenRouter
    chat preset declares ``ReasoningRequestShape.REASONING_OBJECT``, so a
    pinned effort reaches the wire as ``{"reasoning": {"effort": ...}}`` in
    the request body; ``None`` sends no reasoning key at all and leaves the
    route on the provider's default, which is what every call made before
    the pin existed did.

    The effort is a *design* value, not an invocation setting: it changes
    the task model's capability and therefore the thing a study measures.
    It is passed in rather than defaulted here so the one caller that must
    not receive it -- the proposer route -- cannot acquire it by accident.
    """
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
    """Give ``policy`` a reasoning-sized timeout and enough attempts.

    Two changes, both measured against the live Stage 0 rather than
    guessed:

    * ``timeout_seconds`` rises from whetstone's 30 s to
      :data:`TASK_CALL_TIMEOUT_SECONDS`, because ``gpt-5-nano`` spends
      thousands of reasoning tokens per call and a 30 s bound turns an
      ordinary slow call into a timeout that is billed and then retried.
    * ``max_attempts`` *falls* to :data:`DRIVER_MAX_ATTEMPTS` -- one --
      because the retry budget belongs to :class:`RetryingTransport`,
      which is the layer that actually waits between attempts. The five
      attempts a rate limit needs to clear are spent there, once, rather
      than five times over by an outer loop that would multiply both the
      spend and the wall clock while recording neither. See
      :data:`DRIVER_MAX_ATTEMPTS`.

    The retry *eligibility* is left exactly as whetstone sets it: rate
    limits, transport errors, and timeouts are already retryable, and
    provider rejections and malformed responses already are not, which is
    the right split -- re-sending a refused request buys the same refusal.
    It still matters with a single driver attempt, because the wrapper
    reads the same classification off the transport's own
    ``RecoverabilityClass`` and the two must agree about what is worth
    re-sending.

    Like :func:`widened_execution_policy` this changes the policy's
    ``identity_hash`` and not the pre-registration design hash. Both are
    invocation properties: they change how a call is made, never what is
    being measured.
    """
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
    """Return one retrying transport and a factory that always yields it.

    The transport is wrapped in :class:`RetryingTransport` here, at the
    one place a live provider client is constructed, so no caller can
    bind a paid route that retries a 429 without waiting. The wrapper is
    what the whole study talks through: eval and proposal share it, which
    is also what keeps one connection pool rather than one per binding.
    """
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
        self, request: ProviderCallRequest
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
