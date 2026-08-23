from __future__ import annotations

import pytest

pytest.importorskip("dr_providers")

from dr_providers import RequestControl

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    MAX_UNFORCED_PROVIDER_CONCURRENCY,
    bind_openrouter_transport,
    fake_task_reply,
    openrouter_seeded_call_config,
    resolve_provider_concurrency,
    widened_execution_policy,
)


def test_openrouter_preset_advertises_seed() -> None:
    config = openrouter_seeded_call_config(model="openai/gpt-4.1-nano")
    assert config.definition.constraints.supports(RequestControl.SEED)


def test_fake_transport_emits_gold_for_ceiling_prompt() -> None:
    pytest.importorskip("whetstone.experiment.env")
    experiment = prepare_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(2, 2, 0),
        num_seeds=1,
    ).experiment
    task = experiment.eval_configs.internal.tasks[0]
    gold = getattr(task, "gold", None)
    inputs = getattr(task, "prompt_inputs", None)
    assert isinstance(gold, str)
    assert isinstance(inputs, dict)
    contract = c19_render_contract()
    ceiling = contract.render(PROBES.ceiling_template, inputs)
    naive = contract.render(PROBES.naive_template, inputs)
    gold_by_prompt = {ceiling: gold}

    assert fake_task_reply(ceiling, gold_by_prompt) == gold
    assert fake_task_reply(naive, gold_by_prompt) == f"generated: {naive}"


def test_bind_openrouter_transport_reuses_one_client() -> None:
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    transport, factory = bind_openrouter_transport(policy)
    assert factory(policy) is transport


def test_widened_policy_raises_the_connection_pool_to_the_width() -> None:
    """The client's pool is the other bound, and the smaller one wins.

    Fails-before: nothing widened the policy, so whetstone's
    ``default_transport_policy`` fixed the client at ten connections and a
    stage asked to run 32 rows at once would have had 22 workers queued on
    sockets -- recording a width it never actually ran at.
    """
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    assert policy.transport_policy.max_connections == 10

    widened = widened_execution_policy(policy, concurrency=32)
    assert widened.transport_policy.max_connections == 32
    # Raised with the ceiling: these are many short requests to one host,
    # so discarding connections would pay a TLS handshake per call.
    assert widened.transport_policy.max_keepalive_connections == 32
    # Nothing else about the policy moves.
    assert widened.max_attempts == policy.max_attempts
    assert (
        widened.transport_policy.timeout_seconds
        == policy.transport_policy.timeout_seconds
    )


def test_widened_policy_leaves_a_already_sufficient_pool_alone() -> None:
    """A width the pool already holds is not a reason to rewrite identity."""
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    assert widened_execution_policy(policy, concurrency=4) is policy


def test_widened_policy_refuses_a_width_below_one() -> None:
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    with pytest.raises(ValueError, match="at least 1"):
        widened_execution_policy(policy, concurrency=0)


def test_recorded_default_matches_whetstones_own() -> None:
    """The literal this package persists tracks the dependency knowingly.

    ``DEFAULT_PROVIDER_CONCURRENCY`` is stored identity -- a stage record
    written with no flag means "whetstone's default at the time" -- so it
    is a literal rather than an import. This is the test that turns a
    dependency bump into a visible decision instead of a silent
    reinterpretation of every historical record.
    """
    pytest.importorskip("whetstone.eval.runtime_engine")
    from whetstone.eval.runtime_engine import DEFAULT_CONCURRENCY

    assert DEFAULT_PROVIDER_CONCURRENCY == DEFAULT_CONCURRENCY


def test_resolve_refuses_above_the_cap_unless_forced() -> None:
    over = MAX_UNFORCED_PROVIDER_CONCURRENCY + 1
    with pytest.raises(ValueError, match="sanity cap"):
        resolve_provider_concurrency(over, force=False)
    assert resolve_provider_concurrency(over, force=True) == over
    # The lower bound is arithmetic and cannot be forced away.
    with pytest.raises(ValueError, match="at least 1"):
        resolve_provider_concurrency(0, force=True)


# --------------------------------------------------------------------------
# Retrying transport
# --------------------------------------------------------------------------


class _StubEvidence:
    """One transport outcome, shaped only where the retry logic reads it."""

    def __init__(
        self,
        *,
        recoverability: object | None = None,
        retry_after: object | None = None,
    ) -> None:
        self.failure = (
            None
            if recoverability is None
            else type("_F", (), {"recoverability": recoverability})()
        )
        self.retry_after = retry_after


class _Hint:
    def __init__(self, kind: str, value: object) -> None:
        self.kind = kind
        self.value = value


def _scripted(*outcomes: object):
    """A transport returning each outcome in turn, recording its calls."""
    calls: list[object] = []
    remaining = list(outcomes)

    def transport(request: object):
        calls.append(request)
        return remaining.pop(0) if remaining else outcomes[-1]

    return transport, calls


def test_a_rate_limited_call_is_retried_after_waiting() -> None:
    """429 then 200 returns the 200, and the wait actually happened.

    Fails-before: this is the failure that aborted a paid Stage 0. The
    execution policy already classified a rate limit as retryable and
    already computed a backoff, but the eval path never injects a
    ``sleep`` -- ``GraphRolloutEvalDriver`` builds its ``LlmCallContext``
    without one -- so ``run_provider_call`` fell back to ``_no_sleep``
    and every attempt fired within microseconds. Three instant retries
    against a live rate limit are one retry.
    """
    from dr_providers import RecoverabilityClass

    from whetstone_envs.optim.provider import RetryingTransport

    ok = _StubEvidence()
    limited = _StubEvidence(recoverability=RecoverabilityClass.RATE_LIMITED)
    inner, calls = _scripted(limited, ok)
    slept: list[float] = []

    transport = RetryingTransport(inner, sleep=slept.append)
    assert transport("req") is ok
    assert len(calls) == 2
    # The wait is the whole point: one sleep, and a real one.
    assert len(slept) == 1
    assert slept[0] > 0


def test_retries_stop_at_the_attempt_budget() -> None:
    """A limit that never clears is bounded, not retried forever."""
    from dr_providers import RecoverabilityClass

    from whetstone_envs.optim.provider import RetryingTransport

    limited = _StubEvidence(recoverability=RecoverabilityClass.RATE_LIMITED)
    inner, calls = _scripted(limited)
    slept: list[float] = []

    transport = RetryingTransport(inner, max_attempts=4, sleep=slept.append)
    assert transport("req") is limited
    assert len(calls) == 4
    assert len(slept) == 3


def test_a_permanent_rejection_is_not_retried() -> None:
    """Re-sending a refused request buys the same refusal and bills for it."""
    from dr_providers import RecoverabilityClass

    from whetstone_envs.optim.provider import RetryingTransport

    refused = _StubEvidence(recoverability=RecoverabilityClass.PERMANENT)
    inner, calls = _scripted(refused)
    slept: list[float] = []

    transport = RetryingTransport(inner, sleep=slept.append)
    assert transport("req") is refused
    assert len(calls) == 1
    assert slept == []


def test_a_successful_call_is_not_retried() -> None:
    from whetstone_envs.optim.provider import RetryingTransport

    ok = _StubEvidence()
    inner, calls = _scripted(ok)
    transport = RetryingTransport(inner, sleep=lambda _: None)
    assert transport("req") is ok
    assert len(calls) == 1


def test_the_providers_retry_after_wins_over_the_schedule() -> None:
    """The provider knows when it will accept traffic; the schedule guesses."""
    from dr_providers import RecoverabilityClass

    from whetstone_envs.optim.provider import RetryingTransport

    limited = _StubEvidence(
        recoverability=RecoverabilityClass.RATE_LIMITED,
        retry_after=_Hint("delta_seconds", 7),
    )
    inner, _ = _scripted(limited, _StubEvidence())
    slept: list[float] = []

    RetryingTransport(inner, sleep=slept.append)("req")
    assert slept == [7.0]


def test_an_absurd_retry_after_is_bounded() -> None:
    """A header naming an hour must not park a worker for an hour."""
    from dr_providers import RecoverabilityClass

    from whetstone_envs.optim.provider import (
        MAX_HONOURED_RETRY_AFTER_SECONDS,
        RetryingTransport,
    )

    limited = _StubEvidence(
        recoverability=RecoverabilityClass.RATE_LIMITED,
        retry_after=_Hint("delta_seconds", 3600),
    )
    inner, _ = _scripted(limited, _StubEvidence())
    slept: list[float] = []

    RetryingTransport(inner, sleep=slept.append)("req")
    assert slept == [MAX_HONOURED_RETRY_AFTER_SECONDS]


def test_the_backoff_grows_and_is_capped_and_jittered() -> None:
    from whetstone_envs.optim.provider import (
        RETRY_JITTER_FRACTION,
        RETRY_MAX_SECONDS,
        RetryingTransport,
    )

    transport = RetryingTransport(lambda _: None)
    previous = 0.0
    for attempt in range(1, 8):
        delay = transport.delay_for(attempt)
        assert 0 < delay <= RETRY_MAX_SECONDS
        # Every delay sits inside its own jitter window.
        nominal = min(2.0 * 2.0 ** (attempt - 1), RETRY_MAX_SECONDS)
        assert nominal * (1.0 - RETRY_JITTER_FRACTION) <= delay <= nominal
        previous = delay
    assert previous <= RETRY_MAX_SECONDS


def test_the_hardened_policy_covers_a_reasoning_sized_call() -> None:
    """30 s is a chat-completion bound and gpt-5-nano is not one.

    Fails-before: the bound stage used whetstone's 30 s timeout, which the
    live Stage 0's own token counts -- median 4,466 completion tokens,
    maximum 12,335 -- routinely outrun, so an ordinary slow call became a
    billed timeout.
    """
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.provider import (
        TASK_CALL_MAX_ATTEMPTS,
        TASK_CALL_TIMEOUT_SECONDS,
        hardened_execution_policy,
    )

    base = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    assert base.transport_policy.timeout_seconds == 30.0
    assert base.max_attempts == 3

    hardened = hardened_execution_policy(base)
    assert hardened.transport_policy.timeout_seconds == (
        TASK_CALL_TIMEOUT_SECONDS
    )
    assert hardened.transport_policy.idle_timeout_seconds == (
        TASK_CALL_TIMEOUT_SECONDS
    )
    assert hardened.max_attempts == TASK_CALL_MAX_ATTEMPTS
    # Eligibility is whetstone's and is deliberately left alone.
    assert hardened.retry_eligibility == base.retry_eligibility


def test_the_bound_openrouter_transport_retries() -> None:
    """The one place a paid client is built wraps it, so none can skip it."""
    pytest.importorskip("whetstone.eval.reference_runtime")
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.provider import RetryingTransport

    policy = ReferenceEvalRuntimeConfig(
        transport_api_key_env="OPENROUTER_API_KEY",
    ).execution_policy
    transport, factory = bind_openrouter_transport(policy)
    assert isinstance(transport, RetryingTransport)
    assert factory(policy) is transport
