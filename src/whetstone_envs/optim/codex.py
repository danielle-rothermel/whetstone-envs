"""The Codex-direct arm's control and adapter, built per family.

Codex is the study's foreign-agent arm: whetstone runs no search of its
own. The Codex CLI runs out of process under dr-exec containment, reaches
one MCP tool that evaluates a candidate on the internal split, and returns
the ``call_id`` of the call whose candidate it selected. So this module is
much smaller than :mod:`whetstone_envs.optim.gepa` -- there is no
proposer, no transport, and no prompt-services descriptor, because the
agent proposes for itself.

What it does own is the same family-driven binding every other builder
here owns: the render contract and mutation field come from the family's
:class:`~whetstone_envs.optim.families.FamilySpec`, the reward policy and
internal split come from the run's engine, and the arm's capacity cap
becomes the control's ``max_tool_calls``.

**Capacity is the eval budget.** ``max_tool_calls`` is simultaneously the
Step's ``tool_calls`` budget label and the per-run
``ToolCapacity.max_accepted_calls`` the evaluation server admits against.
The study's cap is :data:`CODEX_EVALUATE_CALL_CAP` (D2, 8 admitted
evaluate-calls per run), and it is the default here: whetstone-ai holds no
policy on how large an eval budget a caller should buy, so the number is
the study's to name.

**Preflight is mandatory and unforgeable.** ``prepare_codex_run`` requires
proof of a usable Codex session before it commits capacity, and it has no
default. :func:`build_codex_adapter` runs the real ``codex_auth_preflight``
against this run's executor and process environment before it returns an
adapter at all. The scripted stand-in lives in ``whetstone.testing`` and is
reachable only through :class:`CodexTestSeam`, which no production path
constructs.

**A real Codex session costs money, so it is opt-in.** The preflight
proves a session by *spawning the CLI*, which on an authenticated machine
is itself a billed call -- so "no seam" is not the same as "no spend".
:func:`refuse_unauthorized_real_codex` is the gate every Codex run passes
through before any preflight, adapter, admission, or subprocess exists: a
run either supplies a :class:`CodexTestSeam` or names the two-part opt-in
(:data:`ALLOW_REAL_CODEX_ENV` set to :data:`ALLOW_REAL_CODEX_ENV_VALUE` in
the process environment *and* ``RunSpec.allow_real_codex``), and anything
else raises :class:`RealCodexRefusedError`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY, CodexAdapter
from whetstone.optim.codex.control import (
    CODEX_DEFAULT_BINARY,
    CODEX_DEFAULT_WALL_SECONDS,
    CodexControl,
    CodexReasoningEffort,
    configure_codex,
)
from whetstone.optim.codex.executor import build_codex_executor
from whetstone.optim.codex.preflight import codex_auth_preflight
from whetstone.optim.codex.runner import SubprocessCodexRunner

from whetstone_envs.optim.codex_runtime import (
    ENVS_CODEX_RUNTIME_CONFIG_CLASS,
    EnvsCodexRuntimeConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine, EvalRuntimeConfig
    from whetstone.experiment.env import Experiment
    from whetstone.experiment.reward import RewardPolicy
    from whetstone.optim.codex.runner import CodexPromptContext

    from whetstone_envs.optim.families import FamilySpec

#: The Codex arm's admitted evaluate-call cap (D2). It is the per-run
#: ``ToolCapacity.max_accepted_calls`` and the Step's ``tool_calls``
#: budget at once, so one number bounds both what the agent may buy and
#: what the Step may debit.
#:
#: This is the same constant the study's spec module names; it is
#: re-exported here rather than imported, because ``optim.codex`` is
#: below ``optim.study`` in the dependency direction and the runner must
#: not import the study to run one arm.
CODEX_EVALUATE_CALL_CAP = 8

#: The Codex *agent's* own model. It is deliberately not the run's task
#: model: the two are different products on different routes, and the
#: task model this study evaluates (``openai/gpt-4.1-nano``, an
#: OpenRouter route) is not a model the Codex CLI can run at all -- a
#: ChatGPT subscription refuses it outright with
#: ``"The 'openai/gpt-4.1-nano' model is not supported when using Codex
#: with a ChatGPT account"``, before the agent produces a single token.
#:
#: Naming the CLI's own current default keeps a default run working while
#: staying explicit: ``CodexControl.model`` is identity-bearing and
#: refuses an empty string, so the arm cannot express "whatever the CLI
#: picks" by omission -- it has to say which model it measured. The §6
#: run pins its own through ``--codex-model``, and the manifest records
#: the agent model as uncontrolled either way (OQ1).
CODEX_DEFAULT_AGENT_MODEL = "gpt-5.6-sol"


def resolve_codex_agent_model(codex_model: str | None) -> str:
    """The model a Codex *agent* session runs, given an arm's override.

    The one owner of "which model does the Codex route use". Every caller
    that names a Codex session -- the runner that builds the control, and
    the study stage that preflights the arm before buying the arms ahead
    of it -- resolves it here, so a preflight cannot probe a different
    route than the run it is clearing.

    That drift is not hypothetical: the study preflight passed the run's
    ``task_model``, which is an OpenRouter route the Codex CLI cannot run
    at all. The probe therefore tested a route no arm would use, and a
    real study would clear preflight and then fail on the Codex arm's
    turn -- exactly the late failure the preflight exists to prevent.

    ``None`` means the arm did not override the agent model and takes
    :data:`CODEX_DEFAULT_AGENT_MODEL`. It never means the run's task
    model: the two are different products on different routes.
    """
    return codex_model or CODEX_DEFAULT_AGENT_MODEL


#: Where a Codex run's dr-exec job records live, beneath the run's own
#: output directory. One directory per run, so a completed run's spawn
#: evidence stays beside the artifacts it produced.
CODEX_RUN_ROOT_NAME = "codex-runs"

#: Every reasoning effort the Codex control accepts, as CLI choices. The
#: enum is the owner; this is its ordered projection for argparse.
CODEX_REASONING_EFFORTS: tuple[str, ...] = tuple(
    member.value for member in CodexReasoningEffort
)


@dataclass(frozen=True, slots=True)
class CodexTestSeam:
    """The only way to run the Codex arm without a real Codex session.

    A test drives the scripted fake CLI, which has no session to prove and
    no spend to protect, so it must be able to replace the preflight. But
    a replaceable preflight that any caller could pass is not a preflight:
    the whole point is that no budgeted run starts without a proven
    session.

    So the substitution is a named seam rather than an optional argument.
    :class:`~whetstone_envs.optim.run.RunSpec` has no field that carries
    one, the CLI has no flag that builds one, and
    :func:`build_codex_adapter` defaults it to ``None`` -- meaning a
    production path reaches the real ``codex_auth_preflight`` and cannot
    select anything else. Only a test that constructs this record by name
    gets the stand-in.

    ``preflight`` takes the same keyword arguments as
    ``codex_auth_preflight`` and is expected to ignore them.
    ``environment`` is merged into the run's process environment, which is
    how the fake CLI receives its transcript, and
    ``extra_environment_keys`` widens the runner's allowlist by exactly
    those keys and nothing more.
    """

    preflight: Callable[..., None]
    environment: dict[str, str]
    extra_environment_keys: frozenset[str] = frozenset()


#: The process-environment half of the real-Codex opt-in. It is an
#: environment variable rather than a flag alone because the flag lives in
#: a serializable spec: a study manifest, a saved arm, or a copied command
#: line can carry ``allow_real_codex=True`` into a machine that never
#: intended to spend, and an environment variable does not travel with it.
ALLOW_REAL_CODEX_ENV = "WHETSTONE_ENVS_ALLOW_REAL_CODEX"

#: The one value that opts in. Anything else -- unset, empty, ``"0"``,
#: ``"true"`` -- is not the opt-in, so a half-remembered spelling refuses
#: rather than spends.
ALLOW_REAL_CODEX_ENV_VALUE = "1"

#: The test-process tripwire: while this names any non-empty value, no
#: opt-in of any kind reaches a real Codex session.
#:
#: The opt-in above is *process* state, and a test can set process state.
#: ``monkeypatch.setenv`` is the ordinary way to prove a gate lifts, and a
#: test that lifts the authorization half in order to check the next
#: decision would -- without this -- also lift the last thing standing
#: between the suite and a billed session. That is not hypothetical: the
#: study harness's early guard runs a session probe once the opt-in is
#: satisfied, so an authorization test that supplied no seam spawned the
#: real CLI.
#:
#: So the suite sets this for the whole session and the gate honours it
#: above every other input. A test may monkeypatch the allow variable; it
#: cannot reach a real session while this one is present.
FORBID_REAL_CODEX_ENV = "WHETSTONE_ENVS_FORBID_REAL_CODEX"


class RealCodexRefusedError(RuntimeError):
    """A Codex run would have spawned the real, billed CLI.

    Raised before any preflight, adapter, admission authority, or
    subprocess exists, so a refused run has spent nothing and left no
    durable artifact behind.
    """


def refuse_unauthorized_real_codex(
    *, test_seam: CodexTestSeam | None, allow_real_codex: bool
) -> None:
    """Refuse a Codex run that is neither scripted nor deliberately paid.

    The preflight is not a spend guard. It proves a session by *spawning
    the Codex CLI*, and on a machine with ``~/.codex/auth.json`` that
    spawn succeeds -- and is billed. So "the caller passed no seam" used
    to mean "the run reaches the real CLI", which is exactly the accident
    this refuses: a suite, a study arm, or a parametrization that named
    the Codex optimizer without meaning to buy a session.

    A run is admitted on one of two grounds, and nothing else:

    * a :class:`CodexTestSeam`, which points the run at the scripted fake
      CLI and cannot be built from a spec or a flag; or
    * the deliberate opt-in, which is *both*
      :data:`ALLOW_REAL_CODEX_ENV` set to
      :data:`ALLOW_REAL_CODEX_ENV_VALUE` in this process's environment and
      ``allow_real_codex`` on the run's spec. Requiring both means neither
      a serialized spec nor an exported variable can authorize a paid run
      on its own.

    The two grounds are mutually exclusive in practice but not checked as
    such: a seam already means no real CLI is reachable, so an opt-in
    alongside one buys nothing.

    **The test-process tripwire outranks both.** When
    :data:`FORBID_REAL_CODEX_ENV` names any non-empty value, this refuses
    regardless of the opt-in halves and regardless of any ``monkeypatch``
    that set them: the suite sets it once per session, so a test may lift
    the authorization half to prove a gate lifts and still cannot reach a
    real session. A seam is unaffected, because a seamed run reaches no
    real CLI to forbid -- which is what keeps the scripted path, and the
    tests that drive it, fully exercisable under the tripwire.

    This is the single gate every production path to the real preflight or
    adapter passes through, including the study harness's early stage
    guard, so the tripwire cannot be routed around by reaching the
    preflight another way.
    """
    if test_seam is not None:
        return
    forbidden = os.environ.get(FORBID_REAL_CODEX_ENV)
    if forbidden:
        raise RealCodexRefusedError(
            "a codex run would spawn the real Codex CLI, and "
            f"{FORBID_REAL_CODEX_ENV}={forbidden!r} forbids it in this "
            "process. This is the test-process tripwire: the opt-in is "
            "process state a test can set, so it is not the last line of "
            "defence. Drive the scripted fake CLI through a CodexTestSeam."
        )
    if (
        allow_real_codex
        and os.environ.get(ALLOW_REAL_CODEX_ENV) == ALLOW_REAL_CODEX_ENV_VALUE
    ):
        return
    raise RealCodexRefusedError(
        "a codex run would spawn the real Codex CLI, which costs money: "
        "the authentication preflight is itself a billed session probe, so "
        "it is not a spend guard. Drive the scripted fake CLI through a "
        "CodexTestSeam, or opt in deliberately with both "
        f"{ALLOW_REAL_CODEX_ENV}={ALLOW_REAL_CODEX_ENV_VALUE} in the "
        "environment and --allow-real-codex (RunSpec.allow_real_codex)."
    )


def build_codex_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    family: FamilySpec,
    model: str,
    max_tool_calls: int | None = None,
    codex_binary: str = CODEX_DEFAULT_BINARY,
    reasoning_effort: CodexReasoningEffort = CodexReasoningEffort.MEDIUM,
    wall_seconds: float | None = None,
) -> CodexControl:
    """Resolve one family's Codex control over the engine's internal split.

    ``max_tool_calls`` defaults to :data:`CODEX_EVALUATE_CALL_CAP`, the
    arm's capacity cap. Every other binding is attested rather than
    assumed: the control pins the eval config, reward policy, execution
    policy, task model, and internal split it will be measured against,
    and ``prepare_codex_run`` re-checks each against the live engine, so a
    control that disagreed with the runtime is refused before any capacity
    is committed. Checking the split here as well keeps the failure at
    build time, where the message can say which side is wrong.
    """
    task_hashes = experiment.eval_configs.internal.task_set.task_hashes
    if engine.sampling.task_hashes != task_hashes:
        raise ValueError("Codex evaluation must be the internal eval split")
    return configure_codex(
        model=model,
        max_tool_calls=(
            CODEX_EVALUATE_CALL_CAP
            if max_tool_calls is None
            else max_tool_calls
        ),
        eval_config_ref=engine.eval_config_ref,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        evaluation_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        task_model_identity_hash=engine.task_model_identity_hash(),
        internal_task_hashes=task_hashes,
        reasoning_effort=reasoning_effort,
        codex_binary=codex_binary,
        mutation_field=family.mutation_field,
        wall_seconds=(
            CODEX_DEFAULT_WALL_SECONDS
            if wall_seconds is None
            else wall_seconds
        ),
    )


def build_codex_adapter(  # noqa: PLR0913
    *,
    store: ObjectStore,
    control: CodexControl,
    engine: EvalEngine,
    runtime_config: EnvsCodexRuntimeConfig,
    reward_policy: RewardPolicy,
    store_path: Path,
    run_root: Path,
    test_seam: CodexTestSeam | None = None,
    prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> CodexAdapter:
    """Assemble one family's Codex adapter, session proven first.

    The adapter is not returned until the preflight has proven the binary,
    an auth source, and one cheap structured probe. ``test_seam`` is the
    only way to substitute that check, and no production caller
    constructs one -- see :class:`CodexTestSeam`.

    The returned adapter still awaits its Tool Call Store: bind the
    runtime's exact store with ``bind_tool_store`` once ``build_runtime``
    has produced it, because the adapter reads durable ledger entries from
    the store the harness admits through.

    ``store_path`` must be the run's own sqlite, because the MCP
    evaluation server runs in another process and persists its Tool
    Results there, and ``reward_policy`` is the policy that server scores
    with -- the same one the control pins by hash.

    ``prompt_builder`` replaces the instruction the agent receives. It is
    a *diagnostic* seam, not a production one: no ``RunSpec`` field and no
    CLI flag reaches it, and the only caller is the real-CLI ladder, whose
    capacity and no-tool-call rungs cannot observe what they assert under
    the truthful production prompt -- an agent correctly told it may make
    one call makes one call, and the durable refusal path is never
    exercised. A builder inherits the default's obligations: see
    ``CodexPromptContext``, which carries the ``model_route`` and
    ``base_ref`` the agent can derive from nothing it can see.

    ``runtime_config`` is what that out-of-process server rebuilds its
    engine from, and ``engine`` is the in-process engine it must agree
    with -- see :mod:`whetstone_envs.optim.codex_runtime` for why envs
    ships its own rather than using whetstone-ai's toy-experiment one.
    Three things are asserted equal across the two engines before
    anything is spawned, and each is a distinct failure the run would
    otherwise absorb silently: ``eval_config_ref`` (a mismatch refuses
    every tool call), ``task_model_identity_hash()`` (a mismatch measures
    another model route, which the Eval Config does not pin on the
    openrouter transport), and ``sampling.task_hashes`` (a mismatch
    measures another task set).
    """
    # The MCP evaluation server rebuilds its engine from this config
    # alone, in another process. Proving here that the rebuild lands on
    # the same Eval Config is what keeps a mismatch from becoming a run
    # in which every single tool call is refused and the agent has
    # nothing to select -- the failure mode is silent on the server side,
    # so it is caught on this side before anything is spawned.
    rebuilt = runtime_config.build_engine(store)
    if rebuilt.eval_config_ref != engine.eval_config_ref:
        raise ValueError(
            "the Codex MCP runtime config rebuilds a different Eval "
            "Config than the run's engine, so every tool call it admits "
            "would be refused"
        )
    # The Eval Config alone does not pin the model route. On the
    # openrouter transport the task model is carried by the provider call
    # config, so a runtime config naming a different model can rebuild to
    # the *same* ``eval_config_ref`` and still evaluate on another route --
    # the run would complete, report a coherent trajectory, and have
    # measured a model the study never asked for. Checking the task-model
    # identity closes that; checking the task hashes states the split
    # agreement here rather than leaving it to the control builder alone.
    if rebuilt.task_model_identity_hash() != engine.task_model_identity_hash():
        raise ValueError(
            "the Codex MCP runtime config rebuilds a different task model "
            "than the run's engine, so the agent would be measured on a "
            "route this run did not ask for"
        )
    if rebuilt.sampling.task_hashes != engine.sampling.task_hashes:
        raise ValueError(
            "the Codex MCP runtime config rebuilds a different task set "
            "than the run's engine, so the agent would be measured on "
            "tasks this run did not ask for"
        )
    executor = build_codex_executor(run_root=run_root)
    environment = dict(test_seam.environment) if test_seam else None
    runner = SubprocessCodexRunner(
        executor=executor,
        sqlite_path=str(store_path.resolve()),
        # ``EnvsCodexRuntimeConfig`` satisfies ``EvalRuntimeConfig``
        # behaviorally -- the server round-trips it through
        # ``load_runtime_config`` and calls ``build_engine`` -- but not
        # structurally: pydantic spells ``model_validate_json``'s
        # parameter ``json_data`` while the protocol spells it ``data``,
        # so no pydantic model can satisfy the protocol by name. The
        # cast records that, and the e2e proves the behavior.
        runtime_config=cast("EvalRuntimeConfig", runtime_config),
        runtime_config_class=ENVS_CODEX_RUNTIME_CONFIG_CLASS,
        reward_policy=reward_policy,
        codex_binary=control.codex_binary,
        model=control.model,
        reasoning_effort=control.reasoning_effort.value,
        timeout_seconds=control.wall_seconds,
        max_output_bytes=control.max_output_bytes,
        environment=environment,
        extra_environment_keys=(
            test_seam.extra_environment_keys if test_seam else frozenset()
        ),
        prompt_builder=prompt_builder,
    )
    preflight = test_seam.preflight if test_seam else codex_auth_preflight
    # Proven before the adapter exists, so a broken login cannot reach the
    # harness and start spending the run's capacity.
    preflight(
        executor=executor,
        codex_binary=control.codex_binary,
        environment=runner.codex_process_environment(),
        model=control.model,
    )
    return CodexAdapter(runner, store=store)


def preflight_codex_session(
    *,
    scratch_root: Path,
    codex_binary: str = CODEX_DEFAULT_BINARY,
    model: str = "",
    allow_real_codex: bool = False,
    test_seam: CodexTestSeam | None = None,
) -> None:
    """Prove a usable Codex session ahead of a stage that will need one.

    This is :func:`build_codex_adapter`'s own preflight, reachable without
    an engine, a store, or a control. A study stage whose design names the
    Codex arm needs the answer *before* it starts paying for the arms
    ordered ahead of it: an unsupported platform, a missing binary, or an
    expired session is otherwise discovered on the Codex arm's turn, after
    COPRO, MIPROv2, and GEPA have already been bought.

    It runs the same ``codex_auth_preflight`` through the same executor and
    the same containment path a real run would, so a session it admits is
    one the run can use, and ``test_seam`` substitutes it exactly as it
    does for the adapter -- no production caller can construct one.

    The probe is itself a billed session probe, so this routes through
    :func:`refuse_unauthorized_real_codex` before building anything --
    that gate is where the two-part opt-in and the test-process tripwire
    live, and reaching a real probe around it would be exactly the
    bypass the tripwire exists to prevent. ``allow_real_codex`` is the
    caller's authorization; an unauthorized or forbidden call raises
    :class:`RealCodexRefusedError` here, and an authorized one raises
    whatever the preflight raises, which for the real one is
    ``CodexPreflightError``.
    """
    refuse_unauthorized_real_codex(
        test_seam=test_seam, allow_real_codex=allow_real_codex
    )
    executor = build_codex_executor(run_root=scratch_root)
    runner = SubprocessCodexRunner(
        executor=executor,
        codex_binary=codex_binary,
        model=model,
        environment=dict(test_seam.environment) if test_seam else None,
        extra_environment_keys=(
            test_seam.extra_environment_keys if test_seam else frozenset()
        ),
    )
    preflight = test_seam.preflight if test_seam else codex_auth_preflight
    preflight(
        executor=executor,
        codex_binary=codex_binary,
        environment=runner.codex_process_environment(),
        model=model,
    )


def codex_run_root(output_dir: Path) -> Path:
    """Where one Codex run's dr-exec job records live."""
    return output_dir / CODEX_RUN_ROOT_NAME


__all__ = [
    "ALLOW_REAL_CODEX_ENV",
    "ALLOW_REAL_CODEX_ENV_VALUE",
    "CODEX_ADAPTER_KEY",
    "CODEX_DEFAULT_AGENT_MODEL",
    "CODEX_DEFAULT_BINARY",
    "CODEX_EVALUATE_CALL_CAP",
    "CODEX_REASONING_EFFORTS",
    "CODEX_RUN_ROOT_NAME",
    "FORBID_REAL_CODEX_ENV",
    "CodexReasoningEffort",
    "CodexTestSeam",
    "RealCodexRefusedError",
    "build_codex_adapter",
    "build_codex_control",
    "codex_run_root",
    "preflight_codex_session",
    "refuse_unauthorized_real_codex",
    "resolve_codex_agent_model",
]
