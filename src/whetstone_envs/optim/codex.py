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
"""

from __future__ import annotations

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

    ``runtime_config`` is what that out-of-process server rebuilds its
    engine from, and ``engine`` is the in-process engine it must agree
    with -- see :mod:`whetstone_envs.optim.codex_runtime` for why envs
    ships its own rather than using whetstone-ai's toy-experiment one.
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


def codex_run_root(output_dir: Path) -> Path:
    """Where one Codex run's dr-exec job records live."""
    return output_dir / CODEX_RUN_ROOT_NAME


__all__ = [
    "CODEX_ADAPTER_KEY",
    "CODEX_DEFAULT_BINARY",
    "CODEX_EVALUATE_CALL_CAP",
    "CODEX_REASONING_EFFORTS",
    "CODEX_RUN_ROOT_NAME",
    "CodexReasoningEffort",
    "CodexTestSeam",
    "build_codex_adapter",
    "build_codex_control",
    "codex_run_root",
]
