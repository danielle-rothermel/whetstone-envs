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

from whetstone_envs.optim._bindings import CodexBinding
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

    from whetstone_envs.optim.problem import OptimizationProblem
    from whetstone_envs.optim.spec import CodexSettings

CODEX_EVALUATE_CALL_CAP = 8

CODEX_DEFAULT_AGENT_MODEL = "gpt-5.6-sol"


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
    """Test-only preflight and environment for a scripted Codex CLI.

    Neither RunSpec nor the CLI can select this seam.
    """

    preflight: Callable[..., None]
    environment: dict[str, str]
    extra_environment_keys: frozenset[str] = frozenset()


ALLOW_REAL_CODEX_ENV = "WHETSTONE_ENVS_ALLOW_REAL_CODEX"

#: The one value that opts in. Anything else -- unset, empty, ``"0"``,
#: ``"true"`` -- is not the opt-in, so a half-remembered spelling refuses
#: rather than spends.
ALLOW_REAL_CODEX_ENV_VALUE = "1"

FORBID_REAL_CODEX_ENV = "WHETSTONE_ENVS_FORBID_REAL_CODEX"


class RealCodexRefusedError(RuntimeError):
    pass


def refuse_unauthorized_real_codex(
    *, test_seam: CodexTestSeam | None, allow_real_codex: bool
) -> None:
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
        "environment and --allow-real-codex (CodexSettings.allow_real_codex)."
    )


def build_codex_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    problem: OptimizationProblem,
    model: str,
    max_tool_calls: int | None = None,
    codex_binary: str = CODEX_DEFAULT_BINARY,
    reasoning_effort: CodexReasoningEffort = CodexReasoningEffort.MEDIUM,
    wall_seconds: float | None = None,
) -> CodexControl:
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
        mutation_field=problem.contract.mutation_field,
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
    # Reject a mismatched subprocess runtime before spawning the CLI.
    rebuilt = runtime_config.build_engine(store)
    if rebuilt.eval_config_ref != engine.eval_config_ref:
        raise ValueError(
            "the Codex MCP runtime config rebuilds a different Eval "
            "Config than the run's engine, so every tool call it admits "
            "would be refused"
        )
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
        # cast records that protocol signature mismatch.
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
    "refuse_unauthorized_real_codex",
]


def bind_codex(  # noqa: PLR0913
    *,
    settings: CodexSettings,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    problem: OptimizationProblem,
    runtime_config: EnvsCodexRuntimeConfig,
    output_dir: Path,
    sqlite_path: Path,
    test_seam: CodexTestSeam | None = None,
    prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> CodexBinding:
    control = build_codex_control(
        engine=engine,
        experiment=experiment,
        problem=problem,
        model=settings.model or CODEX_DEFAULT_AGENT_MODEL,
        max_tool_calls=settings.capacity,
        codex_binary=settings.binary,
        reasoning_effort=settings.reasoning_effort,
        wall_seconds=settings.wall_seconds,
    )
    adapter = build_codex_adapter(
        store=store,
        control=control,
        engine=engine,
        runtime_config=runtime_config,
        reward_policy=experiment.reward_policy,
        store_path=sqlite_path,
        run_root=output_dir / CODEX_RUN_ROOT_NAME,
        test_seam=test_seam,
        prompt_builder=prompt_builder,
    )
    return CodexBinding(adapter=adapter, control=control)
