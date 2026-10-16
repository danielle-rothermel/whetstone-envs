from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import ProviderKind
from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    copro_run_request,
    prepare_codex_run,
    prepare_copro_run,
    prepare_gepa_run,
    prepare_miprov2_run,
)
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import ToolAdmissionAuthority

from whetstone_envs.artifacts import (
    default_output_dir,
    durable_run_boundary,
    prepare_output_root,
)
from whetstone_envs.evidence import publish_trajectory_report
from whetstone_envs.optim.codex import (
    CodexTestSeam,
    refuse_unauthorized_real_codex,
)
from whetstone_envs.optim.extraction.trajectory import (
    project_trajectory_report,
)
from whetstone_envs.optim.families import (
    FamilySpec,
    family_spec,
)
from whetstone_envs.optim.miprov2 import (
    build_miprov2_state,
    miprov2_run_ref,
)
from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    widened_execution_policy,
)
from whetstone_envs.optim.spec import (
    CodexSettings,
    CoproSettings,
    GepaSettings,
    RunSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dr_store import ObjectStore
    from dr_store.sync import BlockingObjectStore
    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.core.identity import TypedRef
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.codex.runner import CodexPromptContext


from whetstone_envs.optim._bindings import (
    CodexBinding,
    GepaBinding,
    Miprov2Binding,
    OptimizerBinding,
)
from whetstone_envs.optim.codex import bind_codex
from whetstone_envs.optim.codex_runtime import runtime_config_for_run
from whetstone_envs.optim.copro import bind_copro
from whetstone_envs.optim.experiment import PreparedExperiment
from whetstone_envs.optim.gepa import bind_gepa
from whetstone_envs.optim.miprov2 import bind_miprov2
from whetstone_envs.optim.proposals import ProposalRuntime, prepare_proposer


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    spec: RunSpec
    family: FamilySpec
    prepared: PreparedExperiment
    n_per_category: int
    pool_seed_start: int
    run_id: str


@dataclass(frozen=True, slots=True)
class _RunEvaluation:
    engine: RuntimeEvalEngine
    proposer: ProposalRuntime


def _prepare_run(spec: RunSpec) -> _PreparedRun:
    family = family_spec(spec.family)
    n_per_category = (
        family.default_n_per_category
        if spec.n_per_category is None
        else spec.n_per_category
    )
    pool_seed_start = (
        family.default_pool_seed_start
        if spec.pool_seed_start is None
        else spec.pool_seed_start
    )
    resolved_run_id = spec.run_id or (
        f"{family.run_id_prefix}-{spec.optimizer.kind}-{uuid4().hex[:8]}"
    )
    provider = None
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(
            model=spec.model, reasoning_effort=spec.task_reasoning_effort
        )
    pool = family.generate_pool(
        n_per_category=n_per_category,
        seed_start=pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=spec.num_seeds,
        provider_call_config=provider,
    )
    return _PreparedRun(
        spec=spec,
        family=family,
        prepared=prepared,
        n_per_category=n_per_category,
        pool_seed_start=pool_seed_start,
        run_id=resolved_run_id,
    )


def _build_evaluation(run: _PreparedRun, store: ObjectStore) -> _RunEvaluation:
    spec = run.spec
    family = run.family
    experiment = run.prepared.experiment
    api_key_env = (
        "OPENROUTER_API_KEY"
        if spec.transport == "openrouter"
        else "WHETSTONE_TOY_API_KEY"
    )
    if spec.transport == "openrouter":
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
            provider_kind=ProviderKind.OPENROUTER,
        )
    else:
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
        )
    execution_policy = widened_execution_policy(
        runtime_config.execution_policy,
        concurrency=spec.provider_concurrency,
    )
    if spec.transport == "openrouter":
        execution_policy = hardened_execution_policy(execution_policy)
    live_transport = None
    if spec.transport == "openrouter":
        live_transport, transport_factory = bind_openrouter_transport(
            execution_policy
        )
    else:
        transport_factory = fake_transport_factory(
            gold_by_prompt=fake_gold_by_prompt(
                experiment,
                render_contract=family.render_contract(),
                ceiling_template=family.probes.ceiling_template,
            )
        )
    engine = RuntimeEvalEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.split_for(runtime_config.split_role),
        execution_policy=execution_policy,
        driver=GraphRolloutEvalDriver(
            eval_runner=family.eval_runner(),
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
            transport_factory=transport_factory,
        ),
        concurrency=spec.provider_concurrency,
    )
    proposer = prepare_proposer(
        experiment=experiment,
        model=spec.proposer_model,
        live_transport=live_transport,
        execution_policy=execution_policy,
    )
    return _RunEvaluation(engine=engine, proposer=proposer)


def _bind_optimizer(  # noqa: PLR0913
    *,
    run: _PreparedRun,
    store: ObjectStore,
    evaluation: _RunEvaluation,
    output_dir: Path,
    sqlite_path: Path,
    codex_test_seam: CodexTestSeam | None,
    codex_prompt_builder: Callable[[CodexPromptContext], str] | None,
) -> OptimizerBinding:
    settings = run.spec.optimizer
    engine = evaluation.engine
    experiment = run.prepared.experiment
    family = run.family
    if isinstance(settings, CodexSettings):
        return bind_codex(
            settings=settings,
            store=store,
            engine=engine,
            experiment=experiment,
            family=family,
            runtime_config=runtime_config_for_run(
                spec=run.spec,
                n_per_category=run.n_per_category,
                pool_seed_start=run.pool_seed_start,
            ),
            output_dir=output_dir,
            sqlite_path=sqlite_path,
            test_seam=codex_test_seam,
            prompt_builder=codex_prompt_builder,
        )
    if isinstance(settings, CoproSettings):
        return bind_copro(
            settings=settings,
            engine=engine,
            experiment=experiment,
            family=family,
            proposer=evaluation.proposer,
        )
    if isinstance(settings, GepaSettings):
        return bind_gepa(
            settings=settings,
            store=store,
            engine=engine,
            experiment=experiment,
            family=family,
            run_id=run.run_id,
            proposer=evaluation.proposer,
        )
    return bind_miprov2(
        settings=settings,
        store=store,
        engine=engine,
        experiment=experiment,
        family=family,
        proposer=evaluation.proposer,
    )


def _codex_preflight_already_proven() -> None:
    return


def _codex_tool_executor(
    *,
    engine: EvalEngine,
    experiment: Experiment,
    effect_authority: EffectLeaseAuthority,
    owner_id: str,
) -> EvaluatingToolExecutor:
    return EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        experiment.reward_policy,
        effect_authority,
        owner_id=owner_id,
        replay_policy=ReplayPolicy.IDEMPOTENT,
    )


def _build_run_runtime(  # noqa: PLR0913
    *,
    store: BlockingObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    bound: OptimizerBinding,
    run_id: str,
    sqlite_path: Path,
) -> RegisteredRuntime:
    # Harness and MCP calls share durable effect and admission state.
    effect_authority = EffectLeaseAuthority.sqlite(sqlite_path)
    is_codex = isinstance(bound, CodexBinding)
    owner_id = f"{run_id}-owner"
    runtime = build_runtime(
        store=store,
        engine=engine,
        adapter_registry=MappingAdapterRegistry(
            {bound.adapter_key: bound.adapter}
        ),
        effect_authority=effect_authority,
        owner_id=owner_id,
        admission=(
            ToolAdmissionAuthority.sqlite(sqlite_path) if is_codex else None
        ),
        tool_executor=(
            _codex_tool_executor(
                engine=engine,
                experiment=experiment,
                effect_authority=effect_authority,
                owner_id=owner_id,
            )
            if is_codex
            else None
        ),
    )
    if is_codex:
        assert isinstance(bound, CodexBinding)
        bound.adapter.bind_tool_store(runtime.tool_store)
    return runtime


def _prepare_launch(  # noqa: PLR0913
    *,
    runtime: RegisteredRuntime,
    bound: OptimizerBinding,
    run_id: str,
    experiment: Experiment,
    engine: EvalEngine,
    family: FamilySpec,
) -> OptimRunLaunch:
    render_contract = family.render_contract()
    if isinstance(bound, CodexBinding):
        return prepare_codex_run(
            runtime,
            run_id=run_id,
            control=bound.control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
            # Binding already proved the session; do not run a second probe.
            preflight=_codex_preflight_already_proven,
        )
    if isinstance(bound, GepaBinding):
        return prepare_gepa_run(
            runtime,
            run_id=run_id,
            control=bound.control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
        )
    if isinstance(bound, Miprov2Binding):
        control = bound.control
        miprov2_adapter = bound.adapter
        return prepare_miprov2_run(
            runtime,
            run_id=run_id,
            control=control,
            experiment=experiment,
            initial_state=build_miprov2_state(
                run=miprov2_run_ref(
                    run_id=run_id,
                    control=control,
                    experiment=experiment,
                ),
                control=control,
                engine=engine,
                experiment=experiment,
                adapter=miprov2_adapter,
                family=family,
            ),
            render_contract=render_contract,
            mutation_field=family.mutation_field,
        )
    return prepare_copro_run(
        runtime,
        run_id=run_id,
        control=bound.control,
        experiment=experiment,
        render_contract=render_contract,
        mutation_field=family.mutation_field,
    )


def _publish_result(
    run: _PreparedRun, store: ObjectStore, result_ref: TypedRef, output: Path
) -> None:
    if result_ref.schema_name != OPTIM_RESULT_SCHEMA:
        raise ValueError("optimizer run did not produce an OptimResult")
    result = OptimResult.model_validate(store.get(result_ref.reference))
    (output / "result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    trajectory = project_trajectory_report(
        store=store,
        prepared=run.prepared,
        result_ref=result_ref,
        result=result,
        transport=run.spec.transport,
        model=run.spec.model,
        split_sizes=run.spec.split_sizes,
    )
    publish_trajectory_report(output, trajectory)


def run_optimizer(
    spec: RunSpec,
    *,
    codex_test_seam: CodexTestSeam | None = None,
    codex_prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> Path:
    if codex_test_seam is not None and not isinstance(
        spec.optimizer, CodexSettings
    ):
        raise ValueError("codex_test_seam applies only to --optimizer codex")
    if codex_prompt_builder is not None and not isinstance(
        spec.optimizer, CodexSettings
    ):
        raise ValueError(
            "codex_prompt_builder applies only to --optimizer codex"
        )
    if isinstance(spec.optimizer, CodexSettings):
        refuse_unauthorized_real_codex(
            test_seam=codex_test_seam,
            allow_real_codex=spec.optimizer.allow_real_codex,
        )
    run = _prepare_run(spec)
    output = prepare_output_root(
        spec.output_dir or default_output_dir(run.run_id)
    )
    sqlite_path = output / "runtime.sqlite"
    with durable_run_boundary(output), open_sqlite(str(sqlite_path)) as store:
        object_store = cast("ObjectStore", store)
        evaluation = _build_evaluation(run, object_store)
        bound = _bind_optimizer(
            run=run,
            store=object_store,
            evaluation=evaluation,
            output_dir=output,
            sqlite_path=sqlite_path,
            codex_test_seam=codex_test_seam,
            codex_prompt_builder=codex_prompt_builder,
        )
        with _build_run_runtime(
            store=store,
            engine=evaluation.engine,
            experiment=run.prepared.experiment,
            bound=bound,
            run_id=run.run_id,
            sqlite_path=sqlite_path,
        ) as runtime:
            launch = _prepare_launch(
                runtime=runtime,
                bound=bound,
                run_id=run.run_id,
                experiment=run.prepared.experiment,
                engine=evaluation.engine,
                family=run.family,
            )
            request = copro_run_request(
                launch,
                controller_identity_hash=runtime.controller.runtime_hash,
            )
            result_ref = runtime.controller.drive(request)
            _publish_result(run, object_store, result_ref, output)
    return output


__all__ = ["run_optimizer"]
