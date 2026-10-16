from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import ProviderCallConfig, ProviderKind
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
from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
)
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.core.roles import EvalRole
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.optim.adapters import MappingAdapterRegistry, OptimizerAdapter
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
from whetstone.optim.copro.control import (
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optim.copro.proposal_contract import CoproProposalContractRecord
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    Miprov2Adapter,
)
from whetstone.optim.miprov2.control import Miprov2Control
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    ProviderProposerTransport,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import ToolAdmissionAuthority
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.artifacts import (
    default_output_dir,
    durable_run_boundary,
    prepare_output_root,
)
from whetstone_envs.evidence import publish_trajectory_report
from whetstone_envs.optim.codex import (
    CODEX_ADAPTER_KEY,
    CodexReasoningEffort,
    CodexTestSeam,
    build_codex_adapter,
    build_codex_control,
    codex_run_root,
    refuse_unauthorized_real_codex,
    resolve_codex_agent_model,
)
from whetstone_envs.optim.codex_runtime import EnvsCodexRuntimeConfig
from whetstone_envs.optim.experiment import (
    provider_call_config_ref,
    provider_config_ref,
)
from whetstone_envs.optim.extraction.trajectory import (
    project_trajectory_report,
)
from whetstone_envs.optim.families import (
    FamilySpec,
    family_spec,
)
from whetstone_envs.optim.gepa import build_gepa_adapter
from whetstone_envs.optim.miprov2 import (
    build_miprov2_adapter,
    build_miprov2_control,
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
    Miprov2Settings,
    RunSpec,
)
from whetstone_envs.optim.split import (
    partition_internal_split,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dr_store import ObjectStore
    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.codex.adapter import CodexAdapter
    from whetstone.optim.codex.runner import CodexPromptContext
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.proposal.proposer import ProposerTransport

    from whetstone_envs.optim.codex_runtime import CodexRuntimeTransport


def _proposal_contract(family: FamilySpec) -> CoproProposalContractRecord:
    placeholders = ", ".join(f"{{{field}}}" for field in family.prompt_fields)
    return CoproProposalContractRecord(
        target_name=f"{family.family_id}_prompt_template",
        task_context=family.task_context,
        output_rule=(
            f"Return one non-empty prompt template that uses {placeholders}."
        ),
    )


#: The family-namespaced schema name for COPRO's inline executor policy.
COPRO_EXECUTOR_SCHEMA_SUFFIX = "copro_proposal_executor"


def _copro_adapter(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    family: FamilySpec,
    extra_proposal_bodies: tuple[str, ...] = (),
) -> CoproAdapter:
    """The COPRO adapter this run drives, scripted when transport is fake.

    ``extra_proposal_bodies`` extends the family's scripted bodies so a
    fake run at a wider ``breadth`` proposes genuinely distinct drafts
    rather than re-offering the seed.
    """
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=family.proposal_bodies(extra_proposal_bodies),
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
    )
    return CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema=(f"{family.namespace}.{COPRO_EXECUTOR_SCHEMA_SUFFIX}"),
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedSpec:
    family: FamilySpec
    n_per_category: int
    pool_seed_start: int


def _resolve_spec(spec: RunSpec) -> _ResolvedSpec:
    family = family_spec(spec.family)
    return _ResolvedSpec(
        family=family,
        n_per_category=(
            family.default_n_per_category
            if spec.n_per_category is None
            else spec.n_per_category
        ),
        pool_seed_start=(
            family.default_pool_seed_start
            if spec.pool_seed_start is None
            else spec.pool_seed_start
        ),
    )


@dataclass(frozen=True, slots=True)
class _BoundOptimizer:
    """The one adapter a run registers, plus what its launch needs."""

    adapter_key: str
    adapter: OptimizerAdapter
    gepa_control: GepaControl | None = None
    miprov2_control: Miprov2Control | None = None
    miprov2_adapter: Miprov2Adapter | None = None
    codex_control: CodexControl | None = None


def build_codex_runtime_config(
    *, spec: RunSpec, validated: _ResolvedSpec
) -> EnvsCodexRuntimeConfig:
    """The runtime config the Codex MCP evaluation server rebuilds from.

    Derived from the same spec the in-process engine was built from, so
    the two cannot drift: every generation parameter is read from one
    place. ``build_codex_adapter`` then proves the rebuild lands on the
    same Eval Config before anything is spawned.
    """
    return EnvsCodexRuntimeConfig(
        family_id=validated.family.family_id,
        split_sizes=spec.split_sizes,
        n_per_category=validated.n_per_category,
        pool_seed_start=validated.pool_seed_start,
        num_seeds=spec.num_seeds,
        transport=cast("CodexRuntimeTransport", spec.transport),
        model=spec.model,
        # Forwarded for the same reason ``provider_concurrency`` is, with
        # a sharper consequence: the server rebuilds its engine from this
        # config alone, and ``build_codex_adapter`` refuses a rebuild whose
        # ``task_model_identity_hash`` disagrees with the harness's. An
        # unforwarded effort therefore does not silently run unpinned -- it
        # aborts the Codex arm, which is the correct failure but a
        # confusing one to debug.
        reasoning_effort=spec.task_reasoning_effort,
        # Forwarded rather than defaulted: the server rebuilds from this
        # config alone, so an unforwarded width would leave the Codex arm
        # evaluating at whetstone's default while every other arm ran at
        # the operator's.
        provider_concurrency=spec.provider_concurrency,
    )


def _bind_optimizer(  # noqa: PLR0913
    *,
    spec: RunSpec,
    validated: _ResolvedSpec,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
    copro_control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    proposer_config_ref: IdentityRef,
    output_dir: Path,
    sqlite_path: Path,
    codex_test_seam: CodexTestSeam | None = None,
    codex_prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> _BoundOptimizer:
    """Build exactly the adapter this run drives.

    Registry membership is part of controller identity, so a run registers
    its own optimizer and nothing else.
    """
    if isinstance(spec.optimizer, CodexSettings):
        codex_control = build_codex_control(
            engine=engine,
            experiment=experiment,
            family=validated.family,
            model=resolve_codex_agent_model(spec.optimizer.model),
            max_tool_calls=spec.optimizer.capacity,
            codex_binary=spec.optimizer.binary,
            reasoning_effort=CodexReasoningEffort(
                spec.optimizer.reasoning_effort
            ),
            wall_seconds=spec.optimizer.wall_seconds,
        )
        return _BoundOptimizer(
            adapter_key=CODEX_ADAPTER_KEY,
            # The preflight runs inside this call, so a broken Codex
            # session fails here -- before the runtime exists and before
            # any capacity or eval budget is committed.
            adapter=build_codex_adapter(
                store=store,
                control=codex_control,
                engine=engine,
                runtime_config=build_codex_runtime_config(
                    spec=spec, validated=validated
                ),
                reward_policy=experiment.reward_policy,
                store_path=sqlite_path,
                run_root=codex_run_root(output_dir),
                test_seam=codex_test_seam,
                prompt_builder=codex_prompt_builder,
            ),
            codex_control=codex_control,
        )
    if isinstance(spec.optimizer, CoproSettings):
        return _BoundOptimizer(
            adapter_key=COPRO_ADAPTER_KEY,
            adapter=_copro_adapter(
                engine=engine,
                control=copro_control,
                prompt_adapter=prompt_adapter,
                proposer_transport=proposer_transport,
                family=validated.family,
                extra_proposal_bodies=spec.optimizer.extra_proposal_bodies,
            ),
        )
    assert isinstance(spec.optimizer, (GepaSettings, Miprov2Settings))
    trainset_task_hashes, valset_task_hashes = partition_internal_split(
        tuple(engine.sampling.task_hashes),
        train_size=spec.optimizer.partition.train_size,
        val_size=spec.optimizer.partition.val_size,
    )
    if isinstance(spec.optimizer, GepaSettings):
        gepa_adapter = build_gepa_adapter(
            store=store,
            engine=engine,
            experiment=experiment,
            family=validated.family,
            run_id=run_id,
            proposer_transport=proposer_transport,
            proposer_config_ref=proposer_config_ref,
            max_metric_calls=spec.optimizer.max_metric_calls,
            reflection_minibatch_size=spec.optimizer.reflection_minibatch_size,
            seed=spec.optimizer.seed,
            trainset_task_hashes=trainset_task_hashes,
            valset_task_hashes=valset_task_hashes,
        )
        return _BoundOptimizer(
            adapter_key=GEPA_ADAPTER_KEY,
            adapter=gepa_adapter,
            gepa_control=gepa_adapter.control,
        )
    assert isinstance(spec.optimizer, Miprov2Settings)
    miprov2_control = build_miprov2_control(
        engine=engine,
        experiment=experiment,
        family=validated.family,
        proposer_config_ref=proposer_config_ref,
        demo_mode=spec.optimizer.demo_mode,
        seed=spec.optimizer.seed,
        minibatch=spec.optimizer.minibatch,
        minibatch_size=spec.optimizer.minibatch_size,
        minibatch_full_eval_steps=spec.optimizer.minibatch_full_eval_steps,
        num_trials=spec.optimizer.num_trials,
        num_candidates=spec.optimizer.num_candidates,
        trainset_task_hashes=trainset_task_hashes,
        valset_task_hashes=valset_task_hashes,
    )
    miprov2_adapter = build_miprov2_adapter(
        store=store,
        engine=engine,
        control=miprov2_control,
        family=validated.family,
        proposer_transport=proposer_transport,
    )
    return _BoundOptimizer(
        adapter_key=MIPROV2_ADAPTER_KEY,
        adapter=miprov2_adapter,
        miprov2_control=miprov2_control,
        miprov2_adapter=miprov2_adapter,
    )


def _codex_preflight_already_proven() -> None:
    """The preflight ``prepare_codex_run`` requires, already satisfied.

    ``build_codex_adapter`` runs the real ``codex_auth_preflight`` before
    it returns an adapter at all, so by the time the launch is prepared
    the session has been proven against this run's own executor and
    process environment. ``prepare_codex_run`` still requires a callable --
    it has no default, deliberately, so no caller can forget one -- and
    this names the proof that already happened. Running a second probe
    here would spend wall time re-proving the same session.

    This is not a way to skip the check: the only path that reaches it
    went through the real preflight moments earlier, and a failure there
    raises before the runtime is built.
    """


def _codex_tool_executor(
    *,
    engine: EvalEngine,
    experiment: Experiment,
    effect_authority: EffectLeaseAuthority,
    owner_id: str,
) -> EvaluatingToolExecutor:
    """The in-harness executor that re-issues Codex's out-of-process calls.

    Codex evaluates through its own MCP server, which admits and persists
    each call against the same durable store. The harness still needs an
    executor so the Step's Issued Tool Call ledger can record every call;
    for an already-terminal call it reads the durable result rather than
    evaluating again.
    """
    return EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        experiment.reward_policy,
        effect_authority,
        owner_id=owner_id,
        replay_policy=ReplayPolicy.IDEMPOTENT,
    )


def _build_run_runtime(  # noqa: PLR0913
    *,
    store,
    engine: EvalEngine,
    experiment: Experiment,
    bound: _BoundOptimizer,
    run_id: str,
    sqlite_path: Path,
) -> RegisteredRuntime:
    """Assemble the runtime this run drives its one optimizer through.

    Effect leases are durable, not per-process: they live in the run's own
    ``runtime.sqlite`` beside the object store, mirroring whetstone-ai's
    platform CLI, which hands ``EffectLeaseAuthority.sqlite`` the same path
    it opened the store from. The lease authority owns
    ``dr_store_lease_authority*`` while the object store owns
    ``objects``/``bindings``, so one file carries both without a name
    collision. A memory authority would discard every terminal at process
    exit, so a re-run against a completed run directory would re-execute
    effects that already happened.

    A Codex run needs two more things, and they are the reason this is a
    function rather than one call: its Tool Calls are admitted from an
    out-of-process MCP evaluation server, so its per-run capacity must be
    durable too -- an in-memory authority would let that process admit
    past the cap, and the cap is what bounds paid evaluations -- and being
    the one ``TOOL_USING`` run, it needs a tool executor so the Step's
    Issued Tool Call ledger records every call the agent made outside.
    """
    effect_authority = EffectLeaseAuthority.sqlite(sqlite_path)
    is_codex = bound.codex_control is not None
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
        # The adapter reads durable Tool Call entries from the exact store
        # the harness admits through, so it is bound to the runtime's own
        # rather than to a second one built here. ``bind_tool_store`` is
        # the Codex adapter's, not the ``OptimizerAdapter`` protocol's --
        # only a TOOL_USING adapter has one -- and ``_bind_optimizer``
        # guarantees this branch holds a ``CodexAdapter``.
        codex_adapter = cast("CodexAdapter", bound.adapter)
        codex_adapter.bind_tool_store(runtime.tool_store)
    return runtime


def _prepare_launch(  # noqa: PLR0913
    *,
    runtime: RegisteredRuntime,
    bound: _BoundOptimizer,
    run_id: str,
    experiment: Experiment,
    copro_control: CoproControl,
    engine: EvalEngine,
    family: FamilySpec,
) -> OptimRunLaunch:
    """Bind the run for whichever optimizer this run registered."""
    render_contract = family.render_contract()
    if bound.codex_control is not None:
        return prepare_codex_run(
            runtime,
            run_id=run_id,
            control=bound.codex_control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
            # The session was already proven when the adapter was built,
            # before the runtime existed. ``prepare_codex_run`` requires a
            # callable and has no default, so this names the proof that
            # already happened rather than re-spawning a second probe on
            # the run's wall budget.
            preflight=_codex_preflight_already_proven,
        )
    if bound.gepa_control is not None:
        return prepare_gepa_run(
            runtime,
            run_id=run_id,
            control=bound.gepa_control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
        )
    if bound.miprov2_control is not None:
        control = bound.miprov2_control
        miprov2_adapter = bound.miprov2_adapter
        assert miprov2_adapter is not None
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
        control=copro_control,
        experiment=experiment,
        render_contract=render_contract,
        mutation_field=family.mutation_field,
    )


def run_optimizer(  # noqa: PLR0915
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
    validated = _resolve_spec(spec)
    if isinstance(spec.optimizer, CodexSettings):
        refuse_unauthorized_real_codex(
            test_seam=codex_test_seam,
            allow_real_codex=spec.optimizer.allow_real_codex,
        )
    family = validated.family
    resolved_run_id = spec.run_id or (
        f"{family.run_id_prefix}-{spec.optimizer.kind}-{uuid4().hex[:8]}"
    )
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(
            model=spec.model, reasoning_effort=spec.task_reasoning_effort
        )
        api_key_env = "OPENROUTER_API_KEY"
    pool = family.generate_pool(
        n_per_category=validated.n_per_category,
        seed_start=validated.pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=spec.num_seeds,
        provider_call_config=provider,
    )
    experiment = prepared.experiment
    if spec.transport == "openrouter":
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
            provider_kind=ProviderKind.OPENROUTER,
        )
    else:
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
        )
    # Widened once, here, so the engine's worker pool, the live transport's
    # connection pool, and the proposer all describe one decision.
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
    resolved_output = prepare_output_root(
        spec.output_dir or default_output_dir(resolved_run_id)
    )
    sqlite_path = resolved_output / "runtime.sqlite"
    with (
        durable_run_boundary(resolved_output),
        open_sqlite(str(sqlite_path)) as store,
    ):
        # Built directly rather than through ``build_engine``, which
        # forwards neither a concurrency nor a policy and so would run at
        # whetstone's default width over the unwidened pool. This is the
        # same in-process driver that helper assembles for the default
        # ``driver_mode``.
        engine = RuntimeEvalEngine(
            store=cast("ObjectStore", store),
            experiment=experiment,
            sampling=experiment.eval_configs.split_for(
                runtime_config.split_role
            ),
            execution_policy=execution_policy,
            driver=GraphRolloutEvalDriver(
                eval_runner=family.eval_runner(),
                mutation_field=family.mutation_field,
                render_contract=family.render_contract(),
                transport_factory=transport_factory,
            ),
            concurrency=spec.provider_concurrency,
        )
        prompt_adapter = PlainPromptAdapter()
        # One route object feeds both the reference the optimizer records
        # and the record the transport resolves, so the two cannot name
        # different configs.
        proposer_route = _proposer_route(
            experiment=experiment, proposer_model=spec.proposer_model
        )
        proposer_transport = None
        if live_transport is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=proposer_route.resolver(),
                transport=live_transport,
                execution_policy=execution_policy,
                prompt_adapter=prompt_adapter,
            )
        defaults = CoproInjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=proposer_route.ref,
                temperature=None,
            ),
            proposal_contract=_proposal_contract(family),
            eval_config_ref=engine.eval_config_ref,
            eval_role=EvalRole.INTERNAL,
            provider_execution_policy_ref=(
                engine.provider_execution_policy_ref
            ),
            expected_reward_policy_hash=(
                experiment.reward_policy.identity_hash()
            ),
            provider_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            prompt_adapter=prompt_adapter,
        )
        copro_settings = (
            spec.optimizer
            if isinstance(spec.optimizer, CoproSettings)
            else CoproSettings()
        )
        copro_control = configure_copro(
            breadth=copro_settings.breadth,
            depth=copro_settings.depth,
            track_stats=False,
            defaults=defaults,
        )
        bound = _bind_optimizer(
            spec=spec,
            validated=validated,
            store=cast("ObjectStore", store),
            engine=engine,
            experiment=experiment,
            run_id=resolved_run_id,
            copro_control=copro_control,
            prompt_adapter=prompt_adapter,
            proposer_transport=proposer_transport,
            proposer_config_ref=proposer_route.ref,
            output_dir=resolved_output,
            sqlite_path=sqlite_path,
            codex_test_seam=codex_test_seam,
            codex_prompt_builder=codex_prompt_builder,
        )
        runtime = _build_run_runtime(
            store=store,
            engine=engine,
            experiment=experiment,
            bound=bound,
            run_id=resolved_run_id,
            sqlite_path=sqlite_path,
        )
        # ``RegisteredRuntime.close`` releases the eval engine and the
        # authority's sqlite connection on every exit path, including the
        # failures ``durable_run_boundary`` re-raises.
        with runtime:
            launch = _prepare_launch(
                runtime=runtime,
                bound=bound,
                run_id=resolved_run_id,
                experiment=experiment,
                copro_control=copro_control,
                engine=engine,
                family=family,
            )
            request = copro_run_request(
                launch,
                controller_identity_hash=runtime.controller.runtime_hash,
            )
            result_ref = runtime.controller.drive(request)
            if result_ref.schema_name != OPTIM_RESULT_SCHEMA:
                raise ValueError(
                    "optimizer run did not produce an OptimResult"
                )
            result = OptimResult.model_validate(
                runtime.store.get(result_ref.reference)
            )
            (resolved_output / "result.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            trajectory = project_trajectory_report(
                store=cast("ObjectStore", store),
                prepared=prepared,
                result_ref=result_ref,
                result=result,
                transport=spec.transport,
                model=spec.model,
                split_sizes=spec.split_sizes,
            )
            publish_trajectory_report(resolved_output, trajectory)
    return resolved_output


@dataclass(frozen=True, slots=True)
class _ProposerRoute:
    """The one proposal route a run binds, reference and record together.

    :class:`ProviderProposerTransport` resolves the reference its
    ``ProposerConfig`` carries and then asserts the resolved record matches
    that reference. The reference an optimizer's injected defaults carry and
    the record this route's resolver returns must therefore name the same
    config -- minting the reference from the *task* config while resolving
    to a distinct *proposer* config fails that assertion at draft time, in
    the middle of a durable run. Holding both on one object, derived from
    one config, is what makes them impossible to disagree.
    """

    config: ProviderCallConfig
    ref: IdentityRef

    def resolver(self):
        """The resolver :class:`ProviderProposerTransport` calls."""

        def resolve(_ref: object) -> ProviderCallConfig:
            return self.config

        return resolve


def _proposer_route(
    *,
    experiment: Experiment,
    proposer_model: str | None,
) -> _ProposerRoute:
    if proposer_model is None:
        config = experiment.rollout_graph.provider_call_config
        return _ProposerRoute(
            config=config, ref=provider_call_config_ref(experiment)
        )
    config = openrouter_seeded_call_config(model=proposer_model)
    return _ProposerRoute(config=config, ref=provider_config_ref(config))


__all__ = ["build_codex_runtime_config", "run_optimizer"]
