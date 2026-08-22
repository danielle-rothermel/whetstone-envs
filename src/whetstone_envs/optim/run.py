from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA, ProviderKind
from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    copro_run_request,
    prepare_copro_run,
    prepare_gepa_run,
    prepare_miprov2_run,
)
from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.leasing import EffectLeaseAuthority
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry, OptimizerAdapter
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
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.gepa import build_c19_gepa_adapter
from whetstone_envs.optim.miprov2 import (
    C19_DEMO_MODES,
    Miprov2DemoMode,
    build_c19_miprov2_adapter,
    build_c19_miprov2_control,
    build_c19_miprov2_state,
    c19_miprov2_run_ref,
)
from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    c19_fake_gold_by_prompt,
    c19_fake_transport_factory,
    openrouter_seeded_call_config,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner
from whetstone_envs.reporting.projection import project_trajectory_report
from whetstone_envs.reporting.publication import (
    durable_run_boundary,
    prepare_output_root,
    publish_trajectory_report,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.proposal.proposer import ProposerTransport

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm" / "data" / "runs" / ("whetstone-envs")
)
# Seed COPRO asks for one draft and keeps the naive initial candidate. The
# first body must differ from that seed or COPRO rejects a no-op mutation.
C19_PROPOSAL_BODIES = (
    PROBES.ceiling_template,
    PROBES.naive_template,
)


#: Every optimizer the shared C19 runner can drive.
C19_OPTIMIZERS = ("copro", "gepa", "miprov2")
C19_TRANSPORTS = ("fake", "openrouter")


@dataclass(frozen=True, slots=True)
class C19RunSpec:
    optimizer: str
    transport: str
    split_sizes: tuple[int, int, int] = (2, 2, 0)
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"
    #: MIPROv2's demonstration regime; ignored by COPRO and GEPA.
    demo_mode: str = Miprov2DemoMode.FEWSHOT.value
    #: Repeats per task (K_REPEAT). One seed keeps a run deterministic.
    num_seeds: int = 1


def default_output_dir(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / run_id


def _provider_config_resolver(experiment: Experiment):
    provider_config = experiment.rollout_graph.provider_call_config

    def resolve(_ref: object):
        return provider_config

    return resolve


def _provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    payload = experiment.rollout_graph.provider_call_config.model_dump(
        mode="json"
    )
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def _c19_proposal_contract() -> CoproProposalContractRecord:
    return CoproProposalContractRecord(
        target_name="c19_prompt_template",
        task_context="Predict the MiniGrid fact asked by the question.",
        output_rule=(
            "Return one non-empty prompt template that uses {grid}, "
            "{command}, and {question}."
        ),
    )


_COPRO_EXECUTOR_SCHEMA = "whetstone_envs.c19.copro_proposal_executor"


def _c19_copro_adapter(
    *,
    engine: EvalEngine,
    control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
) -> CoproAdapter:
    """The COPRO adapter this run drives, scripted when transport is fake."""
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=C19_PROPOSAL_BODIES,
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
                schema=_COPRO_EXECUTOR_SCHEMA,
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )


def _validated_demo_mode(spec: C19RunSpec) -> Miprov2DemoMode:
    """Reject an unrunnable spec before any durable effect happens."""
    if spec.optimizer not in set(C19_OPTIMIZERS):
        raise ValueError(f"unsupported optimizer {spec.optimizer!r}")
    if spec.transport not in set(C19_TRANSPORTS):
        raise ValueError(f"unsupported transport {spec.transport!r}")
    if spec.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    try:
        return Miprov2DemoMode(spec.demo_mode)
    except ValueError as error:
        raise ValueError(
            f"unsupported demo mode {spec.demo_mode!r}"
        ) from error


@dataclass(frozen=True, slots=True)
class _BoundOptimizer:
    """The one adapter a run registers, plus what its launch needs."""

    adapter_key: str
    adapter: OptimizerAdapter
    gepa_control: GepaControl | None = None
    miprov2_control: Miprov2Control | None = None
    miprov2_adapter: Miprov2Adapter | None = None


def _bind_optimizer(  # noqa: PLR0913
    *,
    spec: C19RunSpec,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
    demo_mode: Miprov2DemoMode,
    copro_control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
) -> _BoundOptimizer:
    """Build exactly the adapter this run drives.

    Registry membership is part of controller identity, so a run registers
    its own optimizer and nothing else.
    """
    if spec.optimizer == "copro":
        return _BoundOptimizer(
            adapter_key=COPRO_ADAPTER_KEY,
            adapter=_c19_copro_adapter(
                engine=engine,
                control=copro_control,
                prompt_adapter=prompt_adapter,
                proposer_transport=proposer_transport,
            ),
        )
    if spec.optimizer == "gepa":
        gepa_adapter = build_c19_gepa_adapter(
            store=store,
            engine=engine,
            experiment=experiment,
            run_id=run_id,
            proposer_transport=proposer_transport,
        )
        return _BoundOptimizer(
            adapter_key=GEPA_ADAPTER_KEY,
            adapter=gepa_adapter,
            gepa_control=gepa_adapter.control,
        )
    miprov2_control = build_c19_miprov2_control(
        engine=engine,
        experiment=experiment,
        demo_mode=demo_mode,
    )
    miprov2_adapter = build_c19_miprov2_adapter(
        store=store,
        engine=engine,
        control=miprov2_control,
        proposer_transport=proposer_transport,
    )
    return _BoundOptimizer(
        adapter_key=MIPROV2_ADAPTER_KEY,
        adapter=miprov2_adapter,
        miprov2_control=miprov2_control,
        miprov2_adapter=miprov2_adapter,
    )


def _prepare_launch(  # noqa: PLR0913
    *,
    runtime: RegisteredRuntime,
    bound: _BoundOptimizer,
    run_id: str,
    experiment: Experiment,
    copro_control: CoproControl,
    engine: EvalEngine,
) -> OptimRunLaunch:
    """Bind the run for whichever optimizer this run registered."""
    render_contract = c19_render_contract()
    if bound.gepa_control is not None:
        return prepare_gepa_run(
            runtime,
            run_id=run_id,
            control=bound.gepa_control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=C19_MUTATION_FIELD,
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
            initial_state=build_c19_miprov2_state(
                run=c19_miprov2_run_ref(
                    run_id=run_id,
                    control=control,
                    experiment=experiment,
                ),
                control=control,
                engine=engine,
                adapter=miprov2_adapter,
            ),
            render_contract=render_contract,
            mutation_field=C19_MUTATION_FIELD,
        )
    return prepare_copro_run(
        runtime,
        run_id=run_id,
        control=copro_control,
        experiment=experiment,
        render_contract=render_contract,
        mutation_field=C19_MUTATION_FIELD,
    )


def run_c19_optimizer(spec: C19RunSpec) -> Path:
    """Run one C19 optimizer on a small split, writing artifacts off-repo.

    COPRO, GEPA, and MIPROv2 all reach the same runtime entry point; MIPROv2
    additionally binds an opening durable state, and its ``demo_mode``
    selects the demonstration regime.
    """
    demo_mode = _validated_demo_mode(spec)
    resolved_run_id = spec.run_id or (
        f"c19-{spec.optimizer}-{uuid4().hex[:8]}"
    )
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(model=spec.model)
        api_key_env = "OPENROUTER_API_KEY"
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    prepared = prepare_c19_experiment(
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
    live_transport = None
    if spec.transport == "openrouter":
        live_transport, transport_factory = bind_openrouter_transport(
            runtime_config.execution_policy
        )
    else:
        transport_factory = c19_fake_transport_factory(
            gold_by_prompt=c19_fake_gold_by_prompt(experiment)
        )
    resolved_output = prepare_output_root(
        spec.output_dir or default_output_dir(resolved_run_id)
    )
    sqlite_path = resolved_output / "runtime.sqlite"
    with (
        durable_run_boundary(resolved_output),
        open_sqlite(str(sqlite_path)) as store,
    ):
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=transport_factory,
        )
        prompt_adapter = PlainPromptAdapter()
        proposer_transport = None
        if live_transport is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=_provider_config_resolver(
                    experiment
                ),
                transport=live_transport,
                execution_policy=runtime_config.execution_policy,
                prompt_adapter=prompt_adapter,
            )
        defaults = CoproInjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=_provider_call_config_ref(experiment),
                temperature=None,
            ),
            proposal_contract=_c19_proposal_contract(),
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
        copro_control = configure_copro(
            breadth=2,
            depth=1,
            track_stats=False,
            defaults=defaults,
        )
        bound = _bind_optimizer(
            spec=spec,
            store=cast("ObjectStore", store),
            engine=engine,
            experiment=experiment,
            run_id=resolved_run_id,
            demo_mode=demo_mode,
            copro_control=copro_control,
            prompt_adapter=prompt_adapter,
            proposer_transport=proposer_transport,
        )
        # Effect leases are durable, not per-process: they live in the run's
        # own ``runtime.sqlite`` beside the object store, mirroring
        # whetstone-ai's platform CLI, which hands ``EffectLeaseAuthority
        # .sqlite`` the same path it opened the store from. The lease
        # authority owns ``dr_store_lease_authority*`` while the object store
        # owns ``objects``/``bindings``, so one file carries both without a
        # name collision. A memory authority would discard every terminal at
        # process exit, so a re-run against a completed run directory would
        # re-execute effects that already happened.
        runtime = build_runtime(
            store=store,
            engine=engine,
            adapter_registry=MappingAdapterRegistry(
                {bound.adapter_key: bound.adapter}
            ),
            effect_authority=EffectLeaseAuthority.sqlite(sqlite_path),
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


__all__ = [
    "C19_DEMO_MODES",
    "C19_OPTIMIZERS",
    "C19_PROPOSAL_BODIES",
    "C19_TRANSPORTS",
    "DEFAULT_OUTPUT_ROOT",
    "C19RunSpec",
    "default_output_dir",
    "run_c19_optimizer",
]
