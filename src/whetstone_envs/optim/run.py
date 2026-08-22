"""The shared optimizer runner every task family reaches through.

``run_optimizer`` drives one optimizer over one family's prepared
experiment and writes the run's artifacts off-repo. It reads family-specific
knowledge only from the :mod:`whetstone_envs.optim.families` registry, so it
carries no family literal of its own -- that is the C3 generality property
the second family exercises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import ProviderKind
from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    copro_run_request,
    prepare_copro_run,
    prepare_gepa_run,
    prepare_miprov2_run,
)
from whetstone.core.identity import compute_identity_hash
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

from whetstone_envs.optim.experiment import c19_provider_call_config_ref
from whetstone_envs.optim.families import (
    KNOWN_FAMILY_IDS,
    FamilyId,
    FamilySpec,
    family_spec,
    registered_family_ids,
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

    from whetstone_envs.optim.experiment import PreparedC19Experiment

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm" / "data" / "runs" / ("whetstone-envs")
)

#: Every optimizer the shared runner can drive today.
#:
#: ``codex``, ``null-random``, and ``null-identity`` are named by the study
#: protocol and land with their own adapters; they are absent here because
#: their modules are not in this package yet, and admitting a name the runner
#: cannot drive would fail late, inside a durable run boundary, instead of at
#: spec validation.
OPTIMIZERS = ("copro", "gepa", "miprov2")
TRANSPORTS = ("fake", "openrouter")

#: Retained COPRO search shape: two drafts per step, one step of depth.
DEFAULT_COPRO_BREADTH = 2
DEFAULT_COPRO_DEPTH = 1
#: The smallest breadth ``CoproControl`` accepts. A single draft per step
#: leaves nothing to select between, so upstream refuses it.
MIN_COPRO_BREADTH = 2
#: Each optimizer's own seed default, used when a spec names none. These
#: mirror the values ``configure_gepa`` and ``configure_miprov2`` already
#: default to, so an unseeded run keeps the control identity it always had.
GEPA_DEFAULT_SEED = 0
MIPROV2_DEFAULT_SEED = 9

#: The default split, kept small so an unparameterised run stays a smoke run.
DEFAULT_SPLIT_SIZES = (2, 2, 0)


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One optimizer run over one task family.

    Every field the study varies is explicit here, so a run is fully
    described by its spec and nothing is read from module state.
    """

    optimizer: str
    transport: str
    #: The registered task family this run drives.
    family: str = FamilyId.C19.value
    split_sizes: tuple[int, int, int] = DEFAULT_SPLIT_SIZES
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"
    #: The proposer's model. ``None`` reuses ``model`` for both roles.
    proposer_model: str | None = None
    #: MIPROv2's demonstration regime; ignored by COPRO and GEPA.
    demo_mode: str = Miprov2DemoMode.FEWSHOT.value
    #: Repeats per task (K_REPEAT). One seed keeps a run deterministic.
    num_seeds: int = 1
    #: Instances generated per stratum. ``None`` takes the family default.
    n_per_stratum: int | None = None
    #: First generator seed for the pool. ``None`` takes the family default.
    pool_seed_start: int | None = None
    #: This run's algorithmic seed. ``None`` keeps each optimizer's own
    #: default, which is what an unparameterised smoke run wants.
    #:
    #: GEPA and MIPROv2 carry it into their controls as an explicit field.
    #: ``CoproControl`` has no seed: COPRO's stochasticity is the proposer
    #: LM, so its effective seed is the provider ``SEED`` control plus
    #: proposal ordering. The field is still recorded for a COPRO run so the
    #: study manifest can state what was requested and how it was honoured;
    #: :func:`seed_disposition` names that difference. The study assigns
    #: disjoint per-optimizer ranges; choosing them is the study's concern,
    #: not the runner's, so the runner accepts any integer.
    seed: int | None = None
    #: COPRO candidates proposed per step.
    copro_breadth: int = DEFAULT_COPRO_BREADTH
    #: COPRO search depth; step count is ``depth + 1``.
    copro_depth: int = DEFAULT_COPRO_DEPTH
    #: GEPA's paid metric-call ceiling. ``None`` keeps the family default of
    #: one full pass over the trainset plus one reflection minibatch.
    gepa_max_metric_calls: int | None = None
    #: The Codex arm's admitted evaluate-call cap. Carried so a spec is
    #: complete before its adapter lands; rejected on other optimizers so it
    #: cannot look honoured when nothing reads it.
    codex_capacity: int | None = None


#: How a run's ``seed`` reaches the optimizer, recorded per optimizer.
#:
#: These are manifest values, not free-form prose: the study records which
#: arm carried an explicit control seed and which did not.
SEED_DISPOSITION_CONTROL_FIELD = "control-seed-field"
SEED_DISPOSITION_PROVIDER_ONLY = "provider-seed-control-only"


def seed_disposition(optimizer: str) -> str:
    """Name how ``optimizer`` honours a run's requested seed.

    COPRO is the honest exception: ``CoproControl`` carries no seed field,
    so a COPRO run's reproducibility rests on the provider ``SEED`` control
    and proposal ordering rather than on an algorithmic seed. Recording that
    beats faking a seed the control never reads.
    """
    if optimizer == "copro":
        return SEED_DISPOSITION_PROVIDER_ONLY
    return SEED_DISPOSITION_CONTROL_FIELD


def default_output_dir(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / run_id


def _provider_config_resolver(experiment: Experiment):
    provider_config = experiment.rollout_graph.provider_call_config

    def resolve(_ref: object):
        return provider_config

    return resolve


def _proposal_contract(family: FamilySpec) -> CoproProposalContractRecord:
    placeholders = ", ".join(f"{{{field}}}" for field in family.prompt_fields)
    return CoproProposalContractRecord(
        target_name=f"{family.family_id}_prompt_template",
        task_context=family.task_context,
        output_rule=(
            f"Return one non-empty prompt template that uses {placeholders}."
        ),
    )


_COPRO_EXECUTOR_SCHEMA = "whetstone_envs.c19.copro_proposal_executor"


def _copro_adapter(
    *,
    engine: EvalEngine,
    control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    family: FamilySpec,
) -> CoproAdapter:
    """The COPRO adapter this run drives, scripted when transport is fake."""
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=family.proposal_bodies(),
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


@dataclass(frozen=True, slots=True)
class _ValidatedSpec:
    """A spec proven runnable, with its family and demo mode resolved."""

    family: FamilySpec
    demo_mode: Miprov2DemoMode
    n_per_stratum: int
    pool_seed_start: int


def _validate_spec(spec: RunSpec) -> _ValidatedSpec:
    """Reject an unrunnable spec before any durable effect happens."""
    if spec.optimizer not in set(OPTIMIZERS):
        raise ValueError(f"unsupported optimizer {spec.optimizer!r}")
    if spec.transport not in set(TRANSPORTS):
        raise ValueError(f"unsupported transport {spec.transport!r}")
    if spec.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if spec.copro_breadth < MIN_COPRO_BREADTH:
        # Refusing here keeps the failure at spec validation instead of
        # inside the durable run boundary, where it would leave a run
        # directory behind.
        raise ValueError(f"copro_breadth must be at least {MIN_COPRO_BREADTH}")
    if spec.copro_depth < 0:
        raise ValueError("copro_depth must be non-negative")
    if spec.gepa_max_metric_calls is not None:
        if spec.optimizer != "gepa":
            raise ValueError(
                "gepa_max_metric_calls applies only to --optimizer gepa"
            )
        if spec.gepa_max_metric_calls < 1:
            raise ValueError("gepa_max_metric_calls must be at least 1")
    if spec.codex_capacity is not None:
        # The Codex adapter is not in this package yet, so no optimizer can
        # honour a capacity cap. Refusing beats silently ignoring it.
        raise ValueError(
            "codex_capacity applies only to --optimizer codex, "
            "which this runner cannot drive yet"
        )
    family = family_spec(spec.family)
    n_per_stratum = (
        family.default_n_per_stratum
        if spec.n_per_stratum is None
        else spec.n_per_stratum
    )
    if n_per_stratum < 1:
        raise ValueError("n_per_stratum must be at least 1")
    pool_seed_start = (
        family.default_pool_seed_start
        if spec.pool_seed_start is None
        else spec.pool_seed_start
    )
    try:
        demo_mode = Miprov2DemoMode(spec.demo_mode)
    except ValueError as error:
        raise ValueError(
            f"unsupported demo mode {spec.demo_mode!r}"
        ) from error
    return _ValidatedSpec(
        family=family,
        demo_mode=demo_mode,
        n_per_stratum=n_per_stratum,
        pool_seed_start=pool_seed_start,
    )


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
    spec: RunSpec,
    validated: _ValidatedSpec,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
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
            adapter=_copro_adapter(
                engine=engine,
                control=copro_control,
                prompt_adapter=prompt_adapter,
                proposer_transport=proposer_transport,
                family=validated.family,
            ),
        )
    if spec.optimizer == "gepa":
        gepa_adapter = build_c19_gepa_adapter(
            store=store,
            engine=engine,
            experiment=experiment,
            run_id=run_id,
            proposer_transport=proposer_transport,
            max_metric_calls=spec.gepa_max_metric_calls,
            seed=GEPA_DEFAULT_SEED if spec.seed is None else spec.seed,
        )
        return _BoundOptimizer(
            adapter_key=GEPA_ADAPTER_KEY,
            adapter=gepa_adapter,
            gepa_control=gepa_adapter.control,
        )
    miprov2_control = build_c19_miprov2_control(
        engine=engine,
        experiment=experiment,
        demo_mode=validated.demo_mode,
        seed=MIPROV2_DEFAULT_SEED if spec.seed is None else spec.seed,
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
    family: FamilySpec,
) -> OptimRunLaunch:
    """Bind the run for whichever optimizer this run registered."""
    render_contract = family.render_contract()
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
            initial_state=build_c19_miprov2_state(
                run=c19_miprov2_run_ref(
                    run_id=run_id,
                    control=control,
                    experiment=experiment,
                ),
                control=control,
                engine=engine,
                experiment=experiment,
                adapter=miprov2_adapter,
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


def run_optimizer(spec: RunSpec) -> Path:
    """Run one optimizer over one family's split, writing artifacts off-repo.

    COPRO, GEPA, and MIPROv2 all reach the same runtime entry point; MIPROv2
    additionally binds an opening durable state, and its ``demo_mode``
    selects the demonstration regime. Every family-specific decision is read
    from the family registry, so this function names no family.
    """
    validated = _validate_spec(spec)
    family = validated.family
    resolved_run_id = spec.run_id or (
        f"{family.run_id_prefix}-{spec.optimizer}-{uuid4().hex[:8]}"
    )
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(model=spec.model)
        api_key_env = "OPENROUTER_API_KEY"
    pool = family.generate_pool(
        n_per_stratum=validated.n_per_stratum,
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
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
            transport_factory=transport_factory,
        )
        prompt_adapter = PlainPromptAdapter()
        proposer_transport = None
        if live_transport is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=_proposer_config_resolver(
                    experiment=experiment,
                    proposer_model=spec.proposer_model,
                ),
                transport=live_transport,
                execution_policy=runtime_config.execution_policy,
                prompt_adapter=prompt_adapter,
            )
        defaults = CoproInjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=c19_provider_call_config_ref(experiment),
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
        copro_control = configure_copro(
            breadth=spec.copro_breadth,
            depth=spec.copro_depth,
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
                # ``project_trajectory_report`` reads only ``.split`` and
                # ``.experiment``, which is exactly ``PreparedExperiment``,
                # but its annotation still names the concrete c19 type.
                # Widening that annotation belongs to the reporting layer,
                # so the narrowing is stated here rather than there.
                prepared=cast("PreparedC19Experiment", prepared),
                result_ref=result_ref,
                result=result,
                transport=spec.transport,
                model=spec.model,
                split_sizes=spec.split_sizes,
            )
            publish_trajectory_report(resolved_output, trajectory)
    return resolved_output


def _proposer_config_resolver(
    *,
    experiment: Experiment,
    proposer_model: str | None,
):
    """Resolve the proposal route, which may differ from the task route.

    A study runs a cheap task model against a stronger proposer, so the two
    routes are separable. ``None`` reuses the experiment's own route, which
    keeps a single-model run byte-identical to one that never named a
    proposer.
    """
    if proposer_model is None:
        return _provider_config_resolver(experiment)
    proposer_config = openrouter_seeded_call_config(model=proposer_model)

    def resolve(_ref: object):
        return proposer_config

    return resolve


__all__ = [
    "C19_DEMO_MODES",
    "DEFAULT_COPRO_BREADTH",
    "DEFAULT_COPRO_DEPTH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SPLIT_SIZES",
    "GEPA_DEFAULT_SEED",
    "KNOWN_FAMILY_IDS",
    "MIPROV2_DEFAULT_SEED",
    "OPTIMIZERS",
    "SEED_DISPOSITION_CONTROL_FIELD",
    "SEED_DISPOSITION_PROVIDER_ONLY",
    "TRANSPORTS",
    "RunSpec",
    "default_output_dir",
    "registered_family_ids",
    "run_optimizer",
    "seed_disposition",
]
