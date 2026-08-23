from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import compute_identity_hash
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.factory import build_gepa_harness_adapter
from whetstone.optim.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.optim.experiment import provider_call_config_ref
from whetstone_envs.optim.split import require_disjoint_split

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.gepa.harness_adapter import GepaHarnessAdapter
    from whetstone.optim.proposal.proposer import ProposerTransport

    from whetstone_envs.optim.families import FamilySpec

GEPA_COMPONENT_NAME = "generate"
#: Per-family schema names for the two identities GEPA mints. Both are
#: persisted-format strings, so each family's own namespace owns its
#: value rather than one family's name standing in for every family's.
INLINE_EXECUTOR_SCHEMA_SUFFIX = "inline_proposal_executor"
COMPONENT_SCHEMA_SUFFIX = "gepa_component"

#: Traces the reflection proposer consumes per round when a caller pins
#: none. One is the smoke-run shape: enough to exercise the reflection
#: path, small enough to keep an unparameterised run cheap. The study pins
#: its own through ``RunSpec.gepa_reflection_minibatch_size``.
DEFAULT_REFLECTION_MINIBATCH_SIZE = 1


def gepa_prompt_services(family: FamilySpec) -> GepaPromptServices:
    """Reflection prompt services bound to one family's placeholders.

    The shared ``default_gepa_prompt_services`` binds a single ``{prompt}``
    placeholder; a family's templates carry its own set, so the component
    format is declared from the family's contract rather than taken from
    the default.
    """
    component_schema_hash = compute_identity_hash(
        schema=f"{family.namespace}.{COMPONENT_SCHEMA_SUFFIX}",
        schema_version=1,
        payload={"field": family.mutation_field},
    )
    descriptor = GepaPromptFormatDescriptor(
        format_name=f"{family.family_id}_prompt_template",
        components=(
            GepaComponentFormat(
                component_name=GEPA_COMPONENT_NAME,
                component_schema_identity_hash=component_schema_hash,
                allowed_placeholders=family.prompt_fields,
                required_placeholders=family.prompt_fields,
            ),
        ),
    )
    return GepaPromptServices(
        descriptor=descriptor,
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _gepa_transport(
    *,
    engine: EvalEngine,
    prompt_adapter: PlainPromptAdapter,
    proposal_bodies: tuple[str, ...],
    proposer_transport: ProposerTransport | None,
) -> ProposerTransport:
    if proposer_transport is not None:
        return proposer_transport
    adapter_hash = prompt_adapter_identity_hash(prompt_adapter)
    return FakeProposerTransport(
        {("gepa_reflection", 0): proposal_bodies},
        default=proposal_bodies,
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=adapter_hash,
    )


def build_gepa_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    family: FamilySpec,
    prompt_services: GepaPromptServices,
    policy_identity_hash: str,
    seed: int = 0,
    max_metric_calls: int | None = None,
    reflection_minibatch_size: int | None = None,
    trainset_task_hashes: tuple[str, ...],
    valset_task_hashes: tuple[str, ...],
):
    """Resolve one family's GEPA control over the engine's internal split.

    ``seed`` is GEPA's explicit algorithmic seed, carried straight onto the
    control. ``max_metric_calls`` pins the paid metric-call ceiling; ``None``
    keeps the default of one full pass over the trainset plus one reflection
    minibatch, which is what a smoke run needs.

    ``reflection_minibatch_size`` is how many traces the reflection
    proposer consumes per round. ``None`` keeps the single-trace default a
    smoke run wants; a study pins it, because the number of traces the
    reflection step sees is part of the proposer's input rather than a
    runtime detail, and a hardcoded value cannot be audited against a
    design.

    ``trainset_task_hashes`` and ``valset_task_hashes`` are required and
    must be disjoint subsets of the internal split. GEPA reflects over the
    trainset and selects its Pareto frontier on the valset, so a valset
    that repeated the trainset would score candidates on the very tasks
    their reflection was written from -- the selection would then reward
    memorization rather than generalization.
    """
    prompt_adapter = PlainPromptAdapter()
    internal_task_hashes = (
        experiment.eval_configs.internal.task_set.task_hashes
    )
    if engine.sampling.task_hashes != internal_task_hashes:
        raise ValueError("GEPA trainset must be the internal eval split")
    trainset, valset = require_disjoint_split(
        trainset_task_hashes=trainset_task_hashes,
        valset_task_hashes=valset_task_hashes,
        task_hashes=tuple(internal_task_hashes),
        optimizer="GEPA",
    )
    return configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=provider_call_config_ref(experiment),
        ),
        metric=engine.eval_config_ref,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        evaluation_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        proposal_execution_policy_hash=engine.execution_policy_identity_hash(),
        proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
        proposal_durability_policy_identity_hash=policy_identity_hash,
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_format_identity_hash=prompt_services.descriptor.identity_hash(),
        prompt_binding_identity_hash=prompt_services.binding.identity_hash(),
        trainset_task_hashes=trainset,
        valset_task_hashes=valset,
        component_names=(GEPA_COMPONENT_NAME,),
        num_predictors=1,
        # One full pass to score the seed, plus one reflection minibatch.
        # With a distinct valset the seed is scored on both sets, so the
        # full pass is train + val rather than the trainset alone.
        max_metric_calls=(
            len(trainset) + len(valset) + 1
            if max_metric_calls is None
            else max_metric_calls
        ),
        reflection_minibatch_size=(
            DEFAULT_REFLECTION_MINIBATCH_SIZE
            if reflection_minibatch_size is None
            else reflection_minibatch_size
        ),
        seed=seed,
    ).model_copy(update={"mutation_field": family.mutation_field})


def gepa_policy_identity_hash(family: FamilySpec) -> str:
    """The identity of one family's inline GEPA proposal executor policy."""
    return compute_identity_hash(
        schema=f"{family.namespace}.{INLINE_EXECUTOR_SCHEMA_SUFFIX}",
        schema_version=1,
        payload={"mode": "inline"},
    )


def build_gepa_adapter(  # noqa: PLR0913
    *,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    family: FamilySpec,
    run_id: str,
    proposer_transport: ProposerTransport | None,
    seed: int = 0,
    max_metric_calls: int | None = None,
    reflection_minibatch_size: int | None = None,
    trainset_task_hashes: tuple[str, ...],
    valset_task_hashes: tuple[str, ...],
) -> GepaHarnessAdapter:
    """Assemble one family's GEPA adapter on the public factory surface."""
    prompt_adapter = PlainPromptAdapter()
    prompt_services = gepa_prompt_services(family)
    policy_identity_hash = gepa_policy_identity_hash(family)
    control = build_gepa_control(
        engine=engine,
        experiment=experiment,
        family=family,
        prompt_services=prompt_services,
        policy_identity_hash=policy_identity_hash,
        seed=seed,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        trainset_task_hashes=trainset_task_hashes,
        valset_task_hashes=valset_task_hashes,
    )
    return build_gepa_harness_adapter(
        store=store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
        mutation_field=family.mutation_field,
        prompt_services=prompt_services,
        transport=_gepa_transport(
            engine=engine,
            prompt_adapter=prompt_adapter,
            proposal_bodies=family.proposal_bodies(),
            proposer_transport=proposer_transport,
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=policy_identity_hash,
        ),
    )


__all__ = [
    "COMPONENT_SCHEMA_SUFFIX",
    "DEFAULT_REFLECTION_MINIBATCH_SIZE",
    "GEPA_COMPONENT_NAME",
    "INLINE_EXECUTOR_SCHEMA_SUFFIX",
    "build_gepa_adapter",
    "build_gepa_control",
    "gepa_policy_identity_hash",
    "gepa_prompt_services",
]
