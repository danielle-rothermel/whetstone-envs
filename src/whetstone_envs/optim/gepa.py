from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA
from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvalAuthority,
    CanonicalGepaProposalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.factory import CanonicalGepaAdapterFactory
from whetstone.optim.gepa.harness_adapter import (
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
)
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

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_PROMPT_FIELDS,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.proposal.proposer import ProposerTransport

GEPA_COMPONENT_NAME = "generate"
_INLINE_EXECUTOR_SCHEMA = "whetstone_envs.c19.inline_proposal_executor"
_COMPONENT_SCHEMA = "whetstone_envs.c19.gepa_component"


def _c19_prompt_services() -> GepaPromptServices:
    component_schema_hash = compute_identity_hash(
        schema=_COMPONENT_SCHEMA,
        schema_version=1,
        payload={"field": C19_MUTATION_FIELD},
    )
    descriptor = GepaPromptFormatDescriptor(
        format_name="c19_prompt_template",
        components=(
            GepaComponentFormat(
                component_name=GEPA_COMPONENT_NAME,
                component_schema_identity_hash=component_schema_hash,
                allowed_placeholders=C19_PROMPT_FIELDS,
                required_placeholders=C19_PROMPT_FIELDS,
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


def _provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    payload = experiment.rollout_graph.provider_call_config.model_dump(
        mode="json"
    )
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def build_c19_gepa_adapter(
    *,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
    proposer_transport: ProposerTransport | None,
) -> GepaHarnessAdapter:
    """Assemble a real C19 GEPA adapter on the public factory surface."""
    prompt_adapter = PlainPromptAdapter()
    prompt_services = _c19_prompt_services()
    policy_identity_hash = compute_identity_hash(
        schema=_INLINE_EXECUTOR_SCHEMA,
        schema_version=1,
        payload={"mode": "inline"},
    )
    task_hashes = experiment.eval_configs.internal.task_set.task_hashes
    if engine.sampling.task_hashes != task_hashes:
        raise ValueError("GEPA trainset must be the internal eval split")
    control = configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=_provider_call_config_ref(experiment),
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
        trainset_task_hashes=task_hashes,
        valset_task_hashes=None,
        component_names=(GEPA_COMPONENT_NAME,),
        num_predictors=1,
        # One full pass to score the seed, plus one reflection minibatch.
        max_metric_calls=len(task_hashes) + 1,
        reflection_minibatch_size=1,
    ).model_copy(update={"mutation_field": C19_MUTATION_FIELD})
    registry = GepaDataRegistry.from_engine(store=store, engine=engine)
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=candidate_reference(experiment.initial_candidate),
        fields=(
            GepaCandidateFieldBinding(
                component_name=GEPA_COMPONENT_NAME,
                candidate_field=C19_MUTATION_FIELD,
            ),
        ),
    )
    evaluation_authority = CanonicalGepaEvalAuthority(
        store=store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    proposal_authority = CanonicalGepaProposalAuthority(
        store=store,
        control=control,
        prompt_services=prompt_services,
        transport=_gepa_transport(
            engine=engine,
            prompt_adapter=prompt_adapter,
            proposal_bodies=(
                PROBES.ceiling_template,
                PROBES.naive_template,
            ),
            proposer_transport=proposer_transport,
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=policy_identity_hash,
        ),
    )
    factory = CanonicalGepaAdapterFactory(
        store=store,
        run_id=run_id,
        control=control,
        evaluation_authority=evaluation_authority,
        proposal_authority=proposal_authority,
        prompt_services=prompt_services,
    )
    return GepaHarnessAdapter(
        control=control,
        seed_candidate={GEPA_COMPONENT_NAME: PROBES.naive_template},
        trainset=registry.entries,
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )


__all__ = [
    "GEPA_COMPONENT_NAME",
    "build_c19_gepa_adapter",
]
