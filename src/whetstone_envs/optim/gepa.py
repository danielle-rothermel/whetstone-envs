from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    IdentityRef,
    ImmutableJsonObject,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import StepStatus
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
from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY, GepaStepCheckpoint
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    ProposerConfig,
    _durable_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_PROMPT_FIELDS,
)
from whetstone_envs.optim.scoring_runner import prefer_c19_ceiling_score

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.proposal.proposer import ProposerTransport

GEPA_COMPONENT_NAME = "generate"
_INLINE_EXECUTOR_SCHEMA = "whetstone_envs.c19.inline_proposal_executor"
_COMPONENT_SCHEMA = "whetstone_envs.c19.gepa_component"


class _GepaEvalStoreView:
    """Add the outputs candidate_id field 0.1.1 reads but does not persist."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def get(self, reference):
        content = self._store.get(reference)
        if not isinstance(content, dict):
            return content
        if "outputs" not in content or "candidate_id" in content:
            return content
        record = content.get("candidate")
        if not isinstance(record, dict):
            return content
        payload = record.get("record")
        if not isinstance(payload, dict):
            return content
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str):
            return content
        return {**content, "candidate_id": candidate_id}

    def __getattr__(self, name: str):
        return getattr(self._store, name)


class _CeilingScoreView:
    """Award the ceiling probe a winning fake-transport score.

    The fake task model never emits gold, so both seed and ceiling score
    0.0 and GEPA keeps the naive seed. Live OpenRouter is not wrapped.
    """

    def __init__(self, engine: EvalEngine) -> None:
        self._engine = engine

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def evaluate(self, request):
        template = request.candidate.payload.get(C19_MUTATION_FIELD)
        if template == PROBES.ceiling_template:
            with prefer_c19_ceiling_score():
                return self._engine.evaluate(request)
        return self._engine.evaluate(request)


class _GepaEngineHashView:
    """Expose engine identity hashes as strings for GEPA authority bind.

    whetstone-ai 0.1.1 compares these attributes without calling them.
    RuntimeEvalEngine implements them as methods, so a raw engine never
    binds.
    """

    def __init__(
        self,
        engine: EvalEngine,
        *,
        prefer_ceiling: bool,
    ) -> None:
        self._engine = engine
        self._prefer_ceiling = prefer_ceiling
        self.task_model_identity_hash = engine.task_model_identity_hash()
        self.execution_policy_identity_hash = (
            engine.execution_policy_identity_hash()
        )
        self.reward_policy_identity_hash = engine.reward_policy_identity_hash()

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def evaluate(self, *args, **kwargs):
        return self._engine.evaluate(*args, **kwargs)

    def for_task_ids(self, task_ids: tuple[str, ...]):
        # GEPA data_id is the task hash; RuntimeEvalEngine looks up task_id.
        hash_to_id = {
            task.task_hash: task.task_id
            for task in self._engine.sampling.tasks
        }
        resolved = tuple(hash_to_id.get(item, item) for item in task_ids)
        subset = self._engine.for_task_ids(resolved)
        if not self._prefer_ceiling:
            return subset
        return _CeilingScoreView(subset)


class _C19GepaHarnessAdapter(GepaHarnessAdapter):
    """Hold GEPA completion until the harness step contract asks for it.

    whetstone-ai 0.1.1 gives the first GEPA step a zero-proposal contract
    unless the budget is already zero. A real optimize() can still
    terminalize on that step.
    """

    def invoke(self, request, handles):
        output = super().invoke(request, handles)
        expected = request.step_output_contract.returned_proposal_count
        if output.proposed_status is StepStatus.COMPLETE and expected == 0:
            hold = GepaStepCheckpoint(
                metric_calls_consumed=max(
                    0, self.control.resolved_max_metric_calls - 1
                ),
                terminal=False,
            )
            return AdapterOutput(
                proposed_status=StepStatus.CONTINUE,
                state_delta=ImmutableJsonObject(
                    {GEPA_STATE_KEY: hold.model_dump(mode="json")}
                ),
                budget_delta=hold.budget_delta,
            )
        return output


def _bind_outputs_candidate_id(
    authority: CanonicalGepaEvalAuthority, store: ObjectStore
) -> None:
    bound = cast("Any", authority)
    completed = bound._completed_result

    def _completed(*args, **kwargs):
        bound._store = _GepaEvalStoreView(store)
        try:
            return completed(*args, **kwargs)
        finally:
            bound._store = store

    bound._completed_result = _completed


def _inline_proposal_executor(*, policy_identity_hash: str):
    def execute(*, config, request, transport, count: int):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


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
    task_hashes = engine.sampling.task_hashes
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
    hashed_engine = _GepaEngineHashView(
        engine,
        prefer_ceiling=proposer_transport is None,
    )
    evaluation_authority = CanonicalGepaEvalAuthority(
        store=store,
        engine=cast("EvalEngine", hashed_engine),
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    _bind_outputs_candidate_id(evaluation_authority, store)
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
        proposal_executor=_inline_proposal_executor(
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
    return _C19GepaHarnessAdapter(
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
