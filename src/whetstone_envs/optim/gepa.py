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

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_PROMPT_FIELDS,
    c19_provider_call_config_ref,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.gepa.harness_adapter import GepaHarnessAdapter
    from whetstone.optim.proposal.proposer import ProposerTransport

GEPA_COMPONENT_NAME = "generate"
_INLINE_EXECUTOR_SCHEMA = "whetstone_envs.c19.inline_proposal_executor"
_COMPONENT_SCHEMA = "whetstone_envs.c19.gepa_component"


def c19_gepa_prompt_services() -> GepaPromptServices:
    """Reflection prompt services bound to the C19 placeholder contract.

    The shared ``default_gepa_prompt_services`` binds a single ``{prompt}``
    placeholder; C19 templates carry three, so the component format is
    declared here rather than taken from the default.
    """
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


def build_c19_gepa_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    prompt_services: GepaPromptServices,
    policy_identity_hash: str,
    seed: int = 0,
    max_metric_calls: int | None = None,
):
    """Resolve the C19 GEPA control over the engine's internal split.

    ``seed`` is GEPA's explicit algorithmic seed, carried straight onto the
    control. ``max_metric_calls`` pins the paid metric-call ceiling; ``None``
    keeps the default of one full pass over the trainset plus one reflection
    minibatch, which is what a smoke run needs.
    """
    prompt_adapter = PlainPromptAdapter()
    task_hashes = experiment.eval_configs.internal.task_set.task_hashes
    if engine.sampling.task_hashes != task_hashes:
        raise ValueError("GEPA trainset must be the internal eval split")
    return configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=c19_provider_call_config_ref(experiment),
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
        max_metric_calls=(
            len(task_hashes) + 1
            if max_metric_calls is None
            else max_metric_calls
        ),
        reflection_minibatch_size=1,
        seed=seed,
    ).model_copy(update={"mutation_field": C19_MUTATION_FIELD})


def build_c19_gepa_adapter(  # noqa: PLR0913
    *,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
    proposer_transport: ProposerTransport | None,
    seed: int = 0,
    max_metric_calls: int | None = None,
) -> GepaHarnessAdapter:
    """Assemble a real C19 GEPA adapter on the public factory surface."""
    prompt_adapter = PlainPromptAdapter()
    prompt_services = c19_gepa_prompt_services()
    policy_identity_hash = compute_identity_hash(
        schema=_INLINE_EXECUTOR_SCHEMA,
        schema_version=1,
        payload={"mode": "inline"},
    )
    control = build_c19_gepa_control(
        engine=engine,
        experiment=experiment,
        prompt_services=prompt_services,
        policy_identity_hash=policy_identity_hash,
        seed=seed,
        max_metric_calls=max_metric_calls,
    )
    return build_gepa_harness_adapter(
        store=store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
        mutation_field=C19_MUTATION_FIELD,
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


__all__ = [
    "GEPA_COMPONENT_NAME",
    "build_c19_gepa_adapter",
    "build_c19_gepa_control",
    "c19_gepa_prompt_services",
]
