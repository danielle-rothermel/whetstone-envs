"""MIPROv2 wiring on whetstone-ai's public MIPROv2 surface.

MIPROv2 differs from COPRO and GEPA in that its control alone cannot start a
run: the search reads a labeled trainset, rendered proposal examples, and a
durable RNG checkpoint, all of which belong to the opening state. This module
resolves the control, builds the adapter, and derives that opening state from
the same prepared experiment the other optimizers use, for whichever family
the run names.

Demonstrations reach the candidate through MIPROv2's own composed template
(``### Demonstrations``), not through a placeholder in the family's render
contract: the composer emits the section itself and escapes its JSON so the
composed template still satisfies the family's placeholder requirement.
``fewshot`` searches over demo sets and renders the selected set into that
section; ``zeroshot`` and ``ground_only`` keep the section empty while still
bootstrapping demos to ground instruction proposals.

Minibatching is off. Every family's internal split is the valset, and
``configure_miprov2`` refuses a ``minibatch_size`` exceeding it -- C18's
internal split of 24 is below MIPROv2's default 35 -- so a study that turns
minibatching on must size it against the run's own valset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import ImmutableJsonObject, compute_identity_hash
from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import (
    OptimRun,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    Miprov2Adapter,
)
from whetstone.optim.miprov2.control import (
    Miprov2ComponentSpec,
    Miprov2Control,
    Miprov2DemoMode,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optim.miprov2.demo import LabeledTaskDemo
from whetstone.optim.miprov2.engine_binding import EngineEvalBindingResolver
from whetstone.optim.miprov2.proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
)
from whetstone.optim.miprov2.rng import Miprov2DurableBindings
from whetstone.optim.miprov2.runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2State,
)
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.optim.experiment import (
    gold_by_task_hash,
    provider_call_config_ref,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.contracts import OptimRunRef
    from whetstone.optim.proposal.proposer import ProposerTransport

    from whetstone_envs.optim.families import FamilySpec

#: The single optimizable component: every family's rollout graph exposes
#: exactly one provider trace, so MIPROv2 optimizes one prompt component.
MIPROV2_COMPONENT_ID = "generate"
#: MIPROv2 partitions the engine's tasks into a trainset and a valset, so a
#: run needs at least one task for each.
MIN_MIPROV2_TASKS = 2
#: The family-namespaced schema name for MIPROv2's inline executor policy.
INLINE_EXECUTOR_SCHEMA_SUFFIX = "miprov2_proposal_executor"
#: Minibatching is off by default: every trial then evaluates the whole
#: validation split, which is the schedule this runner has always produced.
#: The protocol's auto-light configuration turns it on explicitly.
DEFAULT_MIPROV2_MINIBATCH = False
#: Trials between full-validation re-evaluations of the incumbent. Only
#: observable once ``minibatch`` is on.
DEFAULT_MIPROV2_FULL_EVAL_STEPS = 1

DEMO_MODES = tuple(mode.value for mode in Miprov2DemoMode)


def miprov2_policy_identity_hash(family: FamilySpec) -> str:
    """One family's inline MIPROv2 proposal executor policy identity."""
    return compute_identity_hash(
        schema=f"{family.namespace}.{INLINE_EXECUTOR_SCHEMA_SUFFIX}",
        schema_version=1,
        payload={"mode": "inline"},
    )


def _demo_maxima(
    demo_mode: Miprov2DemoMode,
) -> tuple[int, int]:
    """The demo maxima each mode's control must carry.

    A zero-shot control must carry zero maxima; the bootstrapping modes need
    at least one demo to bootstrap.
    """
    if demo_mode is Miprov2DemoMode.ZEROSHOT:
        return 0, 0
    return 1, 1


def build_miprov2_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    family: FamilySpec,
    demo_mode: Miprov2DemoMode = Miprov2DemoMode.FEWSHOT,
    num_trials: int = 2,
    # Seeds -3/-2 are RESET/LABELS_ONLY; 3 admits seed -1, the first
    # bootstrap candidate.
    num_candidates: int = 3,
    seed: int = 9,
    minibatch: bool = DEFAULT_MIPROV2_MINIBATCH,
    minibatch_size: int | None = None,
    minibatch_full_eval_steps: int = DEFAULT_MIPROV2_FULL_EVAL_STEPS,
) -> Miprov2Control:
    """Resolve one family's MIPROv2 control against the engine.

    ``minibatch_size`` defaults to the whole validation split, which is what
    a non-minibatched run evaluates on every trial. The protocol's
    auto-light configuration turns ``minibatch`` on and sizes the batch
    below that, so the periodic full evaluation becomes observable.
    """
    prompt_adapter = PlainPromptAdapter()
    task_hashes = tuple(engine.sampling.task_hashes)
    if len(task_hashes) < MIN_MIPROV2_TASKS:
        raise ValueError(
            "MIPROv2 needs at least two tasks to split train and val"
        )
    bootstrapped, labeled = _demo_maxima(demo_mode)
    valset = task_hashes[1:]
    resolved_minibatch_size = (
        len(valset) if minibatch_size is None else minibatch_size
    )
    if resolved_minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1")
    if resolved_minibatch_size > len(valset):
        # ``configure_miprov2`` refuses a batch larger than the valset, and
        # refusing here keeps the failure outside the durable run boundary.
        raise ValueError(
            f"minibatch_size {resolved_minibatch_size} exceeds the "
            f"validation split of {len(valset)}"
        )
    if minibatch_full_eval_steps < 1:
        raise ValueError("minibatch_full_eval_steps must be at least 1")
    defaults = Miprov2InjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=provider_call_config_ref(experiment),
            temperature=1.0,
        ),
        bootstrap_eval_source=engine.eval_config_ref,
        validation_eval_source=engine.eval_config_ref,
        reward_policy=experiment.reward_policy,
        eval_role=EvalRole.INTERNAL,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
        provider_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_adapter=prompt_adapter,
        template_render_contract=family.render_contract(),
        mutation_field=family.mutation_field,
        max_errors=4,
        validation_eval_source_is_metric_authority=True,
    )
    layout = Miprov2ProgramLayout(
        layout_id=f"{family.namespace}.miprov2",
        component_specs=(
            Miprov2ComponentSpec(
                component_id=MIPROV2_COMPONENT_ID,
                prompt_format_identity_hash=prompt_adapter_identity_hash(
                    prompt_adapter
                ),
            ),
        ),
    )
    return configure_miprov2(
        base_candidate=candidate_reference(experiment.initial_candidate),
        program_layout=layout,
        trainset=task_hashes[:1],
        valset=task_hashes[1:],
        max_bootstrapped_demos=bootstrapped,
        max_labeled_demos=labeled,
        auto=None,
        num_candidates=num_candidates,
        num_trials=num_trials,
        seed=seed,
        init_temperature=1.0,
        minibatch=minibatch,
        minibatch_size=resolved_minibatch_size,
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        demo_mode=demo_mode,
        defaults=defaults,
    )


def build_miprov2_adapter(
    *,
    store: ObjectStore,
    engine: EvalEngine,
    control: Miprov2Control,
    family: FamilySpec,
    proposer_transport: ProposerTransport | None = None,
) -> Miprov2Adapter:
    """Assemble one family's MIPROv2 adapter on the public adapter surface.

    MIPROv2 rejects an instruction that drops a placeholder the base
    template requires, so the scripted bodies are the family's own probe
    templates, which satisfy its render contract by construction.
    """
    prompt_adapter = PlainPromptAdapter()
    bodies = family.proposal_bodies()
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=bodies,
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
    )
    return Miprov2Adapter(
        store=store,
        proposer_config=control.prompt_model,
        transport=transport,
        eval_config_resolver=EngineEvalBindingResolver(engine=engine),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=miprov2_policy_identity_hash(family),
        ),
    )


def miprov2_budget() -> Miprov2EffectBudget:
    """A ceiling generous enough that a small run ends on its schedule.

    These are budgets, not expectations: the run should terminate because its
    trial schedule is exhausted, so budget exhaustion is a real signal.
    """
    return Miprov2EffectBudget(
        bootstrap_generations=32,
        proposal_calls=32,
        evaluations=32,
        task_rows=256,
    )


def miprov2_run(
    *,
    run_id: str,
    control: Miprov2Control,
    experiment: Experiment,
) -> OptimRun:
    """The exact OptimRun the opening MIPROv2 state must bind."""
    return OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=MIPROV2_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=control.template_render_contract,
        initial_candidate_ref=control.base_candidate,
        mutation_field=control.mutation_field,
        reward_policy=experiment.reward_policy,
    )


def build_miprov2_state(  # noqa: PLR0913
    *,
    run: OptimRunRef,
    control: Miprov2Control,
    engine: EvalEngine,
    experiment: Experiment,
    adapter: Miprov2Adapter,
    family: FamilySpec,
    budget: Miprov2EffectBudget | None = None,
) -> Miprov2State:
    """Derive the opening MIPROv2 state for one run of one family."""
    bindings = Miprov2DurableBindings(
        control_identity_hash=control.identity_hash(),
        prompt_route_identity_hash=control.prompt_model.identity_hash(),
        task_route_identity_hash=control.task_model_identity_hash,
        execution_policy_identity_hash=(
            control.provider_execution_policy_hash
        ),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        proposal_executor_policy_identity_hash=(
            adapter.proposal_executor_policy_identity_hash
        ),
        proposal_transport_durability_identity_hash=(
            adapter.proposal_transport_durability_identity_hash
        ),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        teacher_candidate_identity_hash=(
            control.teacher_candidate.identity_hash
        ),
    )
    component_id = control.component_ids[0]
    inputs_by_hash = {
        task.task_hash: dict(task.prompt_inputs)
        for task in engine.sampling.tasks
    }
    # The engine's sampling view withholds gold, so a labeled demonstration
    # reads its output from the family's own experiment splits. A demo with
    # an empty output would teach the model to answer with nothing.
    gold_by_hash = gold_by_task_hash(experiment)
    missing = set(control.trainset_task_hashes) - set(gold_by_hash)
    if missing:
        raise ValueError(
            "MIPROv2 trainset tasks have no gold in the experiment: "
            f"{', '.join(sorted(missing))}"
        )
    labeled_trainset = tuple(
        LabeledTaskDemo(
            source_task_hash=task_hash,
            inputs_by_component=ImmutableJsonObject(
                {component_id: dict(inputs_by_hash[task_hash])}
            ),
            outputs_by_component=ImmutableJsonObject(
                {
                    component_id: {
                        family.response_field: gold_by_hash[task_hash]
                    }
                }
            ),
        )
        for task_hash in control.trainset_task_hashes
    )
    template = control.base_candidate.record.payload[control.mutation_field]
    proposal_components = (
        Miprov2PromptComponent(
            component_id=component_id,
            template=str(template),
            template_render_contract=control.template_render_contract,
            rendering_rules=family.rendering_rules,
            example_execution=family.example_execution,
        ),
    )
    proposal_trainset = tuple(
        Miprov2DatasetExample(
            task_hash=task_hash,
            rendered_record="\n".join(
                f"{name}: {value}"
                for name, value in sorted(inputs_by_hash[task_hash].items())
            ),
        )
        for task_hash in control.trainset_task_hashes
    )
    return Miprov2Driver().start(
        run=run,
        control=control,
        bindings=bindings,
        labeled_trainset=labeled_trainset,
        proposal_components=proposal_components,
        proposal_trainset=proposal_trainset,
        # MIPROv2 records a component's input field order canonically.
        component_field_order={
            component_id: tuple(sorted(family.prompt_fields))
        },
        budget=budget or miprov2_budget(),
    )


def miprov2_run_ref(
    *,
    run_id: str,
    control: Miprov2Control,
    experiment: Experiment,
) -> OptimRunRef:
    return optimization_run_reference(
        miprov2_run(
            run_id=run_id,
            control=control,
            experiment=experiment,
        )
    )


__all__ = [
    "DEFAULT_MIPROV2_FULL_EVAL_STEPS",
    "DEFAULT_MIPROV2_MINIBATCH",
    "DEMO_MODES",
    "INLINE_EXECUTOR_SCHEMA_SUFFIX",
    "MIPROV2_COMPONENT_ID",
    "Miprov2DemoMode",
    "build_miprov2_adapter",
    "build_miprov2_control",
    "build_miprov2_state",
    "miprov2_budget",
    "miprov2_policy_identity_hash",
    "miprov2_run",
    "miprov2_run_ref",
]
