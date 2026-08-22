"""C19 MIPROv2 wiring on whetstone-ai's public MIPROv2 surface.

MIPROv2 differs from COPRO and GEPA in that its control alone cannot start a
run: the search reads a labeled trainset, rendered proposal examples, and a
durable RNG checkpoint, all of which belong to the opening state. This module
resolves the control, builds the adapter, and derives that opening state from
the same prepared C19 experiment the other optimizers use.

Demonstrations reach the candidate through MIPROv2's own composed template
(``### Demonstrations``), not through a placeholder in the C19 render
contract: the composer emits the section itself and escapes its JSON so the
composed template still satisfies the contract's ``{grid}``/``{command}``/
``{question}`` requirement. ``fewshot`` searches over demo sets and renders
the selected set into that section; ``zeroshot`` and ``ground_only`` keep the
section empty while still bootstrapping demos to ground instruction
proposals.
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

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_NAMESPACE,
    C19_PROMPT_FIELDS,
    c19_render_contract,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.contracts import OptimRunRef
    from whetstone.optim.proposal.proposer import ProposerTransport

#: The single optimizable component: the C19 rollout graph exposes exactly
#: one provider trace, so MIPROv2 optimizes one prompt component.
MIPROV2_COMPONENT_ID = "generate"
#: MIPROv2 partitions the engine's tasks into a trainset and a valset, so a
#: run needs at least one task for each.
MIN_MIPROV2_TASKS = 2
#: MIPROv2 records a component's input field order canonically.
C19_PROMPT_FIELDS_SORTED = tuple(sorted(C19_PROMPT_FIELDS))
_INLINE_EXECUTOR_SCHEMA = "whetstone_envs.c19.miprov2_proposal_executor"

C19_DEMO_MODES = tuple(mode.value for mode in Miprov2DemoMode)


def c19_miprov2_policy_identity_hash() -> str:
    """The identity of C19's inline MIPROv2 proposal executor policy."""
    return compute_identity_hash(
        schema=_INLINE_EXECUTOR_SCHEMA,
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


def build_c19_miprov2_control(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    experiment: Experiment,
    demo_mode: Miprov2DemoMode = Miprov2DemoMode.FEWSHOT,
    num_trials: int = 2,
    # Seeds -3/-2 are RESET/LABELS_ONLY; 3 admits seed -1, the first
    # bootstrap candidate.
    num_candidates: int = 3,
    seed: int = 9,
) -> Miprov2Control:
    """Resolve the C19 MIPROv2 control against the engine's authorities."""
    prompt_adapter = PlainPromptAdapter()
    task_hashes = tuple(engine.sampling.task_hashes)
    if len(task_hashes) < MIN_MIPROV2_TASKS:
        raise ValueError(
            "C19 MIPROv2 needs at least two tasks to split train and val"
        )
    bootstrapped, labeled = _demo_maxima(demo_mode)
    defaults = Miprov2InjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
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
        template_render_contract=c19_render_contract(),
        mutation_field=C19_MUTATION_FIELD,
        max_errors=4,
        validation_eval_source_is_metric_authority=True,
    )
    layout = Miprov2ProgramLayout(
        layout_id=f"{C19_NAMESPACE}.miprov2",
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
        minibatch=False,
        minibatch_size=len(task_hashes[1:]),
        minibatch_full_eval_steps=1,
        demo_mode=demo_mode,
        defaults=defaults,
    )


def c19_miprov2_proposal_bodies() -> tuple[str, ...]:
    """Scripted proposer bodies for a fake-transport C19 MIPROv2 run.

    MIPROv2 rejects an instruction that drops a placeholder the base template
    requires, so every scripted instruction keeps ``{grid}``, ``{command}``,
    and ``{question}``.
    """
    return (PROBES.ceiling_template, PROBES.naive_template)


def build_c19_miprov2_adapter(
    *,
    store: ObjectStore,
    engine: EvalEngine,
    control: Miprov2Control,
    proposer_transport: ProposerTransport | None = None,
) -> Miprov2Adapter:
    """Assemble the C19 MIPROv2 adapter on the public adapter surface."""
    prompt_adapter = PlainPromptAdapter()
    bodies = c19_miprov2_proposal_bodies()
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
            policy_identity_hash=c19_miprov2_policy_identity_hash(),
        ),
    )


def c19_miprov2_budget() -> Miprov2EffectBudget:
    """A ceiling generous enough that a small C19 run ends on its schedule.

    These are budgets, not expectations: the run should terminate because its
    trial schedule is exhausted, so budget exhaustion is a real signal.
    """
    return Miprov2EffectBudget(
        bootstrap_generations=32,
        proposal_calls=32,
        evaluations=32,
        task_rows=256,
    )


def c19_miprov2_run(
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


def build_c19_miprov2_state(
    *,
    run: OptimRunRef,
    control: Miprov2Control,
    engine: EvalEngine,
    adapter: Miprov2Adapter,
    budget: Miprov2EffectBudget | None = None,
) -> Miprov2State:
    """Derive the opening MIPROv2 state for one C19 run."""
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
    labeled_trainset = tuple(
        LabeledTaskDemo(
            source_task_hash=task_hash,
            inputs_by_component=ImmutableJsonObject(
                {component_id: dict(inputs_by_hash[task_hash])}
            ),
            outputs_by_component=ImmutableJsonObject(
                {component_id: {"response": ""}}
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
            rendering_rules=(
                "Render the template with the task's grid, command, and "
                "question substituted for its placeholders."
            ),
            example_execution=(
                "The rendered prompt is sent to the task model and its reply "
                "is scored by exact match against the MiniGrid oracle answer."
            ),
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
        component_field_order={component_id: C19_PROMPT_FIELDS_SORTED},
        budget=budget or c19_miprov2_budget(),
    )


def c19_miprov2_run_ref(
    *,
    run_id: str,
    control: Miprov2Control,
    experiment: Experiment,
) -> OptimRunRef:
    return optimization_run_reference(
        c19_miprov2_run(
            run_id=run_id,
            control=control,
            experiment=experiment,
        )
    )


__all__ = [
    "C19_DEMO_MODES",
    "MIPROV2_COMPONENT_ID",
    "Miprov2DemoMode",
    "build_c19_miprov2_adapter",
    "build_c19_miprov2_control",
    "build_c19_miprov2_state",
    "c19_miprov2_budget",
    "c19_miprov2_policy_identity_hash",
    "c19_miprov2_proposal_bodies",
    "c19_miprov2_run",
    "c19_miprov2_run_ref",
]
