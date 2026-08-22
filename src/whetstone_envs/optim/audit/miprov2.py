"""The eight MIPROv2 fidelity invariants.

MIPROv2's claim is a specific algorithm: bootstrap demonstrations from the
trainset, ground instruction proposals in them, then search the resulting
instruction (and, in ``fewshot``, demo) space with a seeded Optuna TPE
sampler over minibatches, periodically evaluating the incumbent on the full
validation set. These invariants judge that claim against what one run
actually persisted, and nothing else.

Two spellings are load-bearing and are read from whetstone rather than
restated here: ``MIPROV2_ALGORITHM_VERSION`` (``dspy_miprov2/v2``, the
frozen DSPy behaviour the two faithful modes claim) and
``StudyTranscript.whetstone_deviation`` (``demo_mode:ground_only``, the
marker the third mode carries instead of silently claiming faithfulness).

Read paths, verified against whetstone-ai ``miprofix-ai`` @ ``716976f2``
on real fake-transport runs in all three demo modes:

- **Which kind of evaluation an intent is.** Every MIPROv2 evaluation
  reaches the engine as an ``IntentResolution`` whose
  ``optim_eval_request.eval_request.metadata`` carries the purpose written
  by ``metadata_with_purpose``. ``whetstone.eval.metadata.eval_purpose``
  reads it back. The four spellings are ``miprov2_bootstrap``,
  ``miprov2_baseline``, ``miprov2_sample``, and ``miprov2_promotion``.
  This is the only run-level classifier that does not require decoding
  optimizer state, so it is what the ordering and sizing invariants key on.
- **The instruction-proposal calls.** ``Miprov2State.proposal_state``
  carries ``evidence: tuple[Miprov2ProposalEvidence, ...]``; each entry's
  ``request.effect`` names the proposer effect and ``request.effect_ordinal``
  orders them. ``instruction_proposal`` is the effect this module means by
  "instruction proposal"; the dataset and description effects are the
  grounding prelude to it.
- **The demo regime.** ``Miprov2State.control.demo_mode`` and
  ``StudyTranscript.demo_mode`` both record it, and
  ``StudyTranscript.whetstone_deviation`` derives the deviation marker.
  ``Miprov2State.bootstrap_plans`` carries the per-plan demo caps, which is
  where ``zeroshot``'s 3/0 grounding bootstrap is durably witnessed --
  ``control.max_bootstrapped_demos`` stays 0 in that mode by design.
- **The search.** ``StudyTranscript`` is inline on ``Miprov2State``, and
  ``Miprov2Study.reconstruct_study`` replays it into a fresh seeded
  ``TPESampler``, raising ``StudyTranscriptMismatch`` when the recorded
  suggestion sequence is not what the sampler produces.
- **The effect ledger.** ``Miprov2State.completed_effects`` records every
  paid effect with a ``kind`` of ``bootstrap_generations``,
  ``proposal_calls``, or ``evaluations``, so a bootstrap generation billed
  to the proposer transport rather than the engine is visible as a
  ``bootstrap_generations`` effect with no matching engine intent.

Every invariant here FAILs rather than raising when its evidence is absent:
a run that persisted nothing to judge is a fidelity failure, and an audit
that crashed would report nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from whetstone.eval.metadata import eval_purpose
from whetstone.optim.miprov2.control import MIPROV2_ALGORITHM_VERSION
from whetstone.optim.miprov2.demo_mode import Miprov2DemoMode
from whetstone.optim.miprov2.study import (
    Miprov2Study,
    StudyTranscript,
    StudyTranscriptMismatch,
)

from whetstone_envs.optim.audit._evidence import evidence_ref
from whetstone_envs.optim.audit.schema import (
    AuditFinding,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

if TYPE_CHECKING:
    from whetstone.optim.miprov2.runtime import Miprov2State

    from whetstone_envs.optim.audit._evidence import RunEvidence

#: The purposes MIPROv2 stamps on its evaluation intent metadata.
#:
#: These are whetstone's persisted spellings, read back through
#: ``eval_purpose``. They are pinned by a golden test rather than derived
#: from an enum, because a silent respelling upstream would otherwise make
#: every ordering invariant here vacuously pass on an empty match.
BOOTSTRAP_PURPOSE = "miprov2_bootstrap"
BASELINE_PURPOSE = "miprov2_baseline"
SAMPLE_PURPOSE = "miprov2_sample"
PROMOTION_PURPOSE = "miprov2_promotion"

#: The proposer effect that is an instruction proposal, as recorded on
#: ``Miprov2ProposalRequest.effect``. The other effects (dataset summaries,
#: program and component descriptions) are the grounding prelude.
INSTRUCTION_PROPOSAL_EFFECT = "instruction_proposal"

#: ``Miprov2CompletedEffect.kind`` spellings for the two effect classes the
#: bootstrap-routing invariant distinguishes.
BOOTSTRAP_GENERATION_EFFECT = "bootstrap_generations"
PROPOSAL_CALL_EFFECT = "proposal_calls"

#: DSPy's zero-shot grounding bootstrap: three bootstrapped demonstrations
#: and no labeled ones, used to ground instruction proposals and then
#: discarded. Mirrors ``ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL`` and
#: ``ZERO_SHOT_LABELED_DEMOS_IN_PROPOSAL`` in whetstone's bootstrap planner,
#: restated here because those live on a module this package reaches only
#: for the two numbers.
ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS = 3
ZEROSHOT_GROUNDING_LABELED_DEMOS = 0

#: The deviation marker ``ground_only`` must carry. Derived by
#: ``StudyTranscript.whetstone_deviation``; pinned here so the audit fails
#: if the marker's spelling ever drifts.
GROUND_ONLY_DEVIATION = "demo_mode:ground_only"


def _finding(
    invariant_id: InvariantId,
    status: AuditStatus,
    detail: str,
    refs: tuple[EvidenceRef, ...] = (),
) -> AuditFinding:
    return AuditFinding(
        invariant_id=invariant_id,
        status=status,
        detail=detail,
        evidence_refs=refs,
    )


def _missing(invariant_id: InvariantId, what: str) -> AuditFinding:
    """A FAIL for absent evidence.

    An invariant whose evidence is missing has not been satisfied -- the
    run simply cannot show it did the thing. Reporting that as a failure
    rather than raising keeps the rest of the audit running, and keeps a
    run that persisted nothing from reading as validated.
    """
    return _finding(
        invariant_id,
        AuditStatus.FAIL,
        f"{what}, so this invariant cannot be shown to hold",
    )


def _state_refs(evidence: RunEvidence) -> tuple[EvidenceRef, ...]:
    """Cite every step state the MIPROv2 invariants read through."""
    return tuple(
        evidence_ref(entry.step.state_ref)
        for entry in evidence.steps
        if entry.step.state_ref is not None
    )


def _terminal_state(evidence: RunEvidence) -> Miprov2State | None:
    """The last step's MIPROv2 state, which subsumes every earlier one.

    MIPROv2 persists a complete state snapshot per step, so the latest one
    carries the whole run: the full transcript, the whole effect ledger, and
    every bootstrap plan. Walking backwards rather than taking ``steps[-1]``
    outright means a run that terminalized without writing state on its very
    last step is still auditable from the last step that did.
    """
    for entry in reversed(evidence.steps):
        state = entry.miprov2_state()
        if state is not None:
            return state
    return None


def _terminal_transcript(
    evidence: RunEvidence,
) -> StudyTranscript | None:
    state = _terminal_state(evidence)
    if state is None:
        return None
    return state.study_transcript


def _purposed_intents(
    evidence: RunEvidence,
) -> tuple[tuple[int, int, str], ...]:
    """Every evaluation intent as ``(step_index, position, purpose)``.

    Ordered by step index then resolution position, which is the order the
    run issued them: ``OptimResult.step_results`` is the persisted step
    sequence and ``resolved_intents`` is ordered within a step.
    """
    ordered: list[tuple[int, int, str]] = []
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            purpose = eval_purpose(
                resolution.optim_eval_request.eval_request.metadata
            )
            if purpose is None:
                continue
            ordered.append((entry.index, position, purpose))
    return tuple(ordered)


def _proposal_ordinals(state: Miprov2State) -> tuple[int, ...]:
    """The effect ordinals of this run's instruction-proposal calls."""
    proposal_state = state.proposal_state
    if proposal_state is None:
        return ()
    return tuple(
        entry.request.effect_ordinal
        for entry in proposal_state.evidence
        if entry.request.effect == INSTRUCTION_PROPOSAL_EFFECT
    )


def miprov2_bootstrap_before_proposal(
    evidence: RunEvidence,
) -> AuditFinding:
    """Demonstrations are bootstrapped before instructions are proposed.

    This is MIPROv2's grounding claim: the proposer sees bootstrapped
    demonstrations of the base program, so its instructions are conditioned
    on observed behaviour rather than invented from the signature alone.
    Every demo mode bootstraps -- ``zeroshot`` grounds proposals with a 3/0
    bootstrap it then discards, and ``ground_only`` bootstraps fewshot-sized
    pools it never renders into a candidate -- so this holds in all three.

    The check is an ordering one over two independently persisted
    sequences. Bootstrap evaluations are engine intents, ordered by step
    index; instruction proposals are proposer effects, ordered by
    ``effect_ordinal`` inside the durable proposal state. They cannot be
    compared on one clock, so the invariant uses the fact that whetstone
    advances ``Miprov2State.phase`` from ``bootstrap`` to ``proposal``: the
    last step carrying a bootstrap intent must precede the first step whose
    state has entered the proposal phase.

    Deliberately not checked: the *number* of bootstrap rows. The F10
    derivation bounds it at ``7 x min(max_bootstrapped_demos / p_accept,
    |trainset|)`` -- 28 rows best case and 616 worst at ``|trainset| = 88``
    -- a range wide enough that no row count inside it distinguishes a
    faithful run from a broken one. The protocol's ``6 x 4 x 3 = 72`` is
    wrong in method and is not used. Row counts are a budget gate, not a
    fidelity invariant.
    """
    invariant = InvariantId.MIPRO_BOOTSTRAP_BEFORE_PROPOSAL
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")

    refs = _state_refs(evidence)
    bootstrap_steps = [
        step_index
        for step_index, _position, purpose in _purposed_intents(evidence)
        if purpose == BOOTSTRAP_PURPOSE
    ]
    proposal_ordinals = _proposal_ordinals(state)

    if not bootstrap_steps:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"demo mode {state.control.demo_mode.value!r} bootstraps, "
                f"but no evaluation intent carries purpose "
                f"{BOOTSTRAP_PURPOSE!r}"
            ),
            refs,
        )
    if not proposal_ordinals:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(bootstrap_steps)} bootstrap evaluations ran but the "
                f"proposal state records no "
                f"{INSTRUCTION_PROPOSAL_EFFECT!r} effect"
            ),
            refs,
        )

    first_proposal_step: int | None = None
    for entry in evidence.steps:
        step_state = entry.miprov2_state()
        if step_state is None:
            continue
        if step_state.phase != "bootstrap":
            first_proposal_step = entry.index
            break
    if first_proposal_step is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(proposal_ordinals)} instruction proposals were "
                f"recorded but no step state left the bootstrap phase"
            ),
            refs,
        )

    late = [step for step in bootstrap_steps if step >= first_proposal_step]
    if late:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(late)} bootstrap evaluation(s) ran at or after step "
                f"{first_proposal_step}, where the run had already left the "
                f"bootstrap phase to propose instructions (steps {late})"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {len(bootstrap_steps)} bootstrap evaluations completed by "
            f"step {max(bootstrap_steps)}, before the run entered the "
            f"proposal phase at step {first_proposal_step} and issued "
            f"{len(proposal_ordinals)} instruction proposals"
        ),
        refs,
    )


def miprov2_zeroshot_grounding(evidence: RunEvidence) -> AuditFinding:
    """``zeroshot`` bootstraps 3/0 to ground proposals and ships no demos.

    DSPy's 0-shot MIPROv2 is not "skip the bootstrap": it runs the same
    bootstrap with caps of three bootstrapped and zero labeled
    demonstrations so the grounded proposer sees the program's real
    behaviour, then discards the demos rather than rendering them into a
    candidate. Reproducing that -- rather than the simpler thing of not
    bootstrapping at all -- is the fidelity claim.

    The 3/0 caps are durably witnessed on ``Miprov2State.bootstrap_plans``
    rather than on the control: ``control.max_bootstrapped_demos`` and
    ``max_labeled_demos`` both stay ``0`` in ``zeroshot`` precisely so the
    study never searches demos, and the planner substitutes the grounding
    caps. So this reads the plans, and separately requires that no candidate
    in the transcript carries a rendered ``demo_set``.

    ``NOT_APPLICABLE`` outside ``zeroshot``: this is a mode-specific
    behaviour, and the other two modes are judged by their own invariants.

    The predicate itself lives in :func:`zeroshot_grounding_problems` so it
    can be exercised against evidence a real run cannot produce -- see that
    function's note on why the negative fixture is structured that way.
    """
    invariant = InvariantId.MIPRO_ZEROSHOT_GROUNDING
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")

    refs = _state_refs(evidence)
    demo_mode = state.control.demo_mode
    if demo_mode is not Miprov2DemoMode.ZEROSHOT:
        return _finding(
            invariant,
            AuditStatus.NOT_APPLICABLE,
            (
                f"the zero-shot grounding bootstrap is specific to "
                f"{Miprov2DemoMode.ZEROSHOT.value!r}; this run is "
                f"{demo_mode.value!r}"
            ),
            refs,
        )

    problems = zeroshot_grounding_problems(state)
    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            f"zero-shot grounding is not faithful: {'; '.join(problems)}",
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"zero-shot bootstrapped with grounding caps "
            f"{ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS}/"
            f"{ZEROSHOT_GROUNDING_LABELED_DEMOS} across "
            f"{len(state.bootstrap_plans)} plan(s) while control maxima "
            f"stayed 0/0, and no candidate shipped a demo set"
        ),
        refs,
    )


class GroundingCaps(Protocol):
    """The two demonstration caps the grounding predicate reads.

    Both ``Miprov2Control`` and ``FewshotCandidatePlan`` satisfy this, which
    is the point: the predicate compares a control's maxima against the
    plans' grounding caps without needing either concrete type.
    """

    @property
    def max_bootstrapped_demos(self) -> int: ...

    @property
    def max_labeled_demos(self) -> int: ...


class GroundingEvidence(Protocol):
    """Exactly the state slice :func:`zeroshot_grounding_problems` reads.

    Narrowing the parameter to this rather than to ``Miprov2State`` is what
    lets the negative tests exercise the predicate at all: whetstone refuses
    to build a ``Miprov2State`` that violates zero-shot grounding, so a
    conformant stand-in is the only way to prove the predicate would catch
    one. It also states the read surface precisely, which is the audit's
    real contract with whetstone here.
    """

    @property
    def control(self) -> GroundingCaps: ...

    @property
    def bootstrap_plans(self) -> tuple[GroundingCaps, ...]: ...

    @property
    def study_transcript(self) -> StudyTranscript | None: ...


def zeroshot_grounding_problems(
    state: GroundingEvidence,
) -> tuple[str, ...]:
    """Every way this state departs from DSPy's zero-shot grounding.

    Separated from the finding so it can be tested against evidence a real
    run cannot produce. That distinction matters here more than elsewhere:
    ``Miprov2State`` cross-validates its bootstrap plans against a replay of
    the planner and its transcript against the control, so a zeroshot state
    carrying wrong grounding caps or a rendered demo set is *unreachable* --
    whetstone rejects it at load. The predicate must still be right, because
    it is the audit's only guard if that upstream validation is ever
    relaxed, so its negatives are asserted against constructed states rather
    than against a mutated run that cannot exist.
    """
    problems: list[str] = []
    if state.control.max_bootstrapped_demos != 0:
        problems.append(
            f"control.max_bootstrapped_demos is "
            f"{state.control.max_bootstrapped_demos}, not 0"
        )
    if state.control.max_labeled_demos != 0:
        problems.append(
            f"control.max_labeled_demos is "
            f"{state.control.max_labeled_demos}, not 0"
        )

    plans = state.bootstrap_plans
    if not plans:
        problems.append("no bootstrap plan was recorded")
    else:
        labeled = {plan.max_labeled_demos for plan in plans}
        if labeled != {ZEROSHOT_GROUNDING_LABELED_DEMOS}:
            problems.append(
                f"grounding plans use labeled-demo caps {sorted(labeled)}, "
                f"not {ZEROSHOT_GROUNDING_LABELED_DEMOS}"
            )
        peak = max(plan.max_bootstrapped_demos for plan in plans)
        if peak != ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS:
            problems.append(
                f"grounding plans peak at {peak} bootstrapped demos, not "
                f"{ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS}"
            )

    transcript = state.study_transcript
    if transcript is None:
        problems.append("the run persisted no study transcript")
    else:
        if transcript.demo_pool_identity_hashes is not None:
            problems.append(
                "the parameter space carries a searched demo dimension"
            )
        shipped = components_with_demo_sets(transcript)
        if shipped:
            problems.append(
                f"{shipped} candidate component(s) ship a rendered demo set"
            )
    return tuple(problems)


def components_with_demo_sets(transcript: StudyTranscript) -> int:
    """Count candidate components that render a demonstration set."""
    shipped = 0
    for sample in transcript.samples:
        for component in sample.candidate_assembly.rendering.components:
            if component.demo_set is not None:
                shipped += 1
        promotion = sample.promotion
        if promotion is None:
            continue
        for component in promotion.candidate_assembly.rendering.components:
            if component.demo_set is not None:
                shipped += 1
    return shipped


def miprov2_ground_only_deviation(evidence: RunEvidence) -> AuditFinding:
    """``ground_only`` is flagged as a whetstone deviation, not as DSPy.

    ``ground_only`` bootstraps fewshot-sized demonstration pools and grounds
    instruction proposals in them, but excludes the demo dimension from the
    parameter space and never renders demos into a candidate. That is not
    DSPy behaviour, so the transcript must say so: ``whetstone_deviation``
    carries ``demo_mode:ground_only`` while ``algorithm_version`` stays at
    the frozen ``dspy_miprov2/v2``.

    The honesty this protects is asymmetric, so the invariant runs in every
    mode rather than only in ``ground_only``: an unmarked deviation would
    claim faithfulness it does not have, and a marked faithful mode would
    disclaim faithfulness it does have. Both are fidelity defects, and
    checking only the deviating mode would catch just the first.
    """
    invariant = InvariantId.MIPRO_GROUND_ONLY_DEVIATION
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")
    transcript = state.study_transcript
    refs = _state_refs(evidence)
    if transcript is None:
        return _missing(invariant, "the run persisted no study transcript")

    demo_mode = transcript.demo_mode
    deviation = transcript.whetstone_deviation
    is_ground_only = demo_mode is Miprov2DemoMode.GROUND_ONLY
    expected = GROUND_ONLY_DEVIATION if is_ground_only else None

    problems: list[str] = []
    if deviation != expected:
        problems.append(
            f"demo mode {demo_mode.value!r} carries deviation marker "
            f"{deviation!r}, expected {expected!r}"
        )
    if transcript.algorithm_version != MIPROV2_ALGORITHM_VERSION:
        problems.append(
            f"algorithm_version is {transcript.algorithm_version!r}, not "
            f"{MIPROV2_ALGORITHM_VERSION!r}"
        )
    if is_ground_only:
        if transcript.demo_pool_identity_hashes is not None:
            problems.append(
                "the parameter space carries a searched demo dimension"
            )
        shipped = components_with_demo_sets(transcript)
        if shipped:
            problems.append(
                f"{shipped} candidate component(s) ship a rendered demo set"
            )
        if not state.bootstrap_plans:
            problems.append("no bootstrap plan was recorded")

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            f"the deviation record is wrong: {'; '.join(problems)}",
            refs,
        )
    if is_ground_only:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"ground_only bootstrapped across "
                f"{len(state.bootstrap_plans)} plan(s), searched "
                f"instructions only, shipped no demo set, and is flagged "
                f"{GROUND_ONLY_DEVIATION!r} under "
                f"{MIPROV2_ALGORITHM_VERSION!r}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"faithful mode {demo_mode.value!r} claims "
            f"{MIPROV2_ALGORITHM_VERSION!r} with no deviation marker"
        ),
        refs,
    )


def miprov2_tpe_selection(evidence: RunEvidence) -> AuditFinding:
    """The recorded trials replay from a fresh seeded TPE sampler.

    This is the strongest MIPROv2 invariant available and the one that most
    directly tests the algorithm rather than its bookkeeping. Whetstone
    rebuilds the study from the transcript alone -- a fresh
    ``TPESampler(seed=transcript.seed, multivariate=True)``, the baseline
    trial added at trial zero, then each recorded sample re-asked in order
    -- and raises ``StudyTranscriptMismatch`` when the reconstructed
    suggestion differs from what was stored. A run whose parameters were
    chosen by anything other than that sampler cannot replay.

    The audit builds the study from the persisted transcript's own fields,
    so it never supplies a value the run did not record: a transcript that
    disagreed with itself would fail to construct, which is also a FAIL.
    """
    invariant = InvariantId.MIPRO_TPE_SELECTION
    transcript = _terminal_transcript(evidence)
    refs = _state_refs(evidence)
    if transcript is None:
        return _missing(
            invariant, "the run persisted no MIPROv2 study transcript"
        )

    try:
        study = Miprov2Study(
            seed=transcript.seed,
            demo_mode=transcript.demo_mode,
            space=transcript.parameter_space,
            schedule=transcript.schedule,
            run_id=transcript.run_id,
            validation_task_hashes=transcript.validation_task_hashes,
            validation_eval_source=transcript.validation_eval_source,
            reward_policy_hash=transcript.reward_policy_hash,
            optimizer_config=transcript.optimizer_config,
            prompt_adapter_identity_hash=(
                transcript.prompt_adapter_identity_hash
            ),
            expected_base_candidate=transcript.expected_base_candidate,
            program_layout=transcript.program_layout,
            run=transcript.run,
        )
        study.reconstruct_study(transcript)
    except StudyTranscriptMismatch as error:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"replaying {len(transcript.samples)} trial(s) into a fresh "
                f"seeded TPE sampler diverged: {error}"
            ),
            refs,
        )
    except ValueError as error:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the transcript could not be bound to its own study "
                f"contract for replay: {error}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {len(transcript.samples)} recorded trial(s) replay in "
            f"order from a fresh TPESampler(seed={transcript.seed}, "
            f"multivariate=True) over the persisted parameter space"
        ),
        refs,
    )


def miprov2_minibatch_sizing(evidence: RunEvidence) -> AuditFinding:
    """Trial evaluations use the scheduled batch, drawn from the valset.

    MIPROv2's efficiency claim is that a trial is scored on a minibatch of
    the validation set rather than all of it, so a wrong batch is both a
    fidelity defect and a budget one. Each sampled trial's task set must be
    a unique subset of the run's validation tasks of exactly the scheduled
    size -- ``minibatch_size`` when minibatching, the full valset when not.

    **This is also the F16 fan-out assertion.** The concern F16 names is
    that platform deferral row expansion might ignore a per-intent task set
    and silently evaluate the full split, costing roughly 2.5x at the
    study's sizes. That failure is visible right here: the intent's own
    ``task_hashes`` would equal the whole valset. So the invariant checks
    the intent as issued, not only the transcript's record of it, and a
    minibatch intent whose task set covers the valset FAILs.

    Whetstone's own transcript validator already rejects a mis-sized batch,
    so a transcript that reaches this point has usually settled it. Checking
    the intents independently is the point: they are the layer that reaches
    the evaluating side, and they are what a fan-out defect would corrupt.
    """
    invariant = InvariantId.MIPRO_MINIBATCH_SIZING
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")
    transcript = state.study_transcript
    refs = _state_refs(evidence)
    if transcript is None:
        return _missing(invariant, "the run persisted no study transcript")

    valset = set(transcript.validation_task_hashes)
    schedule = transcript.schedule
    expected = (
        min(schedule.minibatch_size, len(valset))
        if schedule.minibatch
        else len(valset)
    )

    problems: list[str] = []
    checked = 0
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            request = resolution.optim_eval_request
            purpose = eval_purpose(request.eval_request.metadata)
            if purpose != SAMPLE_PURPOSE:
                continue
            checked += 1
            where = f"step {entry.index} intent {position}"
            tasks = request.task_hashes
            if tasks is None:
                problems.append(
                    f"{where} declares no task subset, so it evaluates the "
                    f"full task set rather than its scheduled batch"
                )
                continue
            if len(tasks) != expected:
                problems.append(
                    f"{where} evaluates {len(tasks)} task(s), not the "
                    f"scheduled {expected}"
                )
            if len(set(tasks)) != len(tasks):
                problems.append(f"{where} repeats a task within its batch")
            outside = set(tasks) - valset
            if outside:
                problems.append(
                    f"{where} draws {len(outside)} task(s) from outside the "
                    f"validation split"
                )
            if schedule.minibatch and valset <= set(tasks):
                problems.append(
                    f"{where} is a minibatch trial whose task set covers "
                    f"the whole validation split"
                )

    if not checked:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"no evaluation intent carries purpose {SAMPLE_PURPOSE!r}, "
                f"so no trial evaluation was issued to size"
            ),
            refs,
        )
    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} of {checked} trial evaluation(s) are "
                f"mis-sized: {'; '.join(problems[:3])}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {checked} trial evaluation(s) drew a unique "
            f"{expected}-task subset of the {len(valset)}-task validation "
            f"split, matching the persisted schedule "
            f"(minibatch={schedule.minibatch})"
        ),
        refs,
    )


def miprov2_periodic_full_eval(evidence: RunEvidence) -> AuditFinding:
    """The incumbent is evaluated on the full validation set periodically.

    Minibatch scores are noisy, so MIPROv2 does not trust them to rank
    candidates: on a cadence of ``minibatch_full_eval_steps`` trials, and
    once at the end, it re-evaluates the best-so-far candidate on the whole
    validation set and uses that score to choose. Dropping those promotions
    would make the run cheaper and its "best" candidate an artifact of
    minibatch luck.

    ``Miprov2StudySchedule.promotion_due`` owns the cadence, so the audit
    asks it rather than recomputing the arithmetic -- a second copy of the
    rule here would drift. The check is that a promotion is recorded at
    exactly the due trials, that each promotion's evaluation covers the
    whole validation split, and that the engine issued a matching
    ``miprov2_promotion`` intent for every one.

    ``NOT_APPLICABLE`` when the run did not minibatch: with
    ``minibatch=False`` every trial is already a full-valset evaluation, so
    there is no separate periodic evaluation to find, and demanding one
    would be demanding behaviour the algorithm does not have.
    """
    invariant = InvariantId.MIPRO_PERIODIC_FULL_EVAL
    transcript = _terminal_transcript(evidence)
    refs = _state_refs(evidence)
    if transcript is None:
        return _missing(invariant, "the run persisted no study transcript")

    schedule = transcript.schedule
    valset_size = len(transcript.validation_task_hashes)
    if not schedule.minibatch:
        return _finding(
            invariant,
            AuditStatus.NOT_APPLICABLE,
            (
                f"this run did not minibatch, so each of its "
                f"{len(transcript.samples)} trial(s) already evaluated the "
                f"full {valset_size}-task validation split"
            ),
            refs,
        )

    problems: list[str] = []
    promotions = 0
    for sample in transcript.samples:
        due = schedule.promotion_due(optuna_trial_number=sample.trial_number)
        promotion = sample.promotion
        if due and promotion is None:
            problems.append(
                f"trial {sample.trial_number} was due a full evaluation but "
                f"records none"
            )
            continue
        if promotion is None:
            continue
        promotions += 1
        if not due:
            problems.append(
                f"trial {sample.trial_number} records a full evaluation "
                f"off the {schedule.minibatch_full_eval_steps}-trial cadence"
            )
        covered = promotion.evaluation.task_batch_hashes
        if tuple(covered) != transcript.validation_task_hashes:
            problems.append(
                f"trial {sample.trial_number}'s full evaluation covers "
                f"{len(covered)} task(s), not the whole {valset_size}-task "
                f"validation split"
            )

    issued = sum(
        1
        for _step, _position, purpose in _purposed_intents(evidence)
        if purpose == PROMOTION_PURPOSE
    )
    if issued != promotions:
        problems.append(
            f"{promotions} full evaluation(s) are recorded but "
            f"{issued} intent(s) carry purpose {PROMOTION_PURPOSE!r}"
        )
    if not promotions:
        problems.append(
            f"no full evaluation of the incumbent was recorded across "
            f"{len(transcript.samples)} minibatched trial(s)"
        )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"periodic full evaluation is not faithful: "
                f"{'; '.join(problems[:3])}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"{promotions} full evaluation(s) of the incumbent ran on the "
            f"cadence of {schedule.minibatch_full_eval_steps} trial(s), "
            f"each covering all {valset_size} validation tasks, with a "
            f"matching engine intent for every one"
        ),
        refs,
    )


def miprov2_bootstrap_through_engine(
    evidence: RunEvidence,
) -> AuditFinding:
    """Bootstrap generations are paid through the engine, not the proposer.

    A bootstrap generation runs the task model over one trainset task, so it
    is an evaluation and belongs on the evaluation engine: it is then
    budgeted, cached, and scored like every other evaluation, and its cost
    lands under the task-model role. Routing it through the proposer
    transport instead would hide task-model spend inside proposer spend and
    leave the generation unscored -- which would make MIPROv2 look cheaper
    than it is and put its demo acceptance decisions outside the audited
    evidence.

    Both sides are durably recorded, which is what makes this checkable.
    ``Miprov2State.completed_effects`` bills every paid effect with a
    ``kind``, so each ``bootstrap_generations`` entry is a bootstrap the run
    paid for; its ``identity_hash`` is the attempt identity, which also
    appears in the corresponding engine intent's ``request_id``. So the
    invariant is a bijection: every billed bootstrap effect has an engine
    intent, and every bootstrap-purposed intent has a billed effect.
    """
    invariant = InvariantId.MIPRO_BOOTSTRAP_THROUGH_ENGINE
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")
    refs = _state_refs(evidence)

    billed = {
        effect.identity_hash
        for effect in state.completed_effects
        if effect.kind == BOOTSTRAP_GENERATION_EFFECT
    }
    request_ids: list[str] = []
    resolvable = 0
    for entry in evidence.steps:
        for resolution in entry.resolved_intents:
            request = resolution.optim_eval_request
            if eval_purpose(request.eval_request.metadata) != (
                BOOTSTRAP_PURPOSE
            ):
                continue
            request_ids.append(request.eval_request.request_id)
            if resolution.eval_result_ref is not None:
                resolvable += 1

    problems: list[str] = []
    unmatched = [
        identity
        for identity in sorted(billed)
        if not any(identity in request_id for request_id in request_ids)
    ]
    if unmatched:
        problems.append(
            f"{len(unmatched)} billed bootstrap generation(s) have no "
            f"engine evaluation intent, so they were paid elsewhere"
        )
    orphans = [
        request_id
        for request_id in request_ids
        if not any(identity in request_id for identity in billed)
    ]
    if orphans:
        problems.append(
            f"{len(orphans)} bootstrap intent(s) match no billed "
            f"{BOOTSTRAP_GENERATION_EFFECT!r} effect"
        )
    if resolvable != len(request_ids):
        problems.append(
            f"{len(request_ids) - resolvable} bootstrap intent(s) cite no "
            f"evaluation result"
        )
    if not billed and not request_ids:
        problems.append(
            "the run billed no bootstrap generation and issued no bootstrap "
            "intent, but every demo mode bootstraps"
        )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (f"bootstrap routing is not faithful: {'; '.join(problems[:3])}"),
            refs,
        )
    proposal_calls = sum(
        1
        for effect in state.completed_effects
        if effect.kind == PROPOSAL_CALL_EFFECT
    )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {len(billed)} billed bootstrap generation(s) resolve to a "
            f"matching engine evaluation intent with an evaluation result; "
            f"the proposer transport was billed only its "
            f"{proposal_calls} proposal call(s)"
        ),
        refs,
    )


def miprov2_trials_match_control(evidence: RunEvidence) -> AuditFinding:
    """The run completed the trial budget its control asked for.

    ``num_trials`` is the search budget, and a run that quietly recorded
    fewer trials than it was configured for would report an optimizer that
    searched less than the study believes it did -- so its efficacy number
    would be attributed to the wrong budget.

    A truncated run is not a fidelity failure when it is honest about it:
    ``OptimResult.terminal_failure`` records a run that stopped early, and
    the invariant accepts a short transcript in that case. Silent
    truncation is the defect.
    """
    invariant = InvariantId.MIPRO_TRIALS_MATCH_CONTROL
    state = _terminal_state(evidence)
    if state is None:
        return _missing(invariant, "no step persisted MIPROv2 state")
    transcript = state.study_transcript
    refs = _state_refs(evidence)
    if transcript is None:
        return _missing(invariant, "the run persisted no study transcript")

    observed = len(transcript.samples)
    configured = state.control.num_trials
    if observed == configured:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the transcript records all {observed} of the "
                f"{configured} trial(s) the control configured"
            ),
            refs,
        )
    failure = evidence.result.terminal_failure
    if observed < configured and failure is not None:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run recorded {observed} of {configured} configured "
                f"trial(s) and reports the truncation as a terminal failure"
            ),
            refs,
        )
    if observed < configured:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the transcript records {observed} trial(s) but the "
                f"control configured {configured}, and the run reports no "
                f"terminal failure explaining the truncation"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.FAIL,
        (
            f"the transcript records {observed} trial(s), more than the "
            f"{configured} the control configured"
        ),
        refs,
    )


#: Every MIPROv2 invariant, in the order the audit report lists them.
#:
#: Enumerated here for the registry to splice in; ``registry.py`` stays the
#: only place that decides which optimizer runs which set.
MIPROV2_INVARIANTS = (
    miprov2_bootstrap_before_proposal,
    miprov2_zeroshot_grounding,
    miprov2_ground_only_deviation,
    miprov2_tpe_selection,
    miprov2_minibatch_sizing,
    miprov2_periodic_full_eval,
    miprov2_bootstrap_through_engine,
    miprov2_trials_match_control,
)


__all__ = [
    "BASELINE_PURPOSE",
    "BOOTSTRAP_GENERATION_EFFECT",
    "BOOTSTRAP_PURPOSE",
    "GROUND_ONLY_DEVIATION",
    "INSTRUCTION_PROPOSAL_EFFECT",
    "MIPROV2_INVARIANTS",
    "PROMOTION_PURPOSE",
    "PROPOSAL_CALL_EFFECT",
    "SAMPLE_PURPOSE",
    "ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS",
    "ZEROSHOT_GROUNDING_LABELED_DEMOS",
    "GroundingCaps",
    "GroundingEvidence",
    "components_with_demo_sets",
    "miprov2_bootstrap_before_proposal",
    "miprov2_bootstrap_through_engine",
    "miprov2_ground_only_deviation",
    "miprov2_minibatch_sizing",
    "miprov2_periodic_full_eval",
    "miprov2_tpe_selection",
    "miprov2_trials_match_control",
    "miprov2_zeroshot_grounding",
    "zeroshot_grounding_problems",
]
