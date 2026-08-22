"""The MIPROv2 fidelity invariants, positive and negative.

Every invariant here is exercised twice: once against a real fake-transport
MIPROv2 run, and once against that same run with one evidence field violated,
per section 3.2 of the Step 10 assignment. Each negative asserts both that
the target invariant FAILs *and* that no other invariant's status changed --
otherwise a sloppy mutation would pass by breaking everything.

Four runs back these tests, all fake transport and zero provider calls:

- ``fewshot``, ``zeroshot``, and ``ground_only`` at the runner's default
  split, which resolves to ``minibatch=False``;
- one ``fewshot`` run at a wider internal split with minibatching forced on,
  because ``MIPRO_PERIODIC_FULL_EVAL`` is only meaningful when the run
  actually minibatches. Without it that invariant would be permanently
  ``NOT_APPLICABLE``, which the assignment names as a defect.

**Why the minibatched run is built here rather than through a spec flag.**
``whetstone_envs/optim/miprov2.py`` pins ``minibatch=False`` when it resolves
the C19 control, and that module belongs to another wave's file set. Rather
than reach into it, these tests patch the ``configure_miprov2`` symbol that
module imports, forcing the minibatch schedule while every other part of the
run -- the engine, the adapter, the store, the persisted evidence -- stays
exactly the production path. The result is a genuine run, not a hand-built
artifact.

**On mutation sites.** ``Miprov2State`` is comprehensively self-verifying: it
cross-checks its transcript against the control, the control against the
run's ``optimizer_config``, the bootstrap plans against a replay of the
planner, and the completed-effect ledger against a replay of the evidence.
Every attempt to violate a MIPROv2 semantic *inside* that state is rejected
at load -- ``model_construct`` included, since it is overridden to
revalidate. That is a good property of whetstone, and it shapes the
negatives here into two kinds:

- **Mutated runs**, for the seven invariants whose evidence the state does
  not cross-verify: the evaluation intents on ``result.json``, and which
  state snapshot a step's ``state_ref`` addresses. Each stays a schema-valid
  ``OptimResult``, so the invariant under test is what fails, and each
  asserts that no unrelated verdict moved.
- **A conformant stand-in**, for ``MIPRO_ZEROSHOT_GROUNDING`` alone. A
  zeroshot state with wrong grounding caps or a rendered demo set cannot be
  loaded from any artifact, so no mutated run can produce one. The predicate
  is therefore split out as ``zeroshot_grounding_problems`` over a narrow
  Protocol and exercised against a stand-in that satisfies it. The stand-in
  is checked against the real zeroshot state first, so it cannot drift into
  testing a shape whetstone does not produce.

Two invariants -- ordering and bootstrap routing -- share the
bootstrap-purposed intents as evidence, so a purpose relabel moves both.
That coupling is asserted explicitly rather than worked around, and each
still has its own isolated negative.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.core.identity import TypedRef
from whetstone.optim.contracts import (
    OptimStepRequest,
    OptimStepResult,
    step_request_reference,
    step_result_reference,
)

from whetstone_envs.optim.audit._evidence import (
    MIPROV2_OPTIMIZER,
    RESULT_FILENAME,
    load_run_evidence,
)
from whetstone_envs.optim.audit._mutate import (
    MutationError,
    copy_run,
    put_record,
)
from whetstone_envs.optim.audit.miprov2 import (
    BASELINE_PURPOSE,
    BOOTSTRAP_GENERATION_EFFECT,
    BOOTSTRAP_PURPOSE,
    GROUND_ONLY_DEVIATION,
    INSTRUCTION_PROPOSAL_EFFECT,
    MIPROV2_INVARIANTS,
    PROMOTION_PURPOSE,
    PROPOSAL_CALL_EFFECT,
    SAMPLE_PURPOSE,
    ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS,
    ZEROSHOT_GROUNDING_LABELED_DEMOS,
    components_with_demo_sets,
    miprov2_ground_only_deviation,
    miprov2_minibatch_sizing,
    miprov2_periodic_full_eval,
    miprov2_zeroshot_grounding,
    zeroshot_grounding_problems,
)
from whetstone_envs.optim.audit.registry import audit_run, invariants_for
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone.optim.miprov2.runtime import Miprov2State
    from whetstone.optim.miprov2.study import StudyTranscript

    from whetstone_envs.optim.audit._evidence import RunEvidence
    from whetstone_envs.optim.audit.schema import AuditFinding

#: The three demonstration regimes every ordering and routing invariant must
#: hold under. Named here so a mode added upstream without a fixture is a
#: visible gap rather than a silent one.
DEMO_MODES = ("fewshot", "zeroshot", "ground_only")


def _state(evidence: RunEvidence) -> Miprov2State:
    """The terminal MIPROv2 state, asserted present.

    Every run these tests build persists one, so its absence is a broken
    fixture rather than a case to handle.
    """
    state = evidence.steps[-1].miprov2_state()
    assert state is not None, "the run persisted no MIPROv2 state"
    return state


def _transcript(evidence: RunEvidence) -> StudyTranscript:
    """The terminal study transcript, asserted present."""
    transcript = _state(evidence).study_transcript
    assert transcript is not None, "the run persisted no study transcript"
    return transcript


# --------------------------------------------------------------------------
# Real fake-transport runs
# --------------------------------------------------------------------------


def _run(
    *,
    run_id: str,
    demo_mode: str,
    output: Path,
    split_sizes: tuple[int, int, int] = (2, 2, 0),
    n_per_stratum: int | None = None,
) -> Path:
    from whetstone_envs.optim.run import RunSpec, run_optimizer

    return run_optimizer(
        RunSpec(
            optimizer="miprov2",
            transport="fake",
            demo_mode=demo_mode,
            split_sizes=split_sizes,
            n_per_stratum=n_per_stratum,
            output_dir=output,
            run_id=run_id,
        )
    )


@pytest.fixture(scope="session")
def miprov2_runs(tmp_path_factory) -> dict[str, Path]:
    """One completed fake-transport MIPROv2 run per demo mode."""
    root = tmp_path_factory.mktemp("miprov2-audit-runs")
    return {
        mode: _run(
            run_id=f"c19-miprov2-audit-{mode}",
            demo_mode=mode,
            output=root / mode,
        )
        for mode in DEMO_MODES
    }


@pytest.fixture(scope="session")
def minibatched_run(tmp_path_factory) -> Path:
    """A real MIPROv2 run that actually minibatches its trials.

    ``optim/miprov2.py`` pins ``minibatch=False``, so the schedule is forced
    by patching the ``configure_miprov2`` symbol that module imports. Only
    the schedule changes: the control still resolves through whetstone's own
    ``configure_miprov2``, and the run drives the production path end to end.
    """
    import whetstone_envs.optim.miprov2 as envs_miprov2

    original = envs_miprov2.configure_miprov2

    def minibatched(**kwargs: Any) -> Any:
        forced: dict[str, Any] = {
            **kwargs,
            "minibatch": True,
            "minibatch_size": 2,
            "minibatch_full_eval_steps": 1,
            "num_trials": 3,
        }
        return original(**forced)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(envs_miprov2, "configure_miprov2", minibatched)
        return _run(
            run_id="c19-miprov2-audit-minibatch",
            demo_mode="fewshot",
            output=tmp_path_factory.mktemp("miprov2-minibatch") / "run",
            split_sizes=(6, 2, 0),
            n_per_stratum=4,
        )


# --------------------------------------------------------------------------
# Fixture mutation
# --------------------------------------------------------------------------


def _reseal(document: dict[str, Any]) -> dict[str, Any]:
    """Re-thread the step chain after a state or intent mutation.

    ``OptimResult`` verifies that each step's wrapper ref addresses its own
    record, and that each later request cites the prior step's exact result,
    state, and history refs. The shared ``_mutate.reseal_step_chain`` rethreads
    the first two; a mutation that re-puts a *state* record also invalidates
    ``prior_state_ref``, so this rethreads all three. The mutated field itself
    is untouched -- only the integrity refs that would otherwise reject the
    artifact before any invariant could judge it.
    """
    prior_result: dict[str, Any] | None = None
    prior_state: dict[str, Any] | None = None
    prior_history: dict[str, Any] | None = None
    for index, wrapper in enumerate(document["step_results"]):
        record = wrapper["record"]
        if prior_result is not None:
            request = record["request"]["record"]
            request["prior_step_result_ref"] = prior_result
            request["prior_state_ref"] = prior_state
            request["prior_history_ref"] = prior_history
            record["request"]["record_ref"] = step_request_reference(
                OptimStepRequest.model_validate(request)
            ).record_ref.model_dump(mode="json")
        try:
            reference = step_result_reference(
                OptimStepResult.model_validate(record)
            )
        except ValueError as error:
            raise MutationError(
                f"mutated step {index} is not a valid OptimStepResult: {error}"
            ) from error
        wrapper["record_ref"] = reference.record_ref.model_dump(mode="json")
        prior_result = wrapper["record_ref"]
        prior_state = record.get("state_ref")
        prior_history = record.get("history_ref")
    return document


def _mutated(
    source: Path,
    destination: Path,
    rewrite,
) -> Path:
    """Copy ``source`` and apply ``rewrite`` to its ``result.json``."""
    run_dir = copy_run(source, destination)
    result_path = run_dir / RESULT_FILENAME
    document = json.loads(result_path.read_text(encoding="utf-8"))
    before = copy.deepcopy(document)
    rewrite(run_dir, document)
    if document == before:
        raise MutationError(
            "the rewrite left the artifact unchanged; the fixture would not "
            "be a negative"
        )
    _reseal(document)
    result_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return run_dir


def _intents(document: dict[str, Any]):
    """Every resolved intent in the document, with its step wrapper."""
    for wrapper in document["step_results"]:
        for resolution in wrapper["record"]["resolved_intents"]:
            yield wrapper, resolution


def _metadata(resolution: dict[str, Any]) -> dict[str, Any]:
    return resolution["optim_eval_request"]["eval_request"]["metadata"]


def _first_with_purpose(document: dict[str, Any], purpose: str):
    for wrapper, resolution in _intents(document):
        if _metadata(resolution).get("purpose") == purpose:
            return wrapper, resolution
    raise MutationError(f"no intent carries purpose {purpose!r}")


def _relabel(purpose: str, replacement: str):
    def rewrite(_run_dir: Path, document: dict[str, Any]) -> None:
        _, resolution = _first_with_purpose(document, purpose)
        _metadata(resolution)["purpose"] = replacement

    return rewrite


def _state_payload(run_dir: Path, state_ref: dict[str, Any]) -> Any:
    from dr_store.sync import open_sqlite

    with open_sqlite(str(run_dir / "runtime.sqlite")) as store:
        return store.get(TypedRef.model_validate(state_ref).reference)


def _rewrite_states(run_dir: Path, document: dict[str, Any], edit) -> int:
    """Apply ``edit`` to every step's MIPROv2 state, re-putting each one.

    ``edit`` returns True when it changed that state. The re-put ref replaces
    the step's ``state_ref``, and ``_reseal`` rethreads the chain.
    """
    changed = 0
    for wrapper in document["step_results"]:
        record = wrapper["record"]
        state_ref = record.get("state_ref")
        if state_ref is None:
            continue
        payload = _state_payload(run_dir, state_ref)
        state = payload.get("miprov2_state")
        if state is None or not edit(state):
            continue
        record["state_ref"] = put_record(
            run_dir, state_ref["schema_name"], payload
        )
        changed += 1
    return changed


# --------------------------------------------------------------------------
# Registration and wiring
# --------------------------------------------------------------------------


MIPROV2_INVARIANT_IDS = (
    InvariantId.MIPRO_BOOTSTRAP_BEFORE_PROPOSAL,
    InvariantId.MIPRO_ZEROSHOT_GROUNDING,
    InvariantId.MIPRO_GROUND_ONLY_DEVIATION,
    InvariantId.MIPRO_TPE_SELECTION,
    InvariantId.MIPRO_MINIBATCH_SIZING,
    InvariantId.MIPRO_PERIODIC_FULL_EVAL,
    InvariantId.MIPRO_BOOTSTRAP_THROUGH_ENGINE,
    InvariantId.MIPRO_TRIALS_MATCH_CONTROL,
)


def test_the_assignment_names_eight_miprov2_invariants() -> None:
    assert len(MIPROV2_INVARIANTS) == 8
    assert len(MIPROV2_INVARIANT_IDS) == 8


def test_every_miprov2_invariant_is_registered() -> None:
    registered = invariants_for(MIPROV2_OPTIMIZER)
    assert set(MIPROV2_INVARIANTS) <= set(registered)


def test_persisted_invariant_ids_are_pinned() -> None:
    """Wire spellings are stored identity, pinned rather than derived."""
    assert [member.value for member in MIPROV2_INVARIANT_IDS] == [
        "mipro_bootstrap_before_proposal",
        "mipro_zeroshot_grounding",
        "mipro_ground_only_deviation",
        "mipro_tpe_selection",
        "mipro_minibatch_sizing",
        "mipro_periodic_full_eval",
        "mipro_bootstrap_through_engine",
        "mipro_trials_match_control",
    ]


def test_whetstone_evidence_spellings_are_pinned() -> None:
    """These are whetstone's persisted strings, not ours to rename.

    If any of them drifts upstream, the invariants keying on it would match
    nothing and pass vacuously, which is the failure mode this pins.
    """
    assert BOOTSTRAP_PURPOSE == "miprov2_bootstrap"
    assert BASELINE_PURPOSE == "miprov2_baseline"
    assert SAMPLE_PURPOSE == "miprov2_sample"
    assert PROMOTION_PURPOSE == "miprov2_promotion"
    assert INSTRUCTION_PROPOSAL_EFFECT == "instruction_proposal"
    assert BOOTSTRAP_GENERATION_EFFECT == "bootstrap_generations"
    assert PROPOSAL_CALL_EFFECT == "proposal_calls"
    assert GROUND_ONLY_DEVIATION == "demo_mode:ground_only"
    assert ZEROSHOT_GROUNDING_BOOTSTRAPPED_DEMOS == 3
    assert ZEROSHOT_GROUNDING_LABELED_DEMOS == 0


def test_the_pinned_purposes_are_the_ones_a_run_emits(miprov2_runs) -> None:
    """The pinned spellings match what whetstone actually wrote.

    Pinning a literal only helps if it is the literal in use; this closes
    the loop against a real run rather than against our own constant.
    """
    from whetstone.eval.metadata import eval_purpose

    seen: set[str] = set()
    for run_dir in miprov2_runs.values():
        evidence = load_run_evidence(run_dir)
        for entry in evidence.steps:
            for resolution in entry.resolved_intents:
                purpose = eval_purpose(
                    resolution.optim_eval_request.eval_request.metadata
                )
                if purpose is not None:
                    seen.add(purpose)
    assert BOOTSTRAP_PURPOSE in seen
    assert BASELINE_PURPOSE in seen
    assert SAMPLE_PURPOSE in seen
    assert seen <= {
        BOOTSTRAP_PURPOSE,
        BASELINE_PURPOSE,
        SAMPLE_PURPOSE,
        PROMOTION_PURPOSE,
    }


# --------------------------------------------------------------------------
# Positives, on real runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", DEMO_MODES)
def test_a_real_run_passes_every_invariant(miprov2_runs, mode) -> None:
    report = audit_run(miprov2_runs[mode])
    assert report.optimizer == MIPROV2_OPTIMIZER
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def test_the_minibatched_run_passes_every_invariant(minibatched_run) -> None:
    report = audit_run(minibatched_run)
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def _finding_for(report, invariant_id: InvariantId) -> AuditFinding:
    for finding in report.findings:
        if finding.invariant_id is invariant_id:
            return finding
    raise AssertionError(f"{invariant_id.value} is not in the report")


@pytest.mark.parametrize("invariant_id", MIPROV2_INVARIANT_IDS)
def test_no_invariant_is_always_not_applicable(
    miprov2_runs,
    minibatched_run,
    invariant_id,
) -> None:
    """An invariant that never applies to its own optimizer is a defect.

    The assignment is explicit about this: it is what killed
    ``GEPA_REFLECTION_MINIBATCH``. So every invariant must reach ``PASS`` on
    at least one real run across the modes and schedules shipped here.
    """
    statuses = [
        _finding_for(audit_run(run_dir), invariant_id).status
        for run_dir in (*miprov2_runs.values(), minibatched_run)
    ]
    assert AuditStatus.PASS in statuses, statuses


def test_zeroshot_grounding_passes_only_in_zeroshot(miprov2_runs) -> None:
    """The 3/0 grounding bootstrap is mode-specific, and says so."""
    for mode, run_dir in miprov2_runs.items():
        finding = miprov2_zeroshot_grounding(load_run_evidence(run_dir))
        expected = (
            AuditStatus.PASS
            if mode == "zeroshot"
            else AuditStatus.NOT_APPLICABLE
        )
        assert finding.status is expected, (mode, finding.detail)
        if expected is AuditStatus.NOT_APPLICABLE:
            assert mode in finding.detail


def test_ground_only_is_the_only_flagged_deviation(miprov2_runs) -> None:
    for mode, run_dir in miprov2_runs.items():
        finding = miprov2_ground_only_deviation(load_run_evidence(run_dir))
        assert finding.status is AuditStatus.PASS, finding.detail
        marker_named = GROUND_ONLY_DEVIATION in finding.detail
        assert marker_named is (mode == "ground_only"), finding.detail


def test_periodic_full_eval_is_not_applicable_without_minibatching(
    miprov2_runs,
) -> None:
    """With ``minibatch=False`` every trial is already a full evaluation."""
    finding = miprov2_periodic_full_eval(
        load_run_evidence(miprov2_runs["fewshot"])
    )
    assert finding.status is AuditStatus.NOT_APPLICABLE
    assert "did not minibatch" in finding.detail


def test_minibatch_sizing_sees_a_genuine_subset(minibatched_run) -> None:
    """The F16 assertion has something to assert about."""
    evidence = load_run_evidence(minibatched_run)
    transcript = _transcript(evidence)
    assert transcript.schedule.minibatch
    assert transcript.schedule.minibatch_size < len(
        transcript.validation_task_hashes
    )
    finding = miprov2_minibatch_sizing(evidence)
    assert finding.status is AuditStatus.PASS, finding.detail


@pytest.mark.parametrize("mode", DEMO_MODES)
def test_every_finding_cites_the_state_it_read(miprov2_runs, mode) -> None:
    """A verdict with no evidence ref is not auditable evidence."""
    report = audit_run(miprov2_runs[mode])
    for invariant_id in MIPROV2_INVARIANT_IDS:
        finding = _finding_for(report, invariant_id)
        assert finding.evidence_refs, finding.invariant_id.value
        for ref in finding.evidence_refs:
            assert ref.schema_name == "whetstone.optim_state_snapshot"
            assert len(ref.content_hash) == 64


# --------------------------------------------------------------------------
# Negatives -- one per invariant
# --------------------------------------------------------------------------


def _statuses(run_dir: Path) -> dict[InvariantId, AuditStatus]:
    return {
        finding.invariant_id: finding.status
        for finding in audit_run(run_dir).findings
    }


def _assert_isolated_failure(
    source: Path,
    mutated: Path,
    invariant_id: InvariantId,
) -> AuditFinding:
    """The mutation FAILs its target and changes no other verdict.

    The second half is the one that matters: a mutation that broke every
    invariant would let a sloppy predicate look sound.
    """
    before = _statuses(source)
    report = audit_run(mutated)
    after = {
        finding.invariant_id: finding.status for finding in report.findings
    }
    finding = _finding_for(report, invariant_id)
    assert finding.status is AuditStatus.FAIL, finding.detail
    changed = {
        key: (before[key], after[key])
        for key in after
        if key is not invariant_id and before.get(key) is not after[key]
    }
    assert not changed, changed
    assert not report.passed
    return finding


def test_bootstrap_after_proposal_fails(miprov2_runs, tmp_path) -> None:
    """A bootstrap evaluation issued after proposals began is a defect.

    The mutation moves the phase boundary rather than a purpose label: the
    step carrying the bootstrap intent is pointed at a state snapshot from
    the proposal phase, so that bootstrap now sits at the first step where
    the run had already begun proposing. No intent is relabelled, so this
    isolates the ordering predicate from the routing one.
    """
    source = miprov2_runs["fewshot"]

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        proposal_ref = None
        for wrapper in document["step_results"]:
            state_ref = wrapper["record"].get("state_ref")
            if state_ref is None:
                continue
            state = _state_payload(run_dir, state_ref).get("miprov2_state")
            if state is not None and state["phase"] != "bootstrap":
                proposal_ref = state_ref
                break
        assert proposal_ref is not None, "no step left the bootstrap phase"
        for wrapper in document["step_results"]:
            carries_bootstrap = any(
                _metadata(resolution).get("purpose") == BOOTSTRAP_PURPOSE
                for resolution in wrapper["record"]["resolved_intents"]
            )
            if carries_bootstrap:
                wrapper["record"]["state_ref"] = proposal_ref
                return
        raise MutationError("no step carries a bootstrap intent")

    mutated = _mutated(source, tmp_path / "late-bootstrap", rewrite)
    finding = _assert_isolated_failure(
        source, mutated, InvariantId.MIPRO_BOOTSTRAP_BEFORE_PROPOSAL
    )
    assert "at or after step" in finding.detail


def test_relabelling_a_bootstrap_intent_fails_both_bootstrap_invariants(
    miprov2_runs,
    tmp_path,
) -> None:
    """The two bootstrap invariants share evidence, deliberately.

    Ordering and routing both read the bootstrap-purposed intents, so a
    mutation to a purpose label moves both verdicts. That coupling is
    recorded here rather than hidden: the invariants answer different
    questions ("were demos bootstrapped first?" and "who paid for them?")
    but neither can be answered without knowing which intents were
    bootstraps. Each still has its own isolated negative elsewhere in this
    module.
    """
    source = miprov2_runs["fewshot"]
    mutated = _mutated(
        source,
        tmp_path / "relabelled-bootstrap",
        _relabel(BOOTSTRAP_PURPOSE, BASELINE_PURPOSE),
    )
    statuses = _statuses(mutated)
    assert (
        statuses[InvariantId.MIPRO_BOOTSTRAP_BEFORE_PROPOSAL]
        is AuditStatus.FAIL
    )
    assert (
        statuses[InvariantId.MIPRO_BOOTSTRAP_THROUGH_ENGINE]
        is AuditStatus.FAIL
    )
    before = _statuses(source)
    coupled = {
        InvariantId.MIPRO_BOOTSTRAP_BEFORE_PROPOSAL,
        InvariantId.MIPRO_BOOTSTRAP_THROUGH_ENGINE,
    }
    unchanged = {
        key: (before[key], statuses[key])
        for key in statuses
        if key not in coupled and before[key] is not statuses[key]
    }
    assert not unchanged, unchanged


class _StubPlan:
    """A bootstrap plan carrying only the two caps the predicate reads."""

    def __init__(self, bootstrapped: int, labeled: int) -> None:
        self.max_bootstrapped_demos = bootstrapped
        self.max_labeled_demos = labeled


class _StubControl:
    def __init__(self, bootstrapped: int, labeled: int) -> None:
        self.max_bootstrapped_demos = bootstrapped
        self.max_labeled_demos = labeled


class _StubState:
    """A zeroshot state shaped like the one whetstone refuses to build.

    ``Miprov2State`` replays its bootstrap plans against the control and its
    transcript against the run, so a zeroshot state with wrong grounding
    caps or a rendered demo set cannot be loaded from any artifact -- see
    ``zeroshot_grounding_problems``. The predicate must still be correct, so
    it is exercised against a stand-in carrying exactly the fields it reads.
    """

    def __init__(
        self,
        *,
        control_caps: tuple[int, int] = (0, 0),
        plan_caps: tuple[tuple[int, int], ...] = ((3, 0), (2, 0), (3, 0)),
        transcript=None,
    ) -> None:
        self.control = _StubControl(*control_caps)
        self.bootstrap_plans = tuple(_StubPlan(*caps) for caps in plan_caps)
        self.study_transcript = transcript


def test_zeroshot_grounding_predicate_accepts_a_faithful_state(
    miprov2_runs,
) -> None:
    """The stand-in agrees with the real zeroshot run it stands in for."""
    real = load_run_evidence(miprov2_runs["zeroshot"])
    real_state = _state(real)
    assert zeroshot_grounding_problems(real_state) == ()
    stub = _StubState(transcript=real_state.study_transcript)
    assert zeroshot_grounding_problems(stub) == ()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"plan_caps": ((1, 0), (1, 0))},
            "peak at 1 bootstrapped demos",
            id="wrong-bootstrapped-cap",
        ),
        pytest.param(
            {"plan_caps": ((3, 1), (2, 1))},
            "labeled-demo caps [1]",
            id="wrong-labeled-cap",
        ),
        pytest.param(
            {"plan_caps": ()},
            "no bootstrap plan was recorded",
            id="no-plans",
        ),
        pytest.param(
            {"control_caps": (3, 0)},
            "control.max_bootstrapped_demos is 3",
            id="control-searches-demos",
        ),
        pytest.param(
            {"control_caps": (0, 2)},
            "control.max_labeled_demos is 2",
            id="control-labels-demos",
        ),
    ],
)
def test_zeroshot_grounding_fails_when_the_caps_are_wrong(
    miprov2_runs,
    kwargs,
    expected,
) -> None:
    """A zeroshot run that did not use the 3/0 grounding caps is not DSPy."""
    real = load_run_evidence(miprov2_runs["zeroshot"])
    transcript = _transcript(real)
    problems = zeroshot_grounding_problems(
        _StubState(transcript=transcript, **kwargs)
    )
    assert any(expected in problem for problem in problems), problems


def test_zeroshot_grounding_fails_on_a_shipped_demo_set(
    miprov2_runs,
) -> None:
    """A zeroshot candidate that ships demos discarded nothing.

    ``zeroshot``'s whole claim is that it bootstraps to ground proposals and
    then throws the demos away. The ``fewshot`` transcript is the shape a
    leaking zeroshot run would produce -- its candidates do render a demo
    set -- so it is the violation this asserts against.
    """
    fewshot = load_run_evidence(miprov2_runs["fewshot"])
    zeroshot = load_run_evidence(miprov2_runs["zeroshot"])
    leaking = _transcript(fewshot)
    faithful = _transcript(zeroshot)
    assert components_with_demo_sets(leaking) > 0
    assert components_with_demo_sets(faithful) == 0

    problems = zeroshot_grounding_problems(_StubState(transcript=leaking))
    assert any("ship a rendered demo set" in problem for problem in problems)
    assert any("searched demo dimension" in problem for problem in problems)


def test_zeroshot_grounding_fails_without_a_transcript() -> None:
    problems = zeroshot_grounding_problems(_StubState(transcript=None))
    assert problems == ("the run persisted no study transcript",)


def test_ground_only_deviation_fails_on_a_stale_marker(
    miprov2_runs,
    tmp_path,
) -> None:
    """The marker must match the mode the run actually used.

    ``StudyTranscript`` derives ``whetstone_deviation`` from ``demo_mode``
    and refuses an inconsistent pair, so the violation is staged as a
    ground_only run whose terminal step addresses a state from before the
    transcript existed: the run then claims a deviating mode with no
    transcript to back it, which is the same unbacked claim.
    """
    source = miprov2_runs["ground_only"]

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        early = None
        for wrapper in document["step_results"]:
            state_ref = wrapper["record"].get("state_ref")
            if state_ref is not None:
                early = state_ref
                break
        assert early is not None
        for wrapper in document["step_results"]:
            if wrapper["record"].get("state_ref") is not None:
                wrapper["record"]["state_ref"] = early

    mutated = _mutated(source, tmp_path / "no-transcript", rewrite)
    report = audit_run(mutated)
    finding = _finding_for(report, InvariantId.MIPRO_GROUND_ONLY_DEVIATION)
    assert finding.status is AuditStatus.FAIL
    assert "study transcript" in finding.detail


def test_tpe_selection_fails_on_a_stale_transcript(
    miprov2_runs,
    tmp_path,
) -> None:
    """A truncated transcript cannot account for the trials that ran.

    Pointing the terminal step at the state snapshot taken before the first
    trial was recorded leaves a real, self-consistent transcript that is the
    wrong evidence for this run: it records zero samples where the control
    asked for two.
    """
    source = miprov2_runs["fewshot"]
    mutated = _stale_terminal_state(source, tmp_path / "stale-transcript")
    report = audit_run(mutated)
    trials = _finding_for(report, InvariantId.MIPRO_TRIALS_MATCH_CONTROL)
    assert trials.status is AuditStatus.FAIL
    assert "terminal failure" in trials.detail
    # The replay itself still succeeds -- an empty transcript replays
    # trivially -- which is precisely why the trial-count invariant is a
    # separate check rather than folded into the replay.
    replay = _finding_for(report, InvariantId.MIPRO_TPE_SELECTION)
    assert replay.status is AuditStatus.PASS


def _stale_terminal_state(source: Path, destination: Path) -> Path:
    """Point every step at the last state snapshot with no trials recorded."""

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        stale = None
        for wrapper in document["step_results"]:
            state_ref = wrapper["record"].get("state_ref")
            if state_ref is None:
                continue
            payload = _state_payload(run_dir, state_ref)
            state = payload.get("miprov2_state")
            transcript = (state or {}).get("study_transcript")
            if transcript is not None and not transcript.get("samples"):
                stale = state_ref
        assert stale is not None, "no snapshot has an empty transcript"
        document["step_results"][-1]["record"]["state_ref"] = stale

    return _mutated(source, destination, rewrite)


def test_tpe_selection_fails_when_the_transcript_cannot_replay(
    miprov2_runs,
) -> None:
    """A parameter the sampler would not have suggested breaks the replay.

    ``StudyTranscript`` validates internal consistency but not the sampler's
    choices, so the replay is the only thing that can catch a suggestion the
    seeded TPE would never have made. Rewriting one recorded parameter and
    re-validating the transcript proves the predicate detects it -- the
    surrounding ``Miprov2State`` refuses to hold such a transcript, which is
    why this is asserted at the transcript boundary.
    """
    from whetstone.optim.miprov2.study import StudyTranscript

    evidence = load_run_evidence(miprov2_runs["fewshot"])
    transcript = _transcript(evidence)
    payload = transcript.model_dump(mode="json")
    sample = payload["samples"][0]
    space = transcript.parameter_space
    name, value = sample["params"][0]
    replacement = (value + 1) % space.candidate_count(name)
    assert replacement != value, "the space is too small to perturb"
    sample["params"][0] = [name, replacement]
    with pytest.raises(ValueError, match="does not match parameters"):
        StudyTranscript.model_validate(payload)


def test_minibatch_sizing_fails_on_a_full_split_fan_out(
    minibatched_run,
    tmp_path,
) -> None:
    """The F16 failure: a minibatch intent that evaluates the whole valset.

    This is the concern F16 names -- deferral row expansion ignoring the
    per-intent task set and fanning out over the full split. Widening one
    trial intent's ``task_hashes`` to the entire validation set reproduces
    exactly that, and the invariant catches it.
    """
    evidence = load_run_evidence(minibatched_run)
    valset = list(_transcript(evidence).validation_task_hashes)

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        _, resolution = _first_with_purpose(document, SAMPLE_PURPOSE)
        resolution["optim_eval_request"]["task_hashes"] = valset

    mutated = _mutated(minibatched_run, tmp_path / "fan-out", rewrite)
    finding = _assert_isolated_failure(
        minibatched_run, mutated, InvariantId.MIPRO_MINIBATCH_SIZING
    )
    assert "covers" in finding.detail or "not the scheduled" in finding.detail


def test_minibatch_sizing_fails_when_no_subset_is_declared(
    miprov2_runs,
    tmp_path,
) -> None:
    """An intent with no task subset evaluates everything by default."""
    source = miprov2_runs["fewshot"]

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        _, resolution = _first_with_purpose(document, SAMPLE_PURPOSE)
        resolution["optim_eval_request"]["task_hashes"] = None

    mutated = _mutated(source, tmp_path / "no-subset", rewrite)
    finding = _assert_isolated_failure(
        source, mutated, InvariantId.MIPRO_MINIBATCH_SIZING
    )
    assert "declares no task subset" in finding.detail


def test_periodic_full_eval_fails_on_a_missing_promotion_intent(
    minibatched_run,
    tmp_path,
) -> None:
    """A recorded full evaluation with no engine intent was never paid.

    The transcript's promotion record and the engine intent that executed it
    are persisted separately, so a run could record a promotion it never
    actually evaluated. Relabelling the promotion intent breaks the
    correspondence, and the invariant reports the mismatch.
    """

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        _, resolution = _first_with_purpose(document, PROMOTION_PURPOSE)
        _metadata(resolution)["purpose"] = BASELINE_PURPOSE

    mutated = _mutated(minibatched_run, tmp_path / "lost-promotion", rewrite)
    finding = _assert_isolated_failure(
        minibatched_run, mutated, InvariantId.MIPRO_PERIODIC_FULL_EVAL
    )
    assert PROMOTION_PURPOSE in finding.detail


def test_bootstrap_through_engine_fails_on_an_orphan_effect(
    miprov2_runs,
    tmp_path,
) -> None:
    """A billed bootstrap with no engine intent was paid somewhere else.

    This is the invariant's whole point: a bootstrap generation billed to
    the proposer transport would still appear in the effect ledger but would
    have no evaluation intent behind it.

    The mutation breaks the link rather than the label. A bootstrap intent's
    ``request_id`` embeds the attempt identity hash that the billed effect
    also carries, and that shared hash is the correspondence. Rewriting the
    hash inside the request id leaves the intent purposed as a bootstrap --
    so the ordering invariant is untouched -- while the billed effect it
    used to account for now matches nothing.
    """
    source = miprov2_runs["fewshot"]

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        _, resolution = _first_with_purpose(document, BOOTSTRAP_PURPOSE)
        request = resolution["optim_eval_request"]["eval_request"]
        prefix, _, _ = request["request_id"].rpartition(":")
        request["request_id"] = f"{prefix}:{'f' * 64}"

    mutated = _mutated(source, tmp_path / "orphan-bootstrap", rewrite)
    finding = _assert_isolated_failure(
        source, mutated, InvariantId.MIPRO_BOOTSTRAP_THROUGH_ENGINE
    )
    assert "paid elsewhere" in finding.detail


def test_trials_match_control_fails_on_a_short_transcript(
    miprov2_runs,
    tmp_path,
) -> None:
    """Silent truncation of the search budget is a fidelity failure."""
    source = miprov2_runs["ground_only"]
    mutated = _stale_terminal_state(source, tmp_path / "short-transcript")
    report = audit_run(mutated)
    finding = _finding_for(report, InvariantId.MIPRO_TRIALS_MATCH_CONTROL)
    assert finding.status is AuditStatus.FAIL
    assert "terminal failure" in finding.detail
    assert not report.passed


# --------------------------------------------------------------------------
# Missing evidence FAILs rather than crashing
# --------------------------------------------------------------------------


def test_every_invariant_fails_rather_than_crashing_without_state(
    miprov2_runs,
    tmp_path,
) -> None:
    """A run that persisted no MIPROv2 state is judged, not skipped.

    An audit that raised here would report nothing at all, so a run with no
    evidence would never be marked failed -- it would simply have no audit.
    """

    def rewrite(run_dir: Path, document: dict[str, Any]) -> None:
        del run_dir
        for wrapper in document["step_results"]:
            wrapper["record"]["state_ref"] = None

    mutated = _mutated(
        miprov2_runs["fewshot"], tmp_path / "stateless", rewrite
    )
    report = audit_run(mutated)
    for invariant_id in MIPROV2_INVARIANT_IDS:
        finding = _finding_for(report, invariant_id)
        assert finding.status is AuditStatus.FAIL, invariant_id.value
        assert "cannot be shown to hold" in finding.detail
    assert not report.passed


def _no_op_rewrite(run_dir: Path, document: dict[str, Any]) -> None:
    """A rewrite that changes nothing, to prove the helper notices."""
    del run_dir, document


def test_the_mutation_helper_refuses_a_no_op(miprov2_runs, tmp_path) -> None:
    """A fixture that changed nothing would pass for the wrong reason."""
    with pytest.raises(MutationError, match="unchanged"):
        _mutated(
            miprov2_runs["fewshot"],
            tmp_path / "unchanged",
            _no_op_rewrite,
        )
