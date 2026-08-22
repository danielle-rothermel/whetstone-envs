"""The stage harness, and the durable ordering it enforces.

Stage 0 runs end to end here on a real toy c19 population and a fake
transport: real ``EvalEvidence`` behind every number, zero provider calls.
The arm stages run against injected collaborators, because what those tests
check is the *ordering* -- selection persisted before held-out is issued --
and an ordering is proven by observing it, not by running an optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.study.environment import bound_stage_environment
from whetstone_envs.optim.study.manifest import (
    PROVENANCE_AMENDED,
    PROVENANCE_ORIGINAL,
    EvidencePointer,
    RunRecord,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.selection import (
    CandidateScore,
    HeldOutMeasurement,
    ManifestSelectionLog,
    RunCandidate,
    SelectionError,
    SelectionRecord,
)
from whetstone_envs.optim.study.spec import StageId, spec_from_manifest
from whetstone_envs.optim.study.stages import (
    SEED_NOTE_CONTROL_FIELD,
    SEED_NOTE_PROVIDER_ONLY,
    ArmRunResult,
    StageEnvironment,
    StageError,
    run_arm_stage,
    run_stage,
    run_stage0_into_manifest,
)

from .conftest import HELD_OUT_CONFIG, OFFICIAL_CONFIG, toy_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.spec import ArmSpec


# --------------------------------------------------------------------------
# Stage 0, end to end on a fake transport
# --------------------------------------------------------------------------


def test_stage0_records_the_measured_design(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        result = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )

    design = result.manifest.design
    assert design is not None
    # K_CAL is a measurement input and K_REPEAT is the design's own repeat
    # count; the manifest names both because conflating them biases the
    # gate optimistically.
    assert design.k_cal == 4
    assert design.k_repeat == 3
    assert design.k_cal != design.k_repeat
    assert design.mde_formula.startswith("MDE(T, K)")
    assert design.correction == "holm-bonferroni"
    assert design.m == 4
    # The *values* matter, not just the keys: ``k_run_by_arm`` is the
    # pre-registration, so it must be what the design table says -- five
    # runs for a real optimizer, one for null-B -- and never what the arms
    # have run so far, which at Stage 0 is nothing.
    assert design.k_run_by_arm == {"copro": 5, "null-identity": 1}
    assert result.stage0 is not None
    assert result.stage0.k_cal == 4


def test_stage0_records_a_failed_gate_rather_than_erasing_it(
    tmp_path: Path,
) -> None:
    """A failed gate is a finding the study reports, not an error.

    On a fake transport both anchors land on the same score, so the
    calibration is degenerate and the gate must refuse it. What the gate
    must *not* do is discard the calibration it just measured.
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        result = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    assert result.stage0 is not None
    assert not result.stage0.passed
    assert read_study_manifest(study_dir).design is not None


def test_stage0_refuses_a_study_that_declared_no_arms(
    tmp_path: Path,
) -> None:
    """``k_run_by_arm`` is a pre-registration, so it cannot be invented."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest(arms=()))
    with (
        bound_stage_environment(study_dir) as environment,
        pytest.raises(StageError, match="declare the study's arms"),
    ):
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)


def test_stage0_through_the_named_dispatcher(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        manifest = run_stage(
            study_dir=study_dir, stage="stage0", environment=environment
        )
    assert manifest.design is not None


def test_an_unknown_stage_is_named_rather_than_ignored(
    tmp_path: Path,
) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with (
        bound_stage_environment(study_dir) as environment,
        pytest.raises(StageError, match="unknown stage"),
    ):
        run_stage(study_dir=study_dir, stage="stage9", environment=environment)


# --------------------------------------------------------------------------
# Arm stages
# --------------------------------------------------------------------------


def _pointer(char: str) -> EvidencePointer:
    return EvidencePointer(schema_name="s.schema", content_hash=char * 64)


class _Harness:
    """Injected collaborators that record the order they were called in."""

    def __init__(
        self,
        study_dir: Path,
        *,
        scores: dict[str, float],
        task_calls: int = 10,
    ) -> None:
        self.study_dir = study_dir
        self.scores = scores
        self.task_calls = task_calls
        self.events: list[str] = []
        self.selection_seen_at_held_out: list[bool] = []

    def run_optimizer(
        self, *, arm: ArmSpec, seed: int, study_dir: Path
    ) -> ArmRunResult:
        del study_dir
        run_id = f"{arm.arm_id}-{seed}"
        self.events.append(f"run:{run_id}")
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=run_id,
                seed=seed,
                candidate_name=run_id,
                template="{grid} {command} {question}",
            ),
            record=RunRecord(
                run_id=run_id,
                seed=seed,
                artifact_dir=f"/tmp/runs/{run_id}",  # noqa: S108
                result_ref=_pointer("1"),
                audit_ref=_pointer("2"),
                cost_ref=_pointer("3"),
                audit_passed=True,
                spend=(),
            ),
            observed_task_calls=self.task_calls,
        )

    def score_official(self, candidate: RunCandidate) -> CandidateScore:
        self.events.append(f"official:{candidate.run_id}")
        return CandidateScore(
            run_id=candidate.run_id,
            score=self.scores[candidate.run_id],
            per_task=(1.0, 0.0),
            eval_config_hash=OFFICIAL_CONFIG,
            completeness=1.0,
        )

    def evaluate_held_out(
        self, *, candidate_name: str, template: str
    ) -> HeldOutMeasurement:
        del template
        self.events.append(f"held_out:{candidate_name}")
        # Read the manifest off disk at the moment of the call: that is the
        # thing the ordering promises.
        self.selection_seen_at_held_out.append(
            any(
                entry.arm_id == candidate_name
                for entry in read_study_manifest(self.study_dir).selection
            )
        )
        return HeldOutMeasurement(
            candidate_name=candidate_name,
            per_task=(1.0, 0.0),
            mean=0.5,
            eval_config_hash=HELD_OUT_CONFIG,
            repeats=3,
            completeness=1.0,
        )

    def environment(self) -> StageEnvironment:
        with bound_stage_environment(self.study_dir) as base:
            return StageEnvironment(
                bind_engine=base.bind_engine,
                naive_candidate=base.naive_candidate,
                ceiling_candidate=base.ceiling_candidate,
                task_ids_by_role=base.task_ids_by_role,
                pool_ceiling=base.pool_ceiling,
                run_optimizer=self.run_optimizer,
                score_official=self.score_official,
                evaluate_held_out=self.evaluate_held_out,
            )


def _calibrated_study(tmp_path: Path) -> Path:
    """A study directory that has already run Stage 0."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)
    return study_dir


def test_an_arm_stage_refuses_to_run_before_stage0(tmp_path: Path) -> None:
    """Without a measured design there is no MDE to judge a result against."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    harness = _Harness(study_dir, scores={})
    with pytest.raises(StageError, match="run stage0 first"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(),
        )


def test_stage0_is_not_an_arm_stage(tmp_path: Path) -> None:
    study_dir = _calibrated_study(tmp_path)
    harness = _Harness(study_dir, scores={})
    with pytest.raises(StageError, match="runs no optimizers"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE0,
            environment=harness.environment(),
        )


def test_stage1_runs_every_arm_then_selects_then_measures(
    tmp_path: Path,
) -> None:
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.1 * index
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds, start=1)
    }
    harness = _Harness(study_dir, scores=scores)

    result = run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )

    # Every run is scored on official; exactly one candidate per arm
    # reaches held-out.
    held_out_events = [
        event for event in harness.events if event.startswith("held_out:")
    ]
    assert len(held_out_events) == len(spec.arms)
    # The last seed has the highest score, so it is the representative.
    for report in result.arms:
        assert report.selection.selected_run_id == report.representative.run_id
        assert report.selection.rule == "argmax-official"


def test_the_selection_is_durable_before_the_held_out_call(
    tmp_path: Path,
) -> None:
    """The read-back is a filesystem fact, not a variable still in scope."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    assert harness.selection_seen_at_held_out
    assert all(harness.selection_seen_at_held_out)


def test_every_run_is_scored_before_any_arm_is_measured(
    tmp_path: Path,
) -> None:
    """Ordering within an arm: all official, then persist, then held-out."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    for arm in spec.arms:
        official = [
            index
            for index, event in enumerate(harness.events)
            if event.startswith("official:") and arm.arm_id in event
        ]
        held_out = harness.events.index(f"held_out:{arm.arm_id}")
        assert official
        assert max(official) < held_out


def test_an_arm_stage_without_its_collaborators_refuses(
    tmp_path: Path,
) -> None:
    study_dir = _calibrated_study(tmp_path)
    with (
        bound_stage_environment(study_dir) as environment,
        pytest.raises(StageError, match="optimizer runner"),
    ):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=environment,
        )


def test_copro_and_null_b_record_the_honest_seed_note(
    tmp_path: Path,
) -> None:
    """COPRO carries no control seed, so the manifest says so."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    result = run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    notes = {arm.arm_id: arm.seed_note for arm in result.manifest.arms}
    assert notes["copro"] == SEED_NOTE_PROVIDER_ONLY
    assert notes["null-identity"] == SEED_NOTE_PROVIDER_ONLY
    assert SEED_NOTE_CONTROL_FIELD != SEED_NOTE_PROVIDER_ONLY


# --------------------------------------------------------------------------
# The manifest-backed ledger
# --------------------------------------------------------------------------


def test_a_second_selection_for_an_arm_is_refused_on_disk(
    tmp_path: Path,
) -> None:
    """L2, enforced by the durable ledger rather than by a convention."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    # A fresh ledger over the same directory sees the persisted selections,
    # and refuses a second one for an arm that already has one.
    log = ManifestSelectionLog(study_dir)
    assert log.selection_for("copro") is not None
    with pytest.raises(SelectionError, match="already selected"):
        log.record_selection(
            SelectionRecord(
                arm_id="copro",
                selected_run_id=log.require_selection("copro").selected_run_id,
                official_score=0.9,
            )
        )


def test_rerunning_a_finished_stage_refuses_rather_than_repaying(
    tmp_path: Path,
) -> None:
    """Recorded runs are not re-executed, and selection over a subset of an
    arm's runs is refused rather than silently narrowing the arg-max."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    first_pass_runs = len(
        [event for event in harness.events if event.startswith("run:")]
    )

    resumed = _Harness(study_dir, scores=scores)
    with pytest.raises(StageError, match="did not load"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=resumed.environment(),
        )
    # Nothing was re-executed: the recorded seeds were skipped, so the
    # crashed-and-resumed path never re-pays for a run it already has.
    assert not [event for event in resumed.events if event.startswith("run:")]
    assert first_pass_runs > 0


def test_the_ledger_refuses_a_selection_naming_a_run_the_arm_did_not_run(
    tmp_path: Path,
) -> None:
    """The manifest's own rule, surfaced as the protocol violation it is."""
    study_dir = _calibrated_study(tmp_path)
    log = ManifestSelectionLog(study_dir)
    with pytest.raises(SelectionError):
        log.record_selection(
            SelectionRecord(
                arm_id="copro",
                selected_run_id="a-run-that-never-happened",
                official_score=0.4,
            )
        )


def test_reading_back_before_any_selection_refuses(tmp_path: Path) -> None:
    study_dir = _calibrated_study(tmp_path)
    log = ManifestSelectionLog(study_dir)
    with pytest.raises(SelectionError, match="no persisted selection"):
        log.require_selection("copro")


def test_a_held_out_claim_survives_a_lost_process(tmp_path: Path) -> None:
    """The window L3 has to close: paid for, then crashed, then resumed.

    The claim is written *before* the evaluation is issued, so a ledger
    built fresh over the same directory -- which is what a resumed stage
    has -- refuses to measure that candidate again.
    """
    study_dir = _calibrated_study(tmp_path)
    first = ManifestSelectionLog(study_dir)
    first.claim_held_out("naive")
    # Nothing completed the claim: this is the crashed-mid-evaluation state.
    manifest = read_study_manifest(study_dir)
    assert len(manifest.held_out_claims) == 1
    assert not manifest.held_out_claims[0].completed

    resumed = ManifestSelectionLog(study_dir)
    assert resumed.held_out_count("naive") == 1
    with pytest.raises(SelectionError, match="already evaluated on held-out"):
        resumed.claim_held_out("naive")


def test_a_completed_claim_records_what_the_evaluation_returned(
    tmp_path: Path,
) -> None:
    study_dir = _calibrated_study(tmp_path)
    log = ManifestSelectionLog(study_dir)
    log.claim_held_out("naive")
    log.complete_held_out(
        HeldOutMeasurement(
            candidate_name="naive",
            per_task=(1.0, 0.0),
            mean=0.5,
            eval_config_hash=HELD_OUT_CONFIG,
            repeats=3,
            completeness=1.0,
        )
    )
    claim = read_study_manifest(study_dir).held_out_claims[0]
    assert claim.completed
    assert claim.mean == pytest.approx(0.5)
    assert claim.eval_config_hash == HELD_OUT_CONFIG
    assert claim.repeats == 3


def test_completing_an_unclaimed_evaluation_is_refused(
    tmp_path: Path,
) -> None:
    study_dir = _calibrated_study(tmp_path)
    log = ManifestSelectionLog(study_dir)
    with pytest.raises(SelectionError, match="never claimed"):
        log.complete_held_out(
            HeldOutMeasurement(
                candidate_name="naive",
                per_task=(1.0,),
                mean=1.0,
                eval_config_hash=HELD_OUT_CONFIG,
                repeats=3,
                completeness=1.0,
            )
        )


def test_an_arm_stage_leaves_a_completed_claim_per_arm(
    tmp_path: Path,
) -> None:
    """Every held-out number the stage produced is on disk as a claim."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    claims = read_study_manifest(study_dir).held_out_claims
    assert {claim.candidate_name for claim in claims} == {
        arm.arm_id for arm in spec.arms
    }
    assert all(claim.completed for claim in claims)


def test_binding_refuses_a_population_that_no_longer_matches(
    tmp_path: Path,
) -> None:
    """A changed generator must not evaluate different tasks under the
    study's name, so the bind checks content-addressed hashes rather than
    trusting that the sizes still line up."""
    study_dir = tmp_path / "study"
    manifest = toy_manifest()
    # Same shape, different pool: raising n_per_stratum keeps every split
    # size valid while changing which tasks the splits contain.
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "population": manifest.population.model_copy(
                    update={
                        "n_per_stratum": manifest.population.n_per_stratum + 1
                    }
                )
            }
        ),
    )
    with (
        pytest.raises(ValueError, match="does not match the one this study"),
        bound_stage_environment(study_dir),
    ):
        pass


def test_stage1_refuses_a_run_that_looks_like_a_fanned_out_one(
    tmp_path: Path,
) -> None:
    """The gate exists to catch a fan-out bug before Stage 2 pays for it
    five times, so it has to actually run at Stage 1."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores, task_calls=10**6)
    with pytest.raises(StageError, match="fan-out bug"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(),
        )
    # The refusal lands before any held-out evaluation is issued, so the
    # study does not pay for a number it is about to refuse to trust.
    assert not [
        event for event in harness.events if event.startswith("held_out:")
    ]


def test_stage2_does_not_re_gate_what_the_pilot_already_established(
    tmp_path: Path,
) -> None:
    """Stage 1 is where a fan-out bug is caught; re-reporting it at Stage 2
    would just restate a fact the pilot already settled."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores, task_calls=10**6)
    result = run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE2,
        environment=harness.environment(),
    )
    assert result.arms


# --------------------------------------------------------------------------
# The pre-registration Stage 0 pins
# --------------------------------------------------------------------------


def test_stage0_pins_the_pre_registration(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        result = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    pinned = result.manifest.pre_registration
    design = result.manifest.design
    assert pinned is not None
    assert design is not None
    assert pinned.provenance == PROVENANCE_ORIGINAL
    assert pinned.amended_from is None
    # The frozen block and the measured design agree on every field they
    # share, which is what the manifest's own cross-check enforces.
    assert pinned.k_repeat == design.k_repeat
    assert pinned.k_run_by_arm == design.k_run_by_arm
    assert pinned.m == design.m


def test_a_second_stage0_refuses_rather_than_restating_the_design(
    tmp_path: Path,
) -> None:
    """Re-calibrating after results exist would make the design post-hoc."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)
    with (
        bound_stage_environment(study_dir) as environment,
        pytest.raises(StageError, match="already pre-registered"),
    ):
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)


def test_replace_design_records_an_amendment(tmp_path: Path) -> None:
    """An identical re-calibration is not an amendment; a changed one is."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        first = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    original = first.manifest.pre_registration
    assert original is not None

    # Re-running with the same declared design lands on the same hash, so
    # the block is written back unchanged rather than relabelled.
    with bound_stage_environment(study_dir) as environment:
        again = run_stage0_into_manifest(
            study_dir=study_dir,
            environment=environment,
            replace_design=True,
        )
    unchanged = again.manifest.pre_registration
    assert unchanged is not None
    assert unchanged.provenance == PROVENANCE_ORIGINAL
    assert unchanged.design_hash == original.design_hash

    # A genuinely different design amends, and says which hash it replaced.
    manifest = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "design": None,
                "pre_registration": manifest.pre_registration,
            }
        ),
        replace=True,
    )
    changed = read_study_manifest(study_dir).model_copy(
        update={
            "arms": (
                *manifest.arms,
                manifest.arms[0].model_copy(update={"arm_id": "gepa"}),
            )
        }
    )
    write_study_manifest(study_dir, changed, replace=True)
    with bound_stage_environment(study_dir) as environment:
        amended = run_stage0_into_manifest(
            study_dir=study_dir,
            environment=environment,
            replace_design=True,
        )
    block = amended.manifest.pre_registration
    assert block is not None
    assert block.provenance == PROVENANCE_AMENDED
    assert block.amended_from == original.design_hash


def test_replace_design_is_refused_on_an_arm_stage(tmp_path: Path) -> None:
    study_dir = _calibrated_study(tmp_path)
    harness = _Harness(study_dir, scores={})
    with (
        pytest.raises(StageError, match="applies to stage0"),
    ):
        run_stage(
            study_dir=study_dir,
            stage="stage1",
            environment=harness.environment(),
            replace_design=True,
        )
