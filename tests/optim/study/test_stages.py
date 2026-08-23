"""The stage harness, and the durable ordering it enforces.

Stage 0 runs end to end here on a real toy c19 population and a fake
transport: real ``EvalEvidence`` behind every number, zero provider calls.
The arm stages run against injected collaborators, because what those tests
check is the *ordering* -- selection persisted before held-out is issued --
and an ordering is proven by observing it, not by running an optimizer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from whetstone.core.roles import EvalRole

pytest.importorskip("whetstone.experiment.env")

from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.optim.cost import CostRole

from whetstone_envs.optim.provider import DEFAULT_PROVIDER_CONCURRENCY
from whetstone_envs.optim.study.analysis import (
    CEILING_CANDIDATE_NAME,
    NAIVE_CANDIDATE_NAME,
)
from whetstone_envs.optim.study.environment import bound_stage_environment
from whetstone_envs.optim.study.manifest import (
    PROVENANCE_AMENDED,
    PROVENANCE_ORIGINAL,
    ArmKind,
    ArmRecord,
    CallCountGateRecord,
    EvidencePointer,
    OfficialScoreEntry,
    ReportSpendEntry,
    RunRecord,
    RunSpendRecord,
    StageRecord,
    StudyManifest,
    TransportName,
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
from whetstone_envs.optim.study.spec import (
    ArmSpec,
    StageId,
    spec_from_manifest,
)
from whetstone_envs.optim.study.spend import (
    ReportSpendLedger,
    ReportSpendRecord,
)
from whetstone_envs.optim.study.stages import (
    SEED_NOTE_CONTROL_FIELD,
    SEED_NOTE_PROVIDER_ONLY,
    STAGE1_CALL_COUNT_TOLERANCE,
    ArmRunResult,
    StageEnvironment,
    StageError,
    _arm_record,
    _persist_report_spend_to,
    _record_report_spend,
    _transport_change_amendment,
    _without_amended_evidence,
    run_arm_stage,
    run_stage,
    run_stage0_into_manifest,
)

from .conftest import (
    HELD_OUT_CONFIG,
    OFFICIAL_CONFIG,
    toy_arms,
    toy_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone.eval.protocol import EvalEngine

    from whetstone_envs.optim.codex import CodexTestSeam


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


#: The ``K_REPEAT`` the stage fixtures' design pre-registers. The harness
#: mints runs that searched at it, which is what an honest run records.
HARNESS_K_REPEAT = 3


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
        search_num_seeds: int | None = HARNESS_K_REPEAT,
    ) -> None:
        self.study_dir = study_dir
        self.scores = scores
        self.task_calls = task_calls
        #: What each run this harness mints reports having searched at.
        #: Defaults to the design's ``K_REPEAT``, which is what an honest
        #: run records; a test lowers it to drive the stage's refusal.
        self.search_num_seeds = search_num_seeds
        self.events: list[str] = []
        self.selection_seen_at_held_out: list[tuple[str, bool]] = []

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
                transport=TransportName.FAKE.value,
                spend=(),
                search_num_seeds=self.search_num_seeds,
            ),
            observed_task_calls=self.task_calls,
        )

    def load_recorded_run(
        self, *, arm: ArmSpec, run: RunRecord
    ) -> ArmRunResult | None:
        """Re-read a run an earlier stage recorded.

        Stage 2 selects over the union of its own seeds and Stage 1's, so a
        harness driving Stage 2 after a real Stage 1 has to hand back the
        earlier runs. It is rebuilt from the record rather than re-run,
        which is what the production loader does from artifacts.
        """
        del arm
        if run.seed is None:
            return None
        self.events.append(f"load:{run.run_id}")
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=run.run_id,
                seed=run.seed,
                candidate_name=run.run_id,
                template="{grid} {command} {question}",
            ),
            record=run,
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
            (
                candidate_name,
                any(
                    entry.arm_id == candidate_name
                    for entry in read_study_manifest(self.study_dir).selection
                ),
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

    def environment(
        self,
        *,
        real_codex_authorized: bool = False,
        codex_test_seam: CodexTestSeam | None = None,
    ) -> StageEnvironment:
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
                load_recorded_run=self.load_recorded_run,
                real_codex_authorized=real_codex_authorized,
                codex_test_seam=codex_test_seam,
            )


def _calibrated_study(tmp_path: Path) -> Path:
    """A study directory that has already run Stage 0."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)
    return study_dir


def _piloted_study(tmp_path: Path) -> Path:
    """A study whose Stage 1 ran and cleared the call-count gate.

    Stage 2 requires exactly that, so every Stage-2 test that is not about
    the prerequisite itself starts from here rather than from Stage 0.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=_Harness(study_dir, scores=scores).environment(),
    )
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
    # reaches held-out, plus the two anchors the analysis pass measures
    # through the identical procedure (L4).
    held_out_events = [
        event.removeprefix("held_out:")
        for event in harness.events
        if event.startswith("held_out:")
    ]
    assert set(held_out_events) == {
        *(arm.arm_id for arm in spec.arms),
        NAIVE_CANDIDATE_NAME,
        CEILING_CANDIDATE_NAME,
    }
    assert len(held_out_events) == len(set(held_out_events))
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
    # Every *arm* measurement saw its own persisted selection. The anchors
    # have no selection to see -- there is one candidate by construction --
    # so they are excluded rather than asserted about.
    arm_ids = {arm.arm_id for arm in spec.arms}
    seen = [
        seen
        for name, seen in harness.selection_seen_at_held_out
        if name in arm_ids
    ]
    assert len(seen) == len(arm_ids)
    assert all(seen)


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


# --------------------------------------------------------------------------
# A Codex-bearing stage refuses before it spends
# --------------------------------------------------------------------------


def _codex_study(tmp_path: Path) -> Path:
    """A calibrated study whose design names the Codex arm.

    The Codex arm is declared *after* an ordinary arm on purpose: what the
    early refusal buys is that the arms ahead of it are never paid for, and
    an arm order that put Codex first could not show that.
    """
    study_dir = tmp_path / "study"
    codex = ArmRecord(
        arm_id="codex",
        optimizer="codex",
        kind=ArmKind.REAL,
        demo_mode=None,
        train_size=None,
        val_size=None,
        control_identity_hash="f" * 64,
        seed_note="control-seed-field",
        runs=(),
    )
    from whetstone_envs.optim.codex import (
        CODEX_DEFAULT_AGENT_MODEL,
    )

    # A study declaring the Codex arm pre-registers the agent it will run,
    # and the stage guard refuses a resolved control that disagrees.
    write_study_manifest(
        study_dir,
        toy_manifest(
            arms=(*toy_arms(), codex),
            codex_agent_model=CODEX_DEFAULT_AGENT_MODEL,
        ),
    )
    with bound_stage_environment(study_dir) as environment:
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)
    return study_dir


@pytest.mark.parametrize("stage", [StageId.STAGE1, StageId.STAGE2])
def test_an_unauthorized_codex_stage_refuses_before_any_arm_runs(
    tmp_path: Path, stage: StageId
) -> None:
    """The spend-safety property: zero arms dispatched, so zero spend.

    Fails-before: the refusal lived only inside ``run_optimizer``, which
    the Codex arm reaches on *its* turn -- after every arm ordered ahead of
    it had already run and been paid for. The stage then aborted with a
    bill and no result. Asserting the harness recorded no events at all is
    what makes "before any arm runs" checkable rather than asserted.
    """
    from whetstone_envs.optim.codex import RealCodexRefusedError

    study_dir = _codex_study(tmp_path)
    harness = _Harness(study_dir, scores={})
    with pytest.raises(RealCodexRefusedError, match="not authorized"):
        run_arm_stage(
            study_dir=study_dir,
            stage=stage,
            environment=harness.environment(),
        )
    assert harness.events == []


def test_an_authorized_codex_stage_still_needs_the_environment_half(
    tmp_path: Path, monkeypatch
) -> None:
    """The flag alone does not authorize spend, here as in the runner."""
    from whetstone_envs.optim.codex import (
        ALLOW_REAL_CODEX_ENV,
        RealCodexRefusedError,
    )

    monkeypatch.delenv(ALLOW_REAL_CODEX_ENV, raising=False)
    study_dir = _codex_study(tmp_path)
    harness = _Harness(study_dir, scores={})
    with pytest.raises(RealCodexRefusedError):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(real_codex_authorized=True),
        )
    assert harness.events == []


def _scripted_preflight_seam() -> CodexTestSeam:
    """A seam whose preflight succeeds without any Codex at all.

    The early guard's probe is a *real session probe* on the production
    path, so a test that satisfies the opt-in and supplies no seam spawns
    the real CLI -- which is exactly what happened before this seam
    existed. Naming the scripted preflight keeps the test about the gate.
    """
    from whetstone.testing.runtime import scripted_codex_preflight

    from whetstone_envs.optim.codex import CodexTestSeam

    return CodexTestSeam(
        preflight=lambda **_kwargs: scripted_codex_preflight(),
        environment={},
    )


def _failing_preflight_seam(message: str) -> CodexTestSeam:
    """A seam standing in for a Codex this machine cannot run.

    An unsupported platform, an absent binary, and an expired session all
    reach the guard as a raising preflight, so one raising stand-in covers
    the class without needing a real broken Codex to reproduce.
    """
    from whetstone.optim.codex.preflight import CodexPreflightError

    from whetstone_envs.optim.codex import CodexTestSeam

    def _raise(**_kwargs: object) -> None:
        raise CodexPreflightError(message)

    return CodexTestSeam(preflight=_raise, environment={})


def test_both_halves_of_the_opt_in_lift_the_early_refusal(
    tmp_path: Path, monkeypatch
) -> None:
    """With both halves present and a usable session, the stage runs.

    The Codex arm's own run goes through the injected runner and the
    guard's session probe goes through the scripted seam, so nothing here
    reaches a provider or spawns a CLI. Its point is that the gate, once
    satisfied, stops refusing: without this the flag would be
    unfalsifiable.
    """
    from whetstone_envs.optim.codex import (
        ALLOW_REAL_CODEX_ENV,
        ALLOW_REAL_CODEX_ENV_VALUE,
    )

    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)
    study_dir = _codex_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.1 * index
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds, start=1)
    }
    harness = _Harness(study_dir, scores=scores)

    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(
            real_codex_authorized=True,
            codex_test_seam=_scripted_preflight_seam(),
        ),
    )

    assert any(event.startswith("run:codex-") for event in harness.events)


def test_an_unusable_codex_refuses_before_any_arm_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """Authorization is not usability, and the difference costs money.

    Fails-before: both opt-in halves present, the stage proceeded and
    discovered an unusable Codex -- unsupported platform, missing binary,
    expired session -- only when the Codex arm's own turn arrived, after
    COPRO, MIPROv2, and GEPA had been paid for. The empty event list is
    what makes "before any arm runs" checkable rather than asserted.
    """
    from whetstone_envs.optim.codex import (
        ALLOW_REAL_CODEX_ENV,
        ALLOW_REAL_CODEX_ENV_VALUE,
    )

    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)
    study_dir = _codex_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)

    with pytest.raises(StageError, match="cannot run a Codex session") as exc:
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(
                real_codex_authorized=True,
                codex_test_seam=_failing_preflight_seam("no auth source"),
            ),
        )

    assert harness.events == []
    # The preflight's own diagnosis survives as the cause rather than
    # being replaced by the stage's summary of it, so an operator can act
    # on *why* the session is unusable.
    assert "no auth source" in str(exc.value)
    assert exc.value.__cause__ is not None


def test_the_preflight_probes_the_agent_model_not_the_task_model(
    tmp_path: Path, monkeypatch
) -> None:
    """The guard clears the route the Codex arm will actually open.

    The task model and the Codex agent model are different products on
    different routes: the task model is an OpenRouter route the Codex CLI
    cannot run at all, and a subscription session refuses it before the
    agent emits a token.

    Fails-before: the preflight passed ``spec.task_model``, so it probed a
    route no arm would ever use. On this machine the scripted seam accepts
    anything, so the wrong model went unnoticed here -- but a real study
    would clear the guard against a route the CLI refuses and then fail on
    the Codex arm's turn, after the arms ahead of it had been paid for,
    which is the exact late failure this preflight exists to prevent.

    Asserted on the seam's own ``model`` kwarg, which is the value the
    production preflight forwards to the session probe.
    """
    from whetstone_envs.optim.codex import (
        ALLOW_REAL_CODEX_ENV,
        ALLOW_REAL_CODEX_ENV_VALUE,
        CODEX_DEFAULT_AGENT_MODEL,
        CodexTestSeam,
        resolve_codex_agent_model,
    )

    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)
    study_dir = _codex_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.1 * index
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds, start=1)
    }
    harness = _Harness(study_dir, scores=scores)

    probed: list[object] = []

    def _capture(**kwargs: object) -> None:
        from whetstone.testing.runtime import scripted_codex_preflight

        probed.append(kwargs.get("model"))
        scripted_codex_preflight()

    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(
            real_codex_authorized=True,
            codex_test_seam=CodexTestSeam(preflight=_capture, environment={}),
        ),
    )

    assert probed, "the Codex arm's preflight never ran"
    expected = resolve_codex_agent_model(None)
    assert expected == CODEX_DEFAULT_AGENT_MODEL
    assert probed == [expected]
    # The task model is the wrong route, and naming it here is what makes
    # the assertion above a real discrimination rather than a tautology.
    assert spec.task_model != expected
    assert spec.task_model not in probed


def test_a_stage_with_no_codex_arm_needs_no_authorization(
    tmp_path: Path,
) -> None:
    """The guard is scoped to the arm that can bill, and to nothing else."""
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.1 * index
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds, start=1)
    }
    harness = _Harness(study_dir, scores=scores)
    run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=harness.environment(),
    )
    assert harness.events


def test_an_arm_stage_without_its_collaborators_refuses(
    tmp_path: Path,
) -> None:
    """A stage that cannot run, score, or measure refuses before spending.

    The bound environment supplies all three, so the refusal is shown
    against one deliberately stripped of them -- which is the state any
    caller assembling a ``StageEnvironment`` by hand can reach.
    """
    study_dir = _calibrated_study(tmp_path)
    with bound_stage_environment(study_dir) as bound:
        stripped = StageEnvironment(
            bind_engine=bound.bind_engine,
            naive_candidate=bound.naive_candidate,
            ceiling_candidate=bound.ceiling_candidate,
            task_ids_by_role=bound.task_ids_by_role,
            pool_ceiling=bound.pool_ceiling,
        )
        with pytest.raises(StageError, match="optimizer runner"):
            run_arm_stage(
                study_dir=study_dir,
                stage=StageId.STAGE1,
                environment=stripped,
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
    log = ManifestSelectionLog(
        study_dir,
        stage=StageId.STAGE1.value,
        transport=TransportName.FAKE.value,
    )
    assert log.selection_for("copro") is not None
    with pytest.raises(SelectionError, match="already selected"):
        log.record_selection(
            SelectionRecord(
                arm_id="copro",
                selected_run_id=log.require_selection("copro").selected_run_id,
                official_score=0.9,
                stage=StageId.STAGE1.value,
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

    # A stage given *no* loader is the case under test, so the loader the
    # harness otherwise supplies is dropped explicitly rather than by
    # relying on it being absent by default.
    resumed = _Harness(study_dir, scores=scores)
    with pytest.raises(StageError, match="no way to load them"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=replace(resumed.environment(), load_recorded_run=None),
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
    log = ManifestSelectionLog(study_dir, transport=TransportName.FAKE.value)
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
    log = ManifestSelectionLog(study_dir, transport=TransportName.FAKE.value)
    with pytest.raises(SelectionError, match="no persisted selection"):
        log.require_selection("copro")


def test_a_held_out_claim_survives_a_lost_process(tmp_path: Path) -> None:
    """The window L3 has to close: paid for, then crashed, then resumed.

    The claim is written *before* the evaluation is issued, so a ledger
    built fresh over the same directory -- which is what a resumed stage
    has -- refuses to measure that candidate again.
    """
    study_dir = _calibrated_study(tmp_path)
    first = ManifestSelectionLog(study_dir, transport=TransportName.FAKE.value)
    first.claim_held_out("naive")
    # Nothing completed the claim: this is the crashed-mid-evaluation state.
    manifest = read_study_manifest(study_dir)
    assert len(manifest.held_out_claims) == 1
    assert not manifest.held_out_claims[0].completed

    resumed = ManifestSelectionLog(
        study_dir, transport=TransportName.FAKE.value
    )
    assert resumed.held_out_count("naive") == 1
    with pytest.raises(SelectionError, match="already evaluated on held-out"):
        resumed.claim_held_out("naive")


def test_a_completed_claim_records_what_the_evaluation_returned(
    tmp_path: Path,
) -> None:
    study_dir = _calibrated_study(tmp_path)
    log = ManifestSelectionLog(study_dir, transport=TransportName.FAKE.value)
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
    log = ManifestSelectionLog(study_dir, transport=TransportName.FAKE.value)
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
    # Both anchors are claimed too: they reach held-out through the same
    # ledger and the same once-only rule as an arm, which is what makes L3
    # and L4 hold for them rather than only for the arms.
    assert {claim.candidate_name for claim in claims} == {
        *(arm.arm_id for arm in spec.arms),
        NAIVE_CANDIDATE_NAME,
        CEILING_CANDIDATE_NAME,
    }
    assert all(claim.completed for claim in claims)
    assert {claim.stage for claim in claims} == {StageId.STAGE1.value}


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


def test_stage1_records_its_call_count_verdict(tmp_path: Path) -> None:
    """The verdict outlives the process that reached it.

    Fails-before: the gate was evaluated inside Stage 1 and its result was
    kept nowhere, so nothing downstream could tell a cleared pilot from a
    pilot that never ran.
    """
    study_dir = _piloted_study(tmp_path)
    gate = read_study_manifest(study_dir).call_count_gate
    assert gate is not None
    assert gate.stage == StageId.STAGE1.value
    assert gate.passed
    assert gate.overruns == ()


def test_a_failed_stage1_gate_is_recorded_with_its_overruns(
    tmp_path: Path,
) -> None:
    """A failed gate is a finding the study records, not an absence.

    Recording it is what lets the Stage-2 refusal below say the pilot
    *failed* rather than reporting the same missing-pilot message it would
    for a study that never ran one.
    """
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
    gate = read_study_manifest(study_dir).call_count_gate
    assert gate is not None
    assert not gate.passed
    assert gate.overruns


def test_stage2_refuses_when_no_stage1_gate_was_recorded(
    tmp_path: Path,
) -> None:
    """The pilot is a prerequisite, not a suggestion.

    Fails-before: a Stage 2 invoked directly after Stage 0 ran the full
    five-run design without the call-count gate ever having been
    evaluated, which is exactly the five-times-over bill the pilot exists
    to prevent. Asserting the harness recorded no events is what makes
    "before dispatch, zero spend" checkable.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    with pytest.raises(StageError, match="recorded no such gate"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE2,
            environment=harness.environment(),
        )
    assert harness.events == []


def test_stage2_refuses_behind_a_stage1_whose_gate_failed(
    tmp_path: Path,
) -> None:
    """A failed pilot blocks the full design rather than being skipped.

    Fails-before: the Stage-1 failure aborted that stage only. Re-invoking
    the command at Stage 2 skipped ``_check_call_counts`` entirely -- it
    returns early for any stage but Stage 1 -- and paid for the whole
    design behind a gate that had already caught a fan-out.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    failed = _Harness(study_dir, scores=scores, task_calls=10**6)
    with pytest.raises(StageError, match="fan-out bug"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=failed.environment(),
        )
    harness = _Harness(study_dir, scores=scores)
    with pytest.raises(StageError, match="pilot failed it"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE2,
            environment=harness.environment(),
        )
    assert harness.events == []


def test_stage2_does_not_re_gate_what_the_pilot_already_established(
    tmp_path: Path,
) -> None:
    """Stage 1 is where a fan-out bug is caught; re-reporting it at Stage 2
    would just restate a fact the pilot already settled.

    The pilot here passed, so Stage 2 runs -- and it runs even though this
    harness reports call counts that *would* fail the gate, which is the
    point: the gate is the pilot's, not Stage 2's.
    """
    study_dir = _piloted_study(tmp_path)
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


def test_an_amendment_discards_the_previous_pilot_gate(
    tmp_path: Path,
) -> None:
    """A pilot verdict describes the design it was computed against.

    Once ``--replace-design`` records an amendment, the recorded Stage-1
    gate no longer describes the study, and Stage 2 must not be able to
    spend against it. An identical re-calibration keeps the verdict: the
    design did not change, so neither did what the pilot measured.
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        first = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    original = first.manifest.pre_registration
    assert original is not None
    gate = CallCountGateRecord(
        stage=StageId.STAGE1.value,
        passed=True,
        tolerance=STAGE1_CALL_COUNT_TOLERANCE,
        overruns=(),
    )
    manifest = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        manifest.model_copy(update={"call_count_gate": gate}),
        replace=True,
    )

    with bound_stage_environment(study_dir) as environment:
        again = run_stage0_into_manifest(
            study_dir=study_dir,
            environment=environment,
            replace_design=True,
        )
    assert again.manifest.call_count_gate == gate

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
    assert amended.manifest.call_count_gate is None


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


# --------------------------------------------------------------------------
# Crashing between two arms' held-out evaluations
# --------------------------------------------------------------------------


class _CrashingHarness(_Harness):
    """A harness that dies once, after ``crash_after`` arms have reported.

    The crash lands where the deadlock lives: between two arms, with the
    first arm's selection and completed held-out claim already durable and
    the rest of the stage unrun.
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        scores: dict[str, float],
        crash_after: int,
    ) -> None:
        super().__init__(study_dir, scores=scores)
        self._crash_after = crash_after
        self._completed = 0

    def score_official(self, candidate: RunCandidate) -> CandidateScore:
        # Crash while scoring the *next* arm, once the target number of arms
        # have finished. Scoring precedes selection, so the interrupted arm
        # has neither selected nor claimed: this is the between-arms window,
        # not the in-flight-evaluation one.
        if self._completed >= self._crash_after:
            raise _InjectedCrashError(candidate.run_id)
        return super().score_official(candidate)

    def evaluate_held_out(
        self, *, candidate_name: str, template: str
    ) -> HeldOutMeasurement:
        measurement = super().evaluate_held_out(
            candidate_name=candidate_name, template=template
        )
        self._completed += 1
        return measurement


class _InjectedCrashError(RuntimeError):
    """Stands in for the process dying mid-stage."""


def test_a_stage_that_crashed_between_arms_resumes(tmp_path: Path) -> None:
    """**A crash mid-stage must not strand every paid run forever.**

    ``report_arm`` persists each arm's selection as it goes, and the ledger
    refuses a second selection per arm per stage. A resume that re-reported
    every arm would therefore hit that refusal on the arms that already
    selected -- turning a recoverable crash into a permanent failure with
    the study's paid runs stranded behind it.

    The resumed stage must instead rebuild the already-reported arms from
    what is durable, spend nothing further, and return the same selections.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 + index / 100
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds)
    }
    crashing = _CrashingHarness(study_dir, scores=scores, crash_after=1)
    with pytest.raises(_InjectedCrashError):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=crashing.environment(),
        )
    # The crash landed where it was aimed. Selection is persisted before
    # held-out is issued, so the arm the crash interrupted has a durable
    # selection too -- what distinguishes the arm that already *paid* is a
    # completed held-out claim.
    crashed_manifest = read_study_manifest(study_dir)
    stage1 = StageId.STAGE1.value
    completed = tuple(
        entry.candidate_name
        for entry in crashed_manifest.held_out_claims
        if entry.stage == stage1 and entry.completed
    )
    assert len(completed) == 1
    first_arm = completed[0]
    # The reported arm left its selection behind, which is exactly the
    # state that makes a naive re-report hit "already selected".
    assert first_arm in {
        entry.arm_id
        for entry in crashed_manifest.selection
        if entry.stage == stage1
    }
    first_selection = next(
        entry
        for entry in crashed_manifest.selection
        if entry.arm_id == first_arm and entry.stage == stage1
    )

    resumed = _Harness(study_dir, scores=scores)

    def _load_recorded_run(
        *, arm: ArmSpec, run: RunRecord
    ) -> ArmRunResult | None:
        """Re-read a recorded run without re-running it.

        A resumed stage selects over every recorded seed, so the runs paid
        for before the crash have to arrive from their records rather than
        from the runner -- which is the whole point of not re-paying.
        """
        del arm
        assert run.seed is not None, "a recorded run at a stage seed"
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=run.run_id,
                seed=run.seed,
                candidate_name=run.run_id,
                template="{grid} {command} {question}",
            ),
            record=run,
            observed_task_calls=resumed.task_calls,
        )

    environment = resumed.environment()
    result = run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=replace(environment, load_recorded_run=_load_recorded_run),
    )

    # Nothing was re-run: every seed was already recorded before the crash.
    assert not [event for event in resumed.events if event.startswith("run:")]
    # And the arm that already spent its one held-out shot did not spend a
    # second one -- the ledger's claim is what it must be rebuilt from.
    assert f"held_out:{first_arm}" not in resumed.events

    # The resumed report agrees with what was persisted before the crash.
    rebuilt = next(
        report for report in result.arms if report.arm_id == first_arm
    )
    assert rebuilt.selection.selected_run_id == (
        first_selection.selected_run_id
    )
    assert rebuilt.selection.official_score == first_selection.official_score
    # Every arm is reported, and each selected exactly once.
    assert {report.arm_id for report in result.arms} == {
        arm.arm_id for arm in spec.arms
    }
    final = read_study_manifest(study_dir)
    stage1_selections = [
        entry for entry in final.selection if entry.stage == stage1
    ]
    assert len(stage1_selections) == len(
        {entry.arm_id for entry in stage1_selections}
    )


def test_a_resumed_arm_rescores_nothing_it_already_paid_to_score(
    tmp_path: Path,
) -> None:
    """**Resume of a fully reported arm re-buys nothing at all.**

    Official-selection scoring is a provider call per run, and it used to
    run unconditionally on every invocation: a resumed stage re-scored
    every run of every already-reported arm purely to rebuild a report the
    manifest could already answer. That is a second charge for a number
    the study had already bought, and it is invisible in the result --
    the rebuilt report looks identical either way.

    So the assertion is on the calls, not on the report: the arm that
    completed before the crash issues **zero** official scorings on
    resume, and its rebuilt scores equal the ones it recorded.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 + index / 100
        for arm in spec.arms
        for index, seed in enumerate(arm.seeds)
    }
    crashing = _CrashingHarness(study_dir, scores=scores, crash_after=1)
    with pytest.raises(_InjectedCrashError):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=crashing.environment(),
        )
    stage1 = StageId.STAGE1.value
    crashed = read_study_manifest(study_dir)
    first_arm = next(
        entry.candidate_name
        for entry in crashed.held_out_claims
        if entry.stage == stage1 and entry.completed
    )
    # The scores that arm bought are durable, which is what a rebuild
    # reads instead of re-issuing.
    recorded = {
        entry.run_id: entry
        for entry in crashed.official_scores
        if entry.stage == stage1 and entry.arm_id == first_arm
    }
    assert recorded, "a reported arm records the scores it paid for"

    resumed = _Harness(study_dir, scores=scores)

    def _load_recorded_run(
        *, arm: ArmSpec, run: RunRecord
    ) -> ArmRunResult | None:
        del arm
        assert run.seed is not None
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=run.run_id,
                seed=run.seed,
                candidate_name=run.run_id,
                template="{grid} {command} {question}",
            ),
            record=run,
            observed_task_calls=resumed.task_calls,
        )

    environment = resumed.environment()
    result = run_arm_stage(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=replace(environment, load_recorded_run=_load_recorded_run),
    )

    # Zero scorer calls for the runs the crashed invocation already scored.
    rescored = {
        event.removeprefix("official:")
        for event in resumed.events
        if event.startswith("official:")
    }
    assert not (rescored & set(recorded)), (
        "a resumed stage re-bought official scores it had already paid for: "
        f"{sorted(rescored & set(recorded))}"
    )
    # And the rebuilt report carries the recorded numbers, not new ones.
    rebuilt = next(
        report for report in result.arms if report.arm_id == first_arm
    )
    for score in rebuilt.official_scores:
        assert score.score == recorded[score.run_id].score
        assert score.per_task == recorded[score.run_id].per_task


def test_refolding_the_reporting_bill_restates_it_rather_than_doubling_it(
    tmp_path: Path,
) -> None:
    """**The reporting fold is safe to repeat, because a resume repeats it.**

    A stage's row is *merged* rather than replaced, so a resume cannot
    erase what an earlier invocation paid. That rule and an additive
    reporting fold are incompatible: each invocation added the whole pass
    again, so a stage whose reporting ran twice claimed to have spent
    twice -- and the reporting pass is exactly the set of calls every
    efficacy claim is finally made against.

    The fold now reads the manifest's own durable per-evaluation records,
    so it is a function of what is on disk rather than of what this
    process bought. Folding again restates the same total.
    """
    study_dir = _calibrated_study(tmp_path)
    stage1 = StageId.STAGE1.value
    manifest = read_study_manifest(study_dir)
    priced = RunSpendRecord(
        role=CostRole.TASK_MODEL.value,
        calls=4,
        cached_calls=0,
        input_tokens=100,
        output_tokens=20,
        priced_calls=4,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.5,
    )
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "report_spend": (
                    ReportSpendEntry(
                        evidence_schema="whetstone.eval.outputs",
                        evidence_content_hash="a" * 64,
                        purpose="official",
                        candidate_name="copro",
                        stage=stage1,
                        transport=TransportName.OPENROUTER.value,
                        spend=(priced,),
                    ),
                )
            }
        ),
        replace=True,
    )
    environment = _Harness(study_dir, scores={}).environment()
    paid = replace(environment, transport=TransportName.OPENROUTER.value)

    _record_report_spend(
        study_dir=study_dir, stage=StageId.STAGE1, environment=paid
    )
    once = next(
        entry
        for entry in read_study_manifest(study_dir).stages
        if entry.stage == stage1
    )
    assert [entry.usd for entry in once.report_spend] == [0.5]

    # The resume: the same durable records, folded a second time.
    _record_report_spend(
        study_dir=study_dir, stage=StageId.STAGE1, environment=paid
    )
    twice = next(
        entry
        for entry in read_study_manifest(study_dir).stages
        if entry.stage == stage1
    )
    assert twice.report_spend == once.report_spend, (
        "re-folding the reporting pass billed its evaluations a second time"
    )
    assert [entry.calls for entry in twice.report_spend] == [4]


def test_the_reporting_pass_refreshes_the_attempts_of_an_existing_row(
    tmp_path: Path,
) -> None:
    """**A paid stage's row exists before the reporting pass ever runs.**

    **Fails-before: 0 attempts, no outcomes.** The attempt counters were
    read off the transport only in ``_record_report_spend``'s ``existing
    is None`` branch. Stage 0 takes that branch -- it writes its row once,
    at the end -- so the bug was invisible there. Stage 1 and Stage 2 do
    not: the run pass records the stage long before the reporting pass
    folds its bill onto the same row, so the ``else`` branch is the only
    one a paid arm stage ever reaches, and it carried the run pass's
    counters through untouched. Every retry the reporting pass fought --
    the official scoring and held-out evaluation that every efficacy claim
    is made against -- was dropped on the floor.

    **Folded, never assigned.** The row's counters belong to the
    invocations that already ran and the reporter's to this one, and a
    resumed stage's second process starts its transport's tally at zero.
    Assigning would report the resume's retries as the stage's whole
    history and silently drop the original run's, so a stage that fought
    the provider hard, crashed, and resumed cleanly would read as one that
    never retried at all -- which is the crash shape this asserts.
    """
    study_dir = _calibrated_study(tmp_path)
    stage1 = StageId.STAGE1.value
    manifest = read_study_manifest(study_dir)
    priced = RunSpendRecord(
        role=CostRole.TASK_MODEL.value,
        calls=4,
        cached_calls=0,
        input_tokens=100,
        output_tokens=20,
        priced_calls=4,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.5,
    )
    # The row the *run* pass left behind, carrying the attempts that pass
    # made. This is the state a paid Stage 1 is always in by the time its
    # reporting pass starts.
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "report_spend": (
                    ReportSpendEntry(
                        evidence_schema="whetstone.eval.outputs",
                        evidence_content_hash="a" * 64,
                        purpose="official",
                        candidate_name="copro",
                        stage=stage1,
                        transport=TransportName.OPENROUTER.value,
                        spend=(priced,),
                    ),
                ),
                "stages": (
                    StageRecord(
                        stage=stage1,
                        transport=TransportName.OPENROUTER.value,
                        provider_attempts=7,
                        provider_transient_outcomes=("rate_limited",),
                    ),
                ),
            }
        ),
        replace=True,
    )
    environment = _Harness(study_dir, scores={}).environment()
    reporting = replace(
        environment,
        transport=TransportName.OPENROUTER.value,
        # This process's transport, whose tally starts at zero and counts
        # only what the reporting pass itself fought.
        provider_attempts=_StubAttemptReporter(
            attempts=4, transient_outcomes=("server_error", "rate_limited")
        ),
    )

    _record_report_spend(
        study_dir=study_dir, stage=StageId.STAGE1, environment=reporting
    )
    row = next(
        entry
        for entry in read_study_manifest(study_dir).stages
        if entry.stage == stage1
    )
    assert row.provider_attempts == 11, (
        "the reporting pass's attempts did not reach the existing row"
    )
    # In order, run pass first: the sequence is what tells a rate-limit
    # storm apart from a run of transient 5xx.
    assert row.provider_transient_outcomes == (
        "rate_limited",
        "server_error",
        "rate_limited",
    )
    # The reporting bill still landed, and the run-side row is untouched.
    assert [entry.usd for entry in row.report_spend] == [0.5]


def test_a_fake_reporting_pass_leaves_an_existing_rows_attempts_alone(
    tmp_path: Path,
) -> None:
    """Nothing to add is not the same as zero to add.

    A pass that bound no retrying transport reached no provider, so it
    contributes nothing rather than a measured zero -- and must not reset
    a row whose counters an earlier paid invocation wrote. Asserted at the
    fold helper because ``_record_report_spend`` returns early on a
    fake-transport environment, which is a different guard for a different
    reason.
    """
    from whetstone_envs.optim.study.stages import _folded_attempts

    existing = StageRecord(
        stage=StageId.STAGE1.value,
        transport=TransportName.OPENROUTER.value,
        provider_attempts=7,
        provider_transient_outcomes=("rate_limited",),
    )
    environment = _Harness(
        _calibrated_study(tmp_path), scores={}
    ).environment()
    assert environment.provider_attempts is None
    assert _folded_attempts(existing, environment) == {}


def test_a_priced_reporting_evaluation_is_durable_before_the_pass_ends(
    tmp_path: Path,
) -> None:
    """**The spend is on disk the moment it is paid, not at the end.**

    The reporting pass buys an official score per run, a held-out
    measurement per arm, and the anchors, and it writes the stage's row
    only once all of them are done. Spend held in memory across that
    window is lost by a crash inside it -- and lost reporting spend is
    invisible afterwards, because the resume rebuilds its claims without
    re-evaluating and never learns what the crashed invocation bought.

    So the sink is installed before the pass rather than after: each
    priced evaluation appends its own entry, and an evaluation whose
    evidence is already recorded for this stage is a no-op, because one
    evaluation cited twice was paid for once.
    """
    study_dir = _calibrated_study(tmp_path)
    stage = StageId.STAGE1
    environment = _Harness(study_dir, scores={}).environment()
    ledger = ReportSpendLedger(None)
    _persist_report_spend_to(
        study_dir=study_dir,
        stage=stage,
        environment=replace(environment, report_spend=ledger),
    )
    priced = RunSpendRecord(
        role=CostRole.TASK_MODEL.value,
        calls=2,
        cached_calls=0,
        input_tokens=10,
        output_tokens=3,
        priced_calls=2,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.25,
    )
    record = ReportSpendRecord(
        purpose="official",
        candidate_name="copro",
        evidence_key=("whetstone.eval.outputs", "a" * 64),
        spend=(priced,),
    )
    # The ledger calls its sink the moment an evaluation is priced.
    assert ledger._persist is not None
    ledger._persist(record)

    entries = [
        entry
        for entry in read_study_manifest(study_dir).report_spend
        if entry.stage == stage.value
    ]
    assert len(entries) == 1
    assert entries[0].evidence_key == record.evidence_key
    assert entries[0].purpose == "official"
    assert entries[0].candidate_name == "copro"
    assert [item.usd for item in entries[0].spend] == [0.25]

    # The same evaluation, cited again, is one purchase and not two.
    ledger._persist(record)
    assert (
        len(
            [
                entry
                for entry in read_study_manifest(study_dir).report_spend
                if entry.stage == stage.value
            ]
        )
        == 1
    )


# --------------------------------------------------------------------------
# What a cross-transport amendment takes with it
# --------------------------------------------------------------------------


def _reporting_spend() -> RunSpendRecord:
    return RunSpendRecord(
        role=CostRole.TASK_MODEL.value,
        calls=4,
        cached_calls=0,
        input_tokens=100,
        output_tokens=20,
        priced_calls=4,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.5,
    )


def _study_with_a_reported_stage1(tmp_path: Path) -> Path:
    """A fake-transport study whose Stage 1 ran, scored, and was billed.

    Exactly the state a cross-transport ``--replace-design`` invalidates:
    a Stage-1 row, a run under an arm, the official score that run bought,
    and the reporting evaluation that was paid for to produce it.
    """
    study_dir = _calibrated_study(tmp_path)
    stage1 = StageId.STAGE1.value
    manifest = read_study_manifest(study_dir)
    run = RunRecord(
        run_id="copro-0",
        seed=0,
        artifact_dir="/tmp/runs/copro-0",  # noqa: S108
        result_ref=_pointer("1"),
        audit_ref=_pointer("2"),
        cost_ref=_pointer("3"),
        audit_passed=True,
        transport=TransportName.FAKE.value,
        spend=(),
    )
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "stages": (
                    *manifest.stages,
                    StageRecord(
                        stage=stage1, transport=TransportName.FAKE.value
                    ),
                ),
                "arms": tuple(
                    arm.model_copy(update={"runs": (run,)})
                    if arm.arm_id == "copro"
                    else arm
                    for arm in manifest.arms
                ),
                "official_scores": (
                    OfficialScoreEntry(
                        run_id=run.run_id,
                        arm_id="copro",
                        stage=stage1,
                        transport=TransportName.FAKE.value,
                        score=0.5,
                        eval_config_hash=OFFICIAL_CONFIG,
                        completeness=1.0,
                        per_task=(0.5,),
                    ),
                ),
                "report_spend": (
                    ReportSpendEntry(
                        evidence_schema="whetstone.eval.outputs",
                        evidence_content_hash="a" * 64,
                        purpose="official",
                        candidate_name="copro",
                        stage=stage1,
                        transport=TransportName.FAKE.value,
                        spend=(_reporting_spend(),),
                    ),
                ),
            }
        ),
        replace=True,
    )
    return study_dir


def _replaced_across_transports(study_dir: Path) -> StudyManifest:
    """``study_dir``'s manifest after a fake-to-paid re-calibration.

    Driven through the amendment path itself rather than hand-written, so
    what these tests assert is what the stage really produces.
    """
    manifest = read_study_manifest(study_dir)
    paid = replace(
        _Harness(study_dir, scores={}).environment(),
        transport=TransportName.OPENROUTER.value,
    )
    amendment = _transport_change_amendment(
        manifest, environment=paid, replace_design=True
    )
    assert amendment is not None, "the fixture has evidence to drop"
    return _without_amended_evidence(manifest, amendment=amendment)


def test_an_amendment_drops_the_scores_its_dropped_runs_bought(
    tmp_path: Path,
) -> None:
    """An official score is a measurement of the run that bought it.

    Run ids are deterministic, so the replacement stage recomputes the
    very names the amendment just dropped. A score left behind is read
    back by ``official_score_for`` and reused -- a number measured on the
    *previous* transport, presented as this study's selection evidence and
    never re-bought on the transport the study now runs on.
    """
    study_dir = _study_with_a_reported_stage1(tmp_path)

    amended = _replaced_across_transports(study_dir)

    assert amended.official_scores == (), (
        "a score measured on the previous transport survived its run"
    )
    record = amended.amendments[-1]
    assert record.dropped_official_scores == 1, (
        "the amendment records what it dropped, not merely drops it"
    )


def test_a_score_bought_on_another_transport_is_not_read_back(
    tmp_path: Path,
) -> None:
    """Belt and braces: the read-back checks what the score was measured on.

    The drop is the fix; this is the guard that holds if a score ever
    reaches the manifest by another route. Reusing another transport's
    measurement is the failure, whatever put it there.
    """
    study_dir = _study_with_a_reported_stage1(tmp_path)
    entry = read_study_manifest(study_dir).official_scores[0]

    same = ManifestSelectionLog(
        study_dir,
        stage=StageId.STAGE1.value,
        transport=TransportName.FAKE.value,
    )
    assert same.official_score_for(entry.run_id) is not None, (
        "the stage's own measurement is still what a resume reads back"
    )

    other = ManifestSelectionLog(
        study_dir,
        stage=StageId.STAGE1.value,
        transport=TransportName.OPENROUTER.value,
    )
    assert other.official_score_for(entry.run_id) is None, (
        "a score bought on another transport is not this stage's evidence"
    )


def test_an_amendment_drops_the_reporting_spend_it_invalidated(
    tmp_path: Path,
) -> None:
    """Reporting spend belongs to the invocation that bought it.

    The amendment removes the Stage-1 row but the fold is computed from
    the durable per-evaluation records, not from the row. Entries left
    behind are folded by the *next* Stage 1 -- so a paid stage bills
    itself for a fake-transport invocation's evaluations, which is a total
    nobody owes.
    """
    study_dir = _study_with_a_reported_stage1(tmp_path)

    amended = _replaced_across_transports(study_dir)

    assert amended.report_spend == (), (
        "the dropped stage's reporting purchases survived its row"
    )
    record = amended.amendments[-1]
    assert record.dropped_report_spend == 1, (
        "the amendment records what it dropped, not merely drops it"
    )


def test_the_reporting_fold_ignores_another_transports_purchase(
    tmp_path: Path,
) -> None:
    """Belt and braces: the fold keys on ``(stage, transport)``.

    An evaluation bought on one transport is not part of what a stage
    running on another transport spent, so it can never reach that
    stage's row.
    """
    study_dir = _study_with_a_reported_stage1(tmp_path)
    stale = read_study_manifest(study_dir).report_spend[0]
    assert stale.transport == TransportName.FAKE.value

    _record_report_spend(
        study_dir=study_dir,
        stage=StageId.STAGE1,
        environment=replace(
            _Harness(study_dir, scores={}).environment(),
            transport=TransportName.OPENROUTER.value,
        ),
    )

    row = next(
        (
            entry
            for entry in read_study_manifest(study_dir).stages
            if entry.stage == StageId.STAGE1.value
        ),
        None,
    )
    assert row is not None
    assert row.report_spend == (), (
        "a fake-transport purchase was folded into a paid stage's bill"
    )


def test_the_stage_rebuild_keeps_the_pinned_search_shape() -> None:
    """The shape ``init`` pinned must survive the post-stage rebuild.

    ``_arm_record`` *replaces* an arm's record once a stage finishes, so a
    field it does not restate is dropped even though ``init`` wrote it.
    All four search-shape fields were omitted, so the moment Stage 1
    completed, ``copro_breadth``/``copro_depth`` and MIPROv2's
    ``num_trials``/``num_candidates`` fell to ``None`` on every arm.

    ``_recorded_search`` then projected those arms to ``{}`` while the
    pinned block still said ``{'breadth': 6, 'depth': 3}`` and
    ``{'num_candidates': 3, 'num_trials': 10}``, and Stage 2 refused the
    study with ``PreRegistrationViolationError`` -- after Stage 1 had
    already spent an hour running every arm at the correct shape. The
    records were wrong, not the runs.

    Fails-before: both assertions read ``None``.
    """
    arm = ArmSpec(
        arm_id="miprov2",
        optimizer="miprov2",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(2000,),
        miprov2_num_trials=10,
        miprov2_num_candidates=3,
        train_size=44,
        val_size=44,
    )
    copro = ArmSpec(
        arm_id="copro",
        optimizer="copro",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(1000,),
        copro_breadth=6,
        copro_depth=3,
    )
    for spec, expected in (
        (arm, {"num_trials": 10, "num_candidates": 3}),
        (copro, {"breadth": 6, "depth": 3}),
    ):
        prior = ArmRecord(
            arm_id=spec.arm_id,
            optimizer=spec.optimizer,
            kind=spec.kind,
            demo_mode=None,
            train_size=spec.train_size,
            val_size=spec.val_size,
            control_identity_hash="d" * 64,
            seed_note="provider-seed-control-only",
            runs=(
                RunRecord(
                    run_id=f"{spec.arm_id}-{spec.seeds[0]}",
                    seed=spec.seeds[0],
                    artifact_dir="/tmp/runs/x",  # noqa: S108
                    result_ref=_pointer("1"),
                    audit_ref=_pointer("2"),
                    cost_ref=_pointer("3"),
                    audit_passed=True,
                    transport=TransportName.FAKE.value,
                    spend=(),
                ),
            ),
        )
        record = _arm_record(spec, runs=prior.runs, sample=(), prior=prior)
        assert record.miprov2_num_trials == expected.get("num_trials")
        assert record.miprov2_num_candidates == expected.get("num_candidates")
        assert record.copro_breadth == expected.get("breadth")
        assert record.copro_depth == expected.get("depth")


# --------------------------------------------------------------------------
# The design's repeat count, structurally and after the fact
# --------------------------------------------------------------------------


def test_an_arm_stage_refuses_a_run_that_searched_below_k_repeat(
    tmp_path: Path,
) -> None:
    """A run that searched at 1 under K_REPEAT = 3 is not recorded.

    Fails-before: nothing diffed a run's own repeat count against the
    design's. Each optimizer's ``*_repeats_as_recorded`` audit holds a
    run's evaluations to the count that *same run* recorded, so a run that
    consistently recorded and searched at one passed its audit, was
    recorded, and entered selection -- having bought a third of the
    evidence the pre-registration priced. Both numbers are named, because
    either one of them could be the wrong one.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    assert read_study_manifest(study_dir).design is not None
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores, search_num_seeds=1)
    with pytest.raises(StageError, match="K_REPEAT = 3") as caught:
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(),
        )
    assert "searched at 1 repeat(s) per task" in str(caught.value)
    # Refused rather than recorded: the manifest keeps no run bought under
    # a search the study never registered.
    assert all(not arm.runs for arm in read_study_manifest(study_dir).arms)


def test_an_arm_stage_refuses_a_run_whose_repeat_count_is_unknown(
    tmp_path: Path,
) -> None:
    """A run recording no single repeat count is refused, not recorded.

    Fails-before: ``None`` meant "not established", and an unestablished
    count recorded as a run's evidence is indistinguishable from one that
    was checked. A run whose evaluations disagree with each other has no
    search repeat count to hold to the design, which is a refusal rather
    than a number to pick from.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores, search_num_seeds=None)
    with pytest.raises(StageError, match="no single repeat count"):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=harness.environment(),
        )


def test_an_arm_stage_refuses_an_engine_that_would_not_sample_k_repeat(
    tmp_path: Path,
) -> None:
    """The structural half: refused before any arm is dispatched.

    Fails-before: an engine bound at the wrong repeat count was only ever
    caught by the recorded-run diff, which is a whole stage of provider
    spend later. Binding issues no evaluation, so this refusal is free --
    and it is the one that keeps a misbound stage from buying anything at
    all.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    base = harness.environment()

    def bind_at_one(*, role: EvalRole, num_seeds: int) -> EvalEngine:
        """A binder that ignores the count it is asked for."""
        del num_seeds
        return base.bind_engine(role=role, num_seeds=1)

    with pytest.raises(
        StageError,
        match=r"would bind the (internal|official|held_out) engine sampling 1",
    ):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=replace(base, bind_engine=bind_at_one),
        )
    # Before dispatch: no arm ran, so the stage bought nothing.
    assert harness.events == []


def test_an_arm_stage_refuses_an_engine_that_misbinds_only_the_internal_role(
    tmp_path: Path,
) -> None:
    """A binder that honours one role and not another is still refused.

    The optimizers search on the internal engine; a probe of the official
    engine alone would pass this binder and discover the mismatch a full
    stage of spend later.
    """
    study_dir = _calibrated_study(tmp_path)
    spec = spec_from_manifest(read_study_manifest(study_dir))
    scores = {
        f"{arm.arm_id}-{seed}": 0.5 for arm in spec.arms for seed in arm.seeds
    }
    harness = _Harness(study_dir, scores=scores)
    base = harness.environment()

    def bind_at_one(*, role: EvalRole, num_seeds: int) -> EvalEngine:
        """A binder that honours the count for every role but internal."""
        if role is not EvalRole.INTERNAL:
            return base.bind_engine(role=role, num_seeds=num_seeds)
        return base.bind_engine(role=role, num_seeds=1)

    with pytest.raises(
        StageError,
        match=r"would bind the (internal|official|held_out) engine sampling 1",
    ):
        run_arm_stage(
            study_dir=study_dir,
            stage=StageId.STAGE1,
            environment=replace(base, bind_engine=bind_at_one),
        )
    # Before dispatch: no arm ran, so the stage bought nothing.
    assert harness.events == []


# Provider concurrency: bound onto the engine, recorded on the stage
# --------------------------------------------------------------------------


def _runtime_engine(engine: object) -> RuntimeEvalEngine:
    """The concrete engine, so its scheduling width can be read."""
    assert isinstance(engine, RuntimeEvalEngine)
    return engine


def _concurrency_of(engine: RuntimeEvalEngine) -> int:
    """The width the engine schedules rows at.

    Read off the private attribute deliberately: whetstone exposes no
    public accessor for it, and the alternative -- inferring the width by
    timing concurrent rows -- would make a scheduling assertion depend on
    wall-clock behaviour, which is exactly what a test must not do.
    """
    width = engine._concurrency
    assert isinstance(width, int)
    return width


def test_the_bound_engine_runs_at_the_requested_concurrency(
    tmp_path: Path,
) -> None:
    """The width reaches the engine that actually schedules the rows.

    Fails-before: ``ReferenceEvalRuntimeConfig.build_engine`` forwards no
    concurrency, so every engine the study bound fell back to whetstone's
    ``DEFAULT_CONCURRENCY`` of 5 no matter what the operator asked for --
    a setting could be accepted and recorded while the run stayed at 5.

    Read off the engine's own attribute rather than by timing anything:
    what is under test is which number the scheduler holds.
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(
        study_dir, provider_concurrency=17
    ) as environment:
        engine = _runtime_engine(
            environment.bind_engine(role=EvalRole.INTERNAL, num_seeds=1)
        )
        assert _concurrency_of(engine) == 17
        # Derived engines share it: a stage narrows to a task or a seed
        # constantly, and a derivation that dropped back to the default
        # would silently undo the setting partway through the stage.
        derived = engine.for_task_ids(
            (environment.task_ids_by_role[EvalRole.INTERNAL][0],)
        )
        assert _concurrency_of(derived) == 17


def test_an_unnamed_concurrency_binds_the_recorded_default(
    tmp_path: Path,
) -> None:
    """What a stage run with no flag actually ran at."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        engine = _runtime_engine(
            environment.bind_engine(role=EvalRole.INTERNAL, num_seeds=1)
        )
        assert _concurrency_of(engine) == DEFAULT_PROVIDER_CONCURRENCY
        assert environment.provider_concurrency == (
            DEFAULT_PROVIDER_CONCURRENCY
        )


def test_binding_refuses_a_concurrency_below_one(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with (
        pytest.raises(ValueError, match="at least 1"),
        bound_stage_environment(study_dir, provider_concurrency=0),
    ):
        pass


class _StubAttemptReporter:
    """A transport's attempt counters, without a transport.

    ``StageEnvironment`` takes the reporter as a protocol precisely so a
    test can assert the numbers reach the record without standing up an
    HTTP client and a rate limiter to produce them.
    """

    def __init__(
        self, *, attempts: int, transient_outcomes: tuple[str, ...]
    ) -> None:
        self.attempts = attempts
        self.transient_outcomes = transient_outcomes


def test_stage0_records_the_attempts_its_transport_made(
    tmp_path: Path,
) -> None:
    """Retries reach the record as attempts, beside the calls they cost.

    **Fails-before: ``StageRecord`` had no such field.** The wrapper owns
    the whole retry budget and whetstone's driver is pinned to one
    attempt, so a call that survived two 429s persisted one row -- and
    every spend surface re-derives from rows. A stage that spent its
    afternoon fighting a rate limit recorded exactly the same numbers as
    one that sailed through, and nothing in the manifest could tell them
    apart.

    This is the one number that cannot be projected: a retried attempt
    persists nothing, so the transport's own counter is the only record.
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    reporter = _StubAttemptReporter(
        attempts=3, transient_outcomes=("rate_limited", "rate_limited")
    )
    with bound_stage_environment(study_dir) as environment:
        result = run_stage0_into_manifest(
            study_dir=study_dir,
            environment=replace(environment, provider_attempts=reporter),
        )
    stage = next(
        entry for entry in result.manifest.stages if entry.stage == "stage0"
    )
    assert stage.provider_attempts == 3
    assert stage.provider_transient_outcomes == (
        "rate_limited",
        "rate_limited",
    )
    # ``calls`` still counts persisted rows, so the retries did not
    # inflate the bill: the attempts aggregate into the row beside it.
    assert all(entry.calls <= 3 for entry in stage.spend)


def test_a_fake_transport_stage_reports_no_attempts(tmp_path: Path) -> None:
    """No provider was reached, so there is nothing to report -- not zero.

    A fake stage binds no retrying transport, and recording ``0``
    attempts would claim it measured an absence of retries rather than
    never having been in a position to retry. The record's default is
    what says "not applicable".
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(study_dir) as environment:
        assert environment.provider_attempts is None
        result = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    stage = next(
        entry for entry in result.manifest.stages if entry.stage == "stage0"
    )
    assert stage.provider_attempts == 0
    assert stage.provider_transient_outcomes == ()


def test_stage0_records_the_concurrency_it_ran_at(tmp_path: Path) -> None:
    """The width lands on the stage record, like the transport.

    Fails-before: ``StageRecord`` had no such field, so a stage's wall
    time and its rate-limit failures were uninterpretable after the fact
    -- nothing said how wide the run had been.
    """
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    with bound_stage_environment(
        study_dir, provider_concurrency=9
    ) as environment:
        result = run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        )
    stage = next(
        entry for entry in result.manifest.stages if entry.stage == "stage0"
    )
    assert stage.provider_concurrency == 9
    # And it is not design: the pre-registration is unchanged by it.
    assert result.manifest.pre_registration is None or (
        "provider_concurrency"
        not in result.manifest.pre_registration.model_dump()
    )
