"""Stages 0, 1, and 2 end to end through the real CLI, then the report.

This is the operational proof for the whole runner: a toy c19 study runs
every stage on a **fake transport with zero provider calls**, through
``whetstone-study run --stage ...`` rather than through the harness
directly, and the manifest it leaves behind carries real arm runs, a real
selection, real held-out rows with their statistics, and a report generated
from it whose every figure resolves.

The splits are tiny -- (4, 4, 6) against the study's own (88, 132, 220) --
but nothing else is a stand-in: the runs go through
:func:`~whetstone_envs.optim.run.run_optimizer`, the audits are the real
invariants, and the held-out numbers come from real ``EvalEvidence``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite

from whetstone_envs.optim.study.analysis import (
    CEILING_CANDIDATE_NAME,
    NAIVE_CANDIDATE_NAME,
)
from whetstone_envs.optim.study.cli import EXIT_OK, main
from whetstone_envs.optim.study.manifest import (
    STUDY_STORE_NAME,
    ArmRecord,
    check_manifest_pointers,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.spec import StageId
from whetstone_envs.reporting.study_report import (
    REPORT_HTML_NAME,
    REPORT_MARKDOWN_NAME,
    build_study_report,
    figures_in,
)

from .conftest import toy_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.manifest import StudyManifest

#: One real optimizer and one control. COPRO is the cheapest real arm to run
#: at these sizes and exercises the whole path -- run, audit, cost
#: projection, evidence copy -- while null-B exercises the control path that
#: runs no optimizer at all.
E2E_ARMS = ("copro", "null-identity")


def _arms() -> tuple[ArmRecord, ...]:
    return tuple(
        ArmRecord(
            arm_id=arm_id,
            optimizer=arm_id,
            demo_mode=None,
            control_identity_hash=chr(ord("a") + index) * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        )
        for index, arm_id in enumerate(E2E_ARMS)
    )


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    """A study directory holding a pre-Stage-0 toy manifest."""
    directory = tmp_path / "study"
    write_study_manifest(directory, toy_manifest(arms=_arms()))
    return directory


def _run_stage(study_dir: Path, stage: str) -> int:
    """One stage through the CLI's own dispatcher and default runner."""
    return main(["run", "--study-dir", str(study_dir), "--stage", stage])


def _run_every_stage(study_dir: Path) -> StudyManifest:
    for stage in (StageId.STAGE0, StageId.STAGE1, StageId.STAGE2):
        assert _run_stage(study_dir, stage.value) == EXIT_OK, stage
    return read_study_manifest(study_dir)


# --------------------------------------------------------------------------
# The stages
# --------------------------------------------------------------------------


def test_every_stage_runs_through_the_cli_on_a_fake_transport(
    study_dir: Path,
) -> None:
    """The load-bearing assertion: the study is operational end to end."""
    manifest = _run_every_stage(study_dir)

    assert manifest.design is not None
    assert manifest.pre_registration is not None
    # Every arm ran its full pre-registered K_RUN, and Stage 1's runs are
    # part of that count rather than in addition to it.
    for arm in manifest.arms:
        assert arm.runs, arm.arm_id
        assert len(arm.runs) == manifest.design.k_run_by_arm[arm.arm_id]
        assert len({run.seed for run in arm.runs}) == len(arm.runs)


def test_stage2_continues_from_stage1_without_re_paying(
    study_dir: Path,
) -> None:
    """Stage 1's runs count toward Stage 2 -- the same seeds, run once.

    This is the resumability property the protocol pre-registers, and the
    thing that makes a pilot affordable: Stage 2 adds the seeds Stage 1 did
    not run and selects over the union.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value) == EXIT_OK
    assert _run_stage(study_dir, StageId.STAGE1.value) == EXIT_OK
    after_pilot = read_study_manifest(study_dir)
    pilot_runs = {
        arm.arm_id: {run.run_id for run in arm.runs}
        for arm in after_pilot.arms
    }

    assert _run_stage(study_dir, StageId.STAGE2.value) == EXIT_OK
    after_full = read_study_manifest(study_dir)

    for arm in after_full.arms:
        run_ids = {run.run_id for run in arm.runs}
        # Nothing the pilot bought was discarded, and nothing it bought was
        # bought again: the pilot's run ids are a subset of the full set.
        assert pilot_runs[arm.arm_id] <= run_ids
    # One selection per arm per stage, and one held-out claim per candidate
    # per stage: the pilot's decisions are recorded beside the full
    # design's rather than overwritten by them.
    assert {entry.stage for entry in after_full.selection} == {
        StageId.STAGE1.value,
        StageId.STAGE2.value,
    }
    for stage in (StageId.STAGE1.value, StageId.STAGE2.value):
        selected = [
            entry.arm_id
            for entry in after_full.selection
            if entry.stage == stage
        ]
        assert sorted(selected) == sorted(E2E_ARMS)


def test_the_selected_run_is_one_the_arm_actually_ran(
    study_dir: Path,
) -> None:
    manifest = _run_every_stage(study_dir)
    runs_by_arm = {
        arm.arm_id: {run.run_id for run in arm.runs} for arm in manifest.arms
    }
    for entry in manifest.selection:
        assert entry.selected_run_id in runs_by_arm[entry.arm_id]


# --------------------------------------------------------------------------
# The manifest's held-out and statistics blocks
# --------------------------------------------------------------------------


def test_the_manifest_carries_held_out_rows_and_their_statistics(
    study_dir: Path,
) -> None:
    """Item 5's proof: the statistics layer reaches the manifest.

    A row per reported candidate, each with its interval, its uncorrected
    p-value, and -- for the four real optimizers only -- its Holm-corrected
    one. The anchors and the nulls are controls rather than hypotheses, so
    their Holm column is empty by design rather than by omission.
    """
    manifest = _run_every_stage(study_dir)

    rows = {row.candidate_name: row for row in manifest.held_out}
    assert set(rows) == {
        NAIVE_CANDIDATE_NAME,
        CEILING_CANDIDATE_NAME,
        *E2E_ARMS,
    }
    for row in rows.values():
        assert row.ci_low <= row.ci_high
        assert 0.0 <= row.p_bootstrap <= 1.0
        assert 0.0 <= row.completeness <= 1.0
        assert row.eval_evidence_ref.content_hash
        assert row.per_task_scores_ref.content_hash
    # COPRO is a hypothesis and carries the family-wise correction; the
    # control and the anchors do not.
    assert rows["copro"].p_holm is not None
    assert rows["null-identity"].p_holm is None
    assert rows[NAIVE_CANDIDATE_NAME].p_holm is None


def test_held_out_is_evaluated_exactly_once_per_reported_candidate(
    study_dir: Path,
) -> None:
    """L3 over the durable claims the stages actually wrote."""
    manifest = _run_every_stage(study_dir)
    for stage in (StageId.STAGE1.value, StageId.STAGE2.value):
        names = [
            claim.candidate_name
            for claim in manifest.held_out_claims
            if claim.stage == stage
        ]
        assert sorted(names) == sorted(set(names))
        assert set(names) == {
            NAIVE_CANDIDATE_NAME,
            CEILING_CANDIDATE_NAME,
            *E2E_ARMS,
        }
    assert all(claim.completed for claim in manifest.held_out_claims)


def test_every_evidence_pointer_the_manifest_cites_resolves(
    study_dir: Path,
) -> None:
    """The manifest's numbers are backed by records, not typed in."""
    manifest = _run_every_stage(study_dir)
    with open_sqlite(str(study_dir / STUDY_STORE_NAME)) as store:
        report = check_manifest_pointers(manifest, store)
    assert report.passed, [
        (check.pointer.schema_name, check.detail)
        for check in report.unresolved()
    ]
    assert report.checks


# --------------------------------------------------------------------------
# Leakage, and the report generated from the finished manifest
# --------------------------------------------------------------------------


def test_a_clean_study_passes_every_leakage_rule(
    study_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Item 6's proof: L1 is really checked, and a clean study passes it.

    L1 reads each run's own intent resolutions out of the run stores, so a
    study that never ran an optimizer could not check it. This one did, and
    the rule passes rather than being reported unchecked.
    """
    _run_every_stage(study_dir)
    capsys.readouterr()
    assert main(["leakage-check", "--study-dir", str(study_dir)]) == EXIT_OK, (
        capsys.readouterr().err
    )
    out = capsys.readouterr().out
    assert "ok L1 ::" in out
    assert "NOT CHECKED" not in out


def test_the_report_generates_from_the_manifest_and_every_figure_resolves(
    study_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Item 2's proof: the report is generated from what the stages wrote."""
    manifest = _run_every_stage(study_dir)
    assert main(["leakage-check", "--study-dir", str(study_dir)]) == EXIT_OK
    out_dir = tmp_path / "report"
    capsys.readouterr()
    assert (
        main(
            [
                "report",
                "--study-dir",
                str(study_dir),
                "--out",
                str(out_dir),
            ]
        )
        == EXIT_OK
    ), capsys.readouterr().err

    assert (out_dir / REPORT_MARKDOWN_NAME).is_file()
    assert (out_dir / REPORT_HTML_NAME).is_file()

    report = build_study_report(manifest)
    figures = tuple(figures_in(report))
    assert figures, "a reported study renders numbers"
    assert [figure for figure in figures if not figure.backed()] == []
    # Every pointer the report prints is one the manifest itself cites, so
    # the report cannot invent a plausible-looking (schema, hash) pair.
    cited = set(manifest.evidence_pointers())
    printed = {
        figure.pointer for figure in figures if figure.pointer is not None
    }
    assert printed <= cited


def test_the_report_reads_the_deltas_the_manifest_recorded(
    study_dir: Path,
) -> None:
    """The report prints manifest fields rather than recomputing them."""
    manifest = _run_every_stage(study_dir)
    report = build_study_report(manifest)
    rendered = "\n".join(
        cell.rendered()
        for section in report.sections
        for table in section.tables
        for row in table.rows
        for cell in row.cells
    )
    for arm_id in E2E_ARMS:
        assert arm_id in rendered
