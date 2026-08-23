"""The ``whetstone-study`` CLI.

Every subcommand is exercised with injected collaborators, so these tests
run without the stage harness executing and without the report generator
writing a packet. Two exceptions prove the defaults are real rather than
stubs: the ``manifest check`` tests resolve pointers against a store a real
fake-transport optimizer run produced in-test, and
``test_report_defaults_to_the_real_generator`` asserts ``report`` binds the
report package rather than reporting a wiring gap.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import rfc8785

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite

from whetstone_envs.optim.run_cost import RUN_COST_SCHEMA_NAME
from whetstone_envs.optim.split import COPRO_SHAPED_OPTIMIZERS
from whetstone_envs.optim.study.cli import (
    DEFAULT_STORE_NAME,
    ESTIMATE_LABEL,
    EXIT_CHECK_FAILED,
    EXIT_ERROR,
    EXIT_OK,
    MEASURED_BASIS_BY_ARM,
    MEASURED_BASIS_DEFAULT,
    MEASURED_LABEL,
    MEASURED_TASK_CALLS_BY_ARM,
    NO_ESTIMATE,
    NOT_CHECKED,
    OPTIMIZER_BUDGET_HEADING,
    PROGRAM_NAME,
    build_parser,
    main,
    plan_lines,
)
from whetstone_envs.optim.study.manifest import (
    STUDY_MANIFEST_NAME,
    ArmRecord,
    EvidencePointer,
    RunRecord,
    RunSpendRecord,
    StudyManifest,
    TransportName,
    read_study_manifest,
    study_manifest_path,
    write_study_manifest,
)
from whetstone_envs.optim.study.protocols import (
    COPRO_BREADTH,
    COPRO_DEPTH,
)

from .conftest import toy_manifest
from .test_manifest import _minimal_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _Spec:
    """A stand-in for Wave 4a's ``StudySpec``, structural only."""

    def __init__(self) -> None:
        self.study_id = "step10-fixture"
        self.arm_ids = ("copro", "null-identity")
        self.k_run_by_arm: Mapping[str, int] = {
            "copro": 5,
            "null-identity": 1,
        }
        self.k_repeat = 3
        self.split_sizes = (88, 132, 440)

    @property
    def optimizer_by_arm(self) -> Mapping[str, str]:
        """Each arm's optimizer.

        These fixtures name every arm after its optimizer, which is the
        common case; the study's own MIPROv2 demo-mode arms are where the
        two names come apart, and ``test_protocols`` covers that.
        """
        return {arm_id: arm_id for arm_id in self.arm_ids}

    @property
    def copro_shape_by_arm(self) -> Mapping[str, tuple[int, int] | None]:
        """Each arm's pinned COPRO shape, at the protocol's own values.

        Derived from ``arm_ids`` rather than listed, so a fixture that
        renames or adds an arm keeps a shape for every arm it declares --
        which is the block's own rule, and what ``plan`` reads for each.
        Only the COPRO-shaped arms carry one, as in the real design.
        """
        return {
            arm_id: (
                (COPRO_BREADTH, COPRO_DEPTH)
                if arm_id in COPRO_SHAPED_OPTIMIZERS
                else None
            )
            for arm_id in self.arm_ids
        }


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def test_parser_names_the_console_script() -> None:
    assert build_parser().prog == PROGRAM_NAME == "whetstone-study"


def test_the_console_script_is_registered_and_is_this_program() -> None:
    """``whetstone-study`` is installed, and it is this module's ``main``.

    The parser naming itself ``whetstone-study`` proves nothing about
    whether anything installs that name; this reads the distribution's own
    entry points, so a program documented as a console script cannot ship
    without one.
    """
    from importlib.metadata import entry_points

    scripts = {
        entry.name: entry.value
        for entry in entry_points(group="console_scripts")
        if entry.name == PROGRAM_NAME
    }
    assert scripts == {PROGRAM_NAME: "whetstone_envs.optim.study.cli:main"}
    assert entry_points(group="console_scripts")[PROGRAM_NAME].load() is main


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_rejects_an_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run", "--study-dir", "/tmp/s", "--stage", "stage9"]  # noqa: S108
        )


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_plan_lines_derive_the_budget_from_the_matrix() -> None:
    lines = plan_lines(_Spec())
    text = "\n".join(lines)
    assert "study: step10-fixture" in text
    assert "splits: internal=88 official=132 held_out=440" in text
    # 5 runs x 132 official tasks x 3 repeats, and 1 x 132 x 3.
    assert "total official rows: 2376" in text
    # One representative candidate per arm on held-out: 2 x 440 x 3.
    assert "total held-out rows: 2640" in text
    assert "total selection+report rows: 5016" in text


def test_plan_states_the_pre_registered_mde() -> None:
    """The number a reader authorizes spend against, at the design's sizes.

    Pinned as literals at the study's own held-out 440 and ``K_REPEAT`` 3:
    these are the protocol review's recomputed design points, so a plan
    that printed anything else would be quoting a design the review never
    checked.

    Fails-before: ``plan`` printed a run matrix and two budgets and no MDE
    at all, so nothing in the command a reader runs before authorizing
    spend said what effect the design could resolve.
    """
    text = "\n".join(plan_lines(_Spec()))
    assert "pre-registered MDE" in text
    assert "tau^2=0.05" in text
    assert "T=440 K=3" in text
    assert "MDE=0.0622" in text
    assert "tau^2=0.1" in text
    assert "MDE=0.0690" in text


def test_the_plan_mde_is_labelled_as_pre_registered_not_measured() -> None:
    """An estimate and a measurement are never printed in the same voice.

    Stage 0 measures both variances and records the MDE that follows; this
    row is computed at the worst-case ``sigma^2`` beforehand, and a reader
    comparing the two needs to see which is which.
    """
    text = "\n".join(plan_lines(_Spec()))
    assert "worst-case sigma^2=0.25" in text
    assert "from power.py" in text


def test_plan_prints_the_matrix(tmp_path: Path, capsys) -> None:
    code = main(
        ["plan", "--study-dir", str(tmp_path)],
        load_spec=lambda study_dir: _Spec(),  # noqa: ARG005
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "copro" in out
    assert "null-identity" in out
    assert "total selection+report rows: 5016" in out


def test_plan_passes_the_study_directory_to_its_loader(
    tmp_path: Path,
) -> None:
    seen: list[Path] = []

    def load(study_dir: Path) -> _Spec:
        seen.append(study_dir)
        return _Spec()

    assert main(["plan", "--study-dir", str(tmp_path)], load_spec=load) == (
        EXIT_OK
    )
    assert seen == [tmp_path]


def test_plan_defaults_to_the_real_manifest_backed_loader(
    tmp_path: Path, capsys
) -> None:
    """The default wiring is the real loader, not a stub or a gap report."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, _minimal_manifest())
    assert main(["plan", "--study-dir", str(study_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "study: step10-2026-08-22" in out
    # The minimal manifest carries no design, so the loader falls back to
    # the spec's own defaults rather than inventing measured ones.
    assert "K_REPEAT: 3" in out


def test_plan_on_a_missing_manifest_fails_cleanly(
    tmp_path: Path, capsys
) -> None:
    assert main(["plan", "--study-dir", str(tmp_path)]) == EXIT_ERROR
    assert capsys.readouterr().err.strip()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["stage0", "stage1", "stage2"])
def test_run_dispatches_each_stage(tmp_path: Path, capsys, stage: str) -> None:
    seen: list[tuple[Path, str]] = []

    def run_stage(  # noqa: PLR0913
        *,
        study_dir: Path,
        stage: str,
        replace_design: bool = False,
        allow_real_codex: bool = False,
        discard_stale_runs: bool = False,
        transport: str = "fake",
    ) -> StudyManifest:
        assert replace_design is False
        # An unflagged invocation authorizes no spend. Asserted rather than
        # ignored: the default is what stops an accidental Codex bill, and
        # the fake transport is what stops an accidental provider bill.
        assert allow_real_codex is False
        # Nor does it authorize discarding a run directory it cannot
        # claim: that directory may be paid evidence.
        assert discard_stale_runs is False
        assert transport == "fake"
        seen.append((study_dir, stage))
        return _minimal_manifest()

    code = main(
        ["run", "--study-dir", str(tmp_path), "--stage", stage],
        run_stage=run_stage,
    )
    assert code == EXIT_OK
    assert seen == [(tmp_path, stage)]
    out = capsys.readouterr().out
    assert f"{stage} complete for study step10-2026-08-22" in out
    assert str(tmp_path / STUDY_MANIFEST_NAME) in out


def test_run_defaults_to_the_real_stage_harness(
    tmp_path: Path, capsys
) -> None:
    """The default runner reaches the harness, not a wiring-gap report.

    A study directory with no manifest now fails on the manifest read
    inside the harness, which is what proves the default is wired: the old
    behaviour was to refuse before touching the directory at all.
    """
    code = main(["run", "--study-dir", str(tmp_path), "--stage", "stage0"])
    assert code == EXIT_ERROR
    error = capsys.readouterr().err
    assert "not wired into this CLI yet" not in error
    assert STUDY_MANIFEST_NAME in error


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def test_report_hands_the_manifest_to_its_generator(
    tmp_path: Path, capsys
) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, _minimal_manifest())
    packet = tmp_path / "packet"
    seen: list[tuple[str, Path]] = []

    def generate(*, manifest: StudyManifest, out_dir: Path) -> Path:
        seen.append((manifest.study_id, out_dir))
        return out_dir / "report.html"

    code = main(
        [
            "report",
            "--study-dir",
            str(study_dir),
            "--out",
            str(packet),
        ],
        generate_report=generate,
    )
    assert code == EXIT_OK
    assert seen == [("step10-2026-08-22", packet)]
    assert str(packet / "report.html") in capsys.readouterr().out


def test_report_on_a_missing_manifest_fails_cleanly(
    tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "report",
            "--study-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "packet"),
        ],
        generate_report=lambda *, manifest, out_dir: out_dir,  # noqa: ARG005
    )
    assert code == EXIT_ERROR
    assert capsys.readouterr().err.strip()


def test_report_defaults_to_the_real_generator(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """``report`` needs no injected generator: the default is the real one.

    The gap this used to assert is closed, so what is asserted now is that
    an un-injected ``report`` writes an actual packet. The output-root
    guard is relaxed for both writers because a test's ``tmp_path`` is
    legitimately outside a repository on a real run but sits under this
    checkout here.
    """
    from whetstone_envs.reporting.study_report import (
        REPORT_HTML_NAME,
        REPORT_MARKDOWN_NAME,
    )

    for module in (
        "whetstone_envs.optim.study.manifest",
        "whetstone_envs.reporting.study_report",
    ):
        monkeypatch.setattr(
            f"{module}.validate_output_root", lambda path: path.resolve()
        )
    study_dir = tmp_path / "study"
    packet = tmp_path / "packet"
    write_study_manifest(study_dir, _minimal_manifest())

    code = main(
        ["report", "--study-dir", str(study_dir), "--out", str(packet)]
    )
    assert code == EXIT_OK
    assert str(packet.resolve()) in capsys.readouterr().out
    assert (packet / REPORT_HTML_NAME).is_file()
    assert (packet / REPORT_MARKDOWN_NAME).is_file()


def test_the_study_runs_as_a_module_and_as_a_console_script() -> None:
    """``python -m whetstone_envs.optim.study`` is the same program.

    The study is documented under both entry points, and a module
    invocation that did not exist -- or that dispatched separately -- would
    make the documented command a command nobody can run.
    """
    import subprocess
    import sys

    from whetstone_envs.optim.study import __main__ as module_entry

    assert module_entry.main is main
    # Run it the way the documentation says to, in a child process: that is
    # the only check that a missing ``__main__`` would actually fail.
    finished = subprocess.run(
        [sys.executable, "-m", "whetstone_envs.optim.study", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    for subcommand in ("plan", "run", "report", "leakage-check", "manifest"):
        assert subcommand in finished.stdout


# --------------------------------------------------------------------------
# manifest check, against a real fake-transport run artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_run_dir(tmp_path_factory) -> Path:
    """One completed fake-transport COPRO run, reused by the checks below.

    This is a real optimizer run through the shared runner: it writes a
    ``runtime.sqlite`` the checker resolves against, so the pointer check is
    proven against the store shape the study actually produces rather than
    against a hand-built double.
    """
    from whetstone_envs.optim.run import RunSpec, run_optimizer

    output = tmp_path_factory.mktemp("study-cli") / "copro-run"
    return run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            run_id="c19-copro-study-cli",
            output_dir=output,
        )
    )


def _pointer_from(ref: dict[str, str]) -> EvidencePointer:
    return EvidencePointer(
        schema_name=ref["schema_name"],
        content_hash=ref["content_hash"],
    )


def _stored_pointers(run_dir: Path) -> tuple[EvidencePointer, ...]:
    """Three pointers the run itself persisted, all real store records.

    The first step's own result ref stands in for a manifest's
    ``result_ref`` and its first eval-evidence ref for an ``audit_ref``. The
    third is the run's own ``cost.json``, content-addressed into the same
    store by the runner, so ``manifest check`` proves the cost pointer
    resolves rather than merely being well-formed.
    """
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    step = result["step_results"][0]
    return (
        _pointer_from(step["record_ref"]),
        _pointer_from(
            step["record"]["resolved_intents"][0]["eval_result_ref"]
        ),
        _cost_pointer(run_dir),
    )


def _cost_pointer(run_dir: Path) -> EvidencePointer:
    """The ``(schema, hash)`` pair the run's stored cost document has."""
    payload = json.loads((run_dir / "cost.json").read_text(encoding="utf-8"))
    with open_sqlite(str(run_dir / DEFAULT_STORE_NAME)) as store:
        reference, _ = store.put(RUN_COST_SCHEMA_NAME, payload)
    return EvidencePointer(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )


def _manifest_citing(pointers: tuple[EvidencePointer, ...]) -> StudyManifest:
    result_ref, audit_ref, cost_ref = pointers
    return _minimal_manifest().model_copy(
        update={
            "arms": (
                ArmRecord(
                    arm_id="copro",
                    optimizer="copro",
                    demo_mode=None,
                    train_size=None,
                    val_size=None,
                    control_identity_hash="d" * 64,
                    seed_note="provider-seed-control-only",
                    runs=(
                        RunRecord(
                            run_id="c19-copro-study-cli",
                            seed=None,
                            artifact_dir="/tmp/runs/c19-copro",  # noqa: S108
                            result_ref=result_ref,
                            audit_ref=audit_ref,
                            cost_ref=cost_ref,
                            audit_passed=True,
                            transport=TransportName.FAKE.value,
                            spend=(
                                RunSpendRecord(
                                    role="task_model",
                                    calls=4,
                                    cached_calls=0,
                                    input_tokens=40,
                                    output_tokens=8,
                                    priced_calls=0,
                                    unpriced_calls=4,
                                    rows_missing_token_breakdown=4,
                                    usd=None,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        }
    )


def test_pointers_from_a_real_run_resolve_in_its_store(
    tmp_path: Path, fake_run_dir: Path, capsys
) -> None:
    manifest = _manifest_citing(_stored_pointers(fake_run_dir))
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, manifest)
    code = main(
        [
            "manifest",
            "check",
            str(study_manifest_path(study_dir)),
            "--store",
            str(fake_run_dir / DEFAULT_STORE_NAME),
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "3 evidence pointers resolved" in out
    assert out.count("ok ") == 3


def test_a_mutated_pointer_fails_the_check(
    tmp_path: Path, fake_run_dir: Path, capsys
) -> None:
    pointers = _stored_pointers(fake_run_dir)
    mutated = (
        pointers[0].model_copy(update={"content_hash": "0" * 64}),
        pointers[1],
        pointers[2],
    )
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, _manifest_citing(mutated))
    code = main(
        [
            "manifest",
            "check",
            str(study_manifest_path(study_dir)),
            "--store",
            str(fake_run_dir / DEFAULT_STORE_NAME),
        ]
    )
    assert code == EXIT_CHECK_FAILED
    captured = capsys.readouterr()
    assert "MISSING" in captured.out
    assert "1 of 3 evidence pointers did not resolve" in captured.err


def test_check_defaults_to_the_store_beside_the_manifest(
    tmp_path: Path, fake_run_dir: Path, capsys
) -> None:
    manifest = _manifest_citing(_stored_pointers(fake_run_dir))
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, manifest)
    (study_dir / DEFAULT_STORE_NAME).write_bytes(
        (fake_run_dir / DEFAULT_STORE_NAME).read_bytes()
    )
    code = main(["manifest", "check", str(study_dir / STUDY_MANIFEST_NAME)])
    assert code == EXIT_OK
    assert "3 evidence pointers resolved" in capsys.readouterr().out


def test_check_reports_a_missing_store(tmp_path: Path, capsys) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, _minimal_manifest())
    code = main(["manifest", "check", str(study_dir)])
    assert code == EXIT_CHECK_FAILED
    assert "no evidence store at" in capsys.readouterr().err


def test_check_accepts_a_directory(tmp_path: Path, fake_run_dir: Path) -> None:
    manifest = _manifest_citing(_stored_pointers(fake_run_dir))
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, manifest)
    assert read_study_manifest(study_dir) == manifest
    code = main(
        [
            "manifest",
            "check",
            str(study_dir),
            "--store",
            str(fake_run_dir / DEFAULT_STORE_NAME),
        ]
    )
    assert code == EXIT_OK


def test_the_run_store_holds_the_records_the_manifest_cites(
    fake_run_dir: Path,
) -> None:
    """The pointers are real store records, not merely well-formed."""
    pointers = _stored_pointers(fake_run_dir)
    with open_sqlite(str(fake_run_dir / DEFAULT_STORE_NAME)) as store:
        for pointer in pointers:
            assert store.get(pointer.as_object_reference()) is not None


# --------------------------------------------------------------------------
# The default wiring, end to end on a fake transport
# --------------------------------------------------------------------------


def test_stage0_runs_end_to_end_through_the_cli(
    tmp_path: Path, capsys
) -> None:
    """The wiring test: no injected collaborators, real harness, real
    anchors, fake transport, zero provider calls."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())

    assert (
        main(["run", "--study-dir", str(study_dir), "--stage", "stage0"])
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "stage0 complete for study step10-toy" in out
    assert str(study_dir / STUDY_MANIFEST_NAME) in out

    # The design the real harness measured is on disk, not a stub.
    design = read_study_manifest(study_dir).design
    assert design is not None
    assert design.k_cal == 4
    assert design.mde_formula.startswith("MDE(T, K)")
    assert set(design.k_run_by_arm) == {"copro", "null-identity"}


def test_plan_reads_the_design_stage0_measured(tmp_path: Path, capsys) -> None:
    """``plan`` and ``run`` describe one study, because both read one file."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    main(["run", "--study-dir", str(study_dir), "--stage", "stage0"])
    capsys.readouterr()

    assert main(["plan", "--study-dir", str(study_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "splits: internal=4 official=4 held_out=6" in out
    assert "copro" in out
    assert "null-identity" in out


def test_plan_labels_the_optimizer_budget_as_an_estimate(
    tmp_path: Path, capsys
) -> None:
    """The number is large and unmeasured, so the label carries the caveat."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    main(["plan", "--study-dir", str(study_dir)])
    out = capsys.readouterr().out
    assert OPTIMIZER_BUDGET_HEADING in out
    assert "ESTIMATE" in out
    assert "total optimizer-side calls:" in out
    # The derived budget stays separate and stays labelled as derived.
    assert "selection and reporting rows (derived from the matrix):" in out
    assert "total selection+report rows:" in out


def test_plan_states_each_estimate_s_derivation() -> None:
    """A number without a derivation cannot be re-checked when a default
    moves, so ``plan`` prints the basis beside every arm."""
    lines = plan_lines(_Spec())
    text = "\n".join(lines)
    assert "basis:" in text
    # COPRO's derivation is its own search shape.
    assert f"breadth {COPRO_BREADTH}" in text


def test_plan_prints_measured_numbers_beside_the_estimates() -> None:
    """Wave 3's measurements appear, labelled as measurements.

    An estimate and a measurement are known to different degrees, and the
    reader deciding whether to authorize spend is the one who needs to see
    which is which.
    """

    class _MeasuredArmSpec(_Spec):
        def __init__(self) -> None:
            super().__init__()
            self.arm_ids = ("copro", "miprov2", "gepa")
            self.k_run_by_arm = {"copro": 5, "miprov2": 5, "gepa": 5}

    text = "\n".join(plan_lines(_MeasuredArmSpec()))
    assert MEASURED_LABEL in text
    assert ESTIMATE_LABEL in text
    for arm, measured in MEASURED_TASK_CALLS_BY_ARM.items():
        assert arm in text
        assert str(measured) in text
    # COPRO was not measured, so it prints only its estimate.
    copro_lines = [
        line for line in plan_lines(_MeasuredArmSpec()) if "copro" in line
    ]
    assert copro_lines
    assert all(MEASURED_LABEL not in line for line in copro_lines)


def test_plan_prints_the_corrected_per_arm_estimates() -> None:
    """Golden pin: the numbers ``plan`` reports, in task-model rows.

    Both corrections are visible here. GEPA prints 200 rows -- the D3 pin
    -- rather than the 732 metric calls the auto budget resolves to, and
    null-identity prints the report harness rather than COPRO's search
    shape. The split sizes are deliberately *not* the protocol's, because
    at ``(88, 132, 220)`` with ``K_REPEAT = 3`` the harness formula and
    COPRO's shape coincide at 1,056 by accident and the pin would not
    discriminate.
    """

    class _AllArmSpec(_Spec):
        def __init__(self) -> None:
            super().__init__()
            self.arm_ids = ("copro", "miprov2", "gepa", "null-identity")
            self.k_run_by_arm = {
                "copro": 1,
                "miprov2": 1,
                "gepa": 1,
                "null-identity": 1,
            }
            self.k_repeat = 3
            self.split_sizes = (88, 10, 20)

    lines = plan_lines(_AllArmSpec())
    # Scope to the optimizer-side table; the selection table above it has
    # one row per arm too, and those are a different quantity.
    start = lines.index(OPTIMIZER_BUDGET_HEADING)
    optimizer_lines = lines[start:]

    def _row(arm: str) -> str:
        return next(line for line in optimizer_lines if line.startswith(arm))

    # COPRO: (depth 3 + 1) x breadth 6 x 88 internal x 3 repeats, at the
    # protocol's pinned shape rather than the runner's smoke-run default.
    assert str((COPRO_DEPTH + 1) * COPRO_BREADTH * 88 * 3) in _row("copro")
    # MIPROv2: its own control budget, 1870-2458, independent of the splits.
    assert "1870-2458" in _row("miprov2")
    # GEPA: the pinned 200 rows, not 732 metric calls.
    gepa = _row("gepa")
    assert "200" in gepa
    assert "732" not in gepa
    # Null-B: 1 official pass x 10 + 1 held-out pass x 20, at K_REPEAT 3.
    assert "90" in _row("null-identity")


def test_plan_states_the_gepa_basis_in_rows_not_metric_calls() -> None:
    """The unit is what went wrong, so the printed basis has to name it."""

    class _GepaSpec(_Spec):
        def __init__(self) -> None:
            super().__init__()
            self.arm_ids = ("gepa",)
            self.k_run_by_arm = {"gepa": 5}

    text = "\n".join(plan_lines(_GepaSpec()))
    assert "bounds task rows at 200" in text
    assert "no optimizer run" not in text


def test_plan_states_the_null_identity_basis_as_the_harness() -> None:
    """Null-B's printed derivation must not describe a search it never runs."""
    text = "\n".join(plan_lines(_Spec()))
    null_basis = [
        line for line in plan_lines(_Spec()) if "no optimizer run" in line
    ]
    assert null_basis, text
    assert "report harness only" in null_basis[0]


def test_the_gepa_measurement_says_it_was_scaled() -> None:
    """73 is not a number read off a run, and the label must not imply it.

    The run was measured at the retired 732-call budget; what the plan
    prints is that measurement scaled to the pinned 200, so its provenance
    names both budgets rather than claiming a direct measurement.
    """

    class _GepaSpec(_Spec):
        def __init__(self) -> None:
            super().__init__()
            self.arm_ids = ("gepa",)
            self.k_run_by_arm = {"gepa": 5}

    text = "\n".join(plan_lines(_GepaSpec()))
    assert MEASURED_BASIS_BY_ARM["gepa"] in text
    assert "scaled to the pinned 200" in text
    assert "max_metric_calls=732" in text
    # And the generic line is not what GEPA printed.
    assert MEASURED_BASIS_DEFAULT not in text


def test_the_measured_arms_are_the_ones_wave_3_actually_ran() -> None:
    """Only measured arms carry a measurement; the rest say estimate."""
    assert set(MEASURED_TASK_CALLS_BY_ARM) == {"miprov2", "gepa"}


def test_plan_says_no_estimate_rather_than_guessing() -> None:
    """An arm the estimator does not know reports that, not a number."""

    class _UnknownArmSpec(_Spec):
        def __init__(self) -> None:
            super().__init__()
            self.arm_ids = ("copro", "something-new")
            self.k_run_by_arm = {"copro": 2, "something-new": 2}

    text = "\n".join(plan_lines(_UnknownArmSpec()))
    assert NO_ESTIMATE in text


# --------------------------------------------------------------------------
# leakage-check
# --------------------------------------------------------------------------


def test_leakage_check_reports_every_rule(tmp_path: Path, capsys) -> None:
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    code = main(["leakage-check", "--study-dir", str(study_dir)])
    out = capsys.readouterr().out
    for rule in ("L1", "L2", "L3", "L4", "L5", "L6"):
        assert f" {rule} ::" in out
    # A study with no held-out evaluations has not earned a pass.
    assert code == EXIT_CHECK_FAILED


def test_leakage_check_reports_l1_as_unchecked_not_passed(
    tmp_path: Path, capsys
) -> None:
    """An empty observation set is vacuously true, and a vacuous truth is
    not a check -- so it fails the command rather than passing it."""
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest())
    assert main(["leakage-check", "--study-dir", str(study_dir)]) == (
        EXIT_CHECK_FAILED
    )
    captured = capsys.readouterr()
    assert f"{NOT_CHECKED} L1 ::" in captured.out
    assert "could not be checked from the manifest" in captured.err


def test_leakage_check_catches_overlapping_splits(
    tmp_path: Path, capsys
) -> None:
    """L5 over content-addressed hashes, on a manifest that got past its
    own validator because the overlap was introduced after construction."""
    study_dir = tmp_path / "study"
    manifest = toy_manifest()
    write_study_manifest(study_dir, manifest)
    payload = json.loads(
        (study_dir / STUDY_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    payload["splits"]["official"]["task_hashes"][0] = payload["splits"][
        "internal"
    ]["task_hashes"][0]
    (study_dir / STUDY_MANIFEST_NAME).write_bytes(rfc8785.dumps(payload))

    code = main(["leakage-check", "--study-dir", str(study_dir)])
    # The manifest's own validator refuses the overlap on read, which is a
    # stronger guarantee than the check: L5 never gets the chance to fail
    # because the document cannot be loaded at all.
    assert code == EXIT_ERROR
    assert "share" in capsys.readouterr().err


def test_leakage_check_on_a_missing_manifest_fails_cleanly(
    tmp_path: Path, capsys
) -> None:
    assert main(["leakage-check", "--study-dir", str(tmp_path)]) == EXIT_ERROR
    assert capsys.readouterr().err.strip()


# --------------------------------------------------------------------------
# A study id may not claim a design this invocation is not
# --------------------------------------------------------------------------


def test_a_protocol_id_is_allowed_on_the_protocol_it_names() -> None:
    """The full design at full size may be named after itself.

    That is also the default, so passing it changes nothing -- which is
    what keeps the refusal below a refusal of *false* claims rather than a
    ban on the id.
    """
    from whetstone_envs.optim.study.cli import _require_honest_study_id
    from whetstone_envs.optim.study.protocols import STEP10_C19

    _require_honest_study_id(
        STEP10_C19.study_id,
        protocol=STEP10_C19,
        toy=False,
        without_codex_arm=False,
    )
    # An id of its own is always fine, on any invocation.
    _require_honest_study_id(
        "rehearsal-2026-08-23",
        protocol=STEP10_C19,
        toy=True,
        without_codex_arm=True,
    )


@pytest.mark.parametrize(
    ("toy", "without_codex_arm"),
    [(True, False), (False, True), (True, True)],
)
def test_a_reduced_invocation_may_not_claim_a_protocol_id(
    *, toy: bool, without_codex_arm: bool
) -> None:
    """A toy or a projection may not name itself the pre-registration.

    Fails-before: ``--study-id step10-c19`` was accepted on any
    invocation, so a toy or a ``--without-codex`` rehearsal could be
    initialised under the study's own id -- and every artifact, directory,
    and report headline downstream would cite the pre-registration by name
    while holding a smaller design.
    """
    from whetstone_envs.optim.study.cli import _require_honest_study_id
    from whetstone_envs.optim.study.protocols import STEP10_C19, STEP10_C19_ID

    with pytest.raises(SystemExit, match="refusing --study-id"):
        _require_honest_study_id(
            STEP10_C19_ID,
            protocol=STEP10_C19,
            toy=toy,
            without_codex_arm=without_codex_arm,
        )


def test_a_protocol_id_may_not_name_a_different_protocol() -> None:
    """The toy may not be initialised under the real study's id."""
    from whetstone_envs.optim.study.cli import _require_honest_study_id
    from whetstone_envs.optim.study.protocols import (
        STEP10_C19_ID,
        STEP10_C19_TOY,
    )

    with pytest.raises(SystemExit, match="refusing --study-id"):
        _require_honest_study_id(
            STEP10_C19_ID,
            protocol=STEP10_C19_TOY,
            toy=False,
            without_codex_arm=False,
        )
