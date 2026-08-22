"""``python -m whetstone_envs.optim.audit <run_dir>`` end to end."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.optim.audit.test_mutate import EVAL_RESULT_REF_PATH
from whetstone_envs.optim.audit.__main__ import (
    EXIT_FIDELITY_FAILED,
    EXIT_PASSED,
    EXIT_UNREADABLE,
    main,
)
from whetstone_envs.optim.audit._mutate import mutate_run
from whetstone_envs.optim.audit.schema import (
    AUDIT_REPORT_FILENAME,
    AuditReport,
)


def test_a_passing_run_exits_zero_and_writes_audit_json(
    mutable_run_dir, capsys
) -> None:
    assert main([str(mutable_run_dir)]) == EXIT_PASSED
    written = mutable_run_dir / AUDIT_REPORT_FILENAME
    assert written.is_file()
    report = AuditReport.model_validate_json(
        written.read_text(encoding="utf-8")
    )
    assert report.passed
    assert report.optimizer == "copro"
    assert report.findings
    assert "PASS" in capsys.readouterr().out


def test_audit_json_is_beside_result_json(mutable_run_dir) -> None:
    main([str(mutable_run_dir)])
    payload = json.loads(
        (mutable_run_dir / AUDIT_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["schema_name"] == "whetstone_envs.audit_report/v1"


def test_no_write_reports_without_writing(mutable_run_dir, capsys) -> None:
    assert main([str(mutable_run_dir), "--no-write"]) == EXIT_PASSED
    assert not (mutable_run_dir / AUDIT_REPORT_FILENAME).exists()
    assert capsys.readouterr().out.strip()


def test_a_fidelity_failure_exits_one(copro_run_dir, tmp_path, capsys) -> None:
    mutated = mutate_run(
        copro_run_dir,
        tmp_path / "cli-negative",
        EVAL_RESULT_REF_PATH,
        lambda ref: {**ref, "content_hash": "0" * 64},
    )
    assert main([str(mutated)]) == EXIT_FIDELITY_FAILED
    # The report is written even though the audit failed: the evidence is
    # still worth keeping, and the manifest cites it.
    assert (mutated / AUDIT_REPORT_FILENAME).is_file()
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "reported_numbers_resolve" in out


def test_unreadable_evidence_exits_two(tmp_path, capsys) -> None:
    """Distinct from a fidelity failure: nothing was judged at all."""
    assert main([str(tmp_path / "absent")]) == EXIT_UNREADABLE
    assert "could not read" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_is_available(flag: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([flag])
    assert exit_info.value.code == 0


def test_module_is_runnable_as_python_m(mutable_run_dir) -> None:
    """The documented invocation works in a fresh interpreter, offline."""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, literal args
        [
            sys.executable,
            "-m",
            "whetstone_envs.optim.audit",
            str(mutable_run_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == EXIT_PASSED, completed.stderr
    assert (mutable_run_dir / AUDIT_REPORT_FILENAME).is_file()
