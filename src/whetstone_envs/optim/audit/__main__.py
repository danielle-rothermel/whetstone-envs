"""``python -m whetstone_envs.optim.audit <run_dir>``.

Runs every invariant registered for the run's optimizer and writes
``audit.json`` beside ``result.json``. Offline: no API key, no network, no
re-execution.

Exit codes distinguish the two ways an audit can not-pass, because they mean
different things: ``1`` is a fidelity failure -- the audit ran and an
invariant was violated, which is a whetstone implementation defect -- while
``2`` means the evidence could not be read at all and nothing was judged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from whetstone_envs.optim.audit._evidence import AuditEvidenceError
from whetstone_envs.optim.audit.registry import audit_run
from whetstone_envs.optim.audit.schema import (
    AUDIT_REPORT_FILENAME,
    AuditStatus,
)

#: The audit ran and every invariant held.
EXIT_PASSED = 0
#: The audit ran and at least one invariant FAILed.
EXIT_FIDELITY_FAILED = 1
#: The run directory could not be read as evidence; nothing was judged.
EXIT_UNREADABLE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m whetstone_envs.optim.audit",
        description=(
            "Audit one optimizer run's durable evidence for fidelity."
        ),
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=("A run directory containing result.json and runtime.sqlite."),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(f"Report to stdout without writing {AUDIT_REPORT_FILENAME}."),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run_dir
    try:
        report = audit_run(run_dir)
    except AuditEvidenceError as error:
        print(f"audit could not read {run_dir}: {error}", file=sys.stderr)
        return EXIT_UNREADABLE

    if not args.no_write:
        (run_dir / AUDIT_REPORT_FILENAME).write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    for finding in report.findings:
        print(
            f"{finding.status.value.upper():<14} "
            f"{finding.invariant_id.value}: {finding.detail}"
        )
    failed = sum(
        1 for finding in report.findings if finding.status is AuditStatus.FAIL
    )
    verdict = "PASS" if report.passed else "FAIL"
    print(
        f"{verdict} {report.optimizer} run {report.run_id}: "
        f"{len(report.findings)} invariants, {failed} failed"
    )
    return EXIT_PASSED if report.passed else EXIT_FIDELITY_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
