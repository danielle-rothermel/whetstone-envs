"""Offline fidelity audits over one optimizer run's durable evidence.

An audit answers one question: did the optimizer actually do what its
algorithm claims, judged only by what the run persisted. It takes a run
directory holding ``result.json`` and ``runtime.sqlite``, reads them through
whetstone's public API, and writes ``audit.json`` beside them.

    from whetstone_envs.optim.audit import audit_run
    report = audit_run(run_dir)
    report.passed

An audit performs no network access, no re-execution, and no re-scoring, so
it is equally valid in CI against fake-transport artifacts and against a
paid run days later. A failing audit does not abort a run -- the paid
evidence is still worth keeping -- but marks that run's fidelity failed,
which propagates to the study report and downgrades its efficacy number to
descriptive only.
"""

from __future__ import annotations

from whetstone_envs.optim.audit._evidence import (
    RESULT_FILENAME,
    RUNTIME_STORE_FILENAME,
    AuditEvidenceError,
    RunEvidence,
    load_run_evidence,
)
from whetstone_envs.optim.audit.registry import (
    INVARIANTS_BY_OPTIMIZER,
    audit_evidence,
    audit_run,
    invariants_for,
)
from whetstone_envs.optim.audit.schema import (
    AUDIT_REPORT_FILENAME,
    AUDIT_REPORT_SCHEMA,
    AuditFinding,
    AuditReport,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

__all__ = [
    "AUDIT_REPORT_FILENAME",
    "AUDIT_REPORT_SCHEMA",
    "INVARIANTS_BY_OPTIMIZER",
    "RESULT_FILENAME",
    "RUNTIME_STORE_FILENAME",
    "AuditEvidenceError",
    "AuditFinding",
    "AuditReport",
    "AuditStatus",
    "EvidenceRef",
    "InvariantId",
    "RunEvidence",
    "audit_evidence",
    "audit_run",
    "invariants_for",
    "load_run_evidence",
]
