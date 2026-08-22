"""Golden tests over the persisted audit contract.

``audit.json`` is cited by content hash from the study manifest, so its wire
keys are stored identity. These tests pin the exact literals rather than
deriving them from field names -- deriving them is precisely the drift that
would go unnoticed.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from whetstone_envs.optim.audit.schema import (
    AUDIT_REPORT_FILENAME,
    AUDIT_REPORT_SCHEMA,
    AuditFinding,
    AuditReport,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

HASH = "a" * 64


def _finding(status: AuditStatus = AuditStatus.PASS) -> AuditFinding:
    return AuditFinding(
        invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
        status=status,
        detail="checked",
    )


def test_persisted_schema_literals_are_pinned() -> None:
    assert AUDIT_REPORT_SCHEMA == "whetstone_envs.audit_report/v1"
    assert AUDIT_REPORT_FILENAME == "audit.json"


def test_status_values_are_pinned() -> None:
    assert AuditStatus.PASS.value == "pass"
    assert AuditStatus.FAIL.value == "fail"
    assert AuditStatus.NOT_APPLICABLE.value == "not_applicable"
    assert [status.value for status in AuditStatus] == [
        "pass",
        "fail",
        "not_applicable",
    ]


def test_invariant_id_values_are_pinned() -> None:
    """Each id's wire spelling is fixed independently of its member name."""
    assert (
        InvariantId.REPORTED_NUMBERS_RESOLVE.value
        == "reported_numbers_resolve"
    )
    assert len(set(InvariantId)) == len(list(InvariantId))


def test_report_wire_keys_are_pinned() -> None:
    report = AuditReport(
        run_id="run-1",
        optimizer="copro",
        findings=(
            AuditFinding(
                invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
                status=AuditStatus.PASS,
                detail="all 4 reported evaluations resolve",
                evidence_refs=(
                    EvidenceRef(
                        schema_name="whetstone.eval_evidence",
                        content_hash=HASH,
                    ),
                ),
            ),
        ),
    )
    assert json.loads(report.model_dump_json()) == {
        "schema_name": "whetstone_envs.audit_report/v1",
        "run_id": "run-1",
        "optimizer": "copro",
        "findings": [
            {
                "invariant_id": "reported_numbers_resolve",
                "status": "pass",
                "detail": "all 4 reported evaluations resolve",
                "evidence_refs": [
                    {
                        "schema_name": "whetstone.eval_evidence",
                        "content_hash": HASH,
                    }
                ],
            }
        ],
    }


def test_report_round_trips_through_json() -> None:
    report = AuditReport(
        run_id="run-1", optimizer="gepa", findings=(_finding(),)
    )
    restored = AuditReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_passed_is_derived_from_findings() -> None:
    assert AuditReport(
        run_id="r", optimizer="copro", findings=(_finding(),)
    ).passed
    assert AuditReport(
        run_id="r",
        optimizer="copro",
        findings=(_finding(AuditStatus.NOT_APPLICABLE),),
    ).passed
    assert not AuditReport(
        run_id="r",
        optimizer="copro",
        findings=(_finding(AuditStatus.FAIL),),
    ).passed


def test_passed_is_not_a_settable_field() -> None:
    """A stored verdict could disagree with its own findings."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        AuditReport.model_validate(
            {
                "run_id": "r",
                "optimizer": "copro",
                "findings": (),
                "passed": True,
            }
        )


def test_report_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        AuditReport.model_validate(
            {
                "run_id": "r",
                "optimizer": "copro",
                "findings": (),
                "unknown": 1,
            }
        )


def test_report_rejects_a_foreign_schema_name() -> None:
    with pytest.raises(ValidationError, match="audit reports carry schema"):
        AuditReport.model_validate(
            {
                "schema_name": "whetstone_envs.audit_report/v2",
                "run_id": "r",
                "optimizer": "copro",
                "findings": (),
            }
        )


def test_report_rejects_a_duplicated_invariant() -> None:
    with pytest.raises(ValidationError, match="at most once"):
        AuditReport(
            run_id="r",
            optimizer="copro",
            findings=(_finding(), _finding(AuditStatus.FAIL)),
        )


@pytest.mark.parametrize("blank", ["", "   "])
def test_findings_require_a_nonblank_detail(blank: str) -> None:
    with pytest.raises(ValidationError, match="nonblank detail"):
        AuditFinding(
            invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
            status=AuditStatus.PASS,
            detail=blank,
        )


def test_evidence_refs_require_nonblank_parts() -> None:
    with pytest.raises(ValidationError, match="nonblank schema name"):
        EvidenceRef(schema_name="  ", content_hash=HASH)
    with pytest.raises(ValidationError, match="nonblank content hash"):
        EvidenceRef(schema_name="whetstone.eval_evidence", content_hash="")
