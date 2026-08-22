"""Dispatch, and the worked example invariant end to end."""

from __future__ import annotations

from dataclasses import replace

import pytest

from whetstone_envs.optim.audit._evidence import (
    CODEX_OPTIMIZER,
    COPRO_OPTIMIZER,
    GEPA_OPTIMIZER,
    MIPROV2_OPTIMIZER,
    load_run_evidence,
)
from whetstone_envs.optim.audit.registry import (
    INVARIANTS_BY_OPTIMIZER,
    SHARED_INVARIANTS,
    audit_run,
    invariants_for,
    reported_numbers_resolve,
)
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId


def test_every_optimizer_is_registered() -> None:
    assert set(INVARIANTS_BY_OPTIMIZER) == {
        COPRO_OPTIMIZER,
        MIPROV2_OPTIMIZER,
        GEPA_OPTIMIZER,
        CODEX_OPTIMIZER,
    }


def test_shared_invariants_reach_every_optimizer() -> None:
    for optimizer in INVARIANTS_BY_OPTIMIZER:
        registered = invariants_for(optimizer)
        assert set(SHARED_INVARIANTS) <= set(registered)


def test_an_unregistered_optimizer_raises_rather_than_auditing_nothing() -> (
    None
):
    """A vacuous pass would read as a validated run."""
    with pytest.raises(ValueError, match="no invariants registered"):
        invariants_for("gradient-descent")


def test_audit_run_dispatches_on_the_recorded_optimizer(copro_run_dir) -> None:
    report = audit_run(copro_run_dir)
    assert report.optimizer == COPRO_OPTIMIZER
    assert len(report.findings) == len(invariants_for(COPRO_OPTIMIZER))


def test_every_finding_names_a_registered_invariant(copro_run_dir) -> None:
    report = audit_run(copro_run_dir)
    for finding in report.findings:
        assert finding.invariant_id in set(InvariantId)


def test_a_real_run_passes_the_shared_invariant(copro_run_dir) -> None:
    report = audit_run(copro_run_dir)
    assert report.passed, [
        (f.invariant_id, f.detail)
        for f in report.findings
        if f.status is AuditStatus.FAIL
    ]


def test_reported_numbers_resolve_cites_the_evidence_it_checked(
    copro_run_dir,
) -> None:
    finding = reported_numbers_resolve(load_run_evidence(copro_run_dir))
    assert finding.status is AuditStatus.PASS
    assert finding.evidence_refs
    for ref in finding.evidence_refs:
        assert ref.schema_name == "whetstone.eval_evidence"
        assert len(ref.content_hash) == 64


# --------------------------------------------------------------------------
# Zero checked evaluations is never a pass
# --------------------------------------------------------------------------


def test_a_run_that_completed_no_evaluation_fails_rather_than_passing(
    copro_run_dir,
) -> None:
    """ "All 0 of 0 resolve" would read as audited fidelity on an empty run.

    The run kept its persisted steps, so it had every opportunity to report
    a number and reported none. That is a defect in the run, not an
    invariant this run is exempt from.
    """
    evidence = load_run_evidence(copro_run_dir)
    assert evidence.steps, "the fixture run persists steps"
    stripped = replace(
        evidence,
        steps=tuple(
            replace(
                step,
                step=step.step.model_copy(
                    update={"resolved_intents": (), "search_evidence": ()}
                ),
            )
            for step in evidence.steps
        ),
    )
    finding = reported_numbers_resolve(stripped)
    assert finding.status is AuditStatus.FAIL
    assert "completed no evaluation intent" in finding.detail
    assert finding.evidence_refs == ()


def test_a_run_with_no_steps_at_all_is_not_applicable(copro_run_dir) -> None:
    """Nothing to resolve is a different fact from nothing resolving."""
    evidence = replace(load_run_evidence(copro_run_dir), steps=())
    finding = reported_numbers_resolve(evidence)
    assert finding.status is AuditStatus.NOT_APPLICABLE
    assert "persisted no steps" in finding.detail
