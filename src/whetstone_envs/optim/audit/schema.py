"""The persisted audit result contract.

An audit report is durable evidence: it is written to ``audit.json`` beside
the run's ``result.json`` and cited by content hash from the study manifest.
Its wire keys are therefore a persisted format with a named owner here, and
``tests/optim/audit/test_schema.py`` pins every literal against a golden.
Never derive one of these strings from a field name or by iterating the
enum -- only a pinned test catches silent drift of stored identity.
"""

from __future__ import annotations

from enum import UNIQUE, StrEnum, verify

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    model_validator,
)

#: The persisted schema name for an ``audit.json`` document.
AUDIT_REPORT_SCHEMA = "whetstone_envs.audit_report/v1"

#: The file an audit writes beside ``result.json``.
AUDIT_REPORT_FILENAME = "audit.json"


@verify(UNIQUE)
class AuditStatus(StrEnum):
    """How one invariant came out against one run's evidence."""

    PASS = "pass"  # noqa: S105 - an audit verdict, not a credential
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@verify(UNIQUE)
class InvariantId(StrEnum):
    """Every fidelity invariant the audit package can report on.

    Members are added by the per-optimizer waves. The value is the wire
    spelling written into ``audit.json``; it is fixed independently of the
    member name so a rename cannot silently rewrite stored identity.
    """

    #: Generic: every number an optimizer reports resolves to durable
    #: evidence in the run's own store.
    REPORTED_NUMBERS_RESOLVE = "reported_numbers_resolve"

    # --- COPRO -----------------------------------------------------------
    #: Each proposal round measured exactly ``control.breadth`` occurrences.
    COPRO_BREADTH_PER_DEPTH = "copro_breadth_per_depth"
    #: Step count is ``control.depth + 1``, or fewer under a failure.
    COPRO_DEPTH_STEPS = "copro_depth_steps"
    #: Every evaluation used the control's internal Eval Config and role.
    COPRO_INTERNAL_ONLY = "copro_internal_only"
    #: The accepted candidate is the best measured so far, ties to earlier.
    COPRO_BEST_SO_FAR = "copro_best_so_far"
    #: Candidates proposed in one round have pairwise distinct bases.
    COPRO_DISTINCT_BASES = "copro_distinct_bases"
    #: COPRO evaluates through intents only; it runs no search.
    COPRO_NO_SEARCH_EVALS = "copro_no_search_evals"
    #: The terminal candidate was minted in this run, or honestly retained.
    COPRO_TERMINAL_PROVENANCE = "copro_terminal_provenance"
    #: MIPROv2: demonstrations are bootstrapped before any instruction is
    #: proposed, so proposals are grounded in observed behaviour.
    MIPRO_BOOTSTRAP_BEFORE_PROPOSAL = "mipro_bootstrap_before_proposal"
    #: MIPROv2: ``zeroshot`` still runs DSPy's 3/0 grounding bootstrap and
    #: ships no demonstration set on any candidate.
    MIPRO_ZEROSHOT_GROUNDING = "mipro_zeroshot_grounding"
    #: MIPROv2: ``ground_only`` is flagged as a whetstone deviation rather
    #: than claiming frozen DSPy faithfulness.
    MIPRO_GROUND_ONLY_DEVIATION = "mipro_ground_only_deviation"
    #: MIPROv2: the recorded trials replay from a fresh seeded Optuna TPE
    #: sampler in order.
    MIPRO_TPE_SELECTION = "mipro_tpe_selection"
    #: MIPROv2: every trial evaluation drew its scheduled batch from the
    #: validation split.
    MIPRO_MINIBATCH_SIZING = "mipro_minibatch_sizing"
    #: MIPROv2: the incumbent is re-evaluated on the full validation split
    #: on the configured cadence.
    MIPRO_PERIODIC_FULL_EVAL = "mipro_periodic_full_eval"
    #: MIPROv2: bootstrap generations are paid through the evaluation
    #: engine, not the proposer transport.
    MIPRO_BOOTSTRAP_THROUGH_ENGINE = "mipro_bootstrap_through_engine"
    #: MIPROv2: the observed trial count matches the configured budget
    #: unless a terminal failure truncated the run.
    MIPRO_TRIALS_MATCH_CONTROL = "mipro_trials_match_control"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class EvidenceRef(_StrictModel):
    """A ``(schema_name, content_hash)`` pair naming one durable record.

    This mirrors whetstone's ``TypedRef`` on the wire without importing its
    validation, so an audit report stays readable without whetstone-ai
    installed.
    """

    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> EvidenceRef:
        if not self.schema_name.strip():
            raise ValueError("evidence refs require a nonblank schema name")
        if not self.content_hash.strip():
            raise ValueError("evidence refs require a nonblank content hash")
        return self


class AuditFinding(_StrictModel):
    """One invariant's verdict over one run.

    ``detail`` is one sentence naming what was checked and what was seen --
    it is read by a human triaging a failing run, so it states the observed
    value, not merely that a check failed.
    """

    invariant_id: InvariantId
    status: AuditStatus
    detail: StrictStr
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> AuditFinding:
        if not self.detail.strip():
            raise ValueError("every finding requires a nonblank detail")
        return self


class AuditReport(_StrictModel):
    """Every invariant registered for one run's optimizer, with its verdict.

    ``passed`` is derived, never stored as an independent field: a report
    whose stored verdict disagreed with its own findings would be exactly
    the kind of drift this package exists to detect.
    """

    schema_name: StrictStr = AUDIT_REPORT_SCHEMA
    run_id: StrictStr
    optimizer: StrictStr
    findings: tuple[AuditFinding, ...]

    @model_validator(mode="after")
    def _validate(self) -> AuditReport:
        if self.schema_name != AUDIT_REPORT_SCHEMA:
            raise ValueError(
                f"audit reports carry schema {AUDIT_REPORT_SCHEMA!r}"
            )
        if not self.run_id.strip():
            raise ValueError("audit reports require a nonblank run id")
        if not self.optimizer.strip():
            raise ValueError("audit reports require a nonblank optimizer")
        seen = [finding.invariant_id for finding in self.findings]
        if len(seen) != len(set(seen)):
            raise ValueError(
                "an audit report carries each invariant at most once"
            )
        return self

    @property
    def passed(self) -> bool:
        """True when no registered invariant reported ``FAIL``."""
        return all(
            finding.status is not AuditStatus.FAIL for finding in self.findings
        )


__all__ = [
    "AUDIT_REPORT_FILENAME",
    "AUDIT_REPORT_SCHEMA",
    "AuditFinding",
    "AuditReport",
    "AuditStatus",
    "EvidenceRef",
    "InvariantId",
]
