"""The one place an invariant is enumerated, and the audit entry point.

Adding an invariant means adding it to :data:`INVARIANTS_BY_OPTIMIZER` here
and nowhere else. A per-optimizer module owns the predicate; this module
owns which predicates run for which optimizer, so the set an audit covers is
readable in one place rather than assembled by import side effects.

Every invariant is a pure function ``(RunEvidence) -> AuditFinding``. It
must not open a store, re-execute, or re-score -- :mod:`._evidence` has
already resolved everything it may read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.optim.contracts import IntentOutcome

from whetstone_envs.optim.audit._evidence import (
    CODEX_OPTIMIZER,
    COPRO_OPTIMIZER,
    GEPA_OPTIMIZER,
    MIPROV2_OPTIMIZER,
    evidence_ref,
    load_run_evidence,
)
from whetstone_envs.optim.audit.codex import CODEX_INVARIANTS
from whetstone_envs.optim.audit.copro import COPRO_INVARIANTS
from whetstone_envs.optim.audit.gepa import GEPA_INVARIANTS
from whetstone_envs.optim.audit.miprov2 import MIPROV2_INVARIANTS
from whetstone_envs.optim.audit.schema import (
    AuditFinding,
    AuditReport,
    AuditStatus,
    InvariantId,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from whetstone.core.identity import TypedRef

    from whetstone_envs.optim.audit._evidence import RunEvidence
    from whetstone_envs.optim.audit.schema import EvidenceRef

    Invariant = Callable[[RunEvidence], AuditFinding]


class _ResolutionTally:
    """Counts reported evaluations and records which ones do not resolve.

    The three evidence paths -- intent resolutions, search evidence, and
    Tool Results -- differ only in how many refs each reported number
    carries and what to call it in a message. Sharing the resolution
    itself keeps one definition of "resolves", so a Codex run and a COPRO
    run are held to the same standard rather than to two drifting copies.
    """

    __slots__ = ("checked", "evidence", "refs", "unresolved")

    def __init__(self, evidence: RunEvidence) -> None:
        self.evidence = evidence
        self.unresolved: list[str] = []
        self.refs: list[EvidenceRef] = []
        self.checked = 0

    def check_one(self, ref: TypedRef | None, *, where: str) -> None:
        """One reported number, which may cite no ref at all."""
        if ref is None:
            self.checked += 1
            self.unresolved.append(f"{where} cites no eval result")
            return
        self.check_many((ref,), where=where)

    def check_many(self, refs: tuple[TypedRef, ...], *, where: str) -> None:
        """One reported evaluation citing zero or more evidence refs.

        Zero is itself a violation: a completed evaluation that names no
        evidence reported a number nothing backs.
        """
        if not refs:
            self.checked += 1
            self.unresolved.append(f"{where} cites no eval result")
            return
        for ref in refs:
            self.checked += 1
            if self.evidence.eval_evidence(ref) is None:
                self.unresolved.append(
                    f"{where} cites {ref.schema_name} which is not "
                    f"eval evidence"
                )
                continue
            self.refs.append(evidence_ref(ref))


def reported_numbers_resolve(evidence: RunEvidence) -> AuditFinding:
    """Every reported evaluation number resolves to durable evidence.

    An optimizer's search is only auditable if each score it acted on is
    backed by a record in the run's own store. This walks every completed
    intent resolution and every search-evidence entry, and requires that the
    ``eval_result_ref`` it cites actually dereferenced to an ``EvalEvidence``
    carrying the number.

    A rejected or failed intent is exempt: whetstone deliberately writes no
    eval evidence for one, and demanding it would make honest refusal look
    like infidelity. This is the generic invariant every optimizer shares --
    the per-optimizer modules add the algorithm-specific ones on top.

    **A tool-mediated evaluation is a reported number too.** A
    ``TOOL_USING`` run -- Codex is the only one -- resolves no intent and
    mints no search evidence *by design*: its paid evaluations are cited
    from ``tool_evidence`` instead, each Tool Result naming the
    ``EvalEvidence`` it produced. Counting only the intent paths would fail
    every honest Codex run for reporting nothing, while a Codex run that
    genuinely resolved to nothing would be indistinguishable from it. So
    all three paths count, and each is resolved the same way.

    **Zero completed evaluations is never a pass.** A run whose steps
    completed no intent, search, or tool evaluation failed to report the
    numbers this invariant governs, and a run with no steps at all has
    nothing to govern; the first is a ``FAIL`` and the second
    ``NOT_APPLICABLE``. Reporting either as "all 0 of 0 resolve" would let
    an empty run read as audited fidelity.
    """
    tally = _ResolutionTally(evidence)
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            if resolution.outcome is IntentOutcome.COMPLETED:
                tally.check_one(
                    resolution.eval_result_ref,
                    where=f"step {entry.index} intent {position}",
                )
        for position, search in enumerate(entry.search_evidence):
            if search.outcome is IntentOutcome.COMPLETED:
                tally.check_one(
                    search.eval_result_ref,
                    where=f"step {entry.index} search {position}",
                )
        for position, tool in enumerate(entry.tool_evidence):
            record = tool.result.record
            # A tool call that terminalized with a failure reports no
            # number, exactly as a failed intent does. Its admission is
            # still audited -- that is ``codex_no_eval_outside_tools``'s
            # business, not this invariant's.
            if record.terminal_failure is None:
                tally.check_many(
                    record.evaluation_evidence_refs,
                    where=f"step {entry.index} tool call {position}",
                )

    unresolved = tally.unresolved
    refs = tally.refs
    checked = tally.checked

    if unresolved:
        return AuditFinding(
            invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
            status=AuditStatus.FAIL,
            detail=(
                f"{len(unresolved)} of {checked} reported evaluations do "
                f"not resolve to eval evidence: {'; '.join(unresolved[:3])}"
            ),
            evidence_refs=tuple(refs),
        )
    if not checked:
        # "All 0 of 0 resolve" is not a check. A run that reported no
        # evaluation at all resolved nothing, and a PASS here would read as
        # audited fidelity on a run whose numbers nobody verified -- the
        # same vacuity ``invariants_for`` refuses for an unregistered
        # optimizer. Which of the two it is depends on whether the run has
        # any steps at all: a run with steps that completed no intent is a
        # defect, and a run with no steps has nothing this invariant
        # governs.
        if evidence.steps:
            return AuditFinding(
                invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
                status=AuditStatus.FAIL,
                detail=(
                    f"this run persisted {len(evidence.steps)} step(s) but "
                    "completed no evaluation intent, search evaluation, or "
                    "tool evaluation, so no reported number was checked"
                ),
                evidence_refs=(),
            )
        return AuditFinding(
            invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
            status=AuditStatus.NOT_APPLICABLE,
            detail=(
                "this run persisted no steps, so it reported no evaluation "
                "number for this invariant to resolve"
            ),
            evidence_refs=(),
        )
    return AuditFinding(
        invariant_id=InvariantId.REPORTED_NUMBERS_RESOLVE,
        status=AuditStatus.PASS,
        detail=(
            f"all {checked} reported evaluations resolve to eval evidence "
            f"in the run's own store"
        ),
        evidence_refs=tuple(refs),
    )


#: Invariants every optimizer is audited against.
SHARED_INVARIANTS: tuple[Invariant, ...] = (reported_numbers_resolve,)

#: The full invariant set per optimizer. The per-optimizer waves extend the
#: tuples here; the shared set is spliced in so a generic invariant is never
#: forgotten for a newly added optimizer.
INVARIANTS_BY_OPTIMIZER: dict[str, tuple[Invariant, ...]] = {
    COPRO_OPTIMIZER: SHARED_INVARIANTS + COPRO_INVARIANTS,
    MIPROV2_OPTIMIZER: SHARED_INVARIANTS + MIPROV2_INVARIANTS,
    GEPA_OPTIMIZER: SHARED_INVARIANTS + GEPA_INVARIANTS,
    CODEX_OPTIMIZER: SHARED_INVARIANTS + CODEX_INVARIANTS,
}


def invariants_for(optimizer: str) -> tuple[Invariant, ...]:
    """Every invariant registered for ``optimizer``.

    An unregistered optimizer raises rather than auditing vacuously: a run
    that silently passes zero invariants would read as validated.
    """
    try:
        return INVARIANTS_BY_OPTIMIZER[optimizer]
    except KeyError:
        known = ", ".join(sorted(INVARIANTS_BY_OPTIMIZER))
        raise ValueError(
            f"no invariants registered for optimizer {optimizer!r}; "
            f"known optimizers are {known}"
        ) from None


def audit_evidence(evidence: RunEvidence) -> AuditReport:
    """Run every invariant registered for this run's optimizer."""
    findings = tuple(
        invariant(evidence) for invariant in invariants_for(evidence.optimizer)
    )
    return AuditReport(
        run_id=evidence.run_id,
        optimizer=evidence.optimizer,
        findings=findings,
    )


def audit_run(run_dir: Path) -> AuditReport:
    """Audit the completed run in ``run_dir``.

    Takes only a directory holding ``result.json`` and ``runtime.sqlite``.
    The optimizer is read from the result, so a caller cannot audit a run
    against the wrong invariant set.
    """
    return audit_evidence(load_run_evidence(run_dir))


__all__ = [
    "INVARIANTS_BY_OPTIMIZER",
    "SHARED_INVARIANTS",
    "audit_evidence",
    "audit_run",
    "invariants_for",
    "reported_numbers_resolve",
]
