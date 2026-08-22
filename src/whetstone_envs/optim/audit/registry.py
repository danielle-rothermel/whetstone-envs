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

    from whetstone_envs.optim.audit._evidence import RunEvidence

    Invariant = Callable[[RunEvidence], AuditFinding]


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

    **Zero completed evaluations is never a pass.** A run whose steps
    completed no intent failed to report the numbers this invariant governs,
    and a run with no steps at all has nothing to govern; the first is a
    ``FAIL`` and the second ``NOT_APPLICABLE``. Reporting either as "all 0
    of 0 resolve" would let an empty run read as audited fidelity.
    """
    unresolved: list[str] = []
    refs = []
    checked = 0
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            if resolution.outcome is not IntentOutcome.COMPLETED:
                continue
            checked += 1
            ref = resolution.eval_result_ref
            if ref is None:
                unresolved.append(
                    f"step {entry.index} intent {position} cites no "
                    f"eval result"
                )
                continue
            found = evidence.eval_evidence(ref)
            if found is None:
                unresolved.append(
                    f"step {entry.index} intent {position} cites "
                    f"{ref.schema_name} which is not eval evidence"
                )
                continue
            refs.append(evidence_ref(ref))
        for position, search in enumerate(entry.search_evidence):
            if search.outcome is not IntentOutcome.COMPLETED:
                continue
            checked += 1
            ref = search.eval_result_ref
            if ref is None:
                unresolved.append(
                    f"step {entry.index} search {position} cites no "
                    f"eval result"
                )
                continue
            found = evidence.eval_evidence(ref)
            if found is None:
                unresolved.append(
                    f"step {entry.index} search {position} cites "
                    f"{ref.schema_name} which is not eval evidence"
                )
                continue
            refs.append(evidence_ref(ref))

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
                    "completed no evaluation intent and no search "
                    "evaluation, so no reported number was checked"
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
