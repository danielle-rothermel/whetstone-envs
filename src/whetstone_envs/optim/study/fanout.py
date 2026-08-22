"""F16: count the rows a run actually planned, per evaluation intent.

The Stage-1 budget gate divides a run's observed task-model calls by a
pre-spend estimate. That estimate is only meaningful if an optimizer's
*requested* task subset is the set that actually gets executed. The risk
F16 exists to retire is the opposite: that the platform's deferral row
expansion ignores an intent's ``task_hashes`` and fans every evaluation out
over the whole validation split. At MIPROv2's ``minibatch_size=35`` against
an 88-task internal split that is a ``88 / 35 = 2.51x`` multiplier, and it
would land as roughly twelve thousand unbudgeted calls across MIPROv2 and
GEPA at ``K_RUN = 5``.

**The measurement, not an inference.** This module reads a completed run's
durable evidence and reports, per evaluation intent, two numbers side by
side:

``requested`` -- ``len(OptimEvalRequest.task_hashes)``, the subset the
optimizer asked for.

``planned`` -- ``EvalEvidence.row_accounting.planned``, the rows the
platform actually scheduled for that intent.

Not every optimizer routes its evaluations through ``resolved_intents``.
MIPROv2 does; GEPA records its evaluations as ``search_evidence`` instead,
and carries no intents at all. Both are read here, and each eval-evidence
record is counted **once** no matter how many times it is cited -- GEPA
re-emits its whole replayed prefix on every step, so counting citations
would report a run's rows as growing quadratically in its step count when
the paid rows did not move.

Fan-out is ``planned != requested`` on any intent. The per-intent-subset
formula predicts ``sum(requested)``; the ``intents x tasks x seeds``
formula predicts ``len(intents) x len(valset) x num_seeds``. A run tells
you which formula the code implements by which of the two its total
matches, and :func:`measure_fanout` reports both so the comparison is in
the artifact rather than in a reader's head.

The total is cross-checked against ``cost.json``: the task-model role's
``calls`` is an independent count of the same rows, projected from
``OptimResult.cost`` rather than from eval evidence. Two independent paths
agreeing is what makes the measurement evidence instead of an assertion
about one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from whetstone.eval.metadata import eval_purpose

from whetstone_envs.optim.audit._evidence import load_run_evidence

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone.core.identity import TypedRef

    from whetstone_envs.optim.audit._evidence import RunEvidence

__all__ = [
    "INTENT_SOURCE",
    "SEARCH_SOURCE",
    "FanoutMeasurement",
    "IntentRowCount",
    "measure_fanout",
    "measure_run_directory",
]


@dataclass(frozen=True, slots=True)
class IntentRowCount:
    """One evaluation intent's requested subset beside its planned rows.

    ``requested`` is ``None`` when the intent declared no task subset at
    all, which means it evaluates the whole configured task set. That is
    not itself fan-out -- a full-valset pass legitimately declares none --
    so the two cases stay distinguishable rather than collapsing into a
    single integer.
    """

    step_index: int
    position: int
    purpose: str | None
    requested: int | None
    planned: int
    num_seeds: int
    #: How this evaluation reached the run's evidence. MIPROv2 resolves
    #: intents; GEPA emits search evidence. Recorded so a measurement says
    #: which surface it read rather than implying every optimizer has both.
    source: str = "intent"

    @property
    def fanned_out(self) -> bool:
        """Whether the platform scheduled more rows than were asked for."""
        return self.requested is not None and self.planned != self.requested

    @property
    def where(self) -> str:
        return f"step {self.step_index} intent {self.position}"


@dataclass(frozen=True, slots=True)
class FanoutMeasurement:
    """What one run's evidence says about per-intent row expansion.

    ``subset_formula_rows`` is what the per-intent-subset formula predicts;
    ``full_split_formula_rows`` is what the ``intents x tasks x seeds``
    formula predicts. ``planned_rows`` is the measurement. Exactly one of
    the two formulas should match it, and which one is the F16 finding.
    """

    run_id: str
    optimizer: str
    intents: tuple[IntentRowCount, ...]
    #: The largest task set any intent evaluated, standing in for the
    #: validation split's size as the run itself witnessed it.
    widest_task_set: int
    planned_rows: int
    subset_formula_rows: int
    full_split_formula_rows: int

    @property
    def honours_per_intent_subsets(self) -> bool:
        """True when no intent's planned rows exceeded its request.

        This is the F16 precondition of Stage 1. It is deliberately an
        equality over every intent rather than a tolerance on the total:
        two intents fanning out in opposite directions must not cancel.
        """
        return not any(intent.fanned_out for intent in self.intents)

    @property
    def fanout_ratio(self) -> float:
        """Planned rows over the per-intent-subset prediction.

        ``1.0`` means the subset formula is the one the code implements.
        """
        if self.subset_formula_rows == 0:
            return 0.0
        return self.planned_rows / self.subset_formula_rows

    @property
    def minibatch_intents(self) -> int:
        """Intents that evaluated a strict subset of the widest task set."""
        return sum(
            1
            for intent in self.intents
            if intent.requested is not None
            and intent.requested < self.widest_task_set
        )

    @property
    def full_split_intents(self) -> int:
        """Intents that evaluated the whole widest task set."""
        return sum(
            1
            for intent in self.intents
            if intent.requested is not None
            and intent.requested == self.widest_task_set
        )


#: How an evaluation reached the run's evidence.
INTENT_SOURCE = "intent"
SEARCH_SOURCE = "search"


def measure_fanout(evidence: RunEvidence) -> FanoutMeasurement:
    """Count requested-versus-planned rows across ``evidence``'s evaluations.

    Both evidence surfaces are read, because which one an optimizer uses is
    its own choice: MIPROv2 resolves intents, GEPA emits search evidence
    and no intents at all. Measuring only one would report a GEPA run as
    having evaluated nothing.

    Each eval-evidence record is counted once, keyed by its ref. GEPA
    re-emits its entire replayed prefix as search evidence on every step --
    step *i* carries roughly *i* entries -- so counting citations rather
    than records would report rows growing quadratically in the step count
    while the paid rows stayed flat. The ref is a content hash, so
    deduplicating on it is exactly "the same evaluation", not a heuristic.

    An evaluation whose ``eval_result_ref`` did not resolve to eval
    evidence is skipped: a failed evaluation has no row accounting to
    compare, and scoring it as zero would understate the run instead of
    reporting the failure. The audit package's invariants are what report
    a failed evaluation.
    """
    counts: list[IntentRowCount] = []
    seen: set[TypedRef] = set()
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            request = resolution.optim_eval_request
            task_hashes = request.task_hashes
            counted = _row_count(
                evidence=evidence,
                ref=resolution.eval_result_ref,
                seen=seen,
                step_index=entry.index,
                position=position,
                purpose=eval_purpose(request.eval_request.metadata),
                requested=(None if task_hashes is None else len(task_hashes)),
                source=INTENT_SOURCE,
            )
            if counted is not None:
                counts.append(counted)
        for position, search in enumerate(entry.search_evidence):
            counted = _row_count(
                evidence=evidence,
                ref=search.eval_result_ref,
                seen=seen,
                step_index=entry.index,
                position=position,
                purpose=None,
                requested=None,
                source=SEARCH_SOURCE,
            )
            if counted is not None:
                counts.append(counted)
    intents = tuple(counts)
    widest = max(
        (
            intent.requested
            for intent in intents
            if intent.requested is not None
        ),
        default=0,
    )
    if widest == 0:
        # An optimizer that declares no task subsets -- GEPA -- still
        # witnesses the split it evaluated, through the rows it planned.
        widest = max((intent.planned for intent in intents), default=0)
    planned_rows = sum(intent.planned for intent in intents)
    subset_rows = sum(
        intent.planned if intent.requested is None else intent.requested
        for intent in intents
    )
    seeds = max((intent.num_seeds for intent in intents), default=1)
    return FanoutMeasurement(
        run_id=evidence.run_id,
        optimizer=evidence.optimizer,
        intents=intents,
        widest_task_set=widest,
        planned_rows=planned_rows,
        subset_formula_rows=subset_rows,
        full_split_formula_rows=len(intents) * widest * seeds,
    )


def _row_count(  # noqa: PLR0913
    *,
    evidence: RunEvidence,
    ref: TypedRef | None,
    seen: set[TypedRef],
    step_index: int,
    position: int,
    purpose: str | None,
    requested: int | None,
    source: str,
) -> IntentRowCount | None:
    """One evaluation's row count, or None when it is not new evidence."""
    if ref is None or ref in seen:
        return None
    eval_evidence = evidence.eval_evidence(ref)
    if eval_evidence is None:
        return None
    seen.add(ref)
    return IntentRowCount(
        step_index=step_index,
        position=position,
        purpose=purpose,
        requested=requested,
        planned=eval_evidence.row_accounting.planned,
        num_seeds=eval_evidence.num_seeds,
        source=source,
    )


def measure_run_directory(run_dir: Path) -> FanoutMeasurement:
    """Measure fan-out for the completed run at ``run_dir``."""
    return measure_fanout(load_run_evidence(run_dir))
