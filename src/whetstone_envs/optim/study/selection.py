"""Selection on official, reporting on held-out -- as structure, not custom.

``report_arm`` is the single entry point through which any held-out number
enters the study, and it is built so that the two leakage rules it enforces
cannot be violated by a caller who forgets them:

* **L2 (selection on official, once per arm).** The arg-max over an arm's
  runs is computed on the official split, and the resulting selection record
  is persisted into the study's selection log before anything else happens.
  The log refuses a second entry for the same arm.
* **L3 (held-out evaluated once per reported candidate).** The held-out call
  is issued only after the log has been read back and confirmed to contain
  this arm's selection. A second held-out evaluation for the same candidate
  is refused by the ledger rather than merely flagged afterwards.

That ordering is why the held-out evaluation lives inside this function
instead of beside it. A caller cannot reach the held-out engine through this
module without first having persisted a selection, and cannot persist two
selections for one arm, so the sequence "select on official, freeze, then
measure once" is the only sequence expressible.

:mod:`whetstone_envs.optim.study.leakage` re-checks both rules over the
finished artifacts. That is a second line of defence, not the first: L6
detects, and detection after a paid held-out evaluation cannot undo the
leak it found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from whetstone.eval.analysis import bootstrap_paired_delta_ci, holm_adjust

from whetstone_envs.optim.study.power import COMPLETENESS_BACKSTOP
from whetstone_envs.optim.study.spec import CI_LEVEL, RESAMPLES

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "SELECTION_RULE",
    "ArmDelta",
    "ArmReport",
    "ArmStatistics",
    "CandidateScore",
    "HeldOutEvaluator",
    "HeldOutMeasurement",
    "OfficialScorer",
    "RunCandidate",
    "SelectionError",
    "SelectionLog",
    "SelectionRecord",
    "analyze_arms",
    "null_triggers_downgrade",
    "report_arm",
    "report_reference_candidate",
]

#: The pre-registered selection rule, recorded verbatim on every selection.
#: Persisted, so it is a named constant rather than an inline string.
SELECTION_RULE = "argmax-official"


class SelectionError(RuntimeError):
    """A selection or held-out rule was violated.

    Distinct from ``ValueError`` because these are protocol violations, not
    bad arguments: a caller must not catch and continue past one.
    """


@dataclass(frozen=True, slots=True)
class RunCandidate:
    """One run's terminal candidate, as the arm's selection sees it."""

    run_id: str
    seed: int
    candidate_name: str
    template: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run ids must be nonblank")
        if not self.template:
            raise ValueError(
                f"run {self.run_id!r} has no terminal template to score"
            )


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One candidate's official score and its per-task vector."""

    run_id: str
    score: float
    per_task: tuple[float, ...]
    eval_config_hash: str
    completeness: float

    def __post_init__(self) -> None:
        if not self.per_task:
            raise ValueError(
                f"run {self.run_id!r} scored no tasks on official"
            )
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness must be a fraction in [0, 1]")


@dataclass(frozen=True, slots=True)
class HeldOutMeasurement:
    """One candidate's single held-out evaluation."""

    candidate_name: str
    per_task: tuple[float, ...]
    mean: float
    eval_config_hash: str
    repeats: int
    completeness: float
    evidence_ref: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.per_task:
            raise ValueError(
                f"candidate {self.candidate_name!r} scored no held-out tasks"
            )
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness must be a fraction in [0, 1]")


class OfficialScorer(Protocol):
    """Score one candidate on the official split.

    Selection reads this and nothing else. Keeping it a port rather than a
    concrete call is what lets the whole selection path run on a fake
    transport in tests, and what keeps this module free of provider wiring.
    """

    def __call__(self, candidate: RunCandidate) -> CandidateScore: ...


class HeldOutEvaluator(Protocol):
    """Evaluate one candidate on the held-out split, exactly once."""

    def __call__(
        self, *, candidate_name: str, template: str
    ) -> HeldOutMeasurement: ...


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """The persisted decision: which run represents this arm, and why."""

    arm_id: str
    selected_run_id: str
    official_score: float
    rule: str = SELECTION_RULE

    def __post_init__(self) -> None:
        if self.rule != SELECTION_RULE:
            raise ValueError(
                f"selection rule must be {SELECTION_RULE!r}; a different "
                "rule is a different pre-registration"
            )


@dataclass(slots=True)
class SelectionLog:
    """The study's selection ledger, and the guard on held-out spend.

    This is deliberately mutable and deliberately narrow. It holds two facts
    -- which arm has selected, and which candidate has been measured on
    held-out -- and it refuses the second occurrence of either. Wave 4b
    serializes the records into ``study.json``; the refusal lives here
    because that is where the ordering is enforced.
    """

    records: list[SelectionRecord] = field(default_factory=list)
    held_out_measured: list[str] = field(default_factory=list)

    def record_selection(self, record: SelectionRecord) -> None:
        """Persist an arm's selection, refusing a second one (L2)."""
        if any(existing.arm_id == record.arm_id for existing in self.records):
            raise SelectionError(
                f"arm {record.arm_id!r} already selected a representative "
                "run; selection happens exactly once per arm"
            )
        self.records.append(record)

    def selection_for(self, arm_id: str) -> SelectionRecord | None:
        for record in self.records:
            if record.arm_id == arm_id:
                return record
        return None

    def require_selection(self, arm_id: str) -> SelectionRecord:
        """Read back an arm's persisted selection, or refuse to proceed.

        ``report_arm`` calls this immediately before the held-out
        evaluation. Reading the persisted record back -- rather than reusing
        the value in hand -- is what makes the held-out call structurally
        unreachable before the selection is durable.
        """
        record = self.selection_for(arm_id)
        if record is None:
            raise SelectionError(
                f"arm {arm_id!r} has no persisted selection; held-out "
                "evaluation is unreachable before selection is recorded"
            )
        return record

    def claim_held_out(self, candidate_name: str) -> None:
        """Claim the one held-out evaluation this candidate gets (L3)."""
        if candidate_name in self.held_out_measured:
            raise SelectionError(
                f"candidate {candidate_name!r} was already evaluated on "
                "held-out; each reported candidate is evaluated exactly once"
            )
        self.held_out_measured.append(candidate_name)

    def held_out_count(self, candidate_name: str) -> int:
        return self.held_out_measured.count(candidate_name)


@dataclass(frozen=True, slots=True)
class ArmReport:
    """One arm's selection, its official scores, and its held-out number."""

    arm_id: str
    selection: SelectionRecord
    official_scores: tuple[CandidateScore, ...]
    representative: RunCandidate
    held_out: HeldOutMeasurement


def _argmax_official(
    scores: Iterable[CandidateScore],
) -> CandidateScore:
    """The highest official score, ties broken by the earlier run.

    Ties go to the earlier run rather than to an arbitrary one so that
    re-running the selection over the same scores always names the same
    representative.
    """
    best: CandidateScore | None = None
    for score in scores:
        if best is None or score.score > best.score:
            best = score
    if best is None:
        raise ValueError("an arm needs at least one scored run to select from")
    return best


def report_arm(  # noqa: PLR0913
    *,
    arm_id: str,
    runs: tuple[RunCandidate, ...],
    score_official: OfficialScorer,
    evaluate_held_out: HeldOutEvaluator,
    log: SelectionLog,
    candidate_name: str | None = None,
) -> ArmReport:
    """Select this arm's representative on official, then measure it once.

    The four steps happen in exactly this order and no other order is
    reachable through this function:

    1. every run's terminal candidate is scored on **official**;
    2. the arg-max names the representative candidate;
    3. the selection is **persisted** into ``log``, which refuses a second
       selection for this arm;
    4. the persisted selection is **read back**, and only then is the single
       held-out evaluation issued, against a ledger that refuses a second
       evaluation of the same candidate.

    ``candidate_name`` defaults to the arm id, which is what makes the
    held-out ledger's one-per-reported-candidate rule align with the
    one-per-arm selection rule. Anchors and nulls pass their own names and
    reach held-out through this same function, so every held-out number in
    the study is produced by one procedure (L4).
    """
    if not runs:
        raise ValueError(f"arm {arm_id!r} has no runs to report")
    run_ids = tuple(run.run_id for run in runs)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"arm {arm_id!r} repeats a run id")

    official_scores = tuple(score_official(run) for run in runs)
    eval_config_hashes = {score.eval_config_hash for score in official_scores}
    if len(eval_config_hashes) > 1:
        # An arm whose runs were scored under different Eval Configs has no
        # comparable arg-max, so the selection would be meaningless.
        raise SelectionError(
            f"arm {arm_id!r} scored its runs under {len(eval_config_hashes)} "
            "different official Eval Configs; selection requires one"
        )
    best = _argmax_official(official_scores)
    representative = next(run for run in runs if run.run_id == best.run_id)

    log.record_selection(
        SelectionRecord(
            arm_id=arm_id,
            selected_run_id=best.run_id,
            official_score=best.score,
        )
    )
    # Read back rather than reuse: the held-out call must depend on the
    # persisted record, not on a local variable, or "persisted before
    # measured" would be a convention instead of a mechanism.
    selection = log.require_selection(arm_id)
    if selection.selected_run_id != representative.run_id:
        raise SelectionError(
            f"arm {arm_id!r} persisted a selection for run "
            f"{selection.selected_run_id!r} but is about to measure "
            f"{representative.run_id!r}"
        )

    reported_name = candidate_name or arm_id
    log.claim_held_out(reported_name)
    held_out = evaluate_held_out(
        candidate_name=reported_name, template=representative.template
    )
    return ArmReport(
        arm_id=arm_id,
        selection=selection,
        official_scores=official_scores,
        representative=representative,
        held_out=held_out,
    )


def report_reference_candidate(
    *,
    candidate_name: str,
    template: str,
    evaluate_held_out: HeldOutEvaluator,
    log: SelectionLog,
) -> HeldOutMeasurement:
    """Measure an anchor or null seed on held-out, once, with no selection.

    Naive, ceiling, and null-B's seed candidate have nothing to select
    between -- there is one candidate by construction -- so they skip the
    arg-max but keep the identical held-out procedure and the identical
    once-only ledger. Routing them through the same ledger is what makes L3
    and L4 hold for the anchors too, rather than only for the arms.
    """
    log.claim_held_out(candidate_name)
    return evaluate_held_out(candidate_name=candidate_name, template=template)


@dataclass(frozen=True, slots=True)
class ArmDelta:
    """One arm's paired held-out vectors against the naive anchor.

    Both vectors are per task and aligned, which is what makes the bootstrap
    paired: it resamples *tasks*, and each resampled task carries the arm's
    and the anchor's score together.
    """

    arm_id: str
    arm_per_task: tuple[float, ...]
    naive_per_task: tuple[float, ...]
    completeness: float = 1.0

    def __post_init__(self) -> None:
        if len(self.arm_per_task) != len(self.naive_per_task):
            raise ValueError(
                f"arm {self.arm_id!r} has unaligned paired score vectors"
            )
        if not self.arm_per_task:
            raise ValueError(f"arm {self.arm_id!r} has no paired tasks")
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness must be a fraction in [0, 1]")


@dataclass(frozen=True, slots=True)
class ArmStatistics:
    """One arm's held-out delta against naive, with its interval.

    ``claimed`` is the study's three-state verdict input: an interval whose
    completeness fell below the backstop, or which includes zero, is
    reported but never claimed -- the same way a failed fidelity audit makes
    an efficacy number descriptive only.
    """

    arm_id: str
    delta: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_holm: float
    completeness: float
    excludes_zero: bool
    claimed: bool


def analyze_arms(
    arms: tuple[ArmDelta, ...],
    *,
    level: float = CI_LEVEL,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> tuple[ArmStatistics, ...]:
    """Interval, p-value, and Holm correction for every real arm.

    Holm runs over the **real optimizers only** (m = 4). Nulls are controls,
    not hypotheses: correcting them would spend family-wise error budget on
    arms the study never claims, and would make a null harder to trip
    exactly when tripping it matters most. Pass only real arms here and read
    each null's uncorrected interval directly.

    An arm below the completeness backstop keeps its numbers -- they are
    still the best description of what was measured -- but ``claimed`` is
    false, so the report states the delta descriptively and makes no
    significance claim from it.
    """
    if not arms:
        return ()
    intervals = tuple(
        bootstrap_paired_delta_ci(
            arm.naive_per_task,
            arm.arm_per_task,
            level=level,
            resamples=resamples,
            seed=seed,
        )
        for arm in arms
    )
    adjusted = holm_adjust(tuple(ci.p_value for ci in intervals))
    return tuple(
        ArmStatistics(
            arm_id=arm.arm_id,
            delta=ci.point,
            ci_low=ci.low,
            ci_high=ci.high,
            p_bootstrap=ci.p_value,
            p_holm=p_holm,
            completeness=arm.completeness,
            excludes_zero=ci.excludes_zero(),
            claimed=(
                ci.excludes_zero()
                and arm.completeness >= COMPLETENESS_BACKSTOP
            ),
        )
        for arm, ci, p_holm in zip(arms, intervals, adjusted, strict=True)
    )


def null_triggers_downgrade(
    *, null_delta: float, mde_measured: float, excludes_zero: bool
) -> bool:
    """Whether a null's result voids the study's efficacy claims (F12).

    Both conditions are required, exactly as pre-registered: the magnitude
    must exceed the measured MDE **and** the interval must exclude zero. A
    5% coin-flip landing at a tiny significant delta does not void a study,
    and a large but unresolvable delta is noise; only both together mean
    selection over nothing produced a real, detectable movement.
    """
    return abs(null_delta) > mde_measured and excludes_zero
