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

from whetstone_envs.optim.study.manifest import (
    CORRECTION_FAMILY_SIZE,
    SELECTION_RULE_ARGMAX_OFFICIAL,
    OfficialScoreEntry,
    StageId,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.manifest import (
    HeldOutClaimRecord as PersistedHeldOutClaim,
)
from whetstone_envs.optim.study.manifest import (
    SelectionRecord as PersistedSelectionRecord,
)
from whetstone_envs.optim.study.power import COMPLETENESS_BACKSTOP
from whetstone_envs.optim.study.spec import CI_LEVEL, RESAMPLES

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from whetstone_envs.optim.study.manifest import StudyManifest

__all__ = [
    "DEFAULT_SELECTION_STAGE",
    "SELECTION_RULE",
    "ArmDelta",
    "ArmReport",
    "ArmStatistics",
    "CandidateScore",
    "HeldOutEvaluator",
    "HeldOutMeasurement",
    "HeldOutRefusalError",
    "ManifestSelectionLog",
    "OfficialScorer",
    "RunCandidate",
    "SelectionError",
    "SelectionLedger",
    "SelectionLog",
    "SelectionRecord",
    "analyze_arms",
    "null_triggers_downgrade",
    "report_arm",
    "report_reference_candidate",
]

#: The pre-registered selection rule, recorded verbatim on every selection.
#: The manifest owns the persisted literal, so this is an alias rather than
#: a second spelling that could drift from the stored one.
SELECTION_RULE = SELECTION_RULE_ARGMAX_OFFICIAL

#: The stage a selection or claim records when its caller does not name one.
#: Stage 2 is the study's reported stage, so an unnamed selection is the
#: reported one rather than an anonymous one.
DEFAULT_SELECTION_STAGE = StageId.STAGE2.value


class SelectionError(RuntimeError):
    """A selection or held-out rule was violated.

    Distinct from ``ValueError`` because these are protocol violations, not
    bad arguments: a caller must not catch and continue past one.
    """


class HeldOutRefusalError(RuntimeError):
    """A held-out evaluation returned, was billed, and is unfit to report.

    The one outcome that may settle a claim without a measurement, and it
    is deliberately narrow. Two things have to be true before a claim can
    be written off: the provider call **returned**, so the spend is real
    and already ledgered; and the result was **judged** against a
    deterministic rule, so re-issuing would produce the same verdict and
    buy nothing.

    A transient failure satisfies neither. A connection reset, a 503, or
    an OOM mid-evaluation may never have reached the provider at all --
    the spend could be zero -- and the next attempt could well succeed.
    Settling those as refusals would permanently burn a candidate's one
    evaluation over a blip, which is precisely the class of infrastructure
    outcome this wave exists to stop treating as final. They stay
    unsettled, which is crash-shaped, and a resume treats them as the
    in-flight crash they are.

    Raised by the layer that does the judging --
    :meth:`~whetstone_envs.optim.study.arms.RoleScorer.evidence_for`,
    after pricing -- so the narrowing lives with the decision rather than
    being re-derived by matching on messages further out.
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
    #: Rows achieved per task, aligned with ``per_task``. Empty when the
    #: caller has only the aggregate, in which case the completeness
    #: weighting falls back to spreading it evenly -- which is an
    #: approximation, so the measured vector is preferred wherever it
    #: exists.
    per_task_counts: tuple[int, ...] = ()
    evidence_ref: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.per_task:
            raise ValueError(
                f"candidate {self.candidate_name!r} scored no held-out tasks"
            )
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness must be a fraction in [0, 1]")
        if self.per_task_counts and len(self.per_task_counts) != len(
            self.per_task
        ):
            raise ValueError(
                f"candidate {self.candidate_name!r} has "
                f"{len(self.per_task_counts)} per-task row counts for "
                f"{len(self.per_task)} tasks"
            )
        if any(count < 0 for count in self.per_task_counts):
            raise ValueError("per-task row counts are non-negative")


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
    """The persisted decision: which run represents this arm, and why.

    ``stage`` scopes the decision, because the study selects once per arm
    *per stage*: the pilot's arg-max runs over two runs and the full
    design's over five, and only the second is the study's reported
    selection. Both are recorded so the pilot's preliminary delta is
    checkable rather than overwritten.
    """

    arm_id: str
    selected_run_id: str
    official_score: float
    rule: str = SELECTION_RULE
    stage: str = DEFAULT_SELECTION_STAGE

    def __post_init__(self) -> None:
        if self.rule != SELECTION_RULE:
            raise ValueError(
                f"selection rule must be {SELECTION_RULE!r}; a different "
                "rule is a different pre-registration"
            )
        if not self.stage.strip():
            raise ValueError("a selection names the stage that made it")


class SelectionLedger(Protocol):
    """The ordering guard ``report_arm`` reaches held-out through.

    Two implementations satisfy it. :class:`SelectionLog` holds the ledger
    in memory, which is what a test or a dry run wants. :class:`
    ManifestSelectionLog` writes each selection into the study's own
    ``study.json`` and reads it back off disk, which is what a paid stage
    wants: the held-out call then depends on a *durable* record, so a crash
    between selecting and measuring leaves the selection recorded rather
    than lost.

    Both refuse a second selection per arm (L2) and a second held-out
    evaluation per candidate (L3), because that refusal is the mechanism,
    not the storage.
    """

    def record_selection(self, record: SelectionRecord) -> None: ...

    def selection_for(self, arm_id: str) -> SelectionRecord | None: ...

    def require_selection(self, arm_id: str) -> SelectionRecord: ...

    def claim_held_out(self, candidate_name: str) -> None: ...

    def complete_held_out(self, measurement: HeldOutMeasurement) -> None: ...

    def refuse_held_out(self, candidate_name: str, reason: str) -> None:
        """Settle a claimed evaluation that produced no reportable number.

        The counterpart to :meth:`complete_held_out`, for the case where
        the provider was reached and billed but the result was judged
        unfit to report. Both are terminal, and recording the refusal is
        what makes it so: without it the claim stays outstanding, which a
        resume must read as an evaluation that died in flight -- neither
        re-issuable nor writable off.
        """
        ...

    def refused_claim_for(self, candidate_name: str) -> str | None:
        """The reason a settled claim recorded, if it was refused."""
        ...

    def completed_claim_for(
        self, candidate_name: str
    ) -> HeldOutMeasurement | None:
        """The measurement a completed claim recorded, if there is one.

        This is what lets a stage that crashed partway through reporting
        resume: the arm's held-out evaluation is already paid for and its
        result is durable, so the report is rebuilt from the claim instead
        of re-issuing an evaluation the ledger would refuse anyway.

        An *outstanding* claim returns None. A claim is written before the
        evaluation is issued, so an incomplete one means the process died
        with that evaluation in flight -- which is a different fact from a
        completed measurement, and the caller has to treat it as one.
        """
        ...

    def held_out_count(self, candidate_name: str) -> int: ...


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
    #: The stage this log scopes its refusals to. One log per stage, so a
    #: pilot and the full design each select once without either being able
    #: to select twice.
    stage: str = DEFAULT_SELECTION_STAGE

    def record_selection(self, record: SelectionRecord) -> None:
        """Persist an arm's selection, refusing a second one (L2)."""
        if any(
            existing.arm_id == record.arm_id and existing.stage == record.stage
            for existing in self.records
        ):
            raise SelectionError(
                f"arm {record.arm_id!r} already selected a representative "
                f"run at {record.stage}; selection happens exactly once "
                "per arm per stage"
            )
        self.records.append(record)

    def selection_for(self, arm_id: str) -> SelectionRecord | None:
        for record in self.records:
            if record.arm_id == arm_id and record.stage == self.stage:
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

    def complete_held_out(self, measurement: HeldOutMeasurement) -> None:
        """Record that a claimed evaluation returned.

        The in-memory ledger claims and completes in one list because a
        process that lost the claim lost the measurement with it; the
        distinction only earns its keep in the durable ledger, where a
        crash can separate the two.
        """
        if measurement.candidate_name not in self.held_out_measured:
            raise SelectionError(
                f"candidate {measurement.candidate_name!r} completed a "
                "held-out evaluation it never claimed"
            )

    def refuse_held_out(self, candidate_name: str, reason: str) -> None:
        """Record that a claimed evaluation was refused.

        Checked but not stored, for the same reason
        :meth:`complete_held_out` is: a process holding this ledger cannot
        outlive itself, so there is no resume to make the record legible
        to. The claim check still runs, because refusing an evaluation
        that was never claimed is a caller error either way.
        """
        del reason
        if candidate_name not in self.held_out_measured:
            raise SelectionError(
                f"candidate {candidate_name!r} refused a held-out "
                "evaluation it never claimed"
            )

    def refused_claim_for(self, candidate_name: str) -> str | None:
        """Always None: this ledger keeps nothing across a process."""
        del candidate_name
        return None

    def completed_claim_for(
        self, candidate_name: str
    ) -> HeldOutMeasurement | None:
        """Always None: this ledger keeps no measurement to rebuild from.

        Rebuilding an arm's report from a claim is a *durable*-ledger
        capability. A process holding this ledger lost its measurements
        with itself, so reporting one here would be a fabrication.
        """
        del candidate_name
        return None

    def held_out_count(self, candidate_name: str) -> int:
        return self.held_out_measured.count(candidate_name)


@dataclass(frozen=True, slots=True)
class ArmReport:
    """One arm's selection, its official scores, and its held-out number.

    ``held_out`` is ``None`` for an arm whose one held-out evaluation was
    spent and refused. That arm was selected, scored on official, and
    billed, but produced no number anybody may report -- so it carries its
    selection and its official evidence into the report and contributes no
    held-out row, which the report renders as ``VERDICT_UNMEASURED``. The
    alternative is what this replaced: raising, and taking every *other*
    arm's already-paid evidence down with it.
    """

    arm_id: str
    selection: SelectionRecord
    official_scores: tuple[CandidateScore, ...]
    representative: RunCandidate
    held_out: HeldOutMeasurement | None


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
    log: SelectionLedger,
    candidate_name: str | None = None,
    stage: str = DEFAULT_SELECTION_STAGE,
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
            stage=stage,
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

    # The reported candidate keeps the arm's own name: the report keys its
    # held-out rows by it, and a stage-decorated name would leave the
    # study's own result unfindable. The *claim* is what carries the stage,
    # so the pilot and the full design each get exactly one held-out
    # evaluation without either colliding with the other.
    reported_name = candidate_name or arm_id
    log.claim_held_out(reported_name)
    held_out = _evaluate_claimed(
        candidate_name=reported_name,
        template=representative.template,
        evaluate_held_out=evaluate_held_out,
        log=log,
    )
    # The claim is completed with what the evaluation returned, so a claim
    # left outstanding names a crashed evaluation rather than a missing one.
    log.complete_held_out(held_out)
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
    log: SelectionLedger,
) -> HeldOutMeasurement:
    """Measure an anchor or null seed on held-out, once, with no selection.

    Naive, ceiling, and null-B's seed candidate have nothing to select
    between -- there is one candidate by construction -- so they skip the
    arg-max but keep the identical held-out procedure and the identical
    once-only ledger. Routing them through the same ledger is what makes L3
    and L4 hold for the anchors too, rather than only for the arms.
    """
    log.claim_held_out(candidate_name)
    measurement = _evaluate_claimed(
        candidate_name=candidate_name,
        template=template,
        evaluate_held_out=evaluate_held_out,
        log=log,
    )
    log.complete_held_out(measurement)
    return measurement


def _evaluate_claimed(
    *,
    candidate_name: str,
    template: str,
    evaluate_held_out: HeldOutEvaluator,
    log: SelectionLedger,
) -> HeldOutMeasurement:
    """Issue a claimed evaluation, settling the claim if it is refused.

    The claim is written before the call, so between the two there is a
    window where the candidate has spent its one evaluation and no record
    says what came of it. A crash there is genuinely unrecoverable and
    stays that way. A :class:`HeldOutRefusalError` is not: the call returned,
    the spend is real and already ledgered, and the only thing missing was
    a reportable number -- so the claim is settled as refused rather than
    left looking like a crash, and the study resumes into a degraded
    verdict instead of a dead end only a hand-edited manifest could clear.

    **Only that type settles.** Catching every exception here would treat
    a connection reset, a 503, or an OOM -- failures that may never have
    reached the provider, and that a retry could well survive -- as a
    permanent write-off of the candidate's one evaluation. Those propagate
    untouched, leaving the claim outstanding, which is exactly the
    crash-shaped state a resume knows how to refuse safely.

    ``KeyboardInterrupt`` and other ``BaseException``\\ s escape without
    settling, by design: an operator stopping a run has not judged
    anything, and recording a refusal on their behalf would write off an
    evaluation nobody evaluated. That is now a property of the narrowed
    ``except`` rather than an accident of what ``Exception`` happens not
    to cover.

    The exception still propagates in every case. Settling the claim
    records what happened; it does not decide that the caller should
    carry on.
    """
    try:
        return evaluate_held_out(
            candidate_name=candidate_name, template=template
        )
    except HeldOutRefusalError as error:
        log.refuse_held_out(candidate_name, f"{type(error).__name__}: {error}")
        raise


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
    family_size: int = CORRECTION_FAMILY_SIZE,
) -> tuple[ArmStatistics, ...]:
    """Interval, p-value, and Holm correction for every real arm.

    Holm runs over the **real optimizers only** (m = 4). Nulls are controls,
    not hypotheses: correcting them would spend family-wise error budget on
    arms the study never claims, and would make a null harder to trip
    exactly when tripping it matters most. Pass only real arms here and read
    each null's uncorrected interval directly.

    **``m`` is the pre-registered family size, not the number of arms in
    hand.** The family was fixed before any spend, so a study that analyses
    two of its four hypotheses -- a pilot, a partial resume, a stage that
    lost an arm -- still corrects at ``m = 4``. Deriving ``m`` from
    ``len(arms)`` would under-correct exactly the partial studies whose
    multiplicity risk is unchanged, and would make the correction a function
    of how far the study got rather than of what it pre-registered. Passing
    more arms than the family declares is refused rather than silently
    widening the family after the fact.

    An arm below the completeness backstop keeps its numbers -- they are
    still the best description of what was measured -- but ``claimed`` is
    false, so the report states the delta descriptively and makes no
    significance claim from it.
    """
    if not arms:
        return ()
    if family_size < len(arms):
        raise ValueError(
            f"the pre-registered Holm family holds {family_size} "
            f"hypotheses but {len(arms)} arms were analysed; a family is "
            "fixed before spend, not widened to fit its results"
        )
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
    # Holm at the pre-registered ``m``: the unanalysed members of the family
    # enter as p = 1, which is the largest value any hypothesis can take, so
    # they never displace an observed arm in the step-down ordering and the
    # scaling factor stays ``m - rank`` for every arm actually measured.
    # That is exactly Holm over the declared family with the missing arms
    # unrejected, which is the honest reading of an arm the study did not
    # measure.
    padded = tuple(ci.p_value for ci in intervals) + (1.0,) * (
        family_size - len(arms)
    )
    adjusted = holm_adjust(padded)[: len(arms)]
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


class ManifestSelectionLog:
    """The durable ledger: selections live in the study's own manifest.

        Every mutation goes through ``write_study_manifest(replace=True)`` and
        every read goes back through ``read_study_manifest``. Nothing is cached
        between calls, which is the point: ``report_arm`` reads the selection
        back from disk immediately before issuing its held-out evaluation, so
        "persisted before measured" is a property of the filesystem rather than
        of a variable that happened to still be in scope.

    Held-out claims are durable too, and they have to be written *before*
        the evaluation rather than after: a held-out row carries a
        Holm-corrected p-value, which is a whole-study computation that cannot
        exist until every arm is measured, so waiting for the row would leave
        the window between paying for an evaluation and recording that it
        happened completely unguarded. The manifest's ``held_out_claims`` block
        closes it. A claim is written when the evaluation is issued and
        completed with its result when it returns, so a stage that crashes
        mid-evaluation resumes knowing the candidate already spent its one
        shot, and an outstanding claim is legible as a crashed evaluation
        rather than as one that never happened.
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        stage: str = DEFAULT_SELECTION_STAGE,
        transport: str,
    ) -> None:
        self._study_dir = study_dir
        self._stage = stage
        self._transport = transport

    @property
    def study_dir(self) -> Path:
        """Where this ledger persists."""
        return self._study_dir

    @property
    def stage(self) -> str:
        """The stage whose selections and claims this ledger owns."""
        return self._stage

    @property
    def transport(self) -> str:
        """The transport this stage's measurements are bought on.

        Carried because a score is evidence about a run *on a transport*,
        and run ids are deterministic: without it the read-back cannot
        tell this stage's own measurement from one a re-calibrated study
        made somewhere else under the same name.
        """
        return self._transport

    def _read(self) -> StudyManifest:
        return read_study_manifest(self._study_dir)

    def record_selection(self, record: SelectionRecord) -> None:
        """Persist an arm's selection into ``study.json`` (L2)."""
        manifest = self._read()
        if any(
            existing.arm_id == record.arm_id and existing.stage == record.stage
            for existing in manifest.selection
        ):
            raise SelectionError(
                f"arm {record.arm_id!r} already selected a representative "
                f"run at {record.stage}; selection happens exactly once "
                "per arm per stage"
            )
        self._write(
            manifest.model_copy(
                update={
                    "selection": (
                        *manifest.selection,
                        PersistedSelectionRecord(
                            arm_id=record.arm_id,
                            selected_run_id=record.selected_run_id,
                            official_score=record.official_score,
                            rule=record.rule,
                            stage=record.stage,
                        ),
                    )
                }
            )
        )

    def selection_for(self, arm_id: str) -> SelectionRecord | None:
        for entry in self._read().selection:
            if entry.arm_id == arm_id and entry.stage == self._stage:
                return SelectionRecord(
                    arm_id=entry.arm_id,
                    selected_run_id=entry.selected_run_id,
                    official_score=entry.official_score,
                    rule=entry.rule,
                    stage=entry.stage,
                )
        return None

    def require_selection(self, arm_id: str) -> SelectionRecord:
        """Read the persisted selection back, or refuse to proceed."""
        record = self.selection_for(arm_id)
        if record is None:
            raise SelectionError(
                f"arm {arm_id!r} has no persisted selection; held-out "
                "evaluation is unreachable before selection is recorded"
            )
        return record

    def claim_held_out(self, candidate_name: str) -> None:
        """Claim this candidate's one held-out evaluation, durably (L3).

        The write happens before the evaluation is issued. That ordering is
        the guarantee: a crash after the provider call but before any result
        is recorded still leaves the claim on disk, so the candidate cannot
        quietly be measured a second time on resume.
        """
        manifest = self._read()
        if self._claim_index(manifest, candidate_name) is not None:
            raise SelectionError(
                f"candidate {candidate_name!r} was already evaluated on "
                f"held-out at {self._stage}; each reported candidate is "
                "evaluated exactly once"
            )
        self._write(
            manifest.model_copy(
                update={
                    "held_out_claims": (
                        *manifest.held_out_claims,
                        PersistedHeldOutClaim(
                            candidate_name=candidate_name,
                            stage=self._stage,
                        ),
                    )
                }
            )
        )

    def complete_held_out(self, measurement: HeldOutMeasurement) -> None:
        """Attach the returned measurement to its outstanding claim."""
        manifest = self._read()
        index = self._claim_index(manifest, measurement.candidate_name)
        if index is None:
            raise SelectionError(
                f"candidate {measurement.candidate_name!r} completed a "
                "held-out evaluation it never claimed"
            )
        claims = list(manifest.held_out_claims)
        if claims[index].settled:
            raise SelectionError(
                f"candidate {measurement.candidate_name!r} already settled "
                "its held-out evaluation"
            )
        claims[index] = PersistedHeldOutClaim(
            candidate_name=measurement.candidate_name,
            stage=self._stage,
            eval_config_hash=measurement.eval_config_hash,
            repeats=measurement.repeats,
            mean=measurement.mean,
            completeness=measurement.completeness,
            per_task=measurement.per_task,
            per_task_counts=measurement.per_task_counts,
        )
        self._write(
            manifest.model_copy(update={"held_out_claims": tuple(claims)})
        )

    def refuse_held_out(self, candidate_name: str, reason: str) -> None:
        """Settle a claimed evaluation durably as refused.

        Written for the same reason the claim itself is: the evaluation
        was billed, and the fact that it produced nothing reportable has
        to survive the process. Without this the claim stays outstanding
        forever, and a resumed stage reads an outstanding claim as an
        in-flight crash it cannot safely act on -- which is how a refused
        evaluation used to wedge a study that had paid for everything
        else.
        """
        manifest = self._read()
        index = self._claim_index(manifest, candidate_name)
        if index is None:
            raise SelectionError(
                f"candidate {candidate_name!r} refused a held-out "
                "evaluation it never claimed"
            )
        claims = list(manifest.held_out_claims)
        if claims[index].settled:
            raise SelectionError(
                f"candidate {candidate_name!r} already settled its held-out "
                "evaluation"
            )
        claims[index] = PersistedHeldOutClaim(
            candidate_name=candidate_name,
            stage=self._stage,
            refusal=reason,
        )
        self._write(
            manifest.model_copy(update={"held_out_claims": tuple(claims)})
        )

    def refused_claim_for(self, candidate_name: str) -> str | None:
        """The reason this candidate's claim recorded, if it was refused."""
        manifest = self._read()
        index = self._claim_index(manifest, candidate_name)
        if index is None:
            return None
        return manifest.held_out_claims[index].refusal

    def completed_claim_for(
        self, candidate_name: str
    ) -> HeldOutMeasurement | None:
        """The measurement a completed claim recorded, if any.

        A resumed stage rebuilds an already-reported arm from this
        rather than re-issuing an evaluation the ledger refuses.
        """
        manifest = self._read()
        index = self._claim_index(manifest, candidate_name)
        if index is None:
            return None
        claim = manifest.held_out_claims[index]
        if (
            claim.mean is None
            or claim.eval_config_hash is None
            or claim.repeats is None
            or claim.completeness is None
            # Without the vector there is no paired delta to rebuild from,
            # and inventing one would report a number nobody measured.
            or not claim.per_task
        ):
            return None
        return HeldOutMeasurement(
            candidate_name=claim.candidate_name,
            per_task=claim.per_task,
            mean=claim.mean,
            eval_config_hash=claim.eval_config_hash,
            repeats=claim.repeats,
            completeness=claim.completeness,
            per_task_counts=claim.per_task_counts,
        )

    def held_out_count(self, candidate_name: str) -> int:
        """How many held-out evaluations this candidate has claimed.

        A claim counts whether or not it has completed, because the thing
        L3 limits is evaluations issued, not results recorded.
        """
        return sum(
            1
            for entry in self._read().held_out_claims
            if entry.candidate_name == candidate_name
            and entry.stage == self._stage
        )

    def official_score_for(self, run_id: str) -> CandidateScore | None:
        """The official score this run already bought, if it bought one.

        Official scoring is a provider call per run, and it is the one
        reporting cost that was previously re-paid on every resume: an
        already-reported arm was re-scored in full purely to rebuild a
        report the manifest could answer from. Reading the score back is
        what makes a resume free.

        **The transport is part of the match.** Run ids are deterministic,
        so a study re-calibrated onto another transport recomputes the
        same names; a score measured on the transport it left would
        otherwise be read back here and presented as this stage's
        selection evidence. The amendment drops those entries outright --
        this is what holds if one ever reaches the manifest by another
        route.
        """
        for entry in self._read().official_scores:
            if (
                entry.run_id == run_id
                and entry.stage == self._stage
                and entry.transport == self._transport
            ):
                return CandidateScore(
                    run_id=entry.run_id,
                    score=entry.score,
                    per_task=entry.per_task,
                    eval_config_hash=entry.eval_config_hash,
                    completeness=entry.completeness,
                )
        return None

    def record_official_score(
        self, *, arm_id: str, score: CandidateScore
    ) -> None:
        """Persist one run's official score the first time it is bought.

        Idempotent by ``(run_id, stage)``: a re-record of a score already
        on disk is the resume path restating what it read, and rewriting
        it would churn the manifest without changing it. A *disagreeing*
        re-record is refused, because two different scores for one run
        would leave the arg-max unable to say which one it selected on.

        A score this stage cannot read back because it was measured on
        another transport is refused rather than written beside: the
        manifest scores a run once per stage, so the two cannot coexist,
        and the stale entry is an amendment that did not take its evidence
        with it.
        """
        manifest = self._read()
        existing = self.official_score_for(score.run_id)
        if existing is not None:
            if existing != score:
                raise SelectionError(
                    f"run {score.run_id!r} already recorded a different "
                    f"official score at {self._stage}; a run is scored "
                    "once per stage"
                )
            return
        stale = next(
            (
                entry
                for entry in manifest.official_scores
                if entry.run_id == score.run_id and entry.stage == self._stage
            ),
            None,
        )
        if stale is not None:
            raise SelectionError(
                f"run {score.run_id!r} already recorded an official score "
                f"at {self._stage} measured on transport "
                f"{stale.transport!r}, and this stage is running on "
                f"{self._transport!r}; a run is scored once per stage, so "
                "the stale measurement is removed before it is re-bought"
            )
        self._write(
            manifest.model_copy(
                update={
                    "official_scores": (
                        *manifest.official_scores,
                        OfficialScoreEntry(
                            run_id=score.run_id,
                            arm_id=arm_id,
                            stage=self._stage,
                            transport=self._transport,
                            score=score.score,
                            eval_config_hash=score.eval_config_hash,
                            completeness=score.completeness,
                            per_task=score.per_task,
                        ),
                    )
                }
            )
        )

    def _claim_index(self, manifest: StudyManifest, name: str) -> int | None:
        for index, entry in enumerate(manifest.held_out_claims):
            if entry.candidate_name == name and entry.stage == self._stage:
                return index
        return None

    def _write(self, manifest: StudyManifest) -> None:
        try:
            write_study_manifest(self._study_dir, manifest, replace=True)
        except ValueError as error:
            # The manifest's own rules are the same rules this ledger
            # enforces, so a refusal is the protocol violation it names,
            # not a write failure.
            raise SelectionError(str(error)) from error
