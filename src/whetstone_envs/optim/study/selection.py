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
    held_out = evaluate_held_out(
        candidate_name=reported_name, template=representative.template
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
    measurement = evaluate_held_out(
        candidate_name=candidate_name, template=template
    )
    log.complete_held_out(measurement)
    return measurement


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
        self, study_dir: Path, *, stage: str = DEFAULT_SELECTION_STAGE
    ) -> None:
        self._study_dir = study_dir
        self._stage = stage

    @property
    def study_dir(self) -> Path:
        """Where this ledger persists."""
        return self._study_dir

    @property
    def stage(self) -> str:
        """The stage whose selections and claims this ledger owns."""
        return self._stage

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
        if claims[index].completed:
            raise SelectionError(
                f"candidate {measurement.candidate_name!r} already completed "
                "its held-out evaluation"
            )
        claims[index] = PersistedHeldOutClaim(
            candidate_name=measurement.candidate_name,
            stage=self._stage,
            eval_config_hash=measurement.eval_config_hash,
            repeats=measurement.repeats,
            mean=measurement.mean,
            completeness=measurement.completeness,
        )
        self._write(
            manifest.model_copy(update={"held_out_claims": tuple(claims)})
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
