"""The stage harness: run one stage, record what it measured.

This is the seam between the study's design modules and its manifest. Each
stage reads the pre-registered spec off the study directory, does the work
its stage is defined to do, and returns the updated
:class:`~whetstone_envs.optim.study.manifest.StudyManifest` -- so the CLI
reports what the stage recorded rather than re-reading a file the harness
may still be writing.

**Stage 0 measures; Stages 1 and 2 select.** Stage 0 calibrates the naive
and ceiling anchors on all three roles at ``K_CAL`` and evaluates the gate,
writing the ``design`` block that everything downstream reads. Stages 1 and
2 run each arm's optimizer, then route every held-out number through
:func:`~whetstone_envs.optim.study.selection.report_arm` against a
:class:`~whetstone_envs.optim.study.selection.ManifestSelectionLog`, so the
selection is durable in ``study.json`` before the held-out call is issued.

**Provider wiring is injected, not imported.** A stage needs an evaluation
engine per role, an optimizer runner, and a way to build an anchor
candidate; all three arrive as callables on :class:`StageEnvironment`. That
keeps this module runnable end to end on a fake transport with zero
provider calls, which is what the Stage-0 dry run exercises, and it keeps
the family-specific pieces -- which probes anchor the study, how a candidate
is built -- outside the harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from dr_store import ObjectStore
from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import Candidate

from whetstone_envs.optim.study.analysis import (
    AnalysisResult,
    measure_reference_candidates,
    write_held_out_analysis,
)
from whetstone_envs.optim.study.anchors import (
    EngineBinder,
    Stage0Result,
    run_stage0,
)
from whetstone_envs.optim.study.gates import (
    STAGE1_CALL_COUNT_TOLERANCE,
    estimate_optimizer_calls,
)
from whetstone_envs.optim.study.manifest import (
    AMENDMENT_REASON_TRANSPORT_CHANGE,
    COMPLETENESS_BACKSTOP,
    CORRECTION_FAMILY_SIZE,
    CORRECTION_HOLM_BONFERRONI,
    DESIGN_PROJECTION_FULL,
    PROVENANCE_AMENDED,
    PROVENANCE_ORIGINAL,
    STUDY_STORE_NAME,
    AmendmentRecord,
    ArmRecord,
    CallCountGateRecord,
    DesignRecord,
    PreRegistrationRecord,
    ReportSpendEntry,
    RunRecord,
    RunSpendRecord,
    SplitsRecord,
    StageRecord,
    StudyManifest,
    TransportName,
    pre_registration_design_hash,
    read_study_manifest,
    recorded_transport,
    write_study_manifest,
)
from whetstone_envs.optim.study.manifest import StageId as ManifestStageId
from whetstone_envs.optim.study.power import COMPLETENESS_RULE, MDE_FORMULA
from whetstone_envs.optim.study.protocols import PROTOCOL_IDS
from whetstone_envs.optim.study.selection import (
    ArmReport,
    CandidateScore,
    HeldOutEvaluator,
    ManifestSelectionLog,
    OfficialScorer,
    RunCandidate,
    report_arm,
)
from whetstone_envs.optim.study.spec import (
    CODEX_ARM_ID,
    StageId,
    arm_seeds,
    require_pinned_arms,
    require_pinned_codex_agent_model,
    spec_from_manifest,
)
from whetstone_envs.optim.study.spend import (
    ReportSpendLedger,
    ReportSpendRecord,
    run_spend_records,
    stage_spend_records,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from whetstone.eval.schema import EvalEvidence

    from whetstone_envs.optim.study.spec import ArmSpec, StudySpec

__all__ = [
    "SEED_NOTE_CONTROL_FIELD",
    "SEED_NOTE_PROVIDER_ONLY",
    "STAGE0_TRANSPORT_STAGE",
    "STAGE_PREFLIGHT_ROOT_NAME",
    "ArmRunResult",
    "OptimizerRunner",
    "StageEnvironment",
    "StageError",
    "StageResult",
    "call_count_within_estimate",
    "require_matching_transport",
    "run_arm_stage",
    "run_stage",
    "run_stage0_into_manifest",
]


#: Where the early Codex guard's preflight keeps its dr-exec job records,
#: beneath the study directory. A stage's session probe is the study's own
#: evidence rather than any one run's, so it lives beside ``study.json``
#: instead of inside a run directory the stage may never create.
STAGE_PREFLIGHT_ROOT_NAME = "codex-preflight"


class StageError(RuntimeError):
    """A stage cannot run truthfully and must not spend.

    Distinct from ``ValueError`` because these are protocol conditions --
    a stage run out of order, a gate that did not pass, an arm the spec
    never declared -- not bad arguments a caller could fix by retrying.
    """


@dataclass(frozen=True, slots=True)
class ArmRunResult:
    """One optimizer run, as the harness records it.

    ``candidate`` is what selection scores; ``record`` is what the manifest
    stores. Keeping both means the harness never has to re-open a run
    directory to remember what a run produced.
    """

    candidate: RunCandidate
    record: RunRecord
    observed_task_calls: int


class RecordedRunLoader(Protocol):
    """Re-read a run an earlier stage already paid for.

    Stage 2 selects over the union of its own runs and Stage 1's, so it has
    to see the terminal candidate of a run this process never executed.
    Loading it from the run's own artifacts is what makes "Stage 1's runs
    count toward Stage 2" real rather than a refusal: the alternative is
    paying for the new seeds and then declining to select, which is the
    worst of both.

    Returning ``None`` means the recorded run's artifacts are gone. That is
    reported by the caller as the accounting problem it is, rather than
    silently narrowing the arg-max to whatever happened to load.
    """

    def __call__(
        self, *, arm: ArmSpec, run: RunRecord
    ) -> ArmRunResult | None: ...


class OptimizerRunner(Protocol):
    """Run one arm's optimizer at one seed and report what it produced.

    The runner owns the artifact directory, the audit, and the cost
    projection; the stage harness only needs the terminal candidate and the
    record to file. Injecting it is what keeps this module free of provider
    and family wiring.
    """

    def __call__(
        self, *, arm: ArmSpec, seed: int, study_dir: Path
    ) -> ArmRunResult: ...


@dataclass(frozen=True, slots=True)
class StageEnvironment:
    """Everything a stage needs that this module refuses to import.

    ``task_ids_by_role`` and ``pool_ceiling`` come from the prepared
    experiment the caller already built, so the harness never rebuilds a
    population and cannot silently calibrate a different one than the arms
    run against.
    """

    bind_engine: EngineBinder
    naive_candidate: Candidate
    ceiling_candidate: Candidate
    task_ids_by_role: dict[EvalRole, tuple[str, ...]]
    pool_ceiling: int
    run_optimizer: OptimizerRunner | None = None
    score_official: OfficialScorer | None = None
    evaluate_held_out: HeldOutEvaluator | None = None
    #: How a stage re-reads a run an earlier stage recorded. Absent, a
    #: stage that finds one refuses rather than selecting over a subset.
    load_recorded_run: RecordedRunLoader | None = None
    #: Whether this invocation may spend on a real, billed Codex session.
    #:
    #: A spend authorization for one run of one command, not a design
    #: field: it is set from ``whetstone-study run --allow-real-codex``,
    #: never read off the manifest, and never hashed into the
    #: pre-registration. The harness reads it only to refuse a stage whose
    #: design names the Codex arm *before* any arm runs -- see
    #: :func:`_refuse_unauthorized_codex_arm`. The runner behind
    #: ``run_optimizer`` carries the same authorization and remains the
    #: thing that actually gates the spend, together with the opt-in
    #: environment variable.
    real_codex_authorized: bool = False
    #: Where the reporting pass's own spend accumulates.
    #:
    #: Official-selection scoring and held-out evaluation reach the
    #: provider outside any optimizer run, so the run fold cannot see them
    #: and the stage's own evidence route -- which is Stage 0's -- does not
    #: cover them either. The scorers append here as they go and
    #: :func:`run_arm_stage` folds the result into the stage's row once the
    #: pass is done. ``None`` on a caller that supplies its own
    #: collaborators, which ledgers nothing rather than inventing a bill.
    report_spend: ReportSpendLedger | None = None
    #: How a test points the harness's Codex preflight at the scripted
    #: fake CLI instead of a real session.
    #:
    #: ``None`` on every production path -- ``bound_stage_environment``
    #: never sets one, nothing reads it off the manifest, and no CLI flag
    #: builds one -- so the early guard reaches the real
    #: ``codex_auth_preflight``.
    #:
    #: Typed as ``object`` rather than naming
    #: :class:`~whetstone_envs.optim.codex.CodexTestSeam`, because this
    #: module deliberately does not import the optimizer stack -- the one
    #: place it needs Codex, the guard, imports inside the function. The
    #: seam is a concrete frozen record rather than a behavioural port, so
    #: a local Protocol would restate its fields rather than describe a
    #: contract; the guard passes it straight through to
    #: ``preflight_codex_session``, which names the real type.
    codex_test_seam: object | None = None
    #: Which transport this invocation bound. Unlike the Codex
    #: authorization, this one *is* recorded: it does not change what the
    #: study is designed to measure, but it changes what every number a
    #: stage produces is evidence of, so the stage writes it into the
    #: manifest and the cross-stage check reads it back.
    transport: str = TransportName.FAKE.value
    #: The study's evidence store, when the caller bound one. A stage that
    #: evaluates through the engine prices what it evaluated by reading its
    #: own persisted output rows back out of this store; without it the
    #: stage records no spend rather than guessing at one.
    store: ObjectStore | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """What a stage recorded, plus the manifest it wrote.

    ``stage0`` is present only for Stage 0 and ``arms`` only for Stages 1
    and 2, because a stage that measured neither should not be able to
    report an empty one as a result.
    """

    stage: StageId
    manifest: StudyManifest
    stage0: Stage0Result | None = None
    arms: tuple[ArmReport, ...] = ()
    #: What the post-measurement analysis wrote, when one ran. ``None`` for
    #: Stage 0, and for an arm stage whose environment carries no anchors.
    analysis: AnalysisResult | None = None


# --------------------------------------------------------------------------
# Transport: recorded per stage, checked across them
# --------------------------------------------------------------------------

#: The stage whose transport every later stage must match. Stage 0 buys the
#: anchors, and every efficacy number in the study is a delta against them,
#: so it is Stage 0's transport that decides what the whole study measured.
STAGE0_TRANSPORT_STAGE = StageId.STAGE0


def require_matching_transport(
    manifest: StudyManifest,
    *,
    stage: StageId | ManifestStageId,
    transport: str,
) -> None:
    """Refuse a stage whose evidence was not all measured where it runs.

    Every efficacy number is a paired delta against the Stage-0 anchors, so
    anchors measured on the fake transport and arms measured against a
    provider are not two halves of one comparison -- they are two different
    experiments subtracted from each other. There is no flag that makes it
    acceptable, because the resulting number would not mean anything a flag
    could qualify: a toy study that wants the other transport re-runs Stage
    0 on it, which is cheap precisely because it is a toy.

    **Three things are checked, because a study can hold evidence from two
    transports in three ways.** Checking only the first left the other two
    able to reach a paid arg-max over free runs:

    1. **The anchors.** Stage 0's transport, which every held-out delta is
       paired against.
    2. **The target stage's own record.** A stage that already ran on the
       other transport still holds the runs that invocation produced, so
       resuming it here would select across two transports within one
       stage.
    3. **Every surviving arm run.** A run's transport is its own evidence
       rather than its stage's -- a resumed stage keeps runs an earlier
       invocation paid for -- so a stage row can agree while the runs
       beneath it do not. This is the check that reads the evidence.

    Stage 0 is exempt from the check against itself; ``--replace-design``
    is the existing, recorded way to re-calibrate, and onto a different
    transport it drops the stale arm evidence rather than leaving it for
    this guard to trip over.

    Stages are compared by value rather than by identity. Two ``StageId``
    enums with the same members exist in this package -- the spec's and the
    manifest's -- and an identity test would silently pass whichever one it
    was not given, turning a guard into a no-op for half its callers.
    """
    if stage.value == STAGE0_TRANSPORT_STAGE.value:
        return
    anchored = recorded_transport(manifest.stages, STAGE0_TRANSPORT_STAGE)
    if anchored is not None and anchored != transport:
        raise StageError(
            f"{stage.value} was asked to run on transport {transport!r}, "
            f"but this study calibrated its anchors on {anchored!r}; every "
            "held-out delta is paired against those anchors, so the two "
            "cannot be compared. Re-run stage0 on "
            f"{transport!r} with --replace-design, or run {stage.value} on "
            f"{anchored!r}"
        )
    own = recorded_transport(manifest.stages, stage.value)
    if own is not None and own != transport:
        raise StageError(
            f"{stage.value} was asked to run on transport {transport!r}, "
            f"but it already ran on {own!r} and that record still stands. "
            "A resumed stage keeps the runs its earlier invocation "
            "produced, so continuing here would select across two "
            "transports within one stage. Re-run stage0 on "
            f"{transport!r} with --replace-design, which drops this "
            f"stage's records, or run {stage.value} on {own!r}"
        )
    stale = _runs_on_other_transports(manifest, transport=transport)
    if stale:
        names = sorted(stale)
        shown = names[:_STALE_RUNS_SHOWN]
        more = (
            ""
            if len(names) <= _STALE_RUNS_SHOWN
            else f" (+{len(names) - _STALE_RUNS_SHOWN} more)"
        )
        raise StageError(
            f"{stage.value} was asked to run on transport {transport!r}, "
            f"but this study still holds runs measured on another "
            f"transport: {shown}{more}. Those runs are selected over "
            "alongside this stage's, so the arg-max would compare evidence "
            "from two experiments. Re-run stage0 on "
            f"{transport!r} with --replace-design, which drops them, or "
            "run this stage on the transport they were measured on"
        )


#: How many stale run ids a refusal names before summarising the rest. The
#: point is to make the refusal actionable without printing every run of a
#: five-run design across four arms.
_STALE_RUNS_SHOWN = 5


def _runs_on_other_transports(
    manifest: StudyManifest, *, transport: str
) -> set[str]:
    """Every recorded arm run measured on some transport but this one.

    A run's transport is its own evidence: the stage row says what the
    latest invocation of that stage bound, and the runs beneath it may
    predate it. This is the check that reads the evidence rather than the
    summary.
    """
    return {
        run.run_id
        for arm in manifest.arms
        for run in arm.runs
        if run.transport != transport
    }


def _stages_with(
    manifest: StudyManifest, record: StageRecord
) -> tuple[StageRecord, ...]:
    """``manifest.stages`` with ``record`` replacing any same-stage entry.

    A re-run replaces rather than appends: the manifest holds at most one
    record per stage, because two would leave the study unable to say which
    transport its numbers came from.
    """
    others = tuple(
        entry for entry in manifest.stages if entry.stage != record.stage
    )
    return (*others, record)


def _arm_stage_record(
    manifest: StudyManifest, record: StageRecord
) -> tuple[StageRecord, ...]:
    """``manifest.stages`` with an arm stage's record merged in, not over.

    An arm stage's spend is carried by the runs *this invocation*
    executed, which is what keeps a run Stage 1 paid for from being billed
    again on Stage 2's row. That accounting has one gap, and it is the
    expensive one: a stage that crashed after its manifest write has
    already paid for every run and already recorded what they cost, so
    resuming it executes nothing, projects to an empty spend, and -- under
    a plain replacement -- overwrote a measured bill with silence. The
    ledger then rendered a fully paid stage as UNLEDGERED, which is the
    one claim about spend a study must never make falsely.

    So the two are summed rather than swapped. The existing row is what an
    earlier invocation paid, the new record is what this one paid, and
    :func:`~whetstone_envs.optim.study.spend.run_spend_records` folds them
    per role while re-applying its own honesty rules -- most importantly,
    an unknown ``usd`` on either side keeps the total unknown rather than
    letting the priced half stand in for the whole.

    A stage row never shrinks: this is the only writer of an arm stage's
    spend, and it can only add. The transport comes from the new record,
    because it is a property of the invocation that just ran and
    ``require_matching_transport`` has already refused a stage whose
    transport disagrees with the study's.

    Only the *run* side is summed here. The reporting pass accumulates by
    the opposite rule -- it is folded whole from durable records every
    time, so adding it would bill one evaluation once per resume -- and
    :func:`_record_report_spend` owns it. It is carried through unchanged
    rather than dropped, because this writer runs before that one and must
    not erase a total an earlier invocation already recorded.
    """
    existing = next(
        (entry for entry in manifest.stages if entry.stage == record.stage),
        None,
    )
    if existing is None:
        return _stages_with(manifest, record)
    merged = record.model_copy(
        update={
            "spend": (
                run_spend_records((*existing.spend, *record.spend))
                if existing.spend
                else record.spend
            ),
            "report_spend": existing.report_spend,
        }
    )
    return _stages_with(manifest, merged)


def _stage_record(
    *,
    stage: StageId,
    environment: StageEnvironment,
    evidence: Iterable[EvalEvidence] = (),
    run_spend: Iterable[RunSpendRecord] = (),
) -> StageRecord:
    """One stage's record: the transport it ran on, and what it spent.

    The two kinds of stage measure their spend by different routes,
    because they spend by different routes:

    * **Stage 0 evaluates through the engine**, so its bill is re-derived
      from the persisted output rows its anchor evaluations left behind --
      that is what ``evidence`` carries.
    * **An arm stage spends through optimizer runs**, and each run already
      re-derived its own per-role bill. Its stage total is the fold of
      those records, which ``run_spend`` carries. Re-reading the rows here
      would risk counting a call twice and would let the stage row and the
      run rows disagree.

    Two conditions still gate the projection, and both mean "not measured"
    rather than "free":

    * **A fake-transport stage records none.** Its rows are real rows -- a
      generation happened and the row proves it -- so the shared row rule
      counts them as billable-and-unpriced, which is the right answer for a
      provider row and the wrong one for a stage that reached no provider.
      Reporting "112 unpriced calls" for a stage that spent nothing would
      be a bill nobody owes.
    * **A caller that bound no store records none** on the evidence route,
      because the rows are read back out of it and there is nothing to
      read. The run route needs no store: the runs already hold the
      records.

    A paid stage that projects to nothing is *not* silently equated with a
    fake one. Both record an empty tuple here, and the renderers
    distinguish them from the transport -- see
    :func:`~whetstone_envs.optim.study.cli.stage_spend_lines`, which labels
    a paid stage with no records ``UNLEDGERED`` rather than reporting that
    it reached no provider.

    In every case the record still carries the transport, which is the
    fact the cross-stage check and the renderers both need.
    """
    paid = environment.transport != TransportName.FAKE.value
    spend: tuple[RunSpendRecord, ...] = ()
    if paid:
        # The two routes are exclusive by construction -- Stage 0 passes
        # evidence and no runs, an arm stage passes runs and no evidence --
        # and the fold is preferred where both somehow arrive, because a
        # run's own cost report is the narrower, already-attributed claim.
        # Written as an explicit branch rather than an ``or`` chain so that
        # neither route silently stands in for the other when it yields
        # nothing: a paid stage that projects to no records must reach the
        # renderers as UNLEDGERED, not as a stage measured by the route it
        # does not use.
        folded = run_spend_records(run_spend)
        if folded:
            spend = folded
        elif environment.store is not None:
            spend = stage_spend_records(
                store=environment.store, evidence=evidence
            )
    return StageRecord(
        stage=stage.value,
        transport=environment.transport,
        spend=spend,
    )


# --------------------------------------------------------------------------
# Stage 0
# --------------------------------------------------------------------------


def run_stage0_into_manifest(
    *,
    study_dir: Path,
    environment: StageEnvironment,
    replace_design: bool = False,
) -> StageResult:
    """Calibrate the anchors, evaluate the gate, record the design.

    The gate's verdict is recorded whether or not it passed. A failed gate
    is a finding the study reports -- an underpowered design, stated as
    such -- not an error that erases the calibration it just paid for. What
    a failed gate does stop is the *next* stage, which
    :func:`run_arm_stage` refuses to start without a recorded design.

    **Stage 0 pins the pre-registration, and a second Stage 0 refuses.**
    The first run writes the frozen design block; a later one would restate
    the study's power arithmetic after its own results existed, so it is
    refused unless the caller passes ``replace_design``, which records the
    replacement as an ``amended`` pre-registration naming the design hash it
    replaced. Re-calibrating and keeping the same design is *not* an
    amendment: an identical block is written back unchanged and the
    manifest's own immutability check passes it.
    """
    manifest = read_study_manifest(study_dir)
    pinned = manifest.pre_registration
    if pinned is not None and not replace_design:
        raise StageError(
            "this study already pre-registered its design at "
            f"{pinned.design_hash[:12]}; a second stage0 would restate the "
            "power arithmetic after results exist. Pass --replace-design to "
            "record a deliberate amendment"
        )
    # Stage 0 records the *full* design, not the pilot's: ``k_run_by_arm``
    # is what the study pre-registered for Stage 2, and Stage 1 spends a
    # prefix of it rather than a different design.
    spec = spec_from_manifest(manifest, stage=StageId.STAGE2)
    if not spec.arms:
        # The design's ``k_run_by_arm`` is a pre-registration, so a study
        # that has not declared its arms cannot record one. Refusing here
        # beats writing a placeholder that would read as a design.
        raise StageError(
            "stage0 records the pre-registered design, which names every "
            "arm's K_RUN; declare the study's arms before calibrating"
        )
    # Before the calibration spends, not after. A re-calibration onto a
    # different transport invalidates the arm stages, and whether that is
    # allowed at all depends on what they were measured on -- so the
    # refusal is settled here rather than after Stage 0 has paid.
    amendment = _transport_change_amendment(
        manifest, environment=environment, replace_design=replace_design
    )
    result = run_stage0(
        spec=spec,
        bind_engine=environment.bind_engine,
        naive_candidate=environment.naive_candidate,
        ceiling_candidate=environment.ceiling_candidate,
        task_ids_by_role=environment.task_ids_by_role,
        pool_ceiling=environment.pool_ceiling,
    )
    design = _design_record(spec, result)
    pre_registration = _pre_registration_record(
        design, spec=spec, replaced=pinned
    )
    # The stale evidence is dropped *before* the new records are written,
    # so every ``_stages_with`` and every arm list below is built over what
    # survives rather than over what the amendment just invalidated.
    base = (
        manifest
        if amendment is None
        else _without_amended_evidence(manifest, amendment=amendment)
    )
    updated = base.model_copy(
        update={
            "design": design,
            "pre_registration": pre_registration,
            # What this calibration ran on, and what it cost. Written in
            # the same update as the design, because a design recorded
            # without the transport that measured it is exactly the
            # ambiguity the stage block exists to remove.
            "stages": _stages_with(
                base,
                _stage_record(
                    stage=StageId.STAGE0,
                    environment=environment,
                    evidence=result.evidence,
                ),
            ),
            # Stage 0 is where each role's Eval Config first exists, so it
            # is where the manifest learns it. Without this the manifest
            # carries whatever placeholder it was created with, and L1 --
            # which compares each optimizer evaluation's resolved config
            # against the recorded internal one -- can only ever fail.
            "splits": _splits_with_measured_configs(
                base.splits,
                result=result,
                bind_engine=environment.bind_engine,
                k_repeat=spec.k_repeat,
            ),
        }
    )
    # An amendment is only an amendment when it actually changes the pinned
    # block. A re-calibration that lands on the same design writes the
    # original block back untouched, so it goes through the ordinary
    # immutability path rather than being recorded as a design change that
    # did not happen.
    amending = (
        pinned is not None
        and pre_registration.provenance == PROVENANCE_AMENDED
    )
    if amending:
        # The pilot's call-count verdict was computed against the design it
        # replaced. Carrying it across an amendment would let Stage 2 spend
        # the full design on a gate that never saw this design's arms, run
        # counts, or splits, so the amended study owes a fresh Stage 1.
        updated = updated.model_copy(update={"call_count_gate": None})
    write_study_manifest(
        study_dir,
        updated,
        replace=True,
        amend_pre_registration=amending,
    )
    return StageResult(
        stage=StageId.STAGE0,
        manifest=read_study_manifest(study_dir),
        stage0=result,
    )


#: The stages a cross-transport re-calibration invalidates. Stage 0 is not
#: among them: it is the stage being re-run.
AMENDED_STAGES: tuple[StageId, ...] = (StageId.STAGE1, StageId.STAGE2)


def _transport_change_amendment(
    manifest: StudyManifest,
    *,
    environment: StageEnvironment,
    replace_design: bool,
) -> AmendmentRecord | None:
    """What a cross-transport ``--replace-design`` must drop, or ``None``.

    A Stage 0 re-run onto the transport it already used changes nothing
    about what the arm stages measured, so it drops nothing. A re-run onto
    the *other* transport invalidates them twice over -- the design they
    ran against is being replaced, and their evidence was measured
    somewhere else -- and leaving them in place is what let a Stage 2 on a
    paid transport reuse fake runs against freshly bought anchors.

    **Paid evidence is never discarded automatically.** Fake runs cost
    nothing and re-running them is the cheap, obvious recovery. A paid run
    is money already spent, and a command whose stated purpose is
    re-calibrating Stage 0 must not delete it as a side effect, so this
    refuses instead and names the recovery. The refusal is computed before
    the calibration spends, so a study that cannot proceed does not pay to
    find out.

    Returns ``None`` when there is nothing to drop, which covers the
    ordinary same-transport amendment and every first Stage 0.
    """
    if not replace_design:
        return None
    previous = recorded_transport(manifest.stages, StageId.STAGE0)
    if previous is None or previous == environment.transport:
        return None
    dropped_stages = tuple(
        entry.stage
        for entry in manifest.stages
        if entry.stage in {stage.value for stage in AMENDED_STAGES}
    )
    dropped_runs = tuple(run for arm in manifest.arms for run in arm.runs)
    paid = sorted(
        run.run_id
        for run in dropped_runs
        if run.transport != TransportName.FAKE.value
    )
    if paid:
        raise StageError(
            "stage0 --replace-design would move this study from transport "
            f"{previous!r} to {environment.transport!r}, which invalidates "
            f"its arm stages -- but {len(paid)} of their runs were measured "
            f"on a paid transport: {paid[:_STALE_RUNS_SHOWN]}. Paid "
            "evidence is never discarded automatically. Archive this study "
            "directory and calibrate the new transport in a fresh one, or "
            "remove those runs from the manifest deliberately if you have "
            "decided they are worthless"
        )
    if not (
        dropped_stages
        or dropped_runs
        or manifest.selection
        or manifest.held_out_claims
        or manifest.held_out
        or manifest.call_count_gate is not None
    ):
        # The transport changed but nothing downstream of Stage 0 exists
        # yet, so there is nothing to record as dropped. An amendment
        # naming no casualties would be noise in the report.
        return None
    dropped = set(dropped_stages)
    dropped_run_ids = {run.run_id for run in dropped_runs}
    return AmendmentRecord(
        at=datetime.now(UTC).isoformat(),
        amended_stage=StageId.STAGE0.value,
        reason=AMENDMENT_REASON_TRANSPORT_CHANGE,
        from_transport=previous,
        to_transport=environment.transport,
        dropped_stages=dropped_stages,
        dropped_run_ids=tuple(run.run_id for run in dropped_runs),
        # Where those runs still are. The manifest drops them; the disk
        # keeps them, and the next stage to compute one of these names
        # will refuse to reuse what it finds there.
        dropped_run_directories=tuple(
            dict.fromkeys(run.artifact_dir for run in dropped_runs)
        ),
        dropped_selections=len(manifest.selection),
        dropped_held_out_claims=len(manifest.held_out_claims),
        dropped_held_out_rows=len(manifest.held_out),
        dropped_call_count_gate=manifest.call_count_gate is not None,
        # Measured on the transport being left behind, and keyed by names
        # the replacement stage recomputes: counted here so the record
        # says what the study lost, and dropped below so nothing reads it.
        dropped_official_scores=sum(
            1
            for entry in manifest.official_scores
            if entry.run_id in dropped_run_ids or entry.stage in dropped
        ),
        dropped_report_spend=sum(
            1 for entry in manifest.report_spend if entry.stage in dropped
        ),
    )


def _without_amended_evidence(
    manifest: StudyManifest, *, amendment: AmendmentRecord
) -> StudyManifest:
    """The manifest with the amendment's casualties removed and recorded.

    The arms themselves survive with empty run lists: an arm is part of the
    design the study is re-pre-registering, and deleting it would change
    the design rather than clear the evidence for it. What goes is
    everything measured -- the arm stages' records, their runs, the
    selections over those runs, the held-out claims and rows those
    selections produced, and the pilot's call-count verdict.

    **What those runs and stages bought goes with them.** Run ids are
    deterministic, so the replacement stage recomputes the very names this
    drops; an ``official_scores`` entry left behind would be read back by
    :meth:`~whetstone_envs.optim.study.selection.ManifestSelectionLog.official_score_for`
    and reused, presenting a score measured on the previous transport as
    this study's selection evidence -- and never re-buying it on the
    transport the study now runs on. ``report_spend`` is the same shape of
    error in money: the stage's reporting row is folded from those durable
    per-evaluation records rather than from the row, so entries surviving
    their stage are folded by the *next* invocation of it, billing a paid
    stage for evaluations a fake-transport invocation bought. Both are
    counted on the amendment before they go.

    **A verdict computed over dropped evidence goes with it.**
    ``leakage_check`` is L6's mechanical pass over the very run artifacts
    being dropped, so keeping it would leave the manifest asserting a
    clean result about runs the study no longer holds --
    indistinguishable, to a regenerated report, from a study whose
    leakage rules passed over its current runs.
    :func:`~whetstone_envs.reporting.study_report.study_leakage_failed`
    reads an absent block as not-established, so clearing it is what makes
    the report's claim honest rather than merely unstated.

    The other verdicts are deliberately left alone, because they are not
    measurements of these runs: ``gepa_sizing`` and ``fanout_check`` are
    pre-Stage-1 measurements of the optimizer's own mechanics, ``balance``
    is the key's balance at each spend gate rather than a claim about any
    run, and ``c18`` carries its own separate run list which is not among
    the dropped ones.

    The amendment is appended in the same operation, so there is no state
    in which the evidence is gone and the record of its going is not yet
    written.
    """
    dropped = set(amendment.dropped_stages)
    dropped_runs = set(amendment.dropped_run_ids)
    return manifest.model_copy(
        update={
            "amendments": (*manifest.amendments, amendment),
            "stages": tuple(
                entry
                for entry in manifest.stages
                if entry.stage not in dropped
            ),
            "official_scores": tuple(
                entry
                for entry in manifest.official_scores
                if entry.run_id not in dropped_runs
                and entry.stage not in dropped
            ),
            "report_spend": tuple(
                entry
                for entry in manifest.report_spend
                if entry.stage not in dropped
            ),
            "arms": tuple(
                arm.model_copy(update={"runs": ()}) for arm in manifest.arms
            ),
            "selection": (),
            "held_out_claims": (),
            "held_out": (),
            "call_count_gate": None,
            "leakage_check": None,
        }
    )


def _splits_with_measured_configs(
    splits: SplitsRecord,
    *,
    result: Stage0Result,
    bind_engine: EngineBinder,
    k_repeat: int,
) -> SplitsRecord:
    """Record the Eval Config each role is *reported* under.

    The task hashes are left alone: those are the population's identity and
    the bind already refused a run whose regenerated tasks disagreed with
    them. What changes is the config hash, which cannot be known before an
    engine is bound and is therefore the one field a pre-Stage-0 manifest
    can only guess at.

    **The repeat count is part of an Eval Config's identity, so the config
    recorded is the design's, not the calibration's.** Stage 0 calibrates at
    ``K_CAL`` and every later evaluation runs at ``K_REPEAT``; recording the
    calibration's config would make L1 compare each optimizer evaluation
    against a config nothing but Stage 0 ever used, and the rule would fail
    on every clean study. Binding the engine again at ``K_REPEAT`` costs
    nothing -- no evaluation is issued -- and yields the config the runs
    actually resolve.
    """
    del result
    by_role = {
        role: bind_engine(
            role=role, num_seeds=k_repeat
        ).eval_config_ref.config_hash
        for role in (
            EvalRole.INTERNAL,
            EvalRole.OFFICIAL,
            EvalRole.HELD_OUT,
        )
    }
    return SplitsRecord(
        internal=splits.internal.model_copy(
            update={"eval_config_hash": str(by_role[EvalRole.INTERNAL])}
        ),
        official=splits.official.model_copy(
            update={"eval_config_hash": str(by_role[EvalRole.OFFICIAL])}
        ),
        held_out=splits.held_out.model_copy(
            update={"eval_config_hash": str(by_role[EvalRole.HELD_OUT])}
        ),
    )


def _split_by_arm(spec: StudySpec) -> dict[str, tuple[int, int] | None]:
    """Each arm's pre-registered train/val partition, or ``None``.

    ``ArmSpec`` already refuses a split on an arm whose optimizer has no
    train/val concept and requires both halves on one that does, so this is
    a projection rather than a second rule.
    """
    return {
        arm.arm_id: (
            None
            if arm.train_size is None or arm.val_size is None
            else (arm.train_size, arm.val_size)
        )
        for arm in spec.arms
    }


def _minibatch_by_arm(spec: StudySpec) -> dict[str, int | None]:
    """Each arm's pre-registered MIPROv2 minibatch size, or ``None``.

    ``ArmSpec`` already refuses a size on an arm that does not minibatch
    and requires one on an arm that does, so this is a projection rather
    than a second rule.
    """
    return {arm.arm_id: arm.miprov2_minibatch_size for arm in spec.arms}


def _search_by_arm(spec: StudySpec) -> dict[str, dict[str, int]]:
    """Each arm's pre-registered search shape, by field name.

    Only the fields that arm's optimizer actually reads appear, so an arm
    with no search shape pins an empty mapping rather than a row of nulls.
    ``ArmSpec`` already refuses a shape on an optimizer that reads none and
    requires COPRO's on the arms whose search is COPRO's, so this is a
    projection rather than a second rule.
    """
    shapes: dict[str, dict[str, int]] = {}
    for arm in spec.arms:
        shape: dict[str, int] = {}
        if arm.copro_breadth is not None:
            shape["breadth"] = arm.copro_breadth
        if arm.copro_depth is not None:
            shape["depth"] = arm.copro_depth
        if arm.miprov2_num_trials is not None:
            shape["num_trials"] = arm.miprov2_num_trials
        if arm.miprov2_num_candidates is not None:
            shape["num_candidates"] = arm.miprov2_num_candidates
        shapes[arm.arm_id] = shape
    return shapes


def _pre_registration_record(
    design: DesignRecord,
    *,
    spec: StudySpec,
    replaced: PreRegistrationRecord | None,
) -> PreRegistrationRecord:
    """The frozen block for ``design``, amending ``replaced`` if it differs.

    The hash is computed from the design's own values rather than copied,
    so a design and its pinning cannot disagree at the moment they are
    written.

    ``spec`` supplies the per-arm train/val partition and MIPROv2
    minibatch size, which the design block does not carry: both are per
    arm and ``DesignRecord`` records study-wide numbers. They are pinned
    all the same, because an arm rerun at a different split -- or at a
    different batch size -- is measuring a different thing.
    """
    split_by_arm = _split_by_arm(spec)
    minibatch_by_arm = _minibatch_by_arm(spec)
    search_by_arm = _search_by_arm(spec)
    design_hash = pre_registration_design_hash(
        k_repeat=design.k_repeat,
        k_run_by_arm=dict(design.k_run_by_arm),
        split_by_arm=split_by_arm,
        minibatch_by_arm=minibatch_by_arm,
        search_by_arm=search_by_arm,
        ci_level=design.ci_level,
        resamples=design.resamples,
        bootstrap_seed=design.bootstrap_seed,
        correction=design.correction,
        m=design.m,
        completeness_backstop=design.completeness_backstop,
    )
    if replaced is not None and replaced.design_hash == design_hash:
        # Byte-identical to what is already pinned: keep the original
        # provenance rather than relabelling an unchanged design as amended.
        return replaced
    amended = replaced is not None
    return PreRegistrationRecord(
        k_repeat=design.k_repeat,
        k_run_by_arm=dict(design.k_run_by_arm),
        split_by_arm=split_by_arm,
        minibatch_by_arm=minibatch_by_arm,
        search_by_arm=search_by_arm,
        ci_level=design.ci_level,
        resamples=design.resamples,
        bootstrap_seed=design.bootstrap_seed,
        correction=design.correction,
        m=design.m,
        completeness_backstop=design.completeness_backstop,
        design_hash=design_hash,
        provenance=(PROVENANCE_AMENDED if amended else PROVENANCE_ORIGINAL),
        amended_from=replaced.design_hash if replaced is not None else None,
    )


def _design_record(spec: StudySpec, result: Stage0Result) -> DesignRecord:
    """The measured design, as Stage 0 leaves it for every later stage.

    ``k_run_by_arm`` covers every declared arm even before any of them has
    run, because the design is what was pre-registered, not what has
    happened so far.
    """
    return DesignRecord(
        k_cal=result.k_cal,
        k_repeat=spec.k_repeat,
        k_run_by_arm=dict(spec.k_run_by_arm),
        ci_level=spec.ci_level,
        resamples=spec.resamples,
        bootstrap_seed=spec.bootstrap_seed,
        correction=CORRECTION_HOLM_BONFERRONI,
        m=CORRECTION_FAMILY_SIZE,
        mde_formula=MDE_FORMULA,
        mde_measured=result.gate.mde_measured,
        tau_sq=result.inputs.tau_sq,
        sigma_sq=result.inputs.sigma_sq,
        completeness_rule=COMPLETENESS_RULE,
        completeness_backstop=COMPLETENESS_BACKSTOP,
    )


# --------------------------------------------------------------------------
# Stages 1 and 2
# --------------------------------------------------------------------------


def _refuse_unauthorized_codex_arm(
    *,
    spec: StudySpec,
    stage: StageId,
    study_dir: Path,
    environment: StageEnvironment,
) -> None:
    """Refuse a Codex-bearing stage before any arm runs, or not at all.

    The refusal that matters is the *early* one. ``run_optimizer`` already
    declines an unauthorized real Codex run, but it declines it when the
    Codex arm's own turn arrives -- after this stage has paid for every arm
    ordered ahead of it. That is a real bill for a stage that was never
    going to finish, so the same authorization is checked here, against the
    design, while the stage has spent nothing.

    Both halves of the opt-in are required, exactly as
    :func:`~whetstone_envs.optim.codex.refuse_unauthorized_real_codex`
    requires them: the command's ``--allow-real-codex`` and the opt-in
    environment variable. Checking both here means this guard cannot report
    a stage as authorized that the runner would then refuse.

    **Authorization is not usability.** A stage that clears the opt-in can
    still have no Codex to run: an unsupported platform, a binary that is
    not on the run's PATH, an expired or absent session. Those were
    discovered when the adapter was built, on the Codex arm's turn -- again
    after the other arms were paid for -- so the same preflight
    ``run_optimizer`` reaches is run here instead, while the stage has
    spent nothing on the arms ahead of it. The probe is itself a billed
    session probe, which is why it runs strictly *after* the opt-in check
    and never before it.

    Imported inside the function rather than at module scope: this module
    takes its provider-touching collaborators as callables and does not
    import the optimizer stack, and the constants it needs live beside the
    guard that enforces them.
    """
    from whetstone_envs.optim.codex import (  # noqa: PLC0415
        ALLOW_REAL_CODEX_ENV,
        ALLOW_REAL_CODEX_ENV_VALUE,
        CodexTestSeam,
        RealCodexRefusedError,
        preflight_codex_session,
        resolve_codex_agent_model,
    )

    codex_arms = tuple(
        arm.arm_id for arm in spec.arms if arm.optimizer == CODEX_ARM_ID
    )
    if not codex_arms:
        return
    authorized = (
        environment.real_codex_authorized
        and os.environ.get(ALLOW_REAL_CODEX_ENV) == ALLOW_REAL_CODEX_ENV_VALUE
    )
    if not authorized:
        raise RealCodexRefusedError(
            f"{stage.value} declares the Codex arm {list(codex_arms)}, whose "
            "runs spawn a real, billed Codex session, and this invocation is "
            "not authorized to spend on one. Refusing before any arm runs, "
            "so the stage buys nothing it cannot finish. Authorize it with "
            f"{ALLOW_REAL_CODEX_ENV}={ALLOW_REAL_CODEX_ENV_VALUE} in the "
            "environment and --allow-real-codex on the run command."
        )
    seam = environment.codex_test_seam
    if seam is not None and not isinstance(seam, CodexTestSeam):
        # ``StageEnvironment`` types this loosely to keep the optimizer
        # stack out of the module, so the one place that has the real type
        # is the one place that checks it. A wrong object here would
        # otherwise reach the preflight and be read as "no seam", which is
        # the production path.
        raise StageError(
            "codex_test_seam must be a CodexTestSeam; got "
            f"{type(seam).__name__}"
        )
    agent_model = resolve_codex_agent_model(None)
    # Before the probe, because a session opened on the wrong agent is a
    # billed session the design never registered. The pin is design and the
    # resolution is the runner's, so comparing them here is what keeps the
    # arm's proposer the one the manifest names.
    require_pinned_codex_agent_model(spec, resolved=agent_model)
    try:
        preflight_codex_session(
            scratch_root=study_dir / STAGE_PREFLIGHT_ROOT_NAME,
            # The agent model, resolved through the same helper the
            # runner uses, so the probe clears the route the arm will
            # actually open. A study arm does not override the agent
            # model, so this is the arm's default -- but it is resolved
            # rather than assumed, which is what keeps the two in step if
            # an arm ever gains an override.
            #
            # Probing ``spec.task_model`` here tested an OpenRouter route
            # the Codex CLI cannot run at all, so the guard cleared a
            # session no arm would ever open and the real failure waited
            # for the Codex arm's turn -- after COPRO, MIPROv2, and GEPA
            # had been paid for, which is precisely what this preflight
            # exists to prevent.
            model=agent_model,
            allow_real_codex=environment.real_codex_authorized,
            test_seam=seam,
        )
    except RealCodexRefusedError:
        # The gate's own refusal -- an unauthorized call, or the test
        # tripwire -- is already the right error with the right message,
        # so it travels as itself rather than being reworded as a stage
        # failure.
        raise
    except Exception as error:
        # Re-raised as a stage refusal so the caller sees which stage
        # declined and why, with the preflight's own message preserved as
        # the cause rather than replaced by it.
        raise StageError(
            f"{stage.value} declares the Codex arm {list(codex_arms)}, and "
            "this machine cannot run a Codex session: "
            f"{error}. Refusing before any arm runs, so the stage does not "
            "pay for COPRO, MIPROv2, and GEPA and then discover it can "
            "never finish"
        ) from error


def run_arm_stage(
    *, study_dir: Path, stage: StageId, environment: StageEnvironment
) -> StageResult:
    """Run every arm at ``stage``, then select and report each one.

    The order within an arm is the one :func:`report_arm` enforces and this
    function cannot reorder: every run is scored on official, the arg-max is
    persisted into ``study.json``, the persisted record is read back, and
    only then does one held-out evaluation happen. The ledger is the
    manifest itself, so a crash between selecting and measuring leaves the
    selection durable rather than lost.
    """
    if stage is StageId.STAGE0:
        raise StageError("stage0 runs no optimizers; use run_stage0")
    for name, value in (
        ("an optimizer runner", environment.run_optimizer),
        ("an official scorer", environment.score_official),
        ("a held-out evaluator", environment.evaluate_held_out),
    ):
        if value is None:
            raise StageError(f"{stage.value} needs {name}")
    manifest = read_study_manifest(study_dir)
    if manifest.design is None:
        # Stage 0 writes the design; without it there is no measured MDE to
        # judge a result against and no recorded K_REPEAT to run at.
        raise StageError(
            f"{stage.value} requires a recorded design; run stage0 first"
        )
    # Before the arms run, not after: an arm stage on the wrong transport
    # would spend a full pilot or full design before anyone could see that
    # its deltas are paired against anchors from a different experiment.
    require_matching_transport(
        manifest, stage=stage, transport=environment.transport
    )
    # The design's ``k_run_by_arm`` is the *full* pre-registration, so it
    # says how many runs Stage 2 gets, not how many this stage does. Stage 1
    # spends a prefix of the same seeds, which is what makes "Stage 1's runs
    # count toward Stage 2" checkable rather than asserted.
    spec = spec_from_manifest(manifest, stage=stage)
    if not spec.arms:
        raise StageError(f"{stage.value} has no arms to run")
    # An arm declared after the design was pinned would spend on a design
    # nobody registered. Stage 0 tolerates that state -- it is how an
    # amendment is written -- so the check belongs here, not in the loader.
    require_pinned_arms(manifest)
    _refuse_projection_claiming_the_design(manifest=manifest, stage=stage)
    # Both refusals are before dispatch and cost nothing, so the order is
    # about which one a reader should see first. An unauthorized Codex arm
    # is a property of the invocation the operator can fix now; a missing
    # pilot is a property of the study that takes a whole stage to fix.
    _refuse_unauthorized_codex_arm(
        spec=spec,
        stage=stage,
        study_dir=study_dir,
        environment=environment,
    )
    _require_passed_stage1_gate(manifest=manifest, stage=stage)

    arm_records, run_results, executed_runs = _run_every_arm(
        spec=spec,
        stage=stage,
        study_dir=study_dir,
        environment=environment,
        recorded=manifest.arms,
    )
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "arms": arm_records,
                # The stage's transport and its spend, recorded as soon as
                # its arms have run. An arm stage spends through optimizer
                # runs, so the stage total is the fold of the per-run
                # records rather than a second measurement of the same
                # calls -- but it is recorded, because a stage row without
                # it reads as a stage that reached no provider.
                #
                # Only the runs this invocation executed contribute. A
                # run Stage 1 paid for is already billed on Stage 1's row,
                # and Stage 2 selects over it without re-buying it, so
                # counting it again would make the ledger's rows sum to
                # more than the study spent.
                #
                # Merged onto any spend this stage's row already carries
                # rather than replacing it: a resume of a stage that
                # crashed after its manifest write executes nothing, and a
                # plain replacement would discard the bill it already
                # paid. See :func:`_arm_stage_record`.
                "stages": _arm_stage_record(
                    manifest,
                    _stage_record(
                        stage=stage,
                        environment=environment,
                        run_spend=_executed_run_spend(executed_runs),
                    ),
                ),
            }
        ),
        replace=True,
    )

    _check_call_counts(
        study_dir=study_dir,
        spec=spec,
        stage=stage,
        run_results=run_results,
        design=manifest.design,
    )

    # One ledger per stage: the pilot's selection and the full design's are
    # each made once, over different run sets, and neither can be made twice.
    log = ManifestSelectionLog(
        study_dir, stage=stage.value, transport=environment.transport
    )
    # Every reporting evaluation from here on writes its own spend before
    # the pass returns, so a crash mid-pass leaves the bill for what was
    # already bought on disk rather than in a process that is gone.
    _persist_report_spend_to(
        study_dir=study_dir, stage=stage, environment=environment
    )
    reports = tuple(
        _report_or_rebuild_arm(
            arm_id=arm.arm_id,
            runs=tuple(result.candidate for result in run_results[arm.arm_id]),
            score_official=_require(environment.score_official),
            evaluate_held_out=_require(environment.evaluate_held_out),
            log=log,
            stage=stage,
        )
        for arm in spec.arms
    )
    # The anchors and the statistics are a second pass on purpose: a
    # Holm-corrected p-value is a whole-study computation that cannot exist
    # until every arm has been measured, so the rows are written after the
    # last held-out evaluation rather than beside each one.
    analysis = _analyse_stage(
        study_dir=study_dir,
        stage=stage,
        environment=environment,
        reports=reports,
        k_repeat=manifest.design.k_repeat,
        log=log,
    )
    # The reporting pass's own bill, folded in only now. It cannot be
    # written with the arms' row above: official-selection scoring, the
    # held-out evaluations, and the anchors' re-measurement all happen
    # after that write, and every one of them reaches the provider. A
    # stage row that stopped at the run-side total would understate the
    # study by the whole pass its efficacy claims are made against.
    _record_report_spend(
        study_dir=study_dir, stage=stage, environment=environment
    )
    return StageResult(
        stage=stage,
        manifest=read_study_manifest(study_dir),
        arms=reports,
        analysis=analysis,
    )


def _persist_report_spend_to(
    *, study_dir: Path, stage: StageId, environment: StageEnvironment
) -> None:
    """Point this stage's ledger at the manifest, for the pass ahead.

    Each priced evaluation is appended to ``report_spend`` the moment it
    is priced, which is the moment after it was paid for. That ordering is
    the guarantee: the reporting pass buys an official score per run, a
    held-out measurement per arm, and the anchors, and it writes the
    stage's row only once all of them are done -- so anything held only in
    memory is lost by a crash in that window, and lost spend is the one
    error the ledger cannot detect afterwards.

    An entry whose evidence is already recorded for this stage is a no-op:
    one evaluation cited twice was paid for once, and the manifest refuses
    the duplicate structurally.
    """
    ledger = environment.report_spend
    if ledger is None:
        return

    def persist(record: ReportSpendRecord) -> None:
        manifest = read_study_manifest(study_dir)
        if any(
            entry.evidence_key == record.evidence_key
            and entry.stage == stage.value
            for entry in manifest.report_spend
        ):
            return
        schema_name, content_hash = record.evidence_key
        write_study_manifest(
            study_dir,
            manifest.model_copy(
                update={
                    "report_spend": (
                        *manifest.report_spend,
                        ReportSpendEntry(
                            evidence_schema=schema_name,
                            evidence_content_hash=content_hash,
                            purpose=record.purpose,
                            candidate_name=record.candidate_name,
                            stage=stage.value,
                            transport=environment.transport,
                            spend=record.spend,
                        ),
                    )
                }
            ),
            replace=True,
        )

    ledger.persist_to(persist)


def _record_report_spend(
    *, study_dir: Path, stage: StageId, environment: StageEnvironment
) -> None:
    """Merge the reporting pass's spend onto this stage's row.

    Merged rather than replaced, for :func:`_arm_stage_record`'s reason:
    the row already carries what the arms' runs cost, and the reporting
    pass is a second bill on the same stage rather than a correction of
    the first. The fold re-applies the honesty rules, so one unpriced
    reporting evaluation withholds the role's whole ``usd``.

    **Folded from the manifest, not from this process.** Each reporting
    evaluation persisted its own spend as it was bought, and the fold
    reads those records back. That is what makes this idempotent across a
    resume: the row is a function of what is on disk rather than of what
    this invocation happened to buy, so re-folding after a crash restates
    the same total instead of adding a second copy of it -- and an
    evaluation an earlier invocation paid for is still counted even though
    this process never issued it.

    **Keyed on the transport as well as the stage.** An evaluation bought
    on one transport is not part of what a stage running on another spent,
    so a surviving entry from an invalidated invocation -- a fake-transport
    row, costing nothing anyone owes -- can never reach a paid stage's
    bill. :func:`_without_amended_evidence` drops those entries; this is
    what holds if one ever reaches the manifest by another route.

    A fake-transport stage is skipped: its rows are real rows that would
    total to a bill nobody owes, which is the judgement
    :func:`_stage_record` keeps at the call site.
    """
    if environment.transport == TransportName.FAKE.value:
        return
    manifest = read_study_manifest(study_dir)
    folded = run_spend_records(
        entry
        for record in manifest.report_spend
        if record.stage == stage.value
        and record.transport == environment.transport
        for entry in record.spend
    )
    if not folded:
        return
    existing = next(
        (entry for entry in manifest.stages if entry.stage == stage.value),
        None,
    )
    updated = (
        StageRecord(
            stage=stage.value,
            transport=environment.transport,
            report_spend=folded,
        )
        if existing is None
        # Set, never added: ``folded`` is already the whole pass, so the
        # run-side row is carried through untouched and the reporting side
        # is replaced with the total the durable records now describe.
        else existing.model_copy(update={"report_spend": folded})
    )
    if existing == updated:
        return
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={"stages": _stages_with(manifest, updated)}
        ),
        replace=True,
    )


@dataclass(frozen=True, slots=True)
class _DurableOfficialScorer:
    """Score a run once per stage, ever, across every invocation.

    Official scoring reaches the provider, so "score every run" and "score
    every run *again* on resume" are the same code path with very
    different bills. This wraps the real scorer in the manifest's own
    record: a run already scored at this stage is answered from disk and
    issues no call, and a run scored for the first time has its score
    persisted before it is returned.

    Persisted before returning, not after the pass: the arg-max that
    consumes these scores is followed by a held-out evaluation and a
    manifest write, and a crash anywhere in that window would otherwise
    leave the study having paid for scores it kept no record of.
    """

    arm_id: str
    log: ManifestSelectionLog
    score_official: OfficialScorer

    def __call__(self, candidate: RunCandidate) -> CandidateScore:
        recorded = self.log.official_score_for(candidate.run_id)
        if recorded is not None:
            return recorded
        score = self.score_official(candidate)
        self.log.record_official_score(arm_id=self.arm_id, score=score)
        return score


def _report_or_rebuild_arm(  # noqa: PLR0913
    *,
    arm_id: str,
    runs: tuple[RunCandidate, ...],
    score_official: OfficialScorer,
    evaluate_held_out: HeldOutEvaluator,
    log: ManifestSelectionLog,
    stage: StageId,
) -> ArmReport:
    """Report this arm, or rebuild the report it already produced.

    ``report_arm`` persists each arm's selection as it goes and the ledger
    refuses a second selection per arm per stage, so a stage that crashed
    partway through reporting cannot simply re-report every arm on resume:
    the arms that already selected would raise, and the study's paid runs
    would be stranded behind a failure that never clears.

    An arm whose selection *and* completed held-out claim are both durable
    has already been fully reported. Its report is rebuilt entirely from
    persisted records -- the selection, each run's recorded official score,
    and the completed claim -- so a resume of a fully reported arm re-buys
    **nothing**: no second selection, no second official scoring of runs
    the study already paid to score, and no second held-out evaluation of a
    candidate that already spent its one shot.

    Official scoring is durable for exactly that reason. It is a provider
    call per run, and it previously ran unconditionally on every
    invocation, so a resume silently re-bought the whole official pass for
    every arm it was only rebuilding.

    An arm that selected but never claimed crashed in the window *between*
    the two writes, before any provider call. It continues from the
    selection: the held-out evaluation it never issued is still owed, and
    issuing it now costs exactly what the uncrashed stage would have cost.

    An arm with an *outstanding* claim is the one case that cannot be
    continued. The claim is written before the evaluation is issued, so
    whether the provider was billed is not knowable from here -- re-issuing
    would risk paying twice and skipping would report a number nobody
    measured -- and it is refused with the recovery named.
    """
    durable_scorer = _DurableOfficialScorer(
        arm_id=arm_id, log=log, score_official=score_official
    )
    selection = log.selection_for(arm_id)
    if selection is None:
        return report_arm(
            arm_id=arm_id,
            runs=runs,
            score_official=durable_scorer,
            evaluate_held_out=evaluate_held_out,
            log=log,
            stage=stage.value,
        )
    representative = next(
        (run for run in runs if run.run_id == selection.selected_run_id), None
    )
    if representative is None:
        raise StageError(
            f"arm {arm_id!r} persisted a selection for run "
            f"{selection.selected_run_id!r} at {stage.value}, but that run "
            "is not among the runs this stage loaded"
        )
    official_scores = tuple(durable_scorer(run) for run in runs)
    claim = log.completed_claim_for(arm_id)
    if claim is None:
        if log.held_out_count(arm_id) > 0:
            raise StageError(
                f"arm {arm_id!r} claimed a held-out evaluation at "
                f"{stage.value} that never completed, so the process died "
                "with it in flight. That evaluation cannot be re-issued "
                "without risking a second charge for it: complete the "
                "outstanding entry in the manifest's 'held_out_claims' "
                "block from the evaluation's own evidence, or start a "
                "fresh study directory."
            )
        # Selected but never claimed: the crash landed between the two
        # writes, so nothing was issued and nothing was paid for.
        log.claim_held_out(arm_id)
        claim = evaluate_held_out(
            candidate_name=arm_id, template=representative.template
        )
        log.complete_held_out(claim)
    return ArmReport(
        arm_id=arm_id,
        selection=selection,
        official_scores=official_scores,
        representative=representative,
        held_out=claim,
    )


def _analyse_stage(  # noqa: PLR0913
    *,
    study_dir: Path,
    stage: StageId,
    environment: StageEnvironment,
    reports: tuple[ArmReport, ...],
    k_repeat: int,
    log: ManifestSelectionLog,
) -> AnalysisResult | None:
    """Measure the anchors, then write every candidate's held-out row.

    Returns ``None`` when the environment supplies no anchor templates,
    which is the state a test injecting its own collaborators reaches. A
    stage bound through
    :func:`~whetstone_envs.optim.study.environment.bound_stage_environment`
    always has them, so the operational path always analyses.
    """
    naive = _candidate_template(environment.naive_candidate)
    ceiling = _candidate_template(environment.ceiling_candidate)
    if naive is None or ceiling is None:
        return None
    references = measure_reference_candidates(
        naive_template=naive,
        ceiling_template=ceiling,
        evaluate_held_out=_require(environment.evaluate_held_out),
        log=log,
    )
    return write_held_out_analysis(
        study_dir=study_dir,
        store_path=study_dir / STUDY_STORE_NAME,
        arms=reports,
        references=references,
        k_repeat=k_repeat,
        stage=stage.value,
    )


def _candidate_template(candidate: Candidate) -> str | None:
    """The prompt an anchor candidate renders, or None when it carries none.

    Read off the candidate's own payload rather than passed alongside it, so
    the anchor measured on held-out is the same object Stage 0 calibrated
    with rather than a template that merely resembles it.
    """
    payload = getattr(candidate, "payload", None)
    if payload is None:
        return None
    value = payload.get("prompt_template")
    return value if type(value) is str else None


def _run_every_arm(
    *,
    spec: StudySpec,
    stage: StageId,
    study_dir: Path,
    environment: StageEnvironment,
    recorded: tuple[ArmRecord, ...],
) -> tuple[
    tuple[ArmRecord, ...],
    dict[str, tuple[ArmRunResult, ...]],
    tuple[RunRecord, ...],
]:
    """Run each arm's outstanding seeds and merge them with what exists.

    Two rules make this resumable, which matters because this is the path
    that spends. A seed whose run is already recorded is **not** re-run:
    Stage 1's runs count toward Stage 2 by being the same seeds, and a
    Stage 2 that re-ran them would pay twice for identical work. And the
    merged record keeps every previously recorded run rather than replacing
    the arm's list, so a stage that crashed after paying for some runs does
    not discard them on resume.
    """
    runner = _require(environment.run_optimizer)
    by_arm_id = {arm.arm_id: arm for arm in recorded}
    records: list[ArmRecord] = []
    results_by_arm: dict[str, tuple[ArmRunResult, ...]] = {}
    # What *this* invocation executed, which is what its stage row bills
    # for. A run an earlier stage paid for is already in that stage's row;
    # counting it again here would make the ledger's rows sum to more than
    # the study spent.
    executed: list[RunRecord] = []
    for arm in spec.arms:
        stage_seeds = arm_seeds(arm.arm_id, stage=stage)
        existing = by_arm_id.get(arm.arm_id)
        existing_runs = () if existing is None else existing.runs
        done = {run.seed for run in existing_runs if run.seed is not None}
        fresh = tuple(
            runner(arm=arm, seed=seed, study_dir=study_dir)
            for seed in stage_seeds
            if seed not in done
        )
        merged_runs = (*existing_runs, *(result.record for result in fresh))
        if not merged_runs:
            raise StageError(f"arm {arm.arm_id!r} produced no runs")
        results_by_arm[arm.arm_id] = _candidates_for(
            arm=arm,
            stage_seeds=stage_seeds,
            fresh=fresh,
            existing_runs=existing_runs,
            load_recorded_run=environment.load_recorded_run,
        )
        executed.extend(result.record for result in fresh)
        records.append(
            _arm_record(arm, runs=merged_runs, sample=fresh, prior=existing)
        )
    return tuple(records), results_by_arm, tuple(executed)


def _executed_run_spend(
    runs: Iterable[RunRecord],
) -> tuple[RunSpendRecord, ...]:
    """Every per-role record the runs this stage executed reported.

    Flattened rather than folded here: :func:`run_spend_records` owns the
    fold and the honesty rules it re-applies, so this is only the
    selection of *which* records go into it.
    """
    return tuple(entry for run in runs for entry in run.spend)


def _check_call_counts(
    *,
    study_dir: Path,
    spec: StudySpec,
    stage: StageId,
    run_results: dict[str, tuple[ArmRunResult, ...]],
    design: DesignRecord,
) -> None:
    """The Stage-1 budget gate: measured calls against the pre-spend bound.

    This runs at Stage 1 only. It exists to catch a fan-out bug -- an
    optimizer whose minibatch intents silently expanded to the full valset
    -- before Stage 2 pays five times over for the same defect, so catching
    it at the pilot is the whole point and re-running it at Stage 2 would
    just re-report a fact the pilot already established.

    **The verdict is recorded before it is raised on.** Stage 2 requires a
    passed gate to run at all, and a verdict that lived only in this
    process could not be required of a Stage 2 started in a fresh one. The
    record is written for a failed gate too: that is the finding, and it is
    what lets the Stage-2 refusal say the pilot failed rather than that the
    pilot is missing.

    Codex is exempt by construction, because its estimate carries
    ``gated=False``: its agent chooses how much of its cap to spend, and a
    bug detector pointed at a non-deterministic agent invites a false abort
    (OQ3). It is gated on capacity respect and audit pass instead.
    """
    if stage is not StageId.STAGE1:
        return
    internal_size = spec.internal.size
    overruns = tuple(
        f"{arm.arm_id}/{result.candidate.run_id}: "
        f"{result.observed_task_calls} calls"
        for arm in spec.arms
        for result in run_results.get(arm.arm_id, ())
        if not call_count_within_estimate(
            optimizer=arm.optimizer,
            observed_task_calls=result.observed_task_calls,
            internal_size=internal_size,
            k_repeat=design.k_repeat,
            official_size=spec.official.size,
            held_out_size=spec.held_out.size,
            copro_breadth=arm.copro_breadth,
            copro_depth=arm.copro_depth,
        )
    )
    manifest = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "call_count_gate": CallCountGateRecord(
                    stage=StageId.STAGE1.value,
                    passed=not overruns,
                    tolerance=STAGE1_CALL_COUNT_TOLERANCE,
                    overruns=overruns,
                )
            }
        ),
        replace=True,
    )
    if overruns:
        raise StageError(
            "these runs exceeded "
            f"{STAGE1_CALL_COUNT_TOLERANCE}x their pre-spend call estimate, "
            "which is what a fan-out bug looks like: " + "; ".join(overruns)
        )


def _require_passed_stage1_gate(
    *, manifest: StudyManifest, stage: StageId
) -> None:
    """Stage 2 runs only behind a Stage 1 whose call-count gate passed.

    The pilot is not a formality: it is the one place a fan-out bug is
    caught for the price of one run per arm instead of five. A Stage 2
    invoked directly after Stage 0, or after a Stage 1 that failed the
    gate, spent the full design without that check ever having cleared --
    exactly the bill the pilot exists to avoid.

    Refused before dispatch, so a study in either state pays nothing. The
    two cases are named separately because they need different actions: a
    missing pilot has to be run, and a failed one has to be diagnosed.
    """
    if stage is not StageId.STAGE2:
        return
    gate = manifest.call_count_gate
    if gate is None:
        raise StageError(
            "stage2 requires a completed stage1 whose call-count gate "
            "passed, and this study has recorded no such gate; run stage1 "
            "first. Refusing before any arm runs, so the full design buys "
            "nothing the pilot was supposed to clear it for"
        )
    if not gate.passed:
        raise StageError(
            "stage2 requires a stage1 whose call-count gate passed, and "
            "this study's pilot failed it: "
            + "; ".join(gate.overruns)
            + ". Refusing before any arm runs; diagnose the fan-out the "
            "pilot caught rather than paying for it five times over"
        )


def _candidates_for(
    *,
    arm: ArmSpec,
    stage_seeds: tuple[int, ...],
    fresh: tuple[ArmRunResult, ...],
    existing_runs: tuple[RunRecord, ...],
    load_recorded_run: RecordedRunLoader | None,
) -> tuple[ArmRunResult, ...]:
    """The candidates this stage selects between, in seed order.

    Selection is over **every** run at this stage's seeds, including ones an
    earlier stage already paid for. Stage 2 therefore has to see Stage 1's
    terminal candidates, which this process did not produce, so they are
    re-read from their own artifacts through ``load_recorded_run``.

    Two failures are refused rather than worked around, because both would
    quietly turn a ``K_RUN = 5`` arg-max into a smaller one: a stage with no
    loader that finds a recorded run, and a recorded run whose artifacts no
    longer load. Re-running the seed instead is not an option -- it would
    pay a second time for a run the study already bought and recorded.

    The result is ordered by the stage's own seed order rather than by
    "loaded then fresh", so the arg-max's tie-break -- earliest run wins --
    means the earliest *seed*, whichever stage happened to execute it.
    """
    by_seed = {result.candidate.seed: result for result in fresh}
    missing_artifacts: list[int] = []
    unloadable: list[int] = []
    for run in existing_runs:
        if run.seed is None or run.seed in by_seed:
            continue
        if run.seed not in set(stage_seeds):
            # A recorded run outside this stage's seed set is not part of
            # this stage's selection; it stays recorded and unselected.
            continue
        if load_recorded_run is None:
            missing_artifacts.append(run.seed)
            continue
        loaded = load_recorded_run(arm=arm, run=run)
        if loaded is None:
            unloadable.append(run.seed)
            continue
        by_seed[run.seed] = loaded
    if missing_artifacts:
        raise StageError(
            f"arm {arm.arm_id!r} has recorded runs at seeds "
            f"{sorted(missing_artifacts)} and this stage was given no way "
            "to load them; selection would silently run over a subset of "
            "the arm's runs"
        )
    if unloadable:
        raise StageError(
            f"arm {arm.arm_id!r} recorded runs at seeds {sorted(unloadable)} "
            "whose artifacts no longer load; selection would silently run "
            "over a subset of the arm's runs, and re-running them would pay "
            "twice for runs this study already bought"
        )
    selected = tuple(by_seed[seed] for seed in stage_seeds if seed in by_seed)
    if len(selected) != len(stage_seeds):
        absent = sorted(set(stage_seeds) - set(by_seed))
        raise StageError(
            f"arm {arm.arm_id!r} is missing runs at seeds {absent}; every "
            "seed this stage declares must be run or loaded before the "
            "arg-max is taken"
        )
    return selected


def _arm_record(
    arm: ArmSpec,
    *,
    runs: tuple[RunRecord, ...],
    sample: tuple[ArmRunResult, ...],
    prior: ArmRecord | None,
) -> ArmRecord:
    """The arm's merged record, keeping the control identity it first had.

    An arm's control identity is a property of its configuration, not of
    whichever run happened to execute last, so a resumed stage keeps the
    hash the arm was first recorded with rather than restating it from a
    newer run.
    """
    if prior is not None and prior.runs:
        control_identity_hash = prior.control_identity_hash
    elif sample:
        control_identity_hash = sample[0].record.result_ref.content_hash
    else:  # pragma: no cover - guarded by the empty-runs check above
        control_identity_hash = runs[0].result_ref.content_hash
    return ArmRecord(
        arm_id=arm.arm_id,
        optimizer=arm.optimizer,
        demo_mode=arm.demo_mode,
        train_size=arm.train_size,
        val_size=arm.val_size,
        minibatch=arm.miprov2_minibatch,
        minibatch_size=arm.miprov2_minibatch_size,
        control_identity_hash=control_identity_hash,
        seed_note=_seed_note(arm),
        runs=runs,
    )


#: How an arm honoured its requested seed, recorded verbatim. COPRO carries
#: no control seed field, so its runs are seeded only by the provider
#: ``SEED`` control and proposal ordering; saying so beats a manifest that
#: implies every arm was seeded the same way.
SEED_NOTE_CONTROL_FIELD = "control-seed-field"
SEED_NOTE_PROVIDER_ONLY = "provider-seed-control-only"


def _seed_note(arm: ArmSpec) -> str:
    if arm.optimizer in {"copro", "null-identity"}:
        return SEED_NOTE_PROVIDER_ONLY
    return SEED_NOTE_CONTROL_FIELD


def _refuse_projection_claiming_the_design(
    *, manifest: StudyManifest, stage: StageId
) -> None:
    """Refuse an arm stage whose study id claims a design it is not.

    The manifest model already refuses a ``full`` projection with no Codex
    arm, so the two cannot disagree. What is left is the naming: a
    projection whose ``study_id`` is a registered protocol id would spend a
    whole stage and leave every artifact citing the pre-registration by
    name. ``init`` refuses to author that, and this refuses to *spend* on
    one, because a manifest can also arrive hand-edited.
    """
    if manifest.design_projection == DESIGN_PROJECTION_FULL:
        return
    if manifest.study_id not in PROTOCOL_IDS:
        return
    raise StageError(
        f"{stage.value} refuses to run: this manifest declares the "
        f"{manifest.design_projection!r} projection but carries study id "
        f"{manifest.study_id!r}, which is a registered protocol id. Its "
        "runs would be recorded against the pre-registration while "
        "holding a smaller design. Re-initialise the projection under an "
        "id of its own."
    )


def call_count_within_estimate(  # noqa: PLR0913
    *,
    optimizer: str,
    observed_task_calls: int,
    internal_size: int,
    k_repeat: int,
    tolerance: float = STAGE1_CALL_COUNT_TOLERANCE,
    official_size: int = 0,
    held_out_size: int = 0,
    copro_breadth: int | None = None,
    copro_depth: int | None = None,
) -> bool:
    """Whether a run's measured calls land near its pre-spend estimate.

    This is the Stage-1 budget gate, applied per run. **Both sides are
    task-model rows**: ``observed_task_calls`` is what the run's own
    evidence counted, and the estimate is in the same unit by construction
    (see :mod:`~whetstone_envs.optim.study.gates`). A GEPA estimate stated
    in metric calls would not be comparable to it, and a real GEPA run
    would trip the tolerance on the unit mismatch alone.

    Codex is exempt by construction: its estimate carries ``gated=False``
    because its agent chooses how much of its cap to spend, and applying a
    fan-out detector to a non-deterministic agent invites a false abort
    (OQ3). The comparison is against the estimate's **ceiling**, so a
    low-accuracy anchor -- which makes MIPROv2 bootstrap more rows, not
    fewer -- does not read as an overrun (F10).

    ``official_size`` and ``held_out_size`` are needed only by
    ``null-identity``, whose estimate is the report harness rather than an
    optimizer search.

    ``copro_breadth``/``copro_depth`` carry the *arm's own* pinned shape
    into the estimate, because COPRO's whole per-run cost follows from
    them. Gating a 6x3 run against a 2x1 estimate would flag every healthy
    COPRO run as a fourfold overrun.
    """
    estimate = estimate_optimizer_calls(
        optimizer,
        internal_size=internal_size,
        k_repeat=k_repeat,
        official_size=official_size,
        held_out_size=held_out_size,
        **(
            {}
            if copro_breadth is None or copro_depth is None
            else {
                "copro_breadth": copro_breadth,
                "copro_depth": copro_depth,
            }
        ),
    )
    if not estimate.gated:
        return True
    return observed_task_calls <= estimate.high * tolerance


def _require[T](value: T | None) -> T:
    if value is None:  # pragma: no cover - guarded by the caller above
        raise StageError("a required stage collaborator was not provided")
    return value


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def run_stage(
    *,
    study_dir: Path,
    stage: str,
    environment: StageEnvironment,
    replace_design: bool = False,
) -> StudyManifest:
    """Run one named stage and return the manifest it wrote.

    This is the signature the CLI's ``StageRunner`` protocol names, with the
    environment bound by the caller. The manifest is returned rather than a
    path so the CLI reports what the stage recorded without re-reading a
    file the harness may still be writing.

    ``replace_design`` reaches Stage 0 only; an arm stage never rewrites the
    pre-registration, so accepting the flag there would suggest it could.
    """
    try:
        stage_id = StageId(stage)
    except ValueError as error:
        raise StageError(f"unknown stage {stage!r}") from error
    if stage_id is StageId.STAGE0:
        return run_stage0_into_manifest(
            study_dir=study_dir,
            environment=environment,
            replace_design=replace_design,
        ).manifest
    if replace_design:
        raise StageError(
            f"{stage_id.value} does not record a design; --replace-design "
            "applies to stage0"
        )
    return run_arm_stage(
        study_dir=study_dir, stage=stage_id, environment=environment
    ).manifest
