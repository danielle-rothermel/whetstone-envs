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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import Candidate

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
    COMPLETENESS_BACKSTOP,
    CORRECTION_FAMILY_SIZE,
    CORRECTION_HOLM_BONFERRONI,
    ArmRecord,
    DesignRecord,
    RunRecord,
    StudyManifest,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.power import COMPLETENESS_RULE, MDE_FORMULA
from whetstone_envs.optim.study.selection import (
    ArmReport,
    HeldOutEvaluator,
    ManifestSelectionLog,
    OfficialScorer,
    RunCandidate,
    report_arm,
)
from whetstone_envs.optim.study.spec import (
    StageId,
    arm_seeds,
    spec_from_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.spec import ArmSpec, StudySpec

__all__ = [
    "SEED_NOTE_CONTROL_FIELD",
    "SEED_NOTE_PROVIDER_ONLY",
    "ArmRunResult",
    "OptimizerRunner",
    "StageEnvironment",
    "StageError",
    "StageResult",
    "call_count_within_estimate",
    "run_arm_stage",
    "run_stage",
    "run_stage0_into_manifest",
]


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


# --------------------------------------------------------------------------
# Stage 0
# --------------------------------------------------------------------------


def run_stage0_into_manifest(
    *, study_dir: Path, environment: StageEnvironment
) -> StageResult:
    """Calibrate the anchors, evaluate the gate, record the design.

    The gate's verdict is recorded whether or not it passed. A failed gate
    is a finding the study reports -- an underpowered design, stated as
    such -- not an error that erases the calibration it just paid for. What
    a failed gate does stop is the *next* stage, which
    :func:`run_arm_stage` refuses to start without a recorded design.
    """
    manifest = read_study_manifest(study_dir)
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
    result = run_stage0(
        spec=spec,
        bind_engine=environment.bind_engine,
        naive_candidate=environment.naive_candidate,
        ceiling_candidate=environment.ceiling_candidate,
        task_ids_by_role=environment.task_ids_by_role,
        pool_ceiling=environment.pool_ceiling,
    )
    updated = manifest.model_copy(
        update={"design": _design_record(spec, result)}
    )
    write_study_manifest(study_dir, updated, replace=True)
    return StageResult(
        stage=StageId.STAGE0,
        manifest=read_study_manifest(study_dir),
        stage0=result,
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
    # The design's ``k_run_by_arm`` is the *full* pre-registration, so it
    # says how many runs Stage 2 gets, not how many this stage does. Stage 1
    # spends a prefix of the same seeds, which is what makes "Stage 1's runs
    # count toward Stage 2" checkable rather than asserted.
    spec = spec_from_manifest(manifest, stage=stage)
    if not spec.arms:
        raise StageError(f"{stage.value} has no arms to run")

    arm_records, run_results = _run_every_arm(
        spec=spec,
        stage=stage,
        study_dir=study_dir,
        environment=environment,
        recorded=manifest.arms,
    )
    write_study_manifest(
        study_dir,
        manifest.model_copy(update={"arms": arm_records}),
        replace=True,
    )

    _check_call_counts(
        spec=spec, stage=stage, run_results=run_results, design=manifest.design
    )

    log = ManifestSelectionLog(study_dir)
    reports = tuple(
        report_arm(
            arm_id=arm.arm_id,
            runs=tuple(result.candidate for result in run_results[arm.arm_id]),
            score_official=_require(environment.score_official),
            evaluate_held_out=_require(environment.evaluate_held_out),
            log=log,
        )
        for arm in spec.arms
    )
    return StageResult(
        stage=stage,
        manifest=read_study_manifest(study_dir),
        arms=reports,
    )


def _run_every_arm(
    *,
    spec: StudySpec,
    stage: StageId,
    study_dir: Path,
    environment: StageEnvironment,
    recorded: tuple[ArmRecord, ...],
) -> tuple[tuple[ArmRecord, ...], dict[str, tuple[ArmRunResult, ...]]]:
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
    for arm in spec.arms:
        stage_seeds = arm_seeds(arm.optimizer, stage=stage)
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
            arm_id=arm.arm_id,
            stage_seeds=stage_seeds,
            fresh=fresh,
            existing_runs=existing_runs,
        )
        records.append(
            _arm_record(arm, runs=merged_runs, sample=fresh, prior=existing)
        )
    return tuple(records), results_by_arm


def _check_call_counts(
    *,
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

    Codex is exempt by construction, because its estimate carries
    ``gated=False``: its agent chooses how much of its cap to spend, and a
    bug detector pointed at a non-deterministic agent invites a false abort
    (OQ3). It is gated on capacity respect and audit pass instead.
    """
    if stage is not StageId.STAGE1:
        return
    internal_size = spec.internal.size
    overruns = [
        f"{arm.arm_id}/{result.candidate.run_id}: "
        f"{result.observed_task_calls} calls"
        for arm in spec.arms
        for result in run_results.get(arm.arm_id, ())
        if not call_count_within_estimate(
            optimizer=arm.optimizer,
            observed_task_calls=result.observed_task_calls,
            internal_size=internal_size,
            k_repeat=design.k_repeat,
        )
    ]
    if overruns:
        raise StageError(
            "these runs exceeded "
            f"{STAGE1_CALL_COUNT_TOLERANCE}x their pre-spend call estimate, "
            "which is what a fan-out bug looks like: " + "; ".join(overruns)
        )


def _candidates_for(
    *,
    arm_id: str,
    stage_seeds: tuple[int, ...],
    fresh: tuple[ArmRunResult, ...],
    existing_runs: tuple[RunRecord, ...],
) -> tuple[ArmRunResult, ...]:
    """The candidates this stage selects between.

    Selection is over every run at this stage's seeds, including ones an
    earlier stage already paid for. A previously recorded run whose
    terminal candidate this process never saw cannot be scored, so it is
    refused loudly: silently selecting over a subset would quietly turn a
    ``K_RUN = 5`` arg-max into a ``K_RUN = 3`` one.
    """
    if len(fresh) == len(stage_seeds):
        return fresh
    reusable = {run.seed for run in existing_runs if run.seed is not None}
    missing = sorted(
        seed
        for seed in stage_seeds
        if seed in reusable
        and seed not in {result.candidate.seed for result in fresh}
    )
    raise StageError(
        f"arm {arm_id!r} has recorded runs at seeds {missing} whose terminal "
        "candidates this process did not load; selection would silently run "
        "over a subset of the arm's runs"
    )


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


def call_count_within_estimate(
    *,
    optimizer: str,
    observed_task_calls: int,
    internal_size: int,
    k_repeat: int,
    tolerance: float = STAGE1_CALL_COUNT_TOLERANCE,
) -> bool:
    """Whether a run's measured calls land near its pre-spend estimate.

    This is the Stage-1 budget gate, applied per run. Codex is exempt by
    construction: its estimate carries ``gated=False`` because its agent
    chooses how much of its cap to spend, and applying a fan-out detector to
    a non-deterministic agent invites a false abort (OQ3). The comparison is
    against the estimate's **ceiling**, so a low-accuracy anchor -- which
    makes MIPROv2 bootstrap more rows, not fewer -- does not read as an
    overrun (F10).
    """
    estimate = estimate_optimizer_calls(
        optimizer, internal_size=internal_size, k_repeat=k_repeat
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
    *, study_dir: Path, stage: str, environment: StageEnvironment
) -> StudyManifest:
    """Run one named stage and return the manifest it wrote.

    This is the signature the CLI's ``StageRunner`` protocol names, with the
    environment bound by the caller. The manifest is returned rather than a
    path so the CLI reports what the stage recorded without re-reading a
    file the harness may still be writing.
    """
    try:
        stage_id = StageId(stage)
    except ValueError as error:
        raise StageError(f"unknown stage {stage!r}") from error
    if stage_id is StageId.STAGE0:
        return run_stage0_into_manifest(
            study_dir=study_dir, environment=environment
        ).manifest
    return run_arm_stage(
        study_dir=study_dir, stage=stage_id, environment=environment
    ).manifest
