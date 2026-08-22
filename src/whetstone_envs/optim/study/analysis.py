"""Turn a stage's measurements into the manifest's held-out rows.

Selection and measurement happen per arm, one at a time, and each held-out
evaluation is issued the moment its arm's selection is durable. The
*statistics* cannot work that way: a Holm-corrected p-value is a whole-study
computation over the pre-registered family, so it does not exist until every
arm has been measured. This module is that second pass.

It reads what the stage measured -- the arms' held-out vectors and the
anchors' -- runs the pre-registered analysis over them, and writes one
:class:`~whetstone_envs.optim.study.manifest.HeldOutRecord` per reported
candidate into ``study.json``. Every row cites the per-task vector it was
computed from as evidence in the study's own store, so the report resolves a
pointer rather than trusting a number.

**The naive anchor is the comparison, and it is measured here.** Anchors and
nulls reach held-out through
:func:`~whetstone_envs.optim.study.selection.report_reference_candidate` --
the same ledger, the same once-only rule, the same procedure as an arm (L4).
Reporting a delta against an anchor measured any other way would make the
pairing a claim rather than a construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_store.sync import open_sqlite

from whetstone_envs.optim.study.leakage import check_held_out_nesting
from whetstone_envs.optim.study.manifest import (
    CORRECTION_FAMILY_SIZE,
    EvidencePointer,
    HeldOutRecord,
    StudyManifest,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.power import weighted_per_task_delta
from whetstone_envs.optim.study.selection import (
    DEFAULT_SELECTION_STAGE,
    ArmDelta,
    ArmStatistics,
    HeldOutMeasurement,
    analyze_arms,
    null_triggers_downgrade,
    report_reference_candidate,
)
from whetstone_envs.optim.study.spec import (
    NULL_ARM_IDS,
    REAL_OPTIMIZER_ARM_IDS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from whetstone_envs.optim.study.selection import (
        ArmReport,
        HeldOutEvaluator,
        SelectionLedger,
    )

__all__ = [
    "CEILING_CANDIDATE_NAME",
    "HELD_OUT_VECTOR_SCHEMA",
    "NAIVE_CANDIDATE_NAME",
    "AnalysisResult",
    "measure_reference_candidates",
    "write_held_out_analysis",
]

#: The names the two anchors are reported under. Persisted into held-out
#: rows, so they are owned constants rather than inline strings.
NAIVE_CANDIDATE_NAME = "naive"
CEILING_CANDIDATE_NAME = "ceiling"

#: The schema a held-out per-task vector is stored under. A row cites it, so
#: it is a persisted-format literal owned here and pinned by a test.
HELD_OUT_VECTOR_SCHEMA = "whetstone_envs.study_held_out_vector/v1"


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """What the analysis pass computed and wrote.

    ``null_downgrade`` is the F12 verdict: a null whose held-out movement
    exceeds the measured MDE *and* whose interval excludes zero means
    selection over nothing produced a real effect, which voids the study's
    efficacy claims rather than merely being noted.
    """

    manifest: StudyManifest
    rows: tuple[HeldOutRecord, ...]
    null_downgrade: bool
    nesting_checked: bool
    #: Which stage's measurements these rows describe.
    stage: str = DEFAULT_SELECTION_STAGE


def measure_reference_candidates(
    *,
    naive_template: str,
    ceiling_template: str,
    evaluate_held_out: HeldOutEvaluator,
    log: SelectionLedger,
) -> dict[str, HeldOutMeasurement]:
    """Measure both anchors on held-out, once each, through the ledger.

    The anchors have nothing to select between -- there is one candidate by
    construction -- so they skip the arg-max and keep the identical held-out
    procedure and the identical once-only ledger, which is what makes L3 and
    L4 hold for them and not only for the arms.
    """
    return {
        name: report_reference_candidate(
            candidate_name=name,
            template=template,
            evaluate_held_out=evaluate_held_out,
            log=log,
        )
        for name, template in (
            (NAIVE_CANDIDATE_NAME, naive_template),
            (CEILING_CANDIDATE_NAME, ceiling_template),
        )
    }


def write_held_out_analysis(  # noqa: PLR0913
    *,
    study_dir: Path,
    store_path: Path,
    arms: tuple[ArmReport, ...],
    references: dict[str, HeldOutMeasurement],
    k_repeat: int,
    stage: str = DEFAULT_SELECTION_STAGE,
) -> AnalysisResult:
    """Analyse every measured candidate and write its held-out row.

    ``stage`` names which stage's measurements these rows describe. The
    manifest holds one ``held_out`` block, and it is the study's reported
    result, so a pilot's rows are written and then replaced by the full
    design's rather than accumulating two competing answers.

    The order is fixed and load-bearing. Every arm's delta is formed against
    the *same* naive vector, so the bootstrap is genuinely paired; the four
    real optimizers are corrected together at the pre-registered family
    size; and the nulls are analysed uncorrected beside them, because a
    control that had to survive a family-wise correction would be hardest to
    trip exactly when tripping it matters most.
    """
    manifest = read_study_manifest(study_dir)
    naive = references.get(NAIVE_CANDIDATE_NAME)
    if naive is None:
        raise ValueError(
            "the held-out analysis reports every delta against the naive "
            "anchor; measure it before analysing"
        )
    measurements = _measurements_by_name(arms=arms, references=references)
    deltas = {
        name: _delta_for(
            arm_id=name,
            measurement=measurement,
            naive=naive,
            k_repeat=k_repeat,
        )
        for name, measurement in measurements.items()
        if name != NAIVE_CANDIDATE_NAME
    }
    real = tuple(
        deltas[name] for name in REAL_OPTIMIZER_ARM_IDS if name in deltas
    )
    # Holm over the pre-registered family only. The nulls and the ceiling
    # anchor are analysed as their own single-hypothesis families, which is
    # how an uncorrected control interval is computed without letting it
    # borrow the family's error budget.
    corrected = {
        statistic.arm_id: statistic
        for statistic in analyze_arms(
            real,
            level=_ci_level(manifest),
            resamples=_resamples(manifest),
            seed=_bootstrap_seed(manifest),
            family_size=CORRECTION_FAMILY_SIZE,
        )
    }
    uncorrected = {
        name: analyze_arms(
            (delta,),
            level=_ci_level(manifest),
            resamples=_resamples(manifest),
            seed=_bootstrap_seed(manifest),
            family_size=1,
        )[0]
        for name, delta in deltas.items()
        if name not in corrected
    }

    rows: list[HeldOutRecord] = []
    with open_sqlite(str(store_path)) as store:

        def pointer_for(
            name: str, measurement: HeldOutMeasurement
        ) -> tuple[EvidencePointer, EvidencePointer]:
            reference, _status = store.put(
                HELD_OUT_VECTOR_SCHEMA,
                {
                    "schema_version": 1,
                    "candidate_name": name,
                    "eval_config_hash": measurement.eval_config_hash,
                    "repeats": measurement.repeats,
                    "per_task": list(measurement.per_task),
                    "mean": measurement.mean,
                    "completeness": measurement.completeness,
                },
            )
            cited = EvidencePointer(
                schema_name=reference.schema,
                content_hash=reference.content_hash,
            )
            # The evidence pointer and the per-task pointer address the same
            # record here: the vector *is* this study's durable evidence for
            # the number, and citing a second, identical record would imply
            # two independent sources where there is one.
            return cited, cited

        for name, measurement in measurements.items():
            statistic = corrected.get(name) or uncorrected.get(name)
            evidence_ref, vector_ref = pointer_for(name, measurement)
            rows.append(
                HeldOutRecord(
                    candidate_name=name,
                    eval_evidence_ref=evidence_ref,
                    per_task_scores_ref=vector_ref,
                    mean=measurement.mean,
                    ci_low=(0.0 if statistic is None else statistic.ci_low),
                    ci_high=(0.0 if statistic is None else statistic.ci_high),
                    delta_vs_naive=(
                        0.0 if statistic is None else statistic.delta
                    ),
                    p_bootstrap=(
                        1.0 if statistic is None else statistic.p_bootstrap
                    ),
                    p_holm=(
                        corrected[name].p_holm if name in corrected else None
                    ),
                    completeness=measurement.completeness,
                )
            )

    updated = manifest.model_copy(update={"held_out": tuple(rows)})
    write_study_manifest(study_dir, updated, replace=True)
    return AnalysisResult(
        manifest=read_study_manifest(study_dir),
        rows=tuple(rows),
        stage=stage,
        null_downgrade=_null_downgrade(
            manifest=manifest, statistics=uncorrected
        ),
        nesting_checked=_nesting_holds(manifest),
    )


def _measurements_by_name(
    *,
    arms: tuple[ArmReport, ...],
    references: dict[str, HeldOutMeasurement],
) -> dict[str, HeldOutMeasurement]:
    """Every measured candidate, anchors first so ``naive`` is row one."""
    measurements: dict[str, HeldOutMeasurement] = {
        NAIVE_CANDIDATE_NAME: references[NAIVE_CANDIDATE_NAME]
    }
    if CEILING_CANDIDATE_NAME in references:
        measurements[CEILING_CANDIDATE_NAME] = references[
            CEILING_CANDIDATE_NAME
        ]
    for report in arms:
        measurements[report.arm_id] = report.held_out
    return measurements


def _delta_for(
    *,
    arm_id: str,
    measurement: HeldOutMeasurement,
    naive: HeldOutMeasurement,
    k_repeat: int,
) -> ArmDelta:
    """One arm's paired delta against naive, row-completeness weighted (O7).

    The weighting flows into the per-task vector rather than only into a
    variance estimate, so a ragged cell shrinks its own contribution to the
    point estimate as well as to the interval.
    """
    if len(measurement.per_task) != len(naive.per_task):
        raise ValueError(
            f"candidate {arm_id!r} and the naive anchor measured different "
            "numbers of held-out tasks, so their comparison is not paired"
        )
    achieved = tuple(
        round(measurement.completeness * k_repeat)
        for _ in measurement.per_task
    )
    weighted, completeness = weighted_per_task_delta(
        arm_per_task=measurement.per_task,
        naive_per_task=naive.per_task,
        achieved_counts=achieved,
        planned_count=k_repeat,
    )
    # ``ArmDelta`` pairs an arm vector against the anchor's; the weighting
    # has already been applied to the difference, so the anchor side is
    # zeroed rather than re-subtracted.
    return ArmDelta(
        arm_id=arm_id,
        arm_per_task=weighted,
        naive_per_task=(0.0,) * len(weighted),
        completeness=completeness,
    )


def _null_downgrade(
    *,
    manifest: StudyManifest,
    statistics: Mapping[str, ArmStatistics],
) -> bool:
    """F12: whether a null's movement voids the study's efficacy claims.

    Both pre-registered conditions must hold: the magnitude exceeds the
    measured MDE *and* the interval excludes zero. A study with no measured
    design has no MDE to compare against, so it cannot trip the rule --
    which is honest, since there is nothing to void yet.
    """
    design = manifest.design
    if design is None:
        return False
    return any(
        null_triggers_downgrade(
            null_delta=statistic.delta,
            mde_measured=design.mde_measured,
            excludes_zero=statistic.excludes_zero,
        )
        for arm_id in NULL_ARM_IDS
        if (statistic := statistics.get(arm_id)) is not None
    )


def _nesting_holds(manifest: StudyManifest) -> bool:
    """D5's nesting check, run over the split this study actually recorded.

    A study that never grew its held-out split has nothing to nest, and the
    check passes trivially against itself -- which is the honest answer: the
    split before and after the (absent) decision is the same population.
    """
    hashes = manifest.splits.held_out.task_hashes
    return check_held_out_nesting(smaller=hashes, larger=hashes).passed


def _ci_level(manifest: StudyManifest) -> float:
    return 0.95 if manifest.design is None else manifest.design.ci_level


def _resamples(manifest: StudyManifest) -> int:
    return 10_000 if manifest.design is None else manifest.design.resamples


def _bootstrap_seed(manifest: StudyManifest) -> int:
    return 0 if manifest.design is None else manifest.design.bootstrap_seed
