"""The Step 10 study harness: stages, gates, selection, and leakage checks.

The study runs in three stages, each gated, and spends provider budget only
at a gate that passed:

* **Stage 0** calibrates the naive and ceiling anchors on all three roles and
  produces the numbers the whole design rests on -- held-out headroom, the
  variance decomposition, and the re-inverted MDE. Its gate is a hard stop:
  an underpowered design is reported as underpowered, not optimized against.
* **Stage 1** is the pilot at ``K_RUN = 2``, gated on fidelity audits passing
  and per-run call counts landing near their corrected estimates.
* **Stage 2** is the full run at ``K_RUN = 5``, reusing Stage 1's runs.

Two invariants shape the module layout more than anything else:

**Selection and reporting are one function.** Every held-out number in the
study enters through :func:`report_arm`, which scores on official, persists
the arg-max, reads it back, and only then issues one held-out evaluation.
Leakage rules L2 and L3 are therefore properties of the code's shape rather
than rules a caller must remember.

**The gate arithmetic is one line.** :func:`minimum_detectable_effect` is the
study's single pre-registered MDE, computed here rather than read off a power
surface so the gate's design point cannot drift.

This package owns the design, the arithmetic, and the checks. It does not
write ``study.json`` -- the manifest module owns the persisted form, and the
records exposed here are plain dataclasses it serializes.
"""

from __future__ import annotations

from whetstone_envs.optim.study.anchors import (
    ANCHOR_ROLES,
    AnchorPurpose,
    EngineBinder,
    RoleCalibration,
    Stage0Result,
    calibrate_role,
    run_stage0,
)
from whetstone_envs.optim.study.leakage import (
    HeldOutObservation,
    LeakageCheckError,
    LeakageFinding,
    LeakageReport,
    LeakageRule,
    OptimizerEvalObservation,
    SplitIdentity,
    check_held_out_nesting,
    study_leakage_check,
)
from whetstone_envs.optim.study.power import (
    COMPLETENESS_BACKSTOP,
    COMPLETENESS_RULE,
    MDE_FORMULA,
    MDE_MULTIPLIER,
    GateOutcome,
    Stage0Gate,
    Stage0Inputs,
    WithinVarianceCheck,
    evaluate_stage0_gate,
    minimum_detectable_effect,
    nondeterminism_floor,
    split_half_stable,
    weighted_per_task_delta,
    within_variance_divergence,
)
from whetstone_envs.optim.study.selection import (
    SELECTION_RULE,
    ArmDelta,
    ArmReport,
    ArmStatistics,
    CandidateScore,
    HeldOutEvaluator,
    HeldOutMeasurement,
    OfficialScorer,
    RunCandidate,
    SelectionError,
    SelectionLog,
    SelectionRecord,
    analyze_arms,
    null_triggers_downgrade,
    report_arm,
    report_reference_candidate,
)
from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    CORRECTION_RULE,
    HOLM_FAMILY_SIZE,
    K_CAL_CAP,
    K_CAL_INITIAL,
    K_RUN_NULL_A,
    K_RUN_NULL_B,
    K_RUN_PILOT,
    K_RUN_STAGE2,
    NULL_ARM_IDS,
    REAL_OPTIMIZER_ARM_IDS,
    SEED_RANGE_BY_OPTIMIZER,
    ArmKind,
    ArmSpec,
    SplitSpec,
    StageId,
    StudySpec,
    arm_seeds,
    default_arms,
    k_run_for,
    next_k_cal,
)

__all__ = [
    "ANCHOR_ROLES",
    "CODEX_EVALUATE_CALL_CAP",
    "COMPLETENESS_BACKSTOP",
    "COMPLETENESS_RULE",
    "CORRECTION_RULE",
    "HOLM_FAMILY_SIZE",
    "K_CAL_CAP",
    "K_CAL_INITIAL",
    "K_RUN_NULL_A",
    "K_RUN_NULL_B",
    "K_RUN_PILOT",
    "K_RUN_STAGE2",
    "MDE_FORMULA",
    "MDE_MULTIPLIER",
    "NULL_ARM_IDS",
    "REAL_OPTIMIZER_ARM_IDS",
    "SEED_RANGE_BY_OPTIMIZER",
    "SELECTION_RULE",
    "AnchorPurpose",
    "ArmDelta",
    "ArmKind",
    "ArmReport",
    "ArmSpec",
    "ArmStatistics",
    "CandidateScore",
    "EngineBinder",
    "GateOutcome",
    "HeldOutEvaluator",
    "HeldOutMeasurement",
    "HeldOutObservation",
    "LeakageCheckError",
    "LeakageFinding",
    "LeakageReport",
    "LeakageRule",
    "OfficialScorer",
    "OptimizerEvalObservation",
    "RoleCalibration",
    "RunCandidate",
    "SelectionError",
    "SelectionLog",
    "SelectionRecord",
    "SplitIdentity",
    "SplitSpec",
    "Stage0Gate",
    "Stage0Inputs",
    "Stage0Result",
    "StageId",
    "StudySpec",
    "WithinVarianceCheck",
    "analyze_arms",
    "arm_seeds",
    "calibrate_role",
    "check_held_out_nesting",
    "default_arms",
    "evaluate_stage0_gate",
    "k_run_for",
    "minimum_detectable_effect",
    "next_k_cal",
    "nondeterminism_floor",
    "null_triggers_downgrade",
    "report_arm",
    "report_reference_candidate",
    "run_stage0",
    "split_half_stable",
    "study_leakage_check",
    "weighted_per_task_delta",
    "within_variance_divergence",
]
