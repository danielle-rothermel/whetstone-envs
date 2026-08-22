"""Stage 0: calibrate the naive and ceiling anchors on all three roles.

Stage 0 buys the only numbers the rest of the study's design depends on --
the held-out headroom, the variance decomposition, and the re-inverted MDE --
and it buys them with no optimizer involved. It is the first provider spend
and the gate it feeds is a hard stop: an underpowered design is reported as
underpowered rather than optimized against.

The anchors calibrate on **every** role, not just held-out. Internal and
official anchors are what make the leakage check's identical-procedure rule
(L4) checkable and what let the report show that the three splits are drawn
from one population. The gate itself reads held-out only, because held-out is
the split the study reports from and therefore the split whose MDE must be
inverted.

``run_anchor_calibration`` takes ``eval_role`` explicitly, so a mis-bound
engine is a loud error instead of a silently relabelled calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from whetstone.core.roles import EvalRole
from whetstone.eval.analysis import (
    AnchorCalibrationResult,
    run_anchor_calibration,
)
from whetstone.eval.schema import EvalEvidence

from whetstone_envs.optim.study.power import (
    Stage0Gate,
    Stage0Inputs,
    WithinVarianceCheck,
    evaluate_stage0_gate,
    minimum_detectable_effect,
    nondeterminism_floor,
    within_variance_divergence,
)

if TYPE_CHECKING:
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.candidate import Candidate

    from whetstone_envs.optim.study.spec import StudySpec

__all__ = [
    "ANCHOR_ROLES",
    "AnchorPurpose",
    "EngineBinder",
    "RoleCalibration",
    "Stage0Result",
    "calibrate_role",
    "run_stage0",
]

#: Every role Stage 0 calibrates, in the order the manifest records them.
#: Held-out is last because it is the one the gate reads, and reading it last
#: means the cheaper roles have already proven the procedure works.
ANCHOR_ROLES: tuple[EvalRole, ...] = (
    EvalRole.INTERNAL,
    EvalRole.OFFICIAL,
    EvalRole.HELD_OUT,
)


class AnchorPurpose:
    """Purpose strings recorded on each anchor evaluation.

    These reach persisted evidence metadata, so they are named constants
    rather than inline literals: a renamed purpose would silently orphan the
    calibration records a report cites.
    """

    NAIVE = "stage0-naive-anchor"
    CEILING = "stage0-ceiling-anchor"


class EngineBinder(Protocol):
    """Bind an evaluation engine to one role's split.

    Stage 0 needs three engines over the same experiment, one per role, and
    building them is the caller's concern: it owns the store, the transport,
    and the run boundary. Taking a binder keeps this module free of provider
    wiring and makes the whole stage runnable on a fake transport.
    """

    def __call__(self, *, role: EvalRole, num_seeds: int) -> EvalEngine: ...


@dataclass(frozen=True, slots=True)
class RoleCalibration:
    """One role's calibrated anchor pair and the vectors it produced."""

    role: EvalRole
    calibration: AnchorCalibrationResult
    naive_per_task: tuple[float, ...]
    ceiling_per_task: tuple[float, ...]
    eval_config_hash: str
    task_hashes: tuple[str, ...]

    @property
    def naive_mean(self) -> float:
        return self.calibration.power.naive_mean

    @property
    def ceiling_mean(self) -> float:
        return self.calibration.power.ceiling_mean


def _require_success_evidence(
    evidence: object, *, role: EvalRole, anchor: str
) -> EvalEvidence:
    """Narrow one anchor's evidence to the success case, or say why not."""
    if not isinstance(evidence, EvalEvidence):
        raise TypeError(
            f"the {anchor} anchor on role {role.value!r} produced no "
            f"successful evaluation evidence: {type(evidence).__name__}"
        )
    return evidence


def calibrate_role(  # noqa: PLR0913
    *,
    role: EvalRole,
    bind_engine: EngineBinder,
    naive_candidate: Candidate,
    ceiling_candidate: Candidate,
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    k_cal: int,
    bootstrap_seed: int = 0,
) -> RoleCalibration:
    """Calibrate one role's anchors at ``k_cal`` repeats."""
    engine = bind_engine(role=role, num_seeds=k_cal)
    calibration = run_anchor_calibration(
        engine=engine,
        baseline_candidate=naive_candidate,
        ceiling_candidate=ceiling_candidate,
        baseline_purpose=AnchorPurpose.NAIVE,
        ceiling_purpose=AnchorPurpose.CEILING,
        task_ids=task_ids,
        pool_ceiling=pool_ceiling,
        eval_role=role,
        bootstrap_seed=bootstrap_seed,
    )
    # ``run_anchor_calibration`` already refuses a failed or rejected
    # evaluation, so both anchors carry success evidence by the time it
    # returns. Re-narrowing here rather than casting keeps that guarantee
    # checked: if upstream ever widened what it returns, this fails at the
    # calibration rather than at whatever later read the missing field.
    naive_evidence = _require_success_evidence(
        calibration.baseline.evidence, role=role, anchor="naive"
    )
    ceiling_evidence = _require_success_evidence(
        calibration.ceiling.evidence, role=role, anchor="ceiling"
    )
    return RoleCalibration(
        role=role,
        calibration=calibration,
        naive_per_task=naive_evidence.per_task_values,
        ceiling_per_task=ceiling_evidence.per_task_values,
        eval_config_hash=calibration.eval_config_ref.config_hash,
        task_hashes=naive_evidence.task_hashes,
    )


@dataclass(frozen=True, slots=True)
class Stage0Result:
    """Everything Stage 0 measured, plus the gate's verdict.

    ``gate`` is computed from the held-out calibration alone. The other two
    roles are recorded because the report and the leakage check read them,
    not because they enter the gate.
    """

    k_cal: int
    calibrations: tuple[RoleCalibration, ...]
    inputs: Stage0Inputs
    gate: Stage0Gate
    within_variance: WithinVarianceCheck
    null_b_expected_delta: float

    def by_role(self, role: EvalRole) -> RoleCalibration:
        for calibration in self.calibrations:
            if calibration.role is role:
                return calibration
        raise ValueError(f"stage 0 did not calibrate role {role.value!r}")

    @property
    def held_out(self) -> RoleCalibration:
        return self.by_role(EvalRole.HELD_OUT)

    @property
    def passed(self) -> bool:
        return self.gate.passed

    def mde_at(self, *, n_tasks: int, num_seeds: int) -> float:
        """Re-invert the MDE at another design point.

        The Stage-0 gate permits one adjustment of ``K_REPEAT`` and/or the
        held-out size. This is how that adjustment is priced, using the same
        measured variances rather than re-measuring.
        """
        return minimum_detectable_effect(
            tau_sq=self.inputs.tau_sq,
            sigma_sq=self.inputs.sigma_sq,
            n_tasks=n_tasks,
            num_seeds=num_seeds,
        )


def run_stage0(  # noqa: PLR0913
    *,
    spec: StudySpec,
    bind_engine: EngineBinder,
    naive_candidate: Candidate,
    ceiling_candidate: Candidate,
    task_ids_by_role: dict[EvalRole, tuple[str, ...]],
    pool_ceiling: int,
) -> Stage0Result:
    """Calibrate all three roles and evaluate the Stage-0 gate.

    Every role is calibrated at the same ``k_cal`` and through the same
    procedure, which is what L4's identical-procedure rule needs. The gate
    then reads held-out, because that is the split the MDE is inverted on.
    """
    missing = [
        role.value for role in ANCHOR_ROLES if role not in task_ids_by_role
    ]
    if missing:
        raise ValueError(
            f"stage 0 calibrates every role; missing task ids for {missing}"
        )
    calibrations = tuple(
        calibrate_role(
            role=role,
            bind_engine=bind_engine,
            naive_candidate=naive_candidate,
            ceiling_candidate=ceiling_candidate,
            task_ids=task_ids_by_role[role],
            pool_ceiling=pool_ceiling,
            k_cal=spec.k_cal,
            bootstrap_seed=spec.bootstrap_seed,
        )
        for role in ANCHOR_ROLES
    )
    held_out = next(
        calibration
        for calibration in calibrations
        if calibration.role is EvalRole.HELD_OUT
    )
    decomposition = held_out.calibration.power.decomposition
    inputs = Stage0Inputs(
        naive_mean=held_out.naive_mean,
        ceiling_mean=held_out.ceiling_mean,
        tau_sq=decomposition.interaction_var,
        sigma_sq=decomposition.within_sample_var,
        held_out_size=len(held_out.task_hashes),
        k_repeat=spec.k_repeat,
        k_cal=spec.k_cal,
    )
    return Stage0Result(
        k_cal=spec.k_cal,
        calibrations=calibrations,
        inputs=inputs,
        gate=evaluate_stage0_gate(inputs),
        within_variance=within_variance_divergence(
            naive_per_task=held_out.naive_per_task,
            ceiling_per_task=held_out.ceiling_per_task,
        ),
        # Null-B's expected magnitude, computable here because it depends
        # only on the evaluation's own nondeterminism. The report shows the
        # observed null-B delta against this floor.
        null_b_expected_delta=nondeterminism_floor(
            sigma_sq=decomposition.within_sample_var,
            n_tasks=len(held_out.task_hashes),
            num_seeds=spec.k_repeat,
        ),
    )
