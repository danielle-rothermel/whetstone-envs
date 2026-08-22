"""The Stage-0 gate's arithmetic, in one auditable line.

The study pre-registers exactly one minimum detectable effect:

``MDE(T, K) = (z_{1 - alpha/2} + z_power) * sqrt((tau^2 + 2 sigma^2 / K) / T)``

at ``alpha = 0.05`` and ``power = 0.80``, so the multiplier is
``1.9600 + 0.8416 = 2.8016``. :func:`minimum_detectable_effect` is that line
and nothing else, which is what lets a reader check the gate by hand.

**Why the study computes this itself rather than reading a surface point.**
``whetstone.eval.analysis.analyze_power`` computes the same quantity now, and
its ``VarianceDecomposition`` is where ``tau^2`` and ``sigma^2`` come from --
this module consumes that decomposition rather than re-estimating it. What it
does not do is take the gate number from the power surface: the gate inverts
the MDE at one pre-registered design point, and a gate that reads a grid is a
gate whose design point can drift with the grid's resolution.

**The recorded caveat.** ``_decompose_variance`` estimates the within-sample
variance from the **naive arm only**, as ``base_rate * (1 - base_rate)``. When
naive and ceiling sit at very different base rates -- exactly the regime Stage
0 exists to create -- that estimator is not the pooled per-arm one the design
notes specify. :func:`within_variance_divergence` reports the pooled estimate
alongside so the divergence is visible; a divergence above
:data:`WITHIN_VARIANCE_DIVERGENCE_FLAG` is flagged in the manifest rather than
silently corrected, because changing the decomposition is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whetstone.eval.analysis import VarianceDecomposition

__all__ = [
    "COMPLETENESS_BACKSTOP",
    "COMPLETENESS_RULE",
    "MDE_FORMULA",
    "MDE_MULTIPLIER",
    "SIGNIFICANCE_ALPHA",
    "TARGET_POWER",
    "WITHIN_VARIANCE_DIVERGENCE_FLAG",
    "WORST_CASE_SIGMA_SQ",
    "GateOutcome",
    "Stage0Gate",
    "Stage0Inputs",
    "WithinVarianceCheck",
    "evaluate_stage0_gate",
    "from_decomposition",
    "minimum_detectable_effect",
    "nondeterminism_floor",
    "split_half_stable",
    "weighted_per_task_delta",
    "within_variance_divergence",
]

#: The gate's two-sided significance level and its power.
SIGNIFICANCE_ALPHA = 0.05
TARGET_POWER = 0.80

#: ``z_{1 - alpha/2} + z_power`` at the pre-registered alpha and power.
#: Both quantiles are required: ``z_{1 - alpha/2}`` buys the two-sided
#: significance level and ``z_power`` buys the detection probability. Using
#: ``z_power`` alone yields a one-sided detection threshold, not an MDE, and
#: understates the detectable effect by ``2.8016 / 0.8416 = 3.329``.
MDE_MULTIPLIER = NormalDist().inv_cdf(
    1.0 - SIGNIFICANCE_ALPHA / 2.0
) + NormalDist().inv_cdf(TARGET_POWER)

#: The formula as a manifest string. Persisted verbatim so a report never
#: paraphrases the arithmetic it claims to have used.
MDE_FORMULA = (
    "MDE(T, K) = (z_{1-alpha/2} + z_power) * "
    "sqrt((tau^2 + 2 * sigma^2 / K) / T)"
)

#: The worst-case within-task sampling variance for a binary score, at
#: ``p = 0.5``. The protocol review's MDE table is quoted at this value and
#: the plan's pre-registered MDE row recomputes it here, so a design point a
#: reader authorizes spend against is the most pessimistic one rather than a
#: variance the study has not measured yet. Stage 0 replaces it with the
#: measured ``sigma^2``.
WORST_CASE_SIGMA_SQ = 0.25

#: Relative divergence between the naive-only and pooled within-variance
#: estimates above which the study flags the decomposition caveat.
WITHIN_VARIANCE_DIVERGENCE_FLAG = 0.20

#: The Stage-0 gate's three pre-registered thresholds.
MIN_HEADROOM = 0.20
MAX_NAIVE = 0.60
MIN_CEILING = 0.30
#: The MDE must resolve an effect at most half the available headroom.
MDE_HEADROOM_FRACTION = 0.5


def minimum_detectable_effect(
    *, tau_sq: float, sigma_sq: float, n_tasks: int, num_seeds: int
) -> float:
    """``(z_{1-alpha/2} + z_power) * sqrt((tau^2 + 2 sigma^2 / K) / T)``.

    ``tau_sq`` is the task-by-arm interaction variance and ``sigma_sq`` the
    within-task sampling variance; ``n_tasks`` is T and ``num_seeds`` is K.
    """
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if tau_sq < 0.0 or sigma_sq < 0.0:
        raise ValueError("variances must be non-negative")
    if not math.isfinite(tau_sq) or not math.isfinite(sigma_sq):
        raise ValueError("variances must be finite")
    per_task = tau_sq + 2.0 * sigma_sq / num_seeds
    return MDE_MULTIPLIER * math.sqrt(per_task / n_tasks)


def nondeterminism_floor(
    *, sigma_sq: float, n_tasks: int, num_seeds: int
) -> float:
    """``sqrt(2 sigma^2 / K / T)`` -- null-B's expected delta magnitude.

    This is the paired delta's standard error when the two candidates are
    byte-identical, so the only variation left is evaluation nondeterminism.
    Null-B's observed delta is reported against it: a null-B delta near this
    floor is the pipeline behaving, and one far above it is a finding.
    """
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if sigma_sq < 0.0 or not math.isfinite(sigma_sq):
        raise ValueError("sigma_sq must be finite and non-negative")
    return math.sqrt(2.0 * sigma_sq / num_seeds / n_tasks)


@dataclass(frozen=True, slots=True)
class WithinVarianceCheck:
    """The naive-only within-variance estimate against a pooled one."""

    naive_only: float
    pooled: float
    relative_divergence: float
    flagged: bool


def within_variance_divergence(
    *, naive_per_task: tuple[float, ...], ceiling_per_task: tuple[float, ...]
) -> WithinVarianceCheck:
    """Compare the shipped within-variance estimator against a pooled one.

    ``analyze_power`` uses the naive arm's base rate alone. The pooled
    estimate averages ``p(1-p)`` over both arms' base rates. Reporting both
    makes the caveat a measured number instead of a footnote.
    """
    if not naive_per_task or not ceiling_per_task:
        raise ValueError("within-variance comparison needs both arms")
    naive_rate = sum(naive_per_task) / len(naive_per_task)
    ceiling_rate = sum(ceiling_per_task) / len(ceiling_per_task)
    naive_only = naive_rate * (1.0 - naive_rate)
    pooled = (
        naive_rate * (1.0 - naive_rate) + ceiling_rate * (1.0 - ceiling_rate)
    ) / 2.0
    # Divergence is relative to the pooled estimate: it is the reference the
    # design notes specify, so "how wrong is the shipped one" is measured
    # against it rather than against itself.
    divergence = abs(naive_only - pooled) / pooled if pooled > 0.0 else 0.0
    return WithinVarianceCheck(
        naive_only=naive_only,
        pooled=pooled,
        relative_divergence=divergence,
        flagged=divergence > WITHIN_VARIANCE_DIVERGENCE_FLAG,
    )


def split_half_stable(
    first_half: tuple[float, ...],
    second_half: tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    """Whether an even ``K_CAL``'s two halves agree within ``tolerance``.

    This is the doubling rule's stopping check: the calibration is stable
    when the mean measured on the first half of the repeats and the mean on
    the second differ by no more than ``tolerance``. Unstable means double
    ``K_CAL`` and re-measure.
    """
    if not first_half or not second_half:
        raise ValueError("a split-half check needs both halves")
    if len(first_half) != len(second_half):
        raise ValueError("split-half comparison requires equal halves")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    first = sum(first_half) / len(first_half)
    second = sum(second_half) / len(second_half)
    return abs(first - second) <= tolerance


@dataclass(frozen=True, slots=True)
class Stage0Inputs:
    """What Stage 0 measured, as the gate reads it.

    ``tau_sq`` and ``sigma_sq`` come from ``analyze_power``'s
    ``VarianceDecomposition`` on the **held-out** anchors, because held-out
    is the split whose MDE the gate inverts.
    """

    naive_mean: float
    ceiling_mean: float
    tau_sq: float
    sigma_sq: float
    held_out_size: int
    k_repeat: int
    k_cal: int

    def __post_init__(self) -> None:
        if self.held_out_size < 1:
            raise ValueError("held_out_size must be at least 1")
        if self.k_repeat < 1:
            raise ValueError("k_repeat must be at least 1")
        if self.k_cal < 1:
            raise ValueError("k_cal must be at least 1")

    @property
    def headroom(self) -> float:
        """``ceiling - naive`` on held-out, floored at zero."""
        return max(0.0, self.ceiling_mean - self.naive_mean)


def from_decomposition(  # noqa: PLR0913
    decomposition: VarianceDecomposition,
    *,
    naive_mean: float,
    ceiling_mean: float,
    held_out_size: int,
    k_repeat: int,
    k_cal: int,
) -> Stage0Inputs:
    """Read the gate's variance inputs off a power analysis.

    ``interaction_var`` is ``tau^2`` and ``within_sample_var`` is
    ``sigma^2``; the mapping is stated here once so no caller has to
    remember which field is which.
    """
    return Stage0Inputs(
        naive_mean=naive_mean,
        ceiling_mean=ceiling_mean,
        tau_sq=decomposition.interaction_var,
        sigma_sq=decomposition.within_sample_var,
        held_out_size=held_out_size,
        k_repeat=k_repeat,
        k_cal=k_cal,
    )


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """One gate condition and whether it held."""

    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str


@dataclass(frozen=True, slots=True)
class Stage0Gate:
    """The Stage-0 gate's verdict and every condition behind it."""

    passed: bool
    mde_measured: float
    headroom: float
    outcomes: tuple[GateOutcome, ...]

    def failures(self) -> tuple[GateOutcome, ...]:
        return tuple(
            outcome for outcome in self.outcomes if not outcome.passed
        )


def evaluate_stage0_gate(inputs: Stage0Inputs) -> Stage0Gate:
    """Decide whether the design is powered enough to spend on optimizers.

    All four conditions must hold: enough headroom to move, a naive anchor
    that is not already good, a ceiling that is not at the floor, and an MDE
    that resolves at most half the headroom. A failure is not a retry -- it
    is one permitted adjustment of ``K_REPEAT`` and/or held-out size, then a
    recompute, then a stop.
    """
    headroom = inputs.headroom
    mde = minimum_detectable_effect(
        tau_sq=inputs.tau_sq,
        sigma_sq=inputs.sigma_sq,
        n_tasks=inputs.held_out_size,
        num_seeds=inputs.k_repeat,
    )
    mde_threshold = headroom * MDE_HEADROOM_FRACTION
    outcomes = (
        GateOutcome(
            name="headroom",
            passed=headroom >= MIN_HEADROOM,
            observed=headroom,
            threshold=MIN_HEADROOM,
            detail=(
                f"held-out ceiling {inputs.ceiling_mean:.4f} minus naive "
                f"{inputs.naive_mean:.4f} leaves {headroom:.4f} to move"
            ),
        ),
        GateOutcome(
            name="naive_not_saturated",
            passed=inputs.naive_mean <= MAX_NAIVE,
            observed=inputs.naive_mean,
            threshold=MAX_NAIVE,
            detail=(
                f"naive anchor scores {inputs.naive_mean:.4f}; above "
                f"{MAX_NAIVE} there is little left for an optimizer to win"
            ),
        ),
        GateOutcome(
            name="ceiling_not_floored",
            passed=inputs.ceiling_mean >= MIN_CEILING,
            observed=inputs.ceiling_mean,
            threshold=MIN_CEILING,
            detail=(
                f"ceiling anchor scores {inputs.ceiling_mean:.4f}; below "
                f"{MIN_CEILING} the task is too hard to show an effect"
            ),
        ),
        GateOutcome(
            name="mde_resolves_headroom",
            passed=mde <= mde_threshold,
            observed=mde,
            threshold=mde_threshold,
            detail=(
                f"MDE({inputs.held_out_size}, {inputs.k_repeat}) = "
                f"{mde:.4f} against half the headroom {mde_threshold:.4f}"
            ),
        ),
    )
    return Stage0Gate(
        passed=all(outcome.passed for outcome in outcomes),
        mde_measured=mde,
        headroom=headroom,
        outcomes=outcomes,
    )


#: The completeness backstop (O7). Below this fraction of achieved rows an
#: arm is reported incomplete and its interval is not claimed.
COMPLETENESS_BACKSTOP = 0.90

#: How a task's achieved sample count enters the estimate. Persisted, so the
#: report states the rule it applied rather than describing one.
COMPLETENESS_RULE = "achieved-count-weighted-per-task-delta"


def weighted_per_task_delta(
    *,
    arm_per_task: tuple[float, ...],
    naive_per_task: tuple[float, ...],
    achieved_counts: tuple[int, ...],
    planned_count: int,
) -> tuple[tuple[float, ...], float]:
    """Weight each task's delta by the rows it actually achieved.

    Returns the weighted per-task delta vector and the achieved
    completeness. The weighting flows into the bootstrap's per-task vector
    rather than only into a variance estimate, so a ragged cell shrinks its
    own contribution to the point estimate as well as to the interval.

    A task with zero achieved rows contributes a zero-weighted delta rather
    than being dropped: dropping it would silently shrink ``T`` and make the
    interval look tighter than the data supports.
    """
    if not arm_per_task:
        raise ValueError("a delta needs at least one task")
    if not len(arm_per_task) == len(naive_per_task) == len(achieved_counts):
        raise ValueError("per-task vectors and counts must be aligned")
    if planned_count < 1:
        raise ValueError("planned_count must be at least 1")
    if any(count < 0 for count in achieved_counts):
        raise ValueError("achieved counts must be non-negative")
    if any(count > planned_count for count in achieved_counts):
        raise ValueError("a task cannot achieve more rows than planned")
    weighted = tuple(
        (arm - naive) * (count / planned_count)
        for arm, naive, count in zip(
            arm_per_task, naive_per_task, achieved_counts, strict=True
        )
    )
    completeness = sum(achieved_counts) / (
        planned_count * len(achieved_counts)
    )
    return weighted, completeness
