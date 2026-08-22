"""The Stage-0 gate arithmetic, pinned against the review's recomputation."""

from __future__ import annotations

import math

import pytest

from whetstone_envs.optim.study.power import (
    COMPLETENESS_BACKSTOP,
    MDE_MULTIPLIER,
    WORST_CASE_SIGMA_SQ,
    Stage0Inputs,
    evaluate_stage0_gate,
    minimum_detectable_effect,
    nondeterminism_floor,
    split_half_stable,
    weighted_per_task_delta,
    within_variance_divergence,
)


def test_the_worst_case_variance_is_pinned() -> None:
    """The review's table and the plan's MDE row read one constant.

    Two copies could drift, and the plan's pre-registered MDE would then
    quote a number the recomputed table never produced.
    """
    assert WORST_CASE_SIGMA_SQ == 0.25


def test_mde_multiplier_is_two_sided_plus_power() -> None:
    """``z_{0.975} + z_{0.80}``, not ``z_{0.80}`` alone.

    Using the power quantile alone is a one-sided detection threshold and
    understates the detectable effect by 3.329x -- in the optimistic
    direction, which would make the gate look passable when it is not.
    """
    assert pytest.approx(2.8016, abs=1e-4) == MDE_MULTIPLIER


@pytest.mark.parametrize(
    ("n_tasks", "num_seeds", "tau_sq", "expected"),
    [
        (220, 3, 0.05, 0.0879),
        (220, 3, 0.10, 0.0975),
        (220, 3, 0.15, 0.1063),
        (220, 5, 0.05, 0.0732),
        (440, 3, 0.05, 0.0622),
        (440, 3, 0.10, 0.0690),
        (132, 3, 0.05, 0.1135),
        # The c18 second family at its own smaller held-out split.
        (48, 3, 0.05, 0.1882),
    ],
)
def test_mde_matches_recomputed_design_points(
    n_tasks: int, num_seeds: int, tau_sq: float, expected: float
) -> None:
    """Every design point the protocol review recomputed by hand."""
    assert minimum_detectable_effect(
        tau_sq=tau_sq,
        sigma_sq=WORST_CASE_SIGMA_SQ,
        n_tasks=n_tasks,
        num_seeds=num_seeds,
    ) == pytest.approx(expected, abs=5e-5)


def test_mde_shrinks_with_more_tasks_and_more_repeats() -> None:
    base = minimum_detectable_effect(
        tau_sq=0.05, sigma_sq=0.25, n_tasks=220, num_seeds=3
    )
    assert (
        minimum_detectable_effect(
            tau_sq=0.05, sigma_sq=0.25, n_tasks=440, num_seeds=3
        )
        < base
    )
    assert (
        minimum_detectable_effect(
            tau_sq=0.05, sigma_sq=0.25, n_tasks=220, num_seeds=5
        )
        < base
    )


def test_mde_refuses_impossible_designs() -> None:
    with pytest.raises(ValueError, match="n_tasks"):
        minimum_detectable_effect(
            tau_sq=0.05, sigma_sq=0.25, n_tasks=0, num_seeds=3
        )
    with pytest.raises(ValueError, match="num_seeds"):
        minimum_detectable_effect(
            tau_sq=0.05, sigma_sq=0.25, n_tasks=220, num_seeds=0
        )
    with pytest.raises(ValueError, match="non-negative"):
        minimum_detectable_effect(
            tau_sq=-0.01, sigma_sq=0.25, n_tasks=220, num_seeds=3
        )


def test_nondeterminism_floor_is_the_paired_se_without_interaction() -> None:
    """Null-B's expected delta is the MDE's own SE at ``tau^2 = 0``."""
    floor = nondeterminism_floor(sigma_sq=0.25, n_tasks=220, num_seeds=3)
    assert floor == pytest.approx(math.sqrt(2 * 0.25 / 3 / 220), abs=1e-12)
    mde_at_zero_tau = minimum_detectable_effect(
        tau_sq=0.0, sigma_sq=0.25, n_tasks=220, num_seeds=3
    )
    assert mde_at_zero_tau == pytest.approx(MDE_MULTIPLIER * floor, abs=1e-12)


def _inputs(**overrides: float | int) -> Stage0Inputs:
    """A design that passes every gate condition, for targeted mutation."""
    base: dict[str, float | int] = {
        "naive_mean": 0.25,
        "ceiling_mean": 0.75,
        "tau_sq": 0.01,
        "sigma_sq": 0.05,
        "held_out_size": 440,
        "k_repeat": 5,
        "k_cal": 4,
    }
    base.update(overrides)
    return Stage0Inputs(**base)  # type: ignore[arg-type]


def test_gate_passes_a_well_powered_design() -> None:
    gate = evaluate_stage0_gate(_inputs())
    assert gate.passed
    assert gate.failures() == ()
    assert gate.headroom == pytest.approx(0.50)


def test_gate_fails_on_insufficient_headroom() -> None:
    gate = evaluate_stage0_gate(_inputs(ceiling_mean=0.35))
    assert not gate.passed
    assert {failure.name for failure in gate.failures()} == {"headroom"}


def test_gate_fails_on_a_saturated_naive_anchor() -> None:
    """A naive anchor above 0.60 leaves too little for an optimizer to win."""
    gate = evaluate_stage0_gate(_inputs(naive_mean=0.70, ceiling_mean=0.95))
    assert not gate.passed
    assert "naive_not_saturated" in {
        failure.name for failure in gate.failures()
    }


def test_gate_fails_on_a_floored_ceiling() -> None:
    gate = evaluate_stage0_gate(_inputs(naive_mean=0.02, ceiling_mean=0.25))
    assert not gate.passed
    assert "ceiling_not_floored" in {
        failure.name for failure in gate.failures()
    }


def test_gate_fails_when_the_mde_cannot_resolve_half_the_headroom() -> None:
    """The condition that stops the study before optimizer spend."""
    gate = evaluate_stage0_gate(
        _inputs(
            naive_mean=0.25,
            ceiling_mean=0.47,
            tau_sq=0.20,
            sigma_sq=0.25,
            held_out_size=132,
            k_repeat=3,
        )
    )
    assert not gate.passed
    assert "mde_resolves_headroom" in {
        failure.name for failure in gate.failures()
    }
    # The measured MDE is reported whether or not the gate passed, because
    # the one permitted adjustment is priced against it.
    assert gate.mde_measured > gate.headroom / 2


def test_gate_reports_every_condition_even_when_it_passes() -> None:
    gate = evaluate_stage0_gate(_inputs())
    assert {outcome.name for outcome in gate.outcomes} == {
        "headroom",
        "naive_not_saturated",
        "ceiling_not_floored",
        "mde_resolves_headroom",
    }


def test_within_variance_divergence_flags_diverging_base_rates() -> None:
    """The recorded caveat: within is estimated from the naive arm alone."""
    naive = (0.0,) * 18 + (1.0,) * 2
    ceiling = (0.0,) * 10 + (1.0,) * 10
    check = within_variance_divergence(
        naive_per_task=naive, ceiling_per_task=ceiling
    )
    assert check.naive_only == pytest.approx(0.1 * 0.9)
    assert check.pooled == pytest.approx((0.1 * 0.9 + 0.25) / 2)
    assert check.flagged


def test_within_variance_divergence_is_quiet_when_arms_agree() -> None:
    scores = (0.0,) * 10 + (1.0,) * 10
    check = within_variance_divergence(
        naive_per_task=scores, ceiling_per_task=scores
    )
    assert check.relative_divergence == pytest.approx(0.0)
    assert not check.flagged


def test_split_half_stability_drives_the_doubling_rule() -> None:
    assert split_half_stable((0.5, 0.5), (0.5, 0.5), tolerance=0.01)
    assert not split_half_stable((0.9, 0.9), (0.1, 0.1), tolerance=0.05)


def test_split_half_refuses_uneven_halves() -> None:
    with pytest.raises(ValueError, match="equal halves"):
        split_half_stable((0.5,), (0.5, 0.5), tolerance=0.1)


def test_weighted_delta_shrinks_a_ragged_task() -> None:
    """A task that achieved half its rows contributes half its delta."""
    weighted, completeness = weighted_per_task_delta(
        arm_per_task=(1.0, 1.0),
        naive_per_task=(0.0, 0.0),
        achieved_counts=(4, 2),
        planned_count=4,
    )
    assert weighted == (1.0, 0.5)
    assert completeness == pytest.approx(0.75)


def test_weighted_delta_keeps_a_fully_missing_task_in_the_vector() -> None:
    """Dropping it would shrink T and tighten the interval dishonestly."""
    weighted, completeness = weighted_per_task_delta(
        arm_per_task=(1.0, 1.0),
        naive_per_task=(0.0, 0.0),
        achieved_counts=(4, 0),
        planned_count=4,
    )
    assert len(weighted) == 2
    assert weighted[1] == 0.0
    assert completeness == pytest.approx(0.5)
    assert completeness < COMPLETENESS_BACKSTOP


def test_weighted_delta_refuses_impossible_accounting() -> None:
    with pytest.raises(ValueError, match="cannot achieve more rows"):
        weighted_per_task_delta(
            arm_per_task=(1.0,),
            naive_per_task=(0.0,),
            achieved_counts=(5,),
            planned_count=4,
        )
    with pytest.raises(ValueError, match="aligned"):
        weighted_per_task_delta(
            arm_per_task=(1.0, 1.0),
            naive_per_task=(0.0,),
            achieved_counts=(4, 4),
            planned_count=4,
        )
