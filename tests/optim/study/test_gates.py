"""The optimizer-side call estimates and the constants behind them.

The constants are pinned as written-out numbers, not recomputed: every one
of them was wrong at least once in the protocol drafts, and a test that
re-derives a number from the same code that produced it proves nothing. F10
in particular replaced a wrong bound (72) with a correct range (28-616), so
the numbers are the contract here.
"""

from __future__ import annotations

import pytest

from whetstone_envs.optim.study.gates import (
    GEPA_MAX_METRIC_CALLS_PINNED,
    GEPA_MEASURED_FULL_VALSET_PASSES,
    GEPA_MEASURED_REFLECTION_MINIBATCHES,
    GEPA_MEASURED_TASK_CALLS_AT_PIN,
    GEPA_PIN_REASON,
    GEPA_REFLECTION_MINIBATCH_TASKS,
    GEPA_RESOLVED_MAX_METRIC_CALLS,
    GEPA_TASK_CALL_CEILING,
    MEASURED_FANOUT_RATIO,
    MEASURED_GEPA_DISTINCT_EVALUATIONS,
    MEASURED_GEPA_RESULT_JSON_BYTES,
    MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES,
    MEASURED_GEPA_SQLITE_BYTES,
    MEASURED_GEPA_STEPS,
    MEASURED_GEPA_TASK_CALLS,
    MEASURED_GEPA_WALL_SECONDS,
    MEASURED_MIPROV2_BOOTSTRAP_ROWS_FEWSHOT,
    MEASURED_MIPROV2_BOOTSTRAP_ROWS_GROUND_ONLY,
    MEASURED_MIPROV2_BOOTSTRAP_ROWS_ZEROSHOT,
    MEASURED_MIPROV2_FEWSHOT_TASK_CALLS,
    MEASURED_MIPROV2_GROUND_ONLY_TASK_CALLS,
    MEASURED_MIPROV2_MINIBATCH_TASKS,
    MEASURED_MIPROV2_TRAINSET_TASKS,
    MEASURED_MIPROV2_ZEROSHOT_TASK_CALLS,
    MEASUREMENT_N_PER_STRATUM,
    MEASUREMENT_POOL_SEED_START,
    MEASUREMENT_SPLIT_SIZES,
    MIPROV2_BOOTSTRAP_ROWS_BEST_CASE,
    MIPROV2_BOOTSTRAP_ROWS_WORST_CASE,
    MIPROV2_BOOTSTRAPPING_PLANS,
    MIPROV2_FEWSHOT_TASK_CALL_CEILING,
    MIPROV2_FEWSHOT_TASK_CALL_FLOOR,
    MIPROV2_FULL_EVAL_CALLS,
    MIPROV2_MINIBATCH_CALLS,
    NULL_IDENTITY_HELD_OUT_PASSES,
    NULL_IDENTITY_OFFICIAL_PASSES,
    STAGE1_CALL_COUNT_TOLERANCE,
    estimate_optimizer_calls,
    null_identity_report_rows,
)
from whetstone_envs.optim.study.spec import CODEX_EVALUATE_CALL_CAP
from whetstone_envs.optim.study.stages import call_count_within_estimate

# --------------------------------------------------------------------------
# Pinned constants
# --------------------------------------------------------------------------


def test_the_f10_bootstrap_bound_is_the_corrected_one() -> None:
    """28-616, never the protocol's original 72."""
    assert MIPROV2_BOOTSTRAPPING_PLANS == 7
    assert MIPROV2_BOOTSTRAP_ROWS_BEST_CASE == 28
    assert MIPROV2_BOOTSTRAP_ROWS_WORST_CASE == 616
    assert MIPROV2_BOOTSTRAP_ROWS_BEST_CASE != 72
    assert MIPROV2_BOOTSTRAP_ROWS_WORST_CASE != 72


def test_the_fewshot_range_is_pinned_at_1870_to_2458() -> None:
    """The Stage-1 gate divides by the ceiling, so both ends are pinned."""
    assert MIPROV2_FULL_EVAL_CALLS == 1_050
    assert MIPROV2_MINIBATCH_CALLS == 792
    assert MIPROV2_FEWSHOT_TASK_CALL_FLOOR == 1_870
    assert MIPROV2_FEWSHOT_TASK_CALL_CEILING == 2_458


def test_the_gepa_and_codex_constants_are_pinned() -> None:
    assert GEPA_RESOLVED_MAX_METRIC_CALLS == 732
    assert CODEX_EVALUATE_CALL_CAP == 8
    assert STAGE1_CALL_COUNT_TOLERANCE == 1.5


def test_the_codex_cap_has_exactly_one_owner() -> None:
    """D2's cap is a design decision, so ``spec`` owns it and gates reads
    it rather than spelling 8 twice."""
    from whetstone_envs.optim.study import gates

    assert gates.CODEX_EVALUATE_CALL_CAP is CODEX_EVALUATE_CALL_CAP
    assert "CODEX_EVALUATE_CALL_CAP" not in gates.__all__


# --------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------


def test_copro_is_derived_from_its_configured_search_shape() -> None:
    estimate = estimate_optimizer_calls("copro", internal_size=88, k_repeat=3)
    # (depth 1 + 1) steps x breadth 2 x 88 tasks x 3 repeats.
    assert estimate.low == estimate.high == 2 * 2 * 88 * 3
    assert "breadth 2" in estimate.basis
    assert estimate.gated


def test_null_random_evaluates_exactly_as_copro_does() -> None:
    """Null-A perturbs the anchor but drives the same search machinery."""
    null = estimate_optimizer_calls(
        "null-random", internal_size=88, k_repeat=3
    )
    copro = estimate_optimizer_calls("copro", internal_size=88, k_repeat=3)
    assert (null.low, null.high) == (copro.low, copro.high)


def test_null_identity_is_the_report_harness_not_an_optimizer_run() -> None:
    """Null-B runs no search, so COPRO's shape was never its cost.

    ``StudyOptimizerRunner._run_null`` never calls ``run_optimizer``: it
    emits the naive anchor unchanged and reports ``observed_task_calls=0``.
    What null-B costs is the report harness every arm pays.
    """
    null = estimate_optimizer_calls(
        "null-identity",
        internal_size=88,
        k_repeat=3,
        official_size=132,
        held_out_size=220,
    )
    # One official pass over 132 and one held-out pass over 220, at K=3.
    assert null.low == null.high == 3 * (132 + 220) == 1_056
    assert null.gated
    assert "no optimizer run" in null.basis
    # And it does not track the internal split, which it never evaluates.
    wider = estimate_optimizer_calls(
        "null-identity",
        internal_size=8_800,
        k_repeat=3,
        official_size=132,
        held_out_size=220,
    )
    assert wider.low == null.low


def test_null_identity_no_longer_borrows_copros_search_shape() -> None:
    """The regression this replaces: null-B priced as a COPRO run.

    Split sizes where the two formulas genuinely differ, because at the
    protocol's own ``(88, 132, 220)`` at ``K_REPEAT = 3`` they coincide at
    1,056 by arithmetic accident.
    """
    null = estimate_optimizer_calls(
        "null-identity",
        internal_size=88,
        k_repeat=3,
        official_size=10,
        held_out_size=20,
    )
    copro = estimate_optimizer_calls("copro", internal_size=88, k_repeat=3)
    assert null.low == 3 * (10 + 20) == 90
    assert null.low != copro.low


def test_the_null_identity_harness_formula_is_one_pass_each() -> None:
    assert NULL_IDENTITY_OFFICIAL_PASSES == 1
    assert NULL_IDENTITY_HELD_OUT_PASSES == 1
    assert (
        null_identity_report_rows(
            official_size=132, held_out_size=220, k_repeat=3
        )
        == 1_056
    )


@pytest.mark.parametrize(
    ("official", "held_out", "k_repeat"),
    [(-1, 1, 1), (1, -1, 1), (1, 1, 0)],
)
def test_the_harness_formula_refuses_impossible_inputs(
    official: int, held_out: int, k_repeat: int
) -> None:
    with pytest.raises(ValueError, match="are required"):
        null_identity_report_rows(
            official_size=official,
            held_out_size=held_out,
            k_repeat=k_repeat,
        )


def test_miprov2_carries_its_own_budget_not_the_splits() -> None:
    """MIPROv2's volume is its control's, so the splits do not move it."""
    small = estimate_optimizer_calls("miprov2", internal_size=8, k_repeat=1)
    large = estimate_optimizer_calls("miprov2", internal_size=88, k_repeat=3)
    assert (small.low, small.high) == (large.low, large.high)
    assert large.high == MIPROV2_FEWSHOT_TASK_CALL_CEILING


def test_gepa_is_estimated_in_task_rows_at_the_pinned_budget() -> None:
    """The unit fix: rows bounded by the pin, not the 732 auto budget.

    Comparing a row count against a metric-call ceiling is what made a real
    GEPA run trip the 1.5x gate, so the estimate is denominated in the same
    unit ``observed_task_calls`` carries.
    """
    estimate = estimate_optimizer_calls("gepa", internal_size=88, k_repeat=3)
    assert estimate.low == estimate.high == GEPA_TASK_CALL_CEILING == 200
    assert estimate.high != GEPA_RESOLVED_MAX_METRIC_CALLS
    assert "metric calls bounds task rows" in estimate.basis


def test_codex_is_a_cap_and_is_not_gated() -> None:
    """OQ3: a bug-detector gate over a non-deterministic agent false-aborts."""
    estimate = estimate_optimizer_calls("codex", internal_size=88, k_repeat=3)
    assert not estimate.gated
    assert estimate.low == 0
    assert estimate.high == CODEX_EVALUATE_CALL_CAP * 88 * 3


def test_an_unrecognised_optimizer_has_no_estimate() -> None:
    with pytest.raises(ValueError, match="no call estimate"):
        estimate_optimizer_calls("nope", internal_size=88, k_repeat=3)


def test_every_estimate_states_its_derivation() -> None:
    for optimizer in ("copro", "miprov2", "gepa", "codex"):
        estimate = estimate_optimizer_calls(
            optimizer, internal_size=88, k_repeat=3
        )
        assert estimate.basis.strip()


# --------------------------------------------------------------------------
# The Stage-1 comparison
# --------------------------------------------------------------------------


def test_a_low_accuracy_anchor_does_not_read_as_a_budget_overrun() -> None:
    """F10's point: the worst case is the likely one, so it is the gate."""
    assert call_count_within_estimate(
        optimizer="miprov2",
        observed_task_calls=MIPROV2_FEWSHOT_TASK_CALL_CEILING,
        internal_size=88,
        k_repeat=3,
    )
    # And the old, wrong denominator would have failed exactly that run.
    assert not call_count_within_estimate(
        optimizer="miprov2",
        observed_task_calls=int(
            MIPROV2_FEWSHOT_TASK_CALL_CEILING * STAGE1_CALL_COUNT_TOLERANCE
        )
        + 1,
        internal_size=88,
        k_repeat=3,
    )


def test_codex_always_passes_the_call_count_comparison() -> None:
    """It is gated on capacity respect and audit pass, not on agreement."""
    assert call_count_within_estimate(
        optimizer="codex",
        observed_task_calls=10**6,
        internal_size=88,
        k_repeat=3,
    )


def test_a_fanned_out_gepa_run_trips_the_gate() -> None:
    """The gate exists to catch a fan-out bug, so it must catch one."""
    assert not call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=GEPA_TASK_CALL_CEILING * 3,
        internal_size=88,
        k_repeat=3,
    )


# --------------------------------------------------------------------------
# Wave 3 measurements
# --------------------------------------------------------------------------
# These are pinned as literals for the same reason the estimates are: a
# measurement re-derived from the code that produced it proves nothing. The
# provenance for every one of them is in ``gates.py``'s own comments -- which
# run, which splits, which control settings.


def test_the_measurement_provenance_is_the_protocol_splits() -> None:
    """A measurement at other splits would not answer the question asked."""
    assert MEASUREMENT_SPLIT_SIZES == (88, 132, 220)
    assert MEASUREMENT_N_PER_STRATUM == 32
    assert MEASUREMENT_POOL_SEED_START == 1_000_000
    # 22 c19 strata at 32 each is 704 tasks, enough for 88 + 132 + 220.
    assert sum(MEASUREMENT_SPLIT_SIZES) <= MEASUREMENT_N_PER_STRATUM * 22


def test_r6_is_retired_by_a_measured_ratio_of_one() -> None:
    """**The F16 finding.** No fan-out, so R6 does not gate Stage 1.

    The feared multiplier was 88 / 35 = 2.51x. The measured one is 1.0.
    """
    assert MEASURED_FANOUT_RATIO == 1.0
    assert MEASURED_FANOUT_RATIO < 88 / 35


def test_the_measured_miprov2_call_counts_are_pinned() -> None:
    assert MEASURED_MIPROV2_FEWSHOT_TASK_CALLS == 245
    assert MEASURED_MIPROV2_ZEROSHOT_TASK_CALLS == 246
    assert MEASURED_MIPROV2_GROUND_ONLY_TASK_CALLS == 245
    assert MEASURED_MIPROV2_MINIBATCH_TASKS == 35


def test_the_measured_calls_sit_far_inside_the_stage1_gate() -> None:
    """The gate's denominator is a loose upper bound, as intended.

    A loose bound cannot false-abort a run, which is what F10 was worried
    about -- it just turns out to be far looser than F10 expected, because
    the bootstrap term is negligible rather than dominant.
    """
    for observed in (
        MEASURED_MIPROV2_FEWSHOT_TASK_CALLS,
        MEASURED_MIPROV2_ZEROSHOT_TASK_CALLS,
        MEASURED_MIPROV2_GROUND_ONLY_TASK_CALLS,
    ):
        assert observed < MIPROV2_FEWSHOT_TASK_CALL_FLOOR
        assert call_count_within_estimate(
            optimizer="miprov2",
            observed_task_calls=observed,
            internal_size=88,
            k_repeat=1,
        )


def test_the_measured_bootstrap_rows_are_not_the_protocol_bound() -> None:
    """F10's 28-616 does not apply: the trainset is one task.

    ``build_miprov2_control`` slices ``trainset=task_hashes[:1]`` at every
    split size, so bootstrapping walks a single task no matter how large
    the internal split is.
    """
    assert MEASURED_MIPROV2_TRAINSET_TASKS == 1
    assert MEASURED_MIPROV2_BOOTSTRAP_ROWS_FEWSHOT == 1
    assert MEASURED_MIPROV2_BOOTSTRAP_ROWS_GROUND_ONLY == 1
    # Zeroshot emits no LABELS_ONLY plan, so it carries one more
    # bootstrapping plan than the other two modes -- and one more row.
    assert MEASURED_MIPROV2_BOOTSTRAP_ROWS_ZEROSHOT == 2
    for measured in (
        MEASURED_MIPROV2_BOOTSTRAP_ROWS_FEWSHOT,
        MEASURED_MIPROV2_BOOTSTRAP_ROWS_ZEROSHOT,
        MEASURED_MIPROV2_BOOTSTRAP_ROWS_GROUND_ONLY,
    ):
        assert measured < MIPROV2_BOOTSTRAP_ROWS_BEST_CASE


def test_the_measured_gepa_sizing_is_pinned() -> None:
    """556 steps consumed the 732-call budget; steps are not calls."""
    assert MEASURED_GEPA_STEPS == 556
    assert MEASURED_GEPA_STEPS < GEPA_RESOLVED_MAX_METRIC_CALLS
    assert MEASURED_GEPA_WALL_SECONDS == 1_329
    assert MEASURED_GEPA_TASK_CALLS == 265
    assert MEASURED_GEPA_DISTINCT_EVALUATIONS == 91


def test_gepa_search_evidence_is_quadratic_in_the_step_count() -> None:
    """The F9 root cause, pinned as the number that shows it.

    Every step persists its whole replayed prefix as search evidence, so
    the entry count is about ``steps^2 / 2`` while the distinct evaluations
    behind it stay flat. That ratio is why the artifacts are three orders
    of magnitude larger than the work performed.
    """
    assert MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES == 155_956
    quadratic = MEASURED_GEPA_STEPS**2 / 2
    assert (
        0.9 * quadratic
        < MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES
        < (1.2 * quadratic)
    )
    assert (
        MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES
        > 1_000 * MEASURED_GEPA_DISTINCT_EVALUATIONS
    )


def test_d3_pins_gepa_to_the_pre_registered_fallback() -> None:
    """**The D3 decision.** 200, the value registered before measuring.

    Registering the fallback in advance is what makes this a
    pre-registration rather than a number chosen to fit the measurement.
    """
    assert GEPA_MAX_METRIC_CALLS_PINNED == 200
    assert GEPA_MAX_METRIC_CALLS_PINNED < GEPA_RESOLVED_MAX_METRIC_CALLS
    assert GEPA_PIN_REASON.strip()


def test_the_gepa_store_blew_the_sizing_bar() -> None:
    """1.73 GB against a ~1 GB bar is what triggered the pin."""
    one_gigabyte = 1_000_000_000
    assert one_gigabyte < MEASURED_GEPA_SQLITE_BYTES
    assert 0.7 * one_gigabyte < MEASURED_GEPA_RESULT_JSON_BYTES
    # And the wall time did *not* trigger it: under the 60-minute bar.
    assert MEASURED_GEPA_WALL_SECONDS < 60 * 60


# --------------------------------------------------------------------------
# The unit correction: metric calls are not task calls
# --------------------------------------------------------------------------


def test_the_measured_gepa_run_decomposes_into_its_two_eval_shapes() -> None:
    """What makes the unit relation a derivation, not a fitted ratio.

    ``build_gepa_control`` sets ``reflection_minibatch_size=1`` and passes
    ``valset_task_hashes=None``, so a run of this control produces exactly
    two evaluation shapes: a full pass over the 88-task internal split, and
    a one-task reflection minibatch. The measured 91 evaluations and 265
    rows have exactly one decomposition into those shapes.
    """
    valset = 88
    assert GEPA_MEASURED_FULL_VALSET_PASSES == 2
    assert GEPA_MEASURED_REFLECTION_MINIBATCHES == 89
    assert GEPA_REFLECTION_MINIBATCH_TASKS == 1
    assert (
        GEPA_MEASURED_FULL_VALSET_PASSES + GEPA_MEASURED_REFLECTION_MINIBATCHES
        == MEASURED_GEPA_DISTINCT_EVALUATIONS
        == 91
    )
    assert (
        GEPA_MEASURED_FULL_VALSET_PASSES * valset
        + GEPA_MEASURED_REFLECTION_MINIBATCHES
        * GEPA_REFLECTION_MINIBATCH_TASKS
        == MEASURED_GEPA_TASK_CALLS
        == 265
    )


def test_metric_calls_and_task_rows_are_different_units() -> None:
    """The finding: 732 metric calls bought 265 rows, a 2.76x replay factor.

    Every distinct row costs at least one metric call, so the budget is a
    sound upper bound on rows -- and a loose one, which is the property the
    Stage-1 gate needs.
    """
    assert MEASURED_GEPA_TASK_CALLS < GEPA_RESOLVED_MAX_METRIC_CALLS
    replay_factor = GEPA_RESOLVED_MAX_METRIC_CALLS / MEASURED_GEPA_TASK_CALLS
    assert replay_factor == pytest.approx(2.76, abs=0.01)


def test_the_gepa_ceiling_is_sourced_from_the_d3_pin() -> None:
    """D3 pinned 200, so 200 is what the gate bounds -- not the retired 732."""
    assert GEPA_TASK_CALL_CEILING == GEPA_MAX_METRIC_CALLS_PINNED == 200
    assert GEPA_TASK_CALL_CEILING != GEPA_RESOLVED_MAX_METRIC_CALLS


def test_the_measured_run_scaled_to_the_pin_is_inside_the_ceiling() -> None:
    """The cross-check: an upper bound must sit above the scaled measurement.

    265 rows at 732 charged calls scales to 72.4 rows at the pinned 200,
    rounded up to 73.
    """
    assert GEPA_MEASURED_TASK_CALLS_AT_PIN == 73
    assert GEPA_MEASURED_TASK_CALLS_AT_PIN < GEPA_TASK_CALL_CEILING


def test_the_wave3_gepa_run_scaled_to_the_pin_passes_the_gate() -> None:
    """**The bug this fixes.** A real GEPA run must not trip the gate.

    Before the unit fix the estimate was 732 *metric calls* while the
    observed quantity was *task rows*; the run that produced 265 rows at
    the retired budget scales to 73 at the pinned one, and both the scaled
    figure and the raw 265 must be accepted.
    """
    assert call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=GEPA_MEASURED_TASK_CALLS_AT_PIN,
        internal_size=88,
        k_repeat=3,
    )
    # The unscaled measurement is inside 1.5x of the ceiling too, so a run
    # that spent the whole retired budget would not false-abort either.
    assert call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=MEASURED_GEPA_TASK_CALLS,
        internal_size=88,
        k_repeat=3,
    )


def test_a_gepa_run_above_the_tolerance_is_still_rejected() -> None:
    """The gate must keep its teeth: 1.5x of the ceiling is the boundary."""
    boundary = int(GEPA_TASK_CALL_CEILING * STAGE1_CALL_COUNT_TOLERANCE)
    assert call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=boundary,
        internal_size=88,
        k_repeat=3,
    )
    assert not call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=boundary + 1,
        internal_size=88,
        k_repeat=3,
    )


def test_null_identity_reports_zero_calls_and_passes_its_gate() -> None:
    """A control's run-side cost is zero, which is inside any harness bound."""
    assert call_count_within_estimate(
        optimizer="null-identity",
        observed_task_calls=0,
        internal_size=88,
        k_repeat=3,
        official_size=132,
        held_out_size=220,
    )
