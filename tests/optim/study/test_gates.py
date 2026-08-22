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
    GEPA_PIN_REASON,
    GEPA_RESOLVED_MAX_METRIC_CALLS,
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
    STAGE1_CALL_COUNT_TOLERANCE,
    estimate_optimizer_calls,
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


@pytest.mark.parametrize("optimizer", ["null-random", "null-identity"])
def test_the_nulls_evaluate_exactly_as_copro_does(optimizer: str) -> None:
    """A null shares COPRO's selection machinery, so it shares its cost."""
    null = estimate_optimizer_calls(optimizer, internal_size=88, k_repeat=3)
    copro = estimate_optimizer_calls("copro", internal_size=88, k_repeat=3)
    assert (null.low, null.high) == (copro.low, copro.high)


def test_miprov2_carries_its_own_budget_not_the_splits() -> None:
    """MIPROv2's volume is its control's, so the splits do not move it."""
    small = estimate_optimizer_calls("miprov2", internal_size=8, k_repeat=1)
    large = estimate_optimizer_calls("miprov2", internal_size=88, k_repeat=3)
    assert (small.low, small.high) == (large.low, large.high)
    assert large.high == MIPROV2_FEWSHOT_TASK_CALL_CEILING


def test_gepa_reports_its_resolved_metric_call_ceiling() -> None:
    estimate = estimate_optimizer_calls("gepa", internal_size=88, k_repeat=3)
    assert estimate.low == estimate.high == GEPA_RESOLVED_MAX_METRIC_CALLS


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
        observed_task_calls=GEPA_RESOLVED_MAX_METRIC_CALLS * 3,
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
