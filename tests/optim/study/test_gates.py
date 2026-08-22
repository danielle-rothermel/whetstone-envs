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
    GEPA_RESOLVED_MAX_METRIC_CALLS,
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
