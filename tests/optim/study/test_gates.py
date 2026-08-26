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
    COPRO_DEFAULT_BREADTH,
    COPRO_DEFAULT_DEPTH,
    GEPA_MAX_METRIC_CALLS_PINNED,
    GEPA_MEASURED_FULL_VALSET_PASSES,
    GEPA_MEASURED_REFLECTION_MINIBATCHES,
    GEPA_MEASURED_TASK_CALLS_AT_PIN,
    GEPA_PIN_REASON,
    GEPA_REFLECTION_MINIBATCH_TASKS,
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
    MIPROV2_FULL_EVAL_CALLS_MAX,
    MIPROV2_FULL_EVAL_CALLS_MIN,
    MIPROV2_FULL_EVAL_PASSES_ISSUED,
    MIPROV2_FULL_EVAL_PASSES_MAX,
    MIPROV2_FULL_EVAL_PASSES_MIN,
    MIPROV2_FULL_EVAL_STEPS,
    MIPROV2_MINIBATCH_CALLS,
    MIPROV2_MINIBATCH_SIZE,
    MIPROV2_NUM_TRIALS,
    MIPROV2_VALSET_TASKS,
    NULL_IDENTITY_HELD_OUT_PASSES,
    NULL_IDENTITY_OFFICIAL_PASSES,
    STAGE1_CALL_COUNT_TOLERANCE,
    estimate_optimizer_calls,
    gepa_task_call_ceiling,
    miprov2_full_eval_rows,
    miprov2_minibatch_rows,
    null_identity_report_rows,
)
from whetstone_envs.optim.study.protocols import (
    COPRO_BREADTH,
    COPRO_DEPTH,
    GEPA_MAX_METRIC_CALLS,
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


def test_the_fewshot_range_is_pinned_at_1210_to_3118() -> None:
    """The Stage-1 gate divides by the ceiling, so both ends are pinned."""
    assert MIPROV2_MINIBATCH_CALLS == 1_050
    assert MIPROV2_FULL_EVAL_CALLS_MIN == 132
    assert MIPROV2_FULL_EVAL_CALLS_MAX == 1_452
    assert MIPROV2_FEWSHOT_TASK_CALL_FLOOR == 1_210
    assert MIPROV2_FEWSHOT_TASK_CALL_CEILING == 3_118


def test_the_full_valset_pass_count_follows_the_schedule() -> None:
    """F11: the schedule issues 11 passes, not the 6 the old model assumed.

    ``adjusted_num_trials`` is 21 at the registered control, and
    ``promotion_due`` fires on every even display trial -- ten promotions
    -- plus the one baseline full evaluation.
    """
    assert MIPROV2_NUM_TRIALS == 10
    assert MIPROV2_FULL_EVAL_STEPS == 1
    assert MIPROV2_VALSET_TASKS == 44
    assert MIPROV2_MINIBATCH_SIZE == 35
    assert MIPROV2_FULL_EVAL_PASSES_ISSUED == 11
    assert MIPROV2_FULL_EVAL_PASSES_MAX == 11
    # Every promotion may collapse onto the baseline's evidence record.
    assert MIPROV2_FULL_EVAL_PASSES_MIN == 1
    assert miprov2_minibatch_rows(3) == 1_050
    assert miprov2_full_eval_rows(11, 3) == 1_452


@pytest.mark.parametrize(
    ("observed", "bootstrap_attempts", "full_passes"),
    [
        # The five fewshot seeds, bimodal by deduplication.
        (2_370, 44, 9),
        (2_502, 44, 10),
        (1_842, 88, 4),
        (1_710, 44, 4),
    ],
)
def test_the_2b_observations_decompose_and_sit_inside_the_band(
    observed: int, bootstrap_attempts: int, full_passes: int
) -> None:
    """F11's reconciliation, pinned against the 2b run records.

    Each observed count is exactly bootstrap + minibatch + full-valset,
    and the corrected band contains all four. The previous band's high of
    2,458 excluded 2,502 and its low of 1,870 excluded both 1,842 and
    1,710.
    """
    bootstrap = bootstrap_attempts * 1 * 3
    decomposed = (
        bootstrap
        + miprov2_minibatch_rows(3)
        + miprov2_full_eval_rows(full_passes, 3)
    )
    assert decomposed == observed
    assert (
        MIPROV2_FEWSHOT_TASK_CALL_FLOOR
        <= observed
        <= MIPROV2_FEWSHOT_TASK_CALL_CEILING
    )


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
    """The estimate follows the shape it is given, at any shape."""
    estimate = estimate_optimizer_calls(
        "copro",
        internal_size=88,
        k_repeat=3,
        copro_breadth=2,
        copro_depth=1,
    )
    # depth 1 evaluating round x breadth 2 x 88 tasks x 3 repeats.
    assert estimate.low == estimate.high == 1 * 2 * 88 * 3
    assert "breadth 2" in estimate.basis
    assert estimate.gated


def test_the_estimator_defaults_to_the_pinned_shape_not_the_runners() -> None:
    """An unshaped estimate prices the study's search, not a smoke run.

    These defaults were the *runner's* -- 2 and 1 -- which made the
    estimate agree with a run that never received the pinned shape and
    disagree with the design both were meant to describe. COPRO's whole
    per-run cost is ``depth x breadth x T_int x K_REPEAT``, so a default
    that understates the shape understates the budget ninefold.
    """
    assert (COPRO_DEFAULT_BREADTH, COPRO_DEFAULT_DEPTH) == (
        COPRO_BREADTH,
        COPRO_DEPTH,
    )
    estimate = estimate_optimizer_calls("copro", internal_size=88, k_repeat=3)
    assert (
        estimate.low == estimate.high == (COPRO_DEPTH * COPRO_BREADTH * 88 * 3)
    )
    # Which is the protocol's own section 5.1 arithmetic: 6 x 3 x 88 x 3.
    assert estimate.low == 4_752


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
        held_out_size=440,
    )
    # One official pass over 132 and one held-out pass over 440, at K=3.
    assert null.low == null.high == 3 * (132 + 440) == 1_716
    assert null.gated
    assert "no optimizer run" in null.basis
    # And it does not track the internal split, which it never evaluates.
    wider = estimate_optimizer_calls(
        "null-identity",
        internal_size=8_800,
        k_repeat=3,
        official_size=132,
        held_out_size=440,
    )
    assert wider.low == null.low


def test_null_identity_no_longer_borrows_copros_search_shape() -> None:
    """The regression this replaces: null-B priced as a COPRO run.

    Split sizes where the two formulas differ by a wide margin, so the
    pin cannot be satisfied by an arithmetic coincidence.
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
            official_size=132, held_out_size=440, k_repeat=3
        )
        == 1_716
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


def test_miprov2_does_not_read_the_internal_split() -> None:
    """MIPROv2 evaluates its *validation* partition, not the whole split.

    The narrow true statement the old test overshot. Its budget does not
    follow from ``internal_size`` -- but it does follow from the arm's own
    ``val_size`` and minibatch, which is the next test.
    """
    at_24 = estimate_optimizer_calls(
        "miprov2", internal_size=24, k_repeat=3, val_size=44
    )
    at_88 = estimate_optimizer_calls(
        "miprov2", internal_size=88, k_repeat=3, val_size=44
    )
    assert (at_24.low, at_24.high) == (at_88.low, at_88.high)


def test_the_miprov2_estimate_tracks_the_designs_own_shape() -> None:
    """The estimate is the arm's search, not a constant.

    Pinned at the two registered shapes. The c19 band must not move --
    it is what the Stage-1 gate has always divided by -- and the c18 band
    must reflect a design that runs ten trials over a twelve-task valset
    with no minibatch at all.

    Fails before this change: MIPROv2's estimate ignored both parameters,
    so a c18 arm was priced at c19's 1210-3118 and the plan printed a
    "1050 minibatch" volume for a design whose manifest says
    ``minibatch: false``.
    """
    c19 = estimate_optimizer_calls(
        "miprov2",
        internal_size=88,
        k_repeat=3,
        val_size=44,
        miprov2_minibatch_size=35,
    )
    assert (
        (c19.low, c19.high)
        == (
            MIPROV2_FEWSHOT_TASK_CALL_FLOOR,
            MIPROV2_FEWSHOT_TASK_CALL_CEILING,
        )
        == (1210, 3118)
    )
    assert "1050 minibatch" in c19.basis

    c18 = estimate_optimizer_calls(
        "miprov2",
        internal_size=24,
        k_repeat=3,
        val_size=12,
        miprov2_minibatch_size=None,
    )
    assert (c18.low, c18.high) != (c19.low, c19.high)
    # The honest per-run cost of the registered c18 search -- ten trials
    # over the twelve-task valset at three repeats -- lands inside its band.
    assert c18.low <= MIPROV2_NUM_TRIALS * 12 * 3 <= c18.high
    # And the basis says what the design does rather than naming a
    # minibatch volume no c18 run issues.
    assert "unbatched" in c18.basis
    assert "minibatch +" not in c18.basis
    assert "12 val tasks" in c18.basis


def test_the_default_miprov2_estimate_is_c19s_registered_band() -> None:
    """A caller passing no shape still gets exactly the c19 band.

    The constants and every existing call site are unchanged by the
    parameterisation, which is what keeps the c19 study's gate identical.
    """
    default = estimate_optimizer_calls("miprov2", internal_size=88, k_repeat=3)
    assert (default.low, default.high) == (
        MIPROV2_FEWSHOT_TASK_CALL_FLOOR,
        MIPROV2_FEWSHOT_TASK_CALL_CEILING,
    )


def test_gepa_is_estimated_in_task_rows_at_the_pinned_budget() -> None:
    """The unit fix: rows bounded by the pin, not the 732 auto budget.

    Comparing a row count against a metric-call ceiling is what made a real
    GEPA run trip the 1.5x gate, so the estimate is denominated in the same
    unit ``observed_task_calls`` carries.
    """
    estimate = estimate_optimizer_calls("gepa", internal_size=88, k_repeat=3)
    assert estimate.low == estimate.high == gepa_task_call_ceiling(3) == 600
    assert estimate.high != GEPA_RESOLVED_MAX_METRIC_CALLS
    assert "bounds task rows" in estimate.basis


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
        observed_task_calls=gepa_task_call_ceiling(3) * 3,
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


def test_the_measurement_provenance_is_what_wave_3_ran() -> None:
    """Measurement provenance, pinned so it cannot drift to fit the study.

    Wave 3 measured at held-out 220. The study now pre-registers 440, so
    this no longer matches the protocol splits -- and must not be rewritten
    to, because it records what was actually run.
    """
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
    # One owner, not two literals that happen to agree. ``protocols``
    # pins every design value; the gate's name is an alias for it. Two
    # independent 200s are one edit away from a gate that judges a run
    # against a ceiling the run never had.
    assert GEPA_MAX_METRIC_CALLS_PINNED is GEPA_MAX_METRIC_CALLS
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
    """D3 pinned 200 metric calls, and the gate bounds rows -- not the 732."""
    assert GEPA_MAX_METRIC_CALLS_PINNED == 200
    assert gepa_task_call_ceiling(1) == GEPA_MAX_METRIC_CALLS_PINNED
    assert gepa_task_call_ceiling(3) != GEPA_RESOLVED_MAX_METRIC_CALLS


def test_the_gepa_ceiling_scales_with_the_repeat_count() -> None:
    """**The 0.1.11 unit correction.** A metric call bills K_REPEAT rows.

    Upstream made a repeated evaluation bill ``K_REPEAT`` times as many
    provider rows while leaving its metric-call count unchanged, so the pin
    stopped bounding rows on its own. At the design's ``K_REPEAT = 3`` a
    run charged the pinned 200 metric calls may execute up to 600 rows,
    against a gate limit the unscaled ceiling would have put at
    ``200 x 1.5 = 300`` -- the gate would abort the healthy run it exists
    to protect.
    """
    assert gepa_task_call_ceiling(3) == 600
    assert gepa_task_call_ceiling(3) == 3 * gepa_task_call_ceiling(1)
    # The failure the scaling prevents: a run entitled to 600 rows, judged.
    entitled = GEPA_MAX_METRIC_CALLS_PINNED * 3
    assert entitled > int(
        GEPA_MAX_METRIC_CALLS_PINNED * STAGE1_CALL_COUNT_TOLERANCE
    )
    assert call_count_within_estimate(
        optimizer="gepa",
        observed_task_calls=entitled,
        internal_size=88,
        k_repeat=3,
    )


def test_the_gepa_ceiling_refuses_a_non_positive_repeat_count() -> None:
    assert gepa_task_call_ceiling(1) == 200
    with pytest.raises(ValueError, match="k_repeat must be at least 1"):
        gepa_task_call_ceiling(0)


def test_the_measured_run_scaled_to_the_pin_is_inside_the_ceiling() -> None:
    """The cross-check: an upper bound must sit above the scaled measurement.

    265 rows at 732 charged calls scales to 72.4 rows at the pinned 200,
    rounded up to 73.
    """
    assert GEPA_MEASURED_TASK_CALLS_AT_PIN == 73
    assert gepa_task_call_ceiling(1) > GEPA_MEASURED_TASK_CALLS_AT_PIN
    # The measurement was taken at one repeat, so it scales with the run's.
    assert gepa_task_call_ceiling(3) > GEPA_MEASURED_TASK_CALLS_AT_PIN * 3


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
    boundary = int(gepa_task_call_ceiling(3) * STAGE1_CALL_COUNT_TOLERANCE)
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
        held_out_size=440,
    )
