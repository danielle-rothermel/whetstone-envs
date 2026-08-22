"""``report_arm`` selects on official and measures held-out exactly once."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from whetstone_envs.optim.study.selection import (
    SELECTION_RULE,
    ArmDelta,
    CandidateScore,
    HeldOutMeasurement,
    RunCandidate,
    SelectionError,
    SelectionLog,
    SelectionRecord,
    analyze_arms,
    null_triggers_downgrade,
    report_arm,
    report_reference_candidate,
)

OFFICIAL_CONFIG = "official-eval-config"
HELD_OUT_CONFIG = "held-out-eval-config"


@dataclass
class _RecordingScorer:
    """An official scorer that records every candidate it was asked about."""

    scores: dict[str, float]
    seen: list[str] = field(default_factory=list)
    eval_config_hash: str = OFFICIAL_CONFIG

    def __call__(self, candidate: RunCandidate) -> CandidateScore:
        self.seen.append(candidate.run_id)
        score = self.scores[candidate.run_id]
        return CandidateScore(
            run_id=candidate.run_id,
            score=score,
            per_task=(score, score),
            eval_config_hash=self.eval_config_hash,
            completeness=1.0,
        )


@dataclass
class _RecordingHeldOut:
    """A held-out evaluator that records every template it evaluated."""

    seen: list[tuple[str, str]] = field(default_factory=list)
    repeats: int = 3

    def __call__(
        self, *, candidate_name: str, template: str
    ) -> HeldOutMeasurement:
        self.seen.append((candidate_name, template))
        return HeldOutMeasurement(
            candidate_name=candidate_name,
            per_task=(1.0, 0.0),
            mean=0.5,
            eval_config_hash=HELD_OUT_CONFIG,
            repeats=self.repeats,
            completeness=1.0,
        )


def _runs() -> tuple[RunCandidate, ...]:
    return (
        RunCandidate("run-a", 1000, "copro-a", "template A {q}"),
        RunCandidate("run-b", 1001, "copro-b", "template B {q}"),
        RunCandidate("run-c", 1002, "copro-c", "template C {q}"),
    )


def test_selection_takes_the_official_argmax() -> None:
    scorer = _RecordingScorer({"run-a": 0.4, "run-b": 0.7, "run-c": 0.5})
    held_out = _RecordingHeldOut()
    log = SelectionLog()

    report = report_arm(
        arm_id="copro",
        runs=_runs(),
        score_official=scorer,
        evaluate_held_out=held_out,
        log=log,
    )

    assert report.selection.selected_run_id == "run-b"
    assert report.selection.official_score == pytest.approx(0.7)
    assert report.selection.rule == SELECTION_RULE
    assert report.representative.run_id == "run-b"
    # Every run is scored on official; only the winner reaches held-out.
    assert sorted(scorer.seen) == ["run-a", "run-b", "run-c"]


def test_held_out_is_evaluated_exactly_once_and_only_for_the_winner() -> None:
    scorer = _RecordingScorer({"run-a": 0.4, "run-b": 0.7, "run-c": 0.5})
    held_out = _RecordingHeldOut()
    log = SelectionLog()

    report_arm(
        arm_id="copro",
        runs=_runs(),
        score_official=scorer,
        evaluate_held_out=held_out,
        log=log,
    )

    assert held_out.seen == [("copro", "template B {q}")]
    assert log.held_out_count("copro") == 1


def test_a_second_report_of_the_same_arm_is_refused() -> None:
    """L2 as structure: an arm selects once, and the ledger says so."""
    scorer = _RecordingScorer({"run-a": 0.4, "run-b": 0.7, "run-c": 0.5})
    held_out = _RecordingHeldOut()
    log = SelectionLog()

    def run() -> None:
        report_arm(
            arm_id="copro",
            runs=_runs(),
            score_official=scorer,
            evaluate_held_out=held_out,
            log=log,
        )

    run()
    with pytest.raises(SelectionError, match="already selected"):
        run()

    # The refusal happened before the second held-out evaluation, so the
    # study did not pay for the leak it prevented.
    assert len(held_out.seen) == 1


def test_a_second_held_out_evaluation_is_refused_for_any_caller() -> None:
    """L3 as structure, including for callers that skip selection."""
    held_out = _RecordingHeldOut()
    log = SelectionLog()

    report_reference_candidate(
        candidate_name="naive",
        template="naive {q}",
        evaluate_held_out=held_out,
        log=log,
    )
    with pytest.raises(SelectionError, match="already evaluated on held-out"):
        report_reference_candidate(
            candidate_name="naive",
            template="naive {q}",
            evaluate_held_out=held_out,
            log=log,
        )
    assert len(held_out.seen) == 1


def test_held_out_is_unreachable_before_a_selection_is_persisted() -> None:
    """The read-back is what makes the ordering a mechanism, not a habit."""
    log = SelectionLog()
    with pytest.raises(SelectionError, match="no persisted selection"):
        log.require_selection("copro")


def test_selection_is_persisted_before_the_held_out_call_is_issued() -> None:
    """The ordering, observed: the ledger is durable at evaluation time."""
    observed: list[bool] = []
    log = SelectionLog()

    def evaluate(*, candidate_name: str, template: str) -> HeldOutMeasurement:
        del template
        observed.append(log.selection_for("copro") is not None)
        return HeldOutMeasurement(
            candidate_name=candidate_name,
            per_task=(1.0,),
            mean=1.0,
            eval_config_hash=HELD_OUT_CONFIG,
            repeats=3,
            completeness=1.0,
        )

    report_arm(
        arm_id="copro",
        runs=_runs(),
        score_official=_RecordingScorer(
            {"run-a": 0.4, "run-b": 0.7, "run-c": 0.5}
        ),
        evaluate_held_out=evaluate,
        log=log,
    )
    assert observed == [True]


def test_ties_select_the_earlier_run() -> None:
    """A stable rule: re-running selection names the same representative."""
    scorer = _RecordingScorer({"run-a": 0.7, "run-b": 0.7, "run-c": 0.5})
    report = report_arm(
        arm_id="copro",
        runs=_runs(),
        score_official=scorer,
        evaluate_held_out=_RecordingHeldOut(),
        log=SelectionLog(),
    )
    assert report.selection.selected_run_id == "run-a"


def test_mixed_official_eval_configs_are_refused() -> None:
    """An arg-max over incomparable scores is not a selection."""

    class _DriftingScorer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, candidate: RunCandidate) -> CandidateScore:
            self.calls += 1
            return CandidateScore(
                run_id=candidate.run_id,
                score=0.5,
                per_task=(0.5,),
                eval_config_hash=f"config-{self.calls}",
                completeness=1.0,
            )

    held_out = _RecordingHeldOut()
    with pytest.raises(SelectionError, match="different official Eval"):
        report_arm(
            arm_id="copro",
            runs=_runs(),
            score_official=_DriftingScorer(),
            evaluate_held_out=held_out,
            log=SelectionLog(),
        )
    assert held_out.seen == []


def test_an_arm_with_no_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="no runs to report"):
        report_arm(
            arm_id="copro",
            runs=(),
            score_official=_RecordingScorer({}),
            evaluate_held_out=_RecordingHeldOut(),
            log=SelectionLog(),
        )


def test_selection_records_pin_the_pre_registered_rule() -> None:
    with pytest.raises(ValueError, match="different pre-registration"):
        SelectionRecord(
            arm_id="copro",
            selected_run_id="run-a",
            official_score=0.5,
            rule="argmax-held-out",
        )


def test_holm_runs_over_the_real_arms_only() -> None:
    """m = 4: nulls are controls, not hypotheses, and stay uncorrected."""
    arms = tuple(
        ArmDelta(
            arm_id=arm_id,
            arm_per_task=(1.0,) * 30,
            naive_per_task=(0.0,) * 30,
        )
        for arm_id in ("copro", "miprov2", "gepa", "codex")
    )
    stats = analyze_arms(arms, resamples=200, seed=7)
    assert len(stats) == 4
    for statistic in stats:
        assert statistic.p_holm >= statistic.p_bootstrap
        assert statistic.excludes_zero
        assert statistic.claimed


def test_an_incomplete_arm_is_reported_but_never_claimed() -> None:
    """O7's backstop: below 90% achieved rows, the CI is not a claim."""
    stats = analyze_arms(
        (
            ArmDelta(
                arm_id="copro",
                arm_per_task=(1.0,) * 30,
                naive_per_task=(0.0,) * 30,
                completeness=0.5,
            ),
        ),
        resamples=200,
        seed=7,
    )
    assert stats[0].excludes_zero
    assert not stats[0].claimed
    assert stats[0].delta == pytest.approx(1.0)


def test_a_zero_straddling_interval_is_not_claimed() -> None:
    stats = analyze_arms(
        (
            ArmDelta(
                arm_id="copro",
                arm_per_task=(1.0, 0.0) * 15,
                naive_per_task=(0.0, 1.0) * 15,
            ),
        ),
        resamples=200,
        seed=7,
    )
    assert not stats[0].excludes_zero
    assert not stats[0].claimed


def test_null_downgrade_needs_both_magnitude_and_significance() -> None:
    """F12, pre-registered exactly: either condition alone is not enough."""
    assert null_triggers_downgrade(
        null_delta=0.15, mde_measured=0.09, excludes_zero=True
    )
    # Significant but below the MDE: a tiny significant delta is noise the
    # bootstrap resolved, not evidence selection works on nothing.
    assert not null_triggers_downgrade(
        null_delta=0.004, mde_measured=0.09, excludes_zero=True
    )
    # Large but unresolvable: no interval, no claim.
    assert not null_triggers_downgrade(
        null_delta=0.15, mde_measured=0.09, excludes_zero=False
    )


def test_unaligned_paired_vectors_are_refused() -> None:
    with pytest.raises(ValueError, match="unaligned"):
        ArmDelta(
            arm_id="copro",
            arm_per_task=(1.0, 1.0),
            naive_per_task=(0.0,),
        )


# --------------------------------------------------------------------------
# Holm's family size is pre-registered, not counted
# --------------------------------------------------------------------------


def _one_sided_arm(arm_id: str, *, delta: float) -> ArmDelta:
    return ArmDelta(
        arm_id=arm_id,
        arm_per_task=(delta,) * 40,
        naive_per_task=(0.0,) * 40,
    )


def test_holm_corrects_at_the_pre_registered_family_size() -> None:
    """A partial study corrects at m = 4, not at however many arms ran.

    The family was fixed before spend. Deriving ``m`` from the number of
    arms in hand would under-correct exactly the partial studies -- a
    pilot, a resumed stage, an arm that failed -- whose multiplicity risk
    is unchanged by how far the study got.
    """
    two_of_four = analyze_arms(
        (
            _one_sided_arm("copro", delta=1.0),
            _one_sided_arm("gepa", delta=1.0),
        ),
        resamples=200,
        seed=7,
    )
    all_four = analyze_arms(
        tuple(
            _one_sided_arm(arm_id, delta=1.0)
            for arm_id in ("copro", "gepa", "miprov2", "codex")
        ),
        resamples=200,
        seed=7,
    )
    # Both corrections scale the smallest p-value by m = 4, so the partial
    # study's leading adjusted p-value matches the full study's rather than
    # being the halved value an m = 2 correction would produce.
    assert two_of_four[0].p_holm == pytest.approx(all_four[0].p_holm)
    assert two_of_four[0].p_holm == pytest.approx(
        min(1.0, 4 * two_of_four[0].p_bootstrap)
    )


def test_an_explicit_family_size_overrides_the_default() -> None:
    stats = analyze_arms(
        (_one_sided_arm("copro", delta=1.0),),
        resamples=200,
        seed=7,
        family_size=1,
    )
    assert stats[0].p_holm == pytest.approx(stats[0].p_bootstrap)


def test_more_arms_than_the_family_declares_is_refused() -> None:
    """A family is fixed before spend, never widened to fit its results."""
    with pytest.raises(ValueError, match="fixed before spend"):
        analyze_arms(
            tuple(
                _one_sided_arm(arm_id, delta=1.0)
                for arm_id in ("a", "b", "c", "d", "e")
            ),
            resamples=200,
            seed=7,
        )
