"""L1-L6, each with a deliberately leaking fixture that trips exactly it."""

from __future__ import annotations

import pytest

from whetstone_envs.optim.study.cli import NOT_CHECKED, _format_leakage
from whetstone_envs.optim.study.leakage import (
    HeldOutObservation,
    LeakageCheckError,
    LeakageReport,
    LeakageRule,
    OptimizerEvalObservation,
    SplitIdentity,
    check_held_out_nesting,
    check_l1_optimizer_internal_only,
    check_l2_selection_once_per_arm,
    check_l3_held_out_once_per_candidate,
    check_l4_identical_held_out_procedure,
    check_l5_splits_disjoint,
    held_out_observations_from_counts,
    study_leakage_check,
)

INTERNAL_CONFIG = "internal-eval-config"
HELD_OUT_CONFIG = "held-out-eval-config"
ARM_IDS = ("copro", "miprov2", "gepa", "codex")


def _clean_optimizer_observations() -> tuple[OptimizerEvalObservation, ...]:
    return tuple(
        OptimizerEvalObservation(
            run_id="copro-1000",
            step_index=step,
            resolution_index=0,
            eval_role="internal",
            resolved_eval_config_hash=INTERNAL_CONFIG,
        )
        for step in range(3)
    )


def _clean_held_out_observations() -> tuple[HeldOutObservation, ...]:
    return held_out_observations_from_counts(
        dict.fromkeys((*ARM_IDS, "naive", "ceiling"), 1),
        eval_config_hash=HELD_OUT_CONFIG,
        repeats=3,
    )


def _clean_splits() -> tuple[SplitIdentity, ...]:
    return (
        SplitIdentity("internal", tuple(f"h-int-{i}" for i in range(8))),
        SplitIdentity("official", tuple(f"h-off-{i}" for i in range(12))),
        SplitIdentity("held_out", tuple(f"h-held-{i}" for i in range(20))),
    )


def _run_check(  # noqa: PLR0913
    *,
    optimizer_observations: tuple[OptimizerEvalObservation, ...] | None = None,
    selected_arm_ids: list[str] | None = None,
    held_out_observations: tuple[HeldOutObservation, ...] | None = None,
    held_out_candidate_names: list[str] | None = None,
    splits: tuple[SplitIdentity, ...] | None = None,
    strict: bool = True,
) -> LeakageReport:
    """Run L6 over a clean study, with one part optionally mutated.

    ``held_out_candidate_names`` defaults to the names in
    ``held_out_observations``, which is the shape a study without crashed
    evaluations has: every issued evaluation returned.
    """
    observations = (
        _clean_held_out_observations()
        if held_out_observations is None
        else held_out_observations
    )
    return study_leakage_check(
        optimizer_observations=(
            _clean_optimizer_observations()
            if optimizer_observations is None
            else optimizer_observations
        ),
        internal_eval_config_hash=INTERNAL_CONFIG,
        selected_arm_ids=(
            list(ARM_IDS) if selected_arm_ids is None else selected_arm_ids
        ),
        expected_arm_ids=ARM_IDS,
        held_out_candidate_names=(
            [entry.candidate_name for entry in observations]
            if held_out_candidate_names is None
            else held_out_candidate_names
        ),
        held_out_observations=observations,
        splits=_clean_splits() if splits is None else splits,
        strict=strict,
    )


def test_a_clean_study_passes_every_rule() -> None:
    report = _run_check()
    assert report.passed
    assert report.failures() == ()
    assert {finding.rule for finding in report.findings} == set(LeakageRule)


def test_l1_catches_an_optimizer_evaluating_the_official_split() -> None:
    """The leak the whole study design exists to prevent."""
    leaking = (
        *_clean_optimizer_observations(),
        OptimizerEvalObservation(
            run_id="copro-1000",
            step_index=3,
            resolution_index=1,
            eval_role="official",
            resolved_eval_config_hash="official-eval-config",
        ),
    )
    finding = check_l1_optimizer_internal_only(
        leaking, internal_eval_config_hash=INTERNAL_CONFIG
    )
    assert not finding.passed
    assert finding.offenders == ("copro-1000:step3:resolution1",)


def test_l1_catches_the_right_role_reaching_the_wrong_config() -> None:
    """Role alone is not enough: a second internal config is still a leak."""
    leaking = (
        OptimizerEvalObservation(
            run_id="copro-1000",
            step_index=0,
            resolution_index=0,
            eval_role="internal",
            resolved_eval_config_hash="some-other-internal-config",
        ),
    )
    finding = check_l1_optimizer_internal_only(
        leaking, internal_eval_config_hash=INTERNAL_CONFIG
    )
    assert not finding.passed


def test_l2_catches_an_arm_selected_twice() -> None:
    finding = check_l2_selection_once_per_arm(
        selected_arm_ids=["copro", "copro", "miprov2", "gepa", "codex"],
        expected_arm_ids=ARM_IDS,
    )
    assert not finding.passed
    assert finding.offenders == ("copro selected 2 times",)


def test_l2_catches_an_arm_that_never_selected() -> None:
    finding = check_l2_selection_once_per_arm(
        selected_arm_ids=["copro", "miprov2", "gepa"],
        expected_arm_ids=ARM_IDS,
    )
    assert not finding.passed
    assert finding.offenders == ("codex never selected",)


def test_l2_catches_a_selection_for_a_non_arm() -> None:
    finding = check_l2_selection_once_per_arm(
        selected_arm_ids=[*ARM_IDS, "ceiling"],
        expected_arm_ids=ARM_IDS,
    )
    assert not finding.passed
    assert finding.offenders == ("ceiling is not a study arm",)


def test_l3_catches_a_candidate_evaluated_twice_on_held_out() -> None:
    """The leak that would let a study shop for its best held-out number."""
    finding = check_l3_held_out_once_per_candidate(
        ["copro", "copro", "miprov2", "naive"]
    )
    assert not finding.passed
    assert finding.offenders == ("copro evaluated 2 times on held-out",)


def test_l3_counts_an_evaluation_that_was_issued_and_never_returned() -> None:
    """A crashed evaluation still spent the candidate's one shot, so a
    second attempt at it is the leak L3 names."""
    finding = check_l3_held_out_once_per_candidate(["naive", "naive"])
    assert not finding.passed


def test_l4_catches_a_candidate_measured_under_a_different_config() -> None:
    """An unpaired comparison, dressed as a paired one."""
    leaking = (
        HeldOutObservation("copro", HELD_OUT_CONFIG, 3),
        HeldOutObservation("naive", "a-different-held-out-config", 3),
    )
    finding = check_l4_identical_held_out_procedure(leaking)
    assert not finding.passed
    assert any(
        "eval_config_hash" in offender for offender in finding.offenders
    )


def test_l4_catches_a_candidate_measured_with_different_repeats() -> None:
    leaking = (
        HeldOutObservation("copro", HELD_OUT_CONFIG, 3),
        HeldOutObservation("naive", HELD_OUT_CONFIG, 5),
    )
    finding = check_l4_identical_held_out_procedure(leaking)
    assert not finding.passed
    assert any("repeats" in offender for offender in finding.offenders)


def test_l4_fails_when_nothing_was_measured() -> None:
    """An empty held-out set is a broken study, not a vacuously clean one.

    It is reported the way L1 reports its own empty case: **not checked**
    rather than checked-and-failed. Both are refusals -- neither may be
    read as a pass -- but "the rule found a violation" and "the rule had no
    evidence to look at" are different facts, and a reader acts on them
    differently.
    """
    finding = check_l4_identical_held_out_procedure(())
    assert not finding.passed
    assert not finding.checked
    assert "nothing to check" in finding.detail


def test_l4_renders_as_not_checked_rather_than_failed() -> None:
    """The literal a reader sees for an unevaluable rule."""
    report = LeakageReport(
        findings=(check_l4_identical_held_out_procedure(()),)
    )
    line = next(iter(_format_leakage(report)))
    assert line.startswith(NOT_CHECKED)
    assert "FAILED" not in line


def test_l5_catches_a_task_shared_between_two_splits() -> None:
    """The leak that would make the held-out split partly seen."""
    shared = "h-int-0"
    leaking = (
        SplitIdentity("internal", (shared, "h-int-1")),
        SplitIdentity("official", ("h-off-0",)),
        SplitIdentity("held_out", (shared, "h-held-1")),
    )
    finding = check_l5_splits_disjoint(leaking)
    assert not finding.passed
    assert finding.offenders == ("internal and held_out share 1 task hashes",)


def test_l5_catches_a_split_that_repeats_a_task() -> None:
    leaking = (SplitIdentity("held_out", ("h-held-0", "h-held-0")),)
    finding = check_l5_splits_disjoint(leaking)
    assert not finding.passed
    assert finding.offenders == ("held_out repeats a task hash",)


def test_l6_raises_and_names_the_failing_rules() -> None:
    """L6 fails the study loudly rather than letting a report be generated."""
    with pytest.raises(LeakageCheckError, match="L2"):
        _run_check(
            selected_arm_ids=["copro", "copro", "miprov2", "gepa", "codex"]
        )


def test_l6_can_report_without_raising_for_inspection() -> None:
    report = _run_check(
        selected_arm_ids=["copro", "copro", "miprov2", "gepa", "codex"],
        strict=False,
    )
    assert not report.passed
    l2 = report.finding(LeakageRule.L2_SELECTION_ONCE_PER_ARM)
    assert l2.passed is False
    assert report.finding(LeakageRule.L6_CHECK_RAN).passed is False
    # Only the rule that was broken fails, so a fixture cannot pass by
    # breaking everything.
    assert {finding.rule for finding in report.failures()} == {
        LeakageRule.L2_SELECTION_ONCE_PER_ARM,
        LeakageRule.L6_CHECK_RAN,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_rule"),
    [
        (
            "l1",
            LeakageRule.L1_OPTIMIZER_INTERNAL_ONLY,
        ),
        (
            "l3",
            LeakageRule.L3_HELD_OUT_ONCE_PER_CANDIDATE,
        ),
        (
            "l5",
            LeakageRule.L5_SPLITS_DISJOINT,
        ),
    ],
)
def test_each_leak_trips_exactly_its_own_rule(
    mutation: str, expected_rule: LeakageRule
) -> None:
    """One mutation, one failing rule -- plus L6, which aggregates."""
    if mutation == "l1":
        report = _run_check(
            optimizer_observations=(
                *_clean_optimizer_observations(),
                OptimizerEvalObservation(
                    run_id="copro-1000",
                    step_index=9,
                    resolution_index=0,
                    eval_role="held_out",
                    resolved_eval_config_hash=HELD_OUT_CONFIG,
                ),
            ),
            strict=False,
        )
    elif mutation == "l3":
        report = _run_check(
            held_out_observations=held_out_observations_from_counts(
                {
                    name: (2 if name == "copro" else 1)
                    for name in (*ARM_IDS, "naive")
                },
                eval_config_hash=HELD_OUT_CONFIG,
                repeats=3,
            ),
            strict=False,
        )
    else:
        report = _run_check(
            splits=(
                SplitIdentity("internal", ("shared", "h-int-1")),
                SplitIdentity("official", ("h-off-0",)),
                SplitIdentity("held_out", ("shared",)),
            ),
            strict=False,
        )
    assert {finding.rule for finding in report.failures()} == {
        expected_rule,
        LeakageRule.L6_CHECK_RAN,
    }


def test_held_out_growth_requires_the_smaller_split_to_be_nested() -> None:
    """D5: held-220 is a prefix of the pre-registered held-440."""
    smaller = tuple(f"h-{i}" for i in range(220))
    larger = tuple(f"h-{i}" for i in range(440))
    assert check_held_out_nesting(smaller=smaller, larger=larger).passed

    resampled = tuple(f"h-{i}" for i in range(220, 660))
    finding = check_held_out_nesting(smaller=smaller, larger=resampled)
    assert not finding.passed
    assert len(finding.offenders) == 220
