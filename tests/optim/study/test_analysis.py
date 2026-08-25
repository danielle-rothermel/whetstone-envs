"""The post-measurement analysis: held-out rows and the statistics on them.

The stages measure one arm at a time; the statistics cannot, because a
Holm-corrected p-value is a whole-study computation. These tests cover that
second pass directly, over measurements built by hand, so what is under test
is the arithmetic rather than the transport underneath it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.study.analysis import (
    _achieved_counts,
    _delta_for,
)
from whetstone_envs.optim.study.power import COMPLETENESS_BACKSTOP
from whetstone_envs.optim.study.selection import HeldOutMeasurement

HELD_OUT_CONFIG = "held-out-eval-config"


def _measurement(
    name: str,
    per_task: tuple[float, ...],
    *,
    completeness: float = 1.0,
    counts: tuple[int, ...] = (),
    repeats: int = 3,
) -> HeldOutMeasurement:
    return HeldOutMeasurement(
        candidate_name=name,
        per_task=per_task,
        mean=sum(per_task) / len(per_task),
        eval_config_hash=HELD_OUT_CONFIG,
        repeats=repeats,
        completeness=completeness,
        per_task_counts=counts,
    )


def test_measured_row_counts_are_preferred_over_a_spread_average() -> None:
    measurement = _measurement(
        "copro", (1.0, 1.0, 1.0), completeness=7 / 9, counts=(3, 1, 3)
    )
    assert _achieved_counts(measurement, k_repeat=3) == (3, 1, 3)


def test_a_row_count_above_the_plan_is_clamped() -> None:
    """A task cannot achieve more rows than the design scheduled for it."""
    measurement = _measurement("copro", (1.0, 1.0), counts=(5, 3))
    assert _achieved_counts(measurement, k_repeat=3) == (3, 3)


def test_without_counts_the_aggregate_is_spread_evenly() -> None:
    """The documented approximation, used only when nothing better exists."""
    measurement = _measurement("copro", (1.0, 1.0, 1.0), completeness=2 / 3)
    assert _achieved_counts(measurement, k_repeat=3) == (2, 2, 2)


def test_a_ragged_task_shrinks_its_own_contribution() -> None:
    """The load-bearing property of O7's weighting.

    Two arms with identical scores differ only in how many rows their
    second task achieved. The one that measured fewer rows there must
    report a smaller delta, because its evidence for that task is thinner
    -- and the task stays in the vector at reduced weight rather than being
    dropped, which would shrink ``T`` and tighten the interval dishonestly.
    """
    naive = _measurement("naive", (0.0, 0.0, 0.0))
    complete = _delta_for(
        arm_id="copro",
        measurement=_measurement("copro", (1.0, 1.0, 1.0), counts=(3, 3, 3)),
        naive=naive,
        k_repeat=3,
    )
    ragged = _delta_for(
        arm_id="copro",
        measurement=_measurement(
            "copro", (1.0, 1.0, 1.0), completeness=7 / 9, counts=(3, 1, 3)
        ),
        naive=naive,
        k_repeat=3,
    )
    assert complete.arm_per_task == (1.0, 1.0, 1.0)
    assert ragged.arm_per_task == pytest.approx((1.0, 1 / 3, 1.0))
    # The ragged task is still one of three, so the interval is not
    # tightened by dropping it.
    assert len(ragged.arm_per_task) == len(complete.arm_per_task)
    assert ragged.completeness < complete.completeness


def test_a_thin_anchor_downgrades_the_delta_it_anchors() -> None:
    """Completeness is the paired minimum, so both sides count.

    Fails-before: the weighting read the *arm's* achieved counts only. An
    arm that measured every row against an anchor that lost two thirds of
    its own reported completeness 1.0 -- the anchor's thin fallback mean
    was treated as fully observed -- so the delta cleared the 0.90 backstop
    and was claimed on evidence that was mostly missing on one side.
    """
    arm = _measurement("copro", (1.0, 1.0, 1.0), counts=(3, 3, 3))
    thin_anchor = _measurement(
        "naive", (0.0, 0.0, 0.0), completeness=1 / 3, counts=(1, 1, 1)
    )
    delta = _delta_for(
        arm_id="copro", measurement=arm, naive=thin_anchor, k_repeat=3
    )
    assert delta.completeness == pytest.approx(1 / 3)
    assert delta.completeness < COMPLETENESS_BACKSTOP
    # And the arm's own side, measured against a complete anchor, still
    # clears it -- so the downgrade is the anchor's doing, not a blanket
    # tightening of the rule.
    complete = _delta_for(
        arm_id="copro",
        measurement=arm,
        naive=_measurement("naive", (0.0, 0.0, 0.0), counts=(3, 3, 3)),
        k_repeat=3,
    )
    assert complete.completeness == pytest.approx(1.0)


def test_the_paired_minimum_is_taken_per_task_not_in_aggregate() -> None:
    """Tasks do not fail evenly, and neither does the pairing.

    Each side is thin on a *different* task here, so an aggregate rule
    would report both sides at 2/3 and call the comparison 2/3 complete.
    Per task, every one of the three is thin on one side or the other, so
    the honest weight is lower than either side's own completeness.
    """
    delta = _delta_for(
        arm_id="copro",
        measurement=_measurement(
            "copro", (1.0, 1.0, 1.0), completeness=7 / 9, counts=(1, 3, 3)
        ),
        naive=_measurement(
            "naive", (0.0, 0.0, 0.0), completeness=7 / 9, counts=(3, 1, 1)
        ),
        k_repeat=3,
    )
    assert delta.completeness == pytest.approx(3 / 9)


def test_an_unpaired_candidate_is_refused_rather_than_truncated() -> None:
    """A comparison over different task counts is not paired at all."""
    with pytest.raises(ValueError, match="not paired"):
        _delta_for(
            arm_id="copro",
            measurement=_measurement("copro", (1.0, 1.0)),
            naive=_measurement("naive", (0.0, 0.0, 0.0)),
            k_repeat=3,
        )


# --------------------------------------------------------------------------
# A lost task degrades the claim rather than aborting the pass
# --------------------------------------------------------------------------


def test_a_fully_lost_task_pushes_the_delta_below_the_backstop() -> None:
    """The whole point of carrying the loss: it must reach the verdict.

    **Fails-before: the evaluation never got here.** The completeness
    floor refused any fully-lost task upstream, so a stage died before a
    manifest describing this state could be written. Now the loss travels
    as a zero count into O7's weighting, the delta's completeness lands
    under ``COMPLETENESS_BACKSTOP``, and ``_arm_verdict`` reads that as
    ``VERDICT_INCOMPLETE`` -- an arm measured, and measured too shallowly
    to claim.
    """
    # Eight tasks at three repeats; one lost every repeat, so 7 of 8.
    counts = (3,) * 7 + (0,)
    arm = _measurement(
        "copro", (0.8,) * 8, counts=counts, completeness=21 / 24
    )
    naive = _measurement("naive", (0.5,) * 8, counts=(3,) * 8)

    delta = _delta_for(
        arm_id="copro", measurement=arm, naive=naive, k_repeat=3
    )
    # The lost task contributes nothing to the point estimate.
    assert delta.arm_per_task[-1] == 0.0
    assert delta.arm_per_task[0] == pytest.approx(0.3)
    # And the loss shows up where the verdict reads it.
    assert delta.completeness == pytest.approx(7 / 8)
    assert delta.completeness < COMPLETENESS_BACKSTOP


def test_a_single_loss_in_a_large_split_stays_claimable() -> None:
    """The rule is the fraction, not the first lost task.

    At the study's own held-out size one lost task is 0.995 complete,
    comfortably inside the backstop -- which is exactly the outcome the
    old unconditional refusal made impossible.
    """
    tasks = 220
    counts = (4,) * (tasks - 1) + (0,)
    arm = _measurement("copro", (0.8,) * tasks, counts=counts, repeats=4)
    naive = _measurement(
        "naive", (0.5,) * tasks, counts=(4,) * tasks, repeats=4
    )

    delta = _delta_for(
        arm_id="copro", measurement=arm, naive=naive, k_repeat=4
    )
    assert delta.completeness == pytest.approx((tasks - 1) / tasks)
    assert delta.completeness >= COMPLETENESS_BACKSTOP


# --------------------------------------------------------------------------
# Anchors resume from a completed claim
# --------------------------------------------------------------------------


def test_an_anchor_resumes_from_its_completed_claim() -> None:
    """**Fails-before: anchors had no resume branch at all.**

    ``completed_claim_for`` was consulted only for arms, and the anchor
    pass runs *last* -- after every arm has been scored and measured. So a
    crash anywhere in the reporting pass left the anchors in the one state
    with no recovery: their L3 claims durable and refusing a second
    evaluation, and nothing reading those claims back. The resumed pass
    re-issued the call, was refused, and wedged a study that had already
    paid for essentially all of its rows.
    """
    from whetstone_envs.optim.study.analysis import (
        CEILING_CANDIDATE_NAME,
        NAIVE_CANDIDATE_NAME,
        measure_reference_candidates,
    )
    from whetstone_envs.optim.study.selection import SelectionLog

    issued: list[str] = []

    def evaluate(*, candidate_name: str, template: str):
        del template
        issued.append(candidate_name)
        return _measurement(candidate_name, (1.0, 0.0))

    class _ClaimedLog(SelectionLog):
        """A ledger that already holds both anchors' completed claims."""

        def completed_claim_for(self, candidate_name: str):
            return _measurement(candidate_name, (0.9, 0.1))

    references = measure_reference_candidates(
        naive_template="naive {q}",
        ceiling_template="ceiling {q}",
        evaluate_held_out=evaluate,
        log=_ClaimedLog(),
    )

    # Nothing was re-issued, and both anchors came back from their claims.
    assert issued == []
    assert set(references) == {NAIVE_CANDIDATE_NAME, CEILING_CANDIDATE_NAME}
    assert references[NAIVE_CANDIDATE_NAME].per_task == (0.9, 0.1)


def test_an_unclaimed_anchor_is_still_measured_normally() -> None:
    """The resume branch costs nothing on a first run."""
    from whetstone_envs.optim.study.analysis import (
        CEILING_CANDIDATE_NAME,
        NAIVE_CANDIDATE_NAME,
        measure_reference_candidates,
    )
    from whetstone_envs.optim.study.selection import SelectionLog

    issued: list[str] = []

    def evaluate(*, candidate_name: str, template: str):
        del template
        issued.append(candidate_name)
        return _measurement(candidate_name, (1.0, 0.0))

    log = SelectionLog()
    measure_reference_candidates(
        naive_template="naive {q}",
        ceiling_template="ceiling {q}",
        evaluate_held_out=evaluate,
        log=log,
    )
    assert issued == [NAIVE_CANDIDATE_NAME, CEILING_CANDIDATE_NAME]
    # And each spent exactly its one claim (L3 is untouched).
    assert log.held_out_count(NAIVE_CANDIDATE_NAME) == 1
    assert log.held_out_count(CEILING_CANDIDATE_NAME) == 1
