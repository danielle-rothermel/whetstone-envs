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


def test_an_unpaired_candidate_is_refused_rather_than_truncated() -> None:
    """A comparison over different task counts is not paired at all."""
    with pytest.raises(ValueError, match="not paired"):
        _delta_for(
            arm_id="copro",
            measurement=_measurement("copro", (1.0, 1.0)),
            naive=_measurement("naive", (0.0, 0.0, 0.0)),
            k_repeat=3,
        )
