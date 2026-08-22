"""F16: the fan-out measurement, re-measured mechanically on a fake run.

The constants in :mod:`whetstone_envs.optim.study.gates` were measured once,
by hand, at the study's own splits. That measurement retired R6 -- the risk
that minibatch evaluations silently fan out over the full validation split
-- but a number measured once and written down is exactly the thing that
goes stale when a control default moves.

So this module re-measures. It runs a small MIPROv2 fake-transport run with
minibatching on, counts the rows the run actually planned, and asserts the
count against **the formula the code implements**, computed here from the
run's own evidence rather than pinned as a literal. A change that made row
expansion ignore per-intent task sets would fail these tests rather than
appear as an unbudgeted provider bill at Stage 1.

Two independent paths are compared, which is what makes this a measurement
rather than a restatement:

* ``EvalEvidence.row_accounting.planned``, summed over the run's
  evaluations, and
* ``cost.json``'s ``task_model.calls``, projected from ``OptimResult.cost``.

They read different records written by different code. Agreement between
them is evidence; either one alone would only prove itself self-consistent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.run import RunSpec, run_optimizer
from whetstone_envs.optim.run_cost import RUN_COST_NAME
from whetstone_envs.optim.study.fanout import (
    INTENT_SOURCE,
    measure_run_directory,
)
from whetstone_envs.optim.study.gates import (
    MEASURED_FANOUT_RATIO,
    MEASURED_MIPROV2_MINIBATCH_TASKS,
    MEASURED_MIPROV2_TRAINSET_TASKS,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.fanout import FanoutMeasurement

#: Small enough to stay a unit test, large enough that a minibatch is a
#: strict subset of the validation split -- which is the whole point. The
#: runner gives MIPROv2 a one-task trainset, so an internal split of 9
#: leaves an 8-task validation split, and a 3-task minibatch is genuinely
#: a subset of it.
INTERNAL_SPLIT = 9
MINIBATCH_TASKS = 3


@pytest.fixture(scope="module")
def minibatched_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One MIPROv2 fake run with minibatching genuinely engaged."""
    output = tmp_path_factory.mktemp("fanout") / "miprov2-minibatch"
    return run_optimizer(
        RunSpec(
            optimizer="miprov2",
            transport="fake",
            demo_mode="fewshot",
            split_sizes=(INTERNAL_SPLIT, 2, 0),
            n_per_stratum=1,
            num_seeds=1,
            seed=2000,
            miprov2_minibatch=True,
            miprov2_minibatch_size=MINIBATCH_TASKS,
            miprov2_minibatch_full_eval_steps=5,
            run_id="c19-miprov2-fanout",
            output_dir=output,
        )
    )


@pytest.fixture(scope="module")
def measurement(minibatched_run: Path) -> FanoutMeasurement:
    return measure_run_directory(minibatched_run)


def test_the_run_actually_minibatched(measurement: FanoutMeasurement) -> None:
    """Guard the fixture: a full-split run would pass everything below.

    If minibatching silently stopped engaging, every evaluation would be a
    full-validation pass, no intent would be a strict subset, and the
    fan-out assertions would hold vacuously. Asserting the setup first is
    what keeps the rest of this module from becoming a tautology.
    """
    assert measurement.minibatch_intents > 0, (
        "no evaluation drew a strict subset, so this run does not exercise "
        "fan-out at all"
    )
    # Only the trial evaluations, not every strict subset: a bootstrap
    # evaluation is a one-task subset too, and it is sized by the trainset
    # rather than by the minibatch schedule.
    sampled = {
        intent.requested
        for intent in measurement.intents
        if intent.purpose is not None and intent.purpose.endswith("sample")
    }
    assert sampled == {MINIBATCH_TASKS}


def test_planned_rows_match_the_requested_subsets(
    measurement: FanoutMeasurement,
) -> None:
    """**The F16 assertion.** Executed rows equal the subset requested.

    Per evaluation, not merely in total: two evaluations fanning out in
    opposite directions would cancel in a total and pass a tolerance.
    """
    fanned = [
        f"{intent.where} requested {intent.requested} and planned "
        f"{intent.planned}"
        for intent in measurement.intents
        if intent.fanned_out
    ]
    assert not fanned, f"row expansion ignored a task subset: {fanned}"
    assert measurement.honours_per_intent_subsets


def test_the_subset_formula_is_the_one_the_code_implements(
    measurement: FanoutMeasurement,
) -> None:
    """Of the two candidate formulas, the measurement picks one.

    The per-intent-subset formula and the ``intents x tasks x seeds``
    formula disagree whenever any evaluation is a strict subset, so this is
    a real discrimination rather than a restatement -- and the fixture test
    above guarantees at least one such evaluation exists.
    """
    assert measurement.planned_rows == measurement.subset_formula_rows
    assert measurement.fanout_ratio == MEASURED_FANOUT_RATIO
    assert measurement.planned_rows < measurement.full_split_formula_rows


def test_the_cost_projection_independently_agrees(
    minibatched_run: Path, measurement: FanoutMeasurement
) -> None:
    """``cost.json`` counts the same rows by a different path.

    ``OptimResult.cost`` is accumulated by the provider transport; the row
    accounting is written by the eval engine. Two records, two writers, one
    number.
    """
    cost = json.loads((minibatched_run / RUN_COST_NAME).read_text())
    task_calls = next(
        entry["calls"]
        for entry in cost["spend"]
        if entry["role"] == "task_model"
    )
    assert task_calls == measurement.planned_rows


def test_the_trainset_is_the_measured_one_task(
    measurement: FanoutMeasurement,
) -> None:
    """F10's correction, re-measured: bootstrapping walks one task.

    ``build_miprov2_control`` slices ``trainset=task_hashes[:1]``, so the
    bootstrap cost is bounded by one task at any split size. This is why
    the measured bootstrap rows are 1-2 rather than the protocol's 28-616,
    and pinning it here means a change to that slicing fails a test that
    names the budget consequence.
    """
    bootstrap = [
        intent
        for intent in measurement.intents
        if intent.purpose is not None and "bootstrap" in intent.purpose
    ]
    assert bootstrap, "the fewshot run issued no bootstrap evaluation"
    for intent in bootstrap:
        assert intent.requested == MEASURED_MIPROV2_TRAINSET_TASKS
        assert intent.planned == MEASURED_MIPROV2_TRAINSET_TASKS


def test_every_measured_evaluation_names_its_surface(
    measurement: FanoutMeasurement,
) -> None:
    """MIPROv2 evaluates through intents, so that is what was read."""
    assert {intent.source for intent in measurement.intents} == {INTENT_SOURCE}


def test_the_study_splits_minibatch_size_is_pinned() -> None:
    """The measured runs used the protocol's own minibatch size."""
    assert MEASURED_MIPROV2_MINIBATCH_TASKS == 35
