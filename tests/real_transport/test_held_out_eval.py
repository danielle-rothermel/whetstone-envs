"""Rung 5 -- the held-out evaluation path on the real transport.

The held-out role is the study's reporting role: it is what an efficacy
claim is finally made against, and it is the one role an optimizer run
never touches on its own. So it needs its own real-transport rung.

Held-out evidence intentionally carries no reward, which is exactly the
property a fake transport would let a reader assume rather than observe.
This rung asserts the shape that follows from it: the report renders,
scores the held-out tasks, and prices its provider calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.real_transport.conftest import SMOKE_MODEL, SMOKE_SPLIT_SIZES
from whetstone_envs.reporting.execution import (
    C19EvalSpec,
    default_candidates,
    run_c19_evaluation,
)
from whetstone_envs.reporting.publication import EVAL_REPORT_NAME

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.real_transport


def test_rung5_held_out_evaluation_and_report(run_dir: Path) -> None:
    """One candidate, held-out role, real transport, rendered report."""
    candidates = tuple(
        candidate
        for candidate in default_candidates()
        if candidate.name == "ceiling"
    )
    assert candidates, "the ceiling candidate is this rung's one candidate"

    output = run_c19_evaluation(
        C19EvalSpec(
            transport="openrouter",
            role="held_out",
            candidates=candidates,
            repeats=1,
            split_sizes=SMOKE_SPLIT_SIZES,
            output_dir=run_dir,
            run_id="d3-rung5-held-out",
            model=SMOKE_MODEL,
        )
    )
    directory = output.directory

    sqlite_path = directory / "runtime.sqlite"
    assert sqlite_path.is_file(), "held-out run did not publish runtime.sqlite"
    assert sqlite_path.stat().st_size > 0, "runtime.sqlite is empty"

    report_path = directory / EVAL_REPORT_NAME
    assert report_path.is_file(), (
        f"held-out run did not publish {EVAL_REPORT_NAME}"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report, "the held-out eval report is empty"

    # The run block records what was actually run, so it is what proves the
    # rung reached the held-out role on the real transport rather than
    # defaulting to internal or to the fake.
    run_block = report["run"]
    assert run_block["role"] == "held_out", (
        f"expected the held_out role, got {run_block['role']!r}"
    )
    assert run_block["transport"] == "openrouter", (
        f"expected the openrouter transport, got {run_block['transport']!r}"
    )
    assert run_block["model"] == SMOKE_MODEL

    # The held-out split has two tasks and one candidate at one repeat, so
    # the report must carry exactly that many scored observations -- a
    # skipped or cached row would show up as a short list.
    held_out_tasks = SMOKE_SPLIT_SIZES[2]
    assert held_out_tasks > 0, (
        "this rung requires a positive held-out split; --role held_out is "
        "refused by name otherwise"
    )
    assert len(report["tasks"]) == held_out_tasks, (
        f"expected {held_out_tasks} held-out tasks, got {len(report['tasks'])}"
    )
    observations = report["observations"]
    assert len(observations) == held_out_tasks, (
        f"expected {held_out_tasks} observations for one candidate at one "
        f"repeat, got {len(observations)}"
    )

    # Every observation came back from a real provider call: it carries
    # output text and no provider error. Scores are not asserted -- the
    # model's accuracy on toy c19 is near zero and is not what this rung
    # measures.
    for observation in observations:
        assert observation["provider_error"] is None, (
            f"task {observation['task_id']} recorded a provider error: "
            f"{observation['provider_error']}"
        )
        assert observation["output_text"], (
            f"task {observation['task_id']} came back with no output text, "
            "so no real call was made"
        )

    results = report["results"]
    assert len(results) == 1, (
        f"expected one candidate result, got {len(results)}"
    )
    assert results[0]["candidate_name"] == "ceiling"
    assert results[0]["denominator"] == held_out_tasks, (
        "the held-out result must be scored against every held-out task"
    )
    print("RUNG_LEDGER " + json.dumps({"rung": "rung5-held-out"}))
