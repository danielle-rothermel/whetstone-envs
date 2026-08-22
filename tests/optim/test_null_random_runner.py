"""null-A reaches the ordinary runner, on the fake transport.

The regression these tests pin: ``NullRandomTransport`` is a full
``ProposerTransport`` and null-A is defined as *the optimizer's own search
shape with an uninformative proposer substituted*, but the runner had no
way to select it. The control therefore could not produce the same evidence
its treatment arms produce -- no ``result.json``, no ``runtime.sqlite``, no
audit, no priced cost rows -- and "null-A ran through the ordinary runner"
was unfalsifiable.

These run on the fake transport, so they are ordinary suite tests. The
real-transport rung that exercises the same path lives in
``tests/real_transport/``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.audit import AuditStatus, audit_run
from whetstone_envs.optim.nulls import NULL_RANDOM_OPTIMIZER
from whetstone_envs.optim.run import OPTIMIZERS, RunSpec, run_optimizer

if TYPE_CHECKING:
    from pathlib import Path


def _spec(
    tmp_path: Path,
    *,
    output_dir: Path | None = None,
    run_id: str = "null-random-test",
    seed: int | None = None,
) -> RunSpec:
    """null-A at COPRO's retained shape, on the fake transport."""
    return RunSpec(
        optimizer=NULL_RANDOM_OPTIMIZER,
        transport="fake",
        family="c19",
        split_sizes=(4, 2, 0),
        output_dir=output_dir if output_dir is not None else tmp_path / "run",
        run_id=run_id,
        seed=seed,
        copro_breadth=2,
        copro_depth=1,
    )


def test_null_random_is_an_optimizer_the_runner_can_drive() -> None:
    """The control is selectable, so it can be held to the same evidence."""
    assert NULL_RANDOM_OPTIMIZER in OPTIMIZERS


def test_null_random_run_persists_the_same_evidence_a_treatment_does(
    tmp_path: Path,
) -> None:
    """A control run writes a result, a store, and a passing audit.

    This is the whole point of routing null-A through the ordinary runner:
    a control whose evidence has a different shape from its treatment arms
    cannot be compared against them.
    """
    directory = run_optimizer(_spec(tmp_path))

    assert (directory / "result.json").is_file()
    assert (directory / "runtime.sqlite").is_file()
    assert (directory / "trajectory-report.json").is_file()

    report = audit_run(directory)
    failed = [
        finding
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]
    assert not failed, "; ".join(
        f"{finding.invariant_id.value}: {finding.detail}" for finding in failed
    )
    assert report.findings, "the audit judged no invariants"


def test_null_random_spends_the_proposal_budget_it_stands_in_for(
    tmp_path: Path,
) -> None:
    """A control that skipped slots would control for a different thing.

    null-A must fill the same number of proposal slots COPRO would at the
    same breadth and depth, because it is the control for *selection*: the
    comparison is only meaningful if the same number of candidates were
    selected between.
    """
    directory = run_optimizer(_spec(tmp_path))
    result = json.loads((directory / "result.json").read_text())
    report = json.loads(
        (directory / "trajectory-report.json").read_text(encoding="utf-8")
    )
    assert result
    # breadth 2, depth 1 -> one proposal round plus a finalizing step, the
    # same shape rung 1's COPRO run records.
    assert len(report["steps"]) == 2, (
        f"expected COPRO's 2 steps at depth 1, got {len(report['steps'])}"
    )
    assert len(report["candidates"]) >= 2, (
        "null-A must fill the same proposal slots COPRO would, so at least "
        f"breadth-many candidates, got {len(report['candidates'])}"
    )


def test_null_random_candidates_are_perturbations_not_the_seed(
    tmp_path: Path,
) -> None:
    """null-A varies the prompt; an identity null is a different control.

    Distinguishing the two is load-bearing: null-A is the
    selection-on-noise control and null-B is the pipeline-overhead
    control, and a null-A that returned the seed unchanged would silently
    be null-B.
    """
    directory = run_optimizer(_spec(tmp_path))
    report = json.loads(
        (directory / "trajectory-report.json").read_text(encoding="utf-8")
    )
    templates = {
        candidate["mutation_text"] for candidate in report["candidates"]
    }
    assert len(templates) > 1, (
        "every null-A candidate carried the same template, so the "
        "perturbation never varied the prompt and this is null-B"
    )
    # The perturbation preserves the family's placeholders, or the render
    # contract would reject the candidate and null-A would silently stop
    # filling slots.
    for template in templates:
        assert "{grid}" in template
        assert "{command}" in template
        assert "{question}" in template


def test_null_random_records_its_null_kind_in_durable_evidence(
    tmp_path: Path,
) -> None:
    """A control and a treatment are never mistaken for one another.

    null-A drives COPRO's adapter, so the run's top-level ``optimizer`` is
    ``copro`` -- correctly, because COPRO's search shape is exactly what
    the control holds fixed. What separates the two is the transport's own
    durability identity, which the store records as ``null_kind``. Without
    it, a null-A run and a real COPRO run would be indistinguishable in
    persisted evidence, and the study could not prove which arm it read.
    """
    import sqlite3

    directory = run_optimizer(_spec(tmp_path))
    connection = sqlite3.connect(directory / "runtime.sqlite")
    try:
        payloads = [
            payload
            for (payload,) in connection.execute(
                "select canonical from objects"
            )
        ]
    finally:
        connection.close()

    marked = [payload for payload in payloads if '"null_kind"' in payload]
    assert marked, (
        "no persisted record carries null_kind, so this run's evidence "
        "does not say a control produced it"
    )
    for payload in marked:
        assert '"null_kind":"null_random"' in payload, (
            "a null-A run recorded a null_kind other than null_random"
        )


def test_null_random_honours_the_run_seed(tmp_path: Path) -> None:
    """Two seeds draw different perturbations; one seed reproduces.

    This is what makes ``seed_disposition`` honest for null-A. COPRO's
    control carries no seed field, so a COPRO run is seeded only by the
    provider; null-A keys its own RNG off the run seed, so it really does
    have a control seed and the study can vary it across ``K_RUN``.
    """
    from whetstone_envs.optim.run import (
        SEED_DISPOSITION_CONTROL_FIELD,
        seed_disposition,
    )

    assert (
        seed_disposition(NULL_RANDOM_OPTIMIZER)
        == SEED_DISPOSITION_CONTROL_FIELD
    )

    def templates(seed: int, directory_name: str, run_id: str) -> list[str]:
        directory = run_optimizer(
            _spec(
                tmp_path,
                output_dir=tmp_path / directory_name,
                run_id=run_id,
                seed=seed,
            )
        )
        report = json.loads(
            (directory / "trajectory-report.json").read_text(encoding="utf-8")
        )
        return sorted(
            candidate["mutation_text"] for candidate in report["candidates"]
        )

    # Reproducibility is per ``(seed, run_id)``: the per-slot RNG stream is
    # keyed by the proposal request's identity hash, and the run id is part
    # of that identity. So a re-run of *the same run* reproduces exactly,
    # which is what a replay or a resumed stage needs.
    first = templates(5000, "a", "null-seed-run")
    repeat = templates(5000, "b", "null-seed-run")
    assert first == repeat, (
        "re-running one run id at one seed drew different perturbations, "
        "so null-A is not reproducible"
    )

    # A different seed at the same run id draws a different stream, which
    # is what lets the study vary the control across ``K_RUN``.
    other_seed = templates(7777, "c", "null-seed-run")
    assert first != other_seed, (
        "two seeds drew the same perturbations, so null-A does not read "
        "the run seed and cannot vary across K_RUN"
    )


def test_null_random_refuses_a_train_val_split(tmp_path: Path) -> None:
    """A control has no train/val concept, like the COPRO it stands in for."""
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="train_size"):
        run_optimizer(replace(spec, train_size=2, val_size=2))
