"""One real-transport rung per optimizer arm, on toy c19 splits.

Standing rule: nothing reaches the real experiments having been tested only
against mocks. Each rung drives ``run_optimizer`` through its public entry
point on the REAL OpenRouter transport and asserts on **persisted state** --
``result.json``, ``runtime.sqlite``, a passing ``audit_run``, priced
``run_cost`` rows, and a rendered trajectory report -- rather than on
in-process return values, because persisted state is what a study stage and
the audit actually read.

Rungs are deliberately minimal in shape (two-candidate breadth, two trials,
a four-task internal split) and single-seed. They exist to surface
mock-only assumptions, not to measure anything: a rung asserts that the
machinery ran and recorded truthful evidence, never that the model scored
well. ``openai/gpt-4.1-nano`` scores near zero on these toy c19 tasks, so
any reward-magnitude assertion here would be a flake.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.real_transport.conftest import (
    SMOKE_MODEL,
    SMOKE_SPLIT_SIZES,
    assert_persisted_run,
    rung_ledger,
)
from whetstone_envs.optim.run import (
    DEFAULT_COPRO_BREADTH,
    DEFAULT_COPRO_DEPTH,
    DEFAULT_MIPROV2_MINIBATCH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    RunSpec,
    run_optimizer,
)
from whetstone_envs.reporting.publication import TRAJECTORY_REPORT_NAME

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.real_transport


def _ledger(directory: Path, *, rung: str) -> None:
    """Print one rung's ledger so the transcript carries the cost."""
    print("RUNG_LEDGER " + json.dumps(rung_ledger(directory, rung=rung)))


def _spec(  # noqa: PLR0913
    *,
    optimizer: str,
    run_id: str,
    output_dir: Path,
    family: str = "c19",
    demo_mode: str = "fewshot",
    copro_breadth: int = DEFAULT_COPRO_BREADTH,
    copro_depth: int = DEFAULT_COPRO_DEPTH,
    train_size: int | None = None,
    val_size: int | None = None,
    gepa_max_metric_calls: int | None = None,
    miprov2_num_trials: int = DEFAULT_MIPROV2_NUM_TRIALS,
    miprov2_num_candidates: int = DEFAULT_MIPROV2_NUM_CANDIDATES,
    miprov2_minibatch: bool = DEFAULT_MIPROV2_MINIBATCH,
    miprov2_minibatch_size: int | None = None,
    n_per_stratum: int | None = None,
    split_sizes: tuple[int, int, int] = SMOKE_SPLIT_SIZES,
) -> RunSpec:
    """A rung's spec: real transport, toy split, one seed, minimal shape.

    Spelled out rather than assembled from a mapping so the runner's own
    field types are checked at each call site: a rung that passed a
    wrongly-typed knob would otherwise only fail once it had spent.
    """
    return RunSpec(
        optimizer=optimizer,
        transport="openrouter",
        family=family,
        split_sizes=split_sizes,
        output_dir=output_dir,
        run_id=run_id,
        model=SMOKE_MODEL,
        demo_mode=demo_mode,
        num_seeds=1,
        n_per_stratum=n_per_stratum,
        copro_breadth=copro_breadth,
        copro_depth=copro_depth,
        gepa_max_metric_calls=gepa_max_metric_calls,
        miprov2_minibatch=miprov2_minibatch,
        miprov2_minibatch_size=miprov2_minibatch_size,
        miprov2_num_trials=miprov2_num_trials,
        miprov2_num_candidates=miprov2_num_candidates,
        train_size=train_size,
        val_size=val_size,
    )


def _assert_trajectory_renders(directory: Path, *, optimizer: str) -> None:
    """The trajectory report published and parses as the strict document.

    Rendering is part of every rung because the report is what a reader
    sees; a run whose evidence is sound but whose report cannot render is
    still a blocker for Stage 0.
    """
    path = directory / TRAJECTORY_REPORT_NAME
    assert path.is_file(), (
        f"{optimizer}: {TRAJECTORY_REPORT_NAME} was not published"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["steps"], f"{optimizer}: trajectory report recorded no steps"
    assert report["candidates"], (
        f"{optimizer}: trajectory report recorded no candidates"
    )
    assert report["spend"], (
        f"{optimizer}: trajectory report carries no per-role spend block"
    )


# --------------------------------------------------------------------------
# Rung 1 -- COPRO
# --------------------------------------------------------------------------


def test_rung1_copro(run_dir: Path) -> None:
    """COPRO at the retained search shape: breadth 2, depth 1."""
    directory = run_optimizer(
        _spec(
            optimizer="copro",
            output_dir=run_dir,
            run_id="d3-rung1-copro",
            copro_breadth=2,
            copro_depth=1,
        )
    )
    assert_persisted_run(directory, optimizer="copro")
    _assert_trajectory_renders(directory, optimizer="copro")
    _ledger(directory, rung="rung1-copro")


# --------------------------------------------------------------------------
# Rung 2 -- GEPA
# --------------------------------------------------------------------------


def test_rung2_gepa(run_dir: Path) -> None:
    """GEPA on a required train/val split with a small metric-call cap.

    The valset Pareto selection and incremental search evidence are the
    two things a mock cannot vouch for, so both are asserted through the
    audit's own invariants rather than restated here.
    """
    directory = run_optimizer(
        _spec(
            optimizer="gepa",
            output_dir=run_dir,
            run_id="d3-rung2-gepa",
            train_size=2,
            val_size=2,
            gepa_max_metric_calls=6,
        )
    )
    assert_persisted_run(directory, optimizer="gepa")
    _assert_trajectory_renders(directory, optimizer="gepa")

    report = _audit(directory)
    _assert_invariant_passed(report, "gepa_pareto_front", optimizer="gepa")
    _assert_invariant_passed(
        report, "gepa_step_evidence_present", optimizer="gepa"
    )
    _assert_invariant_passed(
        report, "gepa_train_val_disjoint", optimizer="gepa"
    )
    _ledger(directory, rung="rung2-gepa")


# --------------------------------------------------------------------------
# Rung 3 -- MIPROv2 across every demonstration regime
# --------------------------------------------------------------------------


@pytest.mark.parametrize("demo_mode", ["fewshot", "zeroshot", "ground_only"])
def test_rung3_miprov2(run_dir: Path, demo_mode: str) -> None:
    """MIPROv2 in each demo regime, minibatched, on a required split.

    ``mipro_train_val_disjoint`` is the invariant that carries the
    assignment's "bootstrapped demos carry gold from train tasks only"
    claim: it holds the trainset and valset disjoint and checks that every
    paid evaluation touched only those tasks.

    Two shape constraints are load-bearing here and are not free choices:

    * ``miprov2_minibatch_size`` must be strictly smaller than the valset,
      or ``mipro_minibatch_sizing`` FAILs -- a "minibatch" that covers the
      whole validation split is exactly the fan-out defect that invariant
      exists to catch. That forces a valset larger than the default toy
      split, hence the wider internal split.
    * ``miprov2_num_candidates`` is 3 rather than 2: at 2 candidates with
      minibatching, upstream MIPROv2 raises ``No valid program found in
      param_score_dict``. See the findings note; this is a whetstone-ai
      defect, reproducible on the fake transport, not an envs one.
    """
    directory = run_optimizer(
        _spec(
            optimizer="miprov2",
            output_dir=run_dir,
            run_id=f"d3-rung3-miprov2-{demo_mode}",
            demo_mode=demo_mode,
            split_sizes=(8, 2, 2),
            train_size=3,
            val_size=4,
            miprov2_num_trials=2,
            miprov2_num_candidates=3,
            miprov2_minibatch=True,
            miprov2_minibatch_size=2,
        )
    )
    assert_persisted_run(directory, optimizer="miprov2")
    _assert_trajectory_renders(directory, optimizer="miprov2")

    report = _audit(directory)
    _assert_invariant_passed(
        report, "mipro_train_val_disjoint", optimizer="miprov2"
    )
    _assert_invariant_passed(
        report, "mipro_bootstrap_through_engine", optimizer="miprov2"
    )
    _ledger(directory, rung=f"rung3-miprov2-{demo_mode}")


# --------------------------------------------------------------------------
# Rung 4 -- null-A through the ordinary runner
# --------------------------------------------------------------------------


def test_rung4_null_random(run_dir: Path) -> None:
    """The selection-on-noise control, on the same path a treatment takes.

    null-A only controls for selection if it spends the same proposal
    budget and produces the same evidence as the arm it stands in for, so
    it runs at rung 1's shape and is held to rung 1's assertions. The
    perturbed drafts are uninformative by construction; that they are
    still *evaluated and selected between* is the property under test.
    """
    directory = run_optimizer(
        _spec(
            optimizer="null-random",
            output_dir=run_dir,
            run_id="d3-rung4-null-random",
            copro_breadth=2,
            copro_depth=1,
        )
    )
    assert_persisted_run(directory, optimizer="null-random")
    _assert_trajectory_renders(directory, optimizer="null-random")

    # The control's drafts must actually vary, or null-A has silently
    # become null-B (the pipeline-overhead control) and the study would be
    # reporting the wrong control's delta.
    report = json.loads(
        (directory / "trajectory-report.json").read_text(encoding="utf-8")
    )
    templates = {
        candidate["mutation_text"] for candidate in report["candidates"]
    }
    assert len(templates) > 1, (
        "every null-A candidate carried the same template on the real "
        "transport, so the perturbation never varied the prompt"
    )
    _ledger(directory, rung="rung4-null-random")


# --------------------------------------------------------------------------
# Rung 6 -- the c18 family on the real transport
# --------------------------------------------------------------------------


def test_rung6_copro_c18(run_dir: Path) -> None:
    """Rung 1's shape on the second family, to prove the swap is real.

    c18 has four strata, so ``n_per_stratum=2`` is the smallest pool that
    still fills this rung's ``(4, 2, 2)`` split -- at 1 the pool holds four
    instances and the split is refused. The family adapter is the only
    thing that differs from rung 1.
    """
    directory = run_optimizer(
        _spec(
            optimizer="copro",
            family="c18",
            output_dir=run_dir,
            run_id="d3-rung6-copro-c18",
            copro_breadth=2,
            copro_depth=1,
            n_per_stratum=2,
        )
    )
    assert_persisted_run(directory, optimizer="copro")
    _assert_trajectory_renders(directory, optimizer="copro")
    _ledger(directory, rung="rung6-copro-c18")


# --------------------------------------------------------------------------
# Audit helpers
# --------------------------------------------------------------------------


def _audit(directory: Path):
    from whetstone_envs.optim.audit import audit_run

    return audit_run(directory)


def _assert_invariant_passed(
    report, invariant: str, *, optimizer: str
) -> None:
    """One named invariant was judged and passed.

    Asserting presence as well as status matters: an invariant that
    silently stopped being registered would otherwise let a rung pass by
    checking nothing.
    """
    from whetstone_envs.optim.audit import AuditStatus

    findings = [
        finding
        for finding in report.findings
        if finding.invariant_id.value == invariant
    ]
    assert findings, (
        f"{optimizer}: the audit judged no {invariant!r} invariant, so "
        "this rung asserted nothing about it"
    )
    for finding in findings:
        assert finding.status is not AuditStatus.FAIL, (
            f"{optimizer}: {invariant} FAILed: {finding.detail}"
        )
