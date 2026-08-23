from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from whetstone_envs.c19 import PROBES
from whetstone_envs.reporting.execution import (
    C19EvalSpec,
    CandidateInput,
    run_c19_evaluation,
)
from whetstone_envs.reporting.publication import (
    DurableRunError,
    load_eval_report,
)
from whetstone_envs.reporting.schema import EvalRoleName, EvalSuccess

if TYPE_CHECKING:
    from dr_store import ObjectStore


def test_fake_naive_ceiling_e2e_and_reload(fake_eval_output) -> None:
    report = load_eval_report(fake_eval_output.directory)
    assert tuple(candidate.name for candidate in report.candidates) == (
        "naive",
        "ceiling",
    )
    assert all(isinstance(result, EvalSuccess) for result in report.results)
    assert tuple(row.candidate_name for row in report.observations) == (
        "naive",
        "naive",
        "ceiling",
        "ceiling",
    )
    results = tuple(
        result for result in report.results if isinstance(result, EvalSuccess)
    )
    assert tuple(result.score for result in results) == (0.0, 1.0)


def test_official_eval_has_successful_rewardless_evidence(tmp_path) -> None:
    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="official",
            candidates=(
                CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
            ),
            split_sizes=(1, 1, 0),
            output_dir=tmp_path / "official",
            run_id="official-test",
        )
    )
    result = output.report.results[0]
    assert isinstance(result, EvalSuccess)
    assert result.score == 1.0
    assert result.evidence.reward_ref is None
    assert output.report.run.role == "official"


def test_custom_utf8_template_and_occurrence_order(tmp_path) -> None:
    custom = "λ {grid}\n{command}\n{question}\n"
    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="internal",
            candidates=(
                CandidateInput("custom-λ", "custom", custom),
                CandidateInput("naive", "naive", PROBES.naive_template),
            ),
            split_sizes=(1, 1, 0),
            output_dir=tmp_path / "custom",
        )
    )
    assert tuple(candidate.name for candidate in output.report.candidates) == (
        "custom-λ",
        "naive",
    )
    assert output.report.candidates[0].prompt_template == custom


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            (
                CandidateInput("same", "custom", PROBES.naive_template),
                CandidateInput("same", "custom", PROBES.ceiling_template),
            ),
            "duplicate candidate name",
        ),
        (
            (CandidateInput("bad", "custom", "{grid} {missing}"),),
            "unavailable fields",
        ),
    ],
)
def test_candidate_validation_precedes_output_creation(
    tmp_path, candidates, message
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=message):
        run_c19_evaluation(
            C19EvalSpec(
                transport="openrouter",
                role="internal",
                candidates=candidates,
                split_sizes=(1, 1, 0),
                output_dir=output,
            )
        )
    assert not output.exists()


def test_held_out_role_runs_through_the_same_path_as_official(
    tmp_path,
) -> None:
    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="held_out",
            candidates=(
                CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
            ),
            split_sizes=(1, 1, 2),
            output_dir=tmp_path / "held-out",
            run_id="held-out-test",
        )
    )
    result = output.report.results[0]
    assert isinstance(result, EvalSuccess)
    assert output.report.run.role == "held_out"
    assert len(output.report.tasks) == 2
    reloaded = load_eval_report(output.directory)
    assert reloaded.run.role == "held_out"
    assert reloaded == output.report


def test_each_role_reports_its_own_split_and_eval_config(tmp_path) -> None:
    reports = {
        role: run_c19_evaluation(
            C19EvalSpec(
                transport="fake",
                role=role,
                candidates=(
                    CandidateInput(
                        "ceiling", "ceiling", PROBES.ceiling_template
                    ),
                ),
                split_sizes=(1, 1, 2),
                output_dir=tmp_path / role,
                run_id=f"role-{role}",
            )
        ).report
        for role in ("internal", "official", "held_out")
    }
    config_hashes = {
        report.run.eval_config_hash for report in reports.values()
    }
    assert len(config_hashes) == 3
    task_ids = {
        role: {task.task_id for task in report.tasks}
        for role, report in reports.items()
    }
    assert task_ids["held_out"].isdisjoint(task_ids["internal"])
    assert task_ids["held_out"].isdisjoint(task_ids["official"])


def test_absent_held_out_split_refuses_rather_than_falling_back(
    tmp_path,
) -> None:
    """A two-role experiment must not silently report official as held-out."""
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="no held_out split"):
        run_c19_evaluation(
            C19EvalSpec(
                transport="fake",
                role="held_out",
                candidates=(
                    CandidateInput(
                        "ceiling", "ceiling", PROBES.ceiling_template
                    ),
                ),
                split_sizes=(1, 1, 0),
                output_dir=output,
                run_id="held-out-absent",
            )
        )
    assert not output.exists()


def test_unsupported_role_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        run_c19_evaluation(
            C19EvalSpec(
                transport="fake",
                role=cast("EvalRoleName", "validation"),
                split_sizes=(1, 1, 1),
            )
        )


# --------------------------------------------------------------------------
# The standalone eval path ledgers what it spent (D3 defect (c))
# --------------------------------------------------------------------------


def test_the_eval_report_carries_a_spend_block(tmp_path) -> None:
    """Every efficacy claim's own evaluation now reports its bill.

    Fails-before: `EvalReport` had no `spend` field at all. Optimizer runs
    projected `cost.json` and a spend block; the standalone eval path --
    the one held-out evaluation every claim is finally made against --
    published `eval-report.json` and `runtime.sqlite` and nothing else, so
    a study's reported cost understated true spend by the whole reporting
    pass.
    """
    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="official",
            candidates=(
                CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
            ),
            split_sizes=(1, 1, 0),
            output_dir=tmp_path / "spend",
            run_id="spend-test",
        )
    )
    spend = output.report.spend
    assert spend is not None
    # One role only: an evaluation runs no proposer, and an all-zero
    # proposer row would claim it measured one and found it free.
    assert spend.task_model.role == "task_model"
    assert spend.task_model.calls == 1
    # The fake transport reports no price, so the honesty rule withholds
    # the total rather than printing a zero that looks authoritative.
    assert spend.task_model.usd is None
    assert spend.task_model.unpriced_calls == spend.task_model.calls


def test_the_eval_path_writes_a_cost_document(tmp_path) -> None:
    """`cost.json` beside the report, in the same shape runs publish.

    Fails-before: `run_c19_evaluation` wrote no cost document at all, so
    there was nothing for a study manifest to cite or a reader to open.
    """
    from whetstone_envs.optim.run_cost import RUN_COST_NAME, read_run_cost

    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="official",
            candidates=(
                CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
            ),
            split_sizes=(1, 1, 0),
            output_dir=tmp_path / "cost",
            run_id="cost-test",
        )
    )
    assert (output.directory / RUN_COST_NAME).is_file()
    document = read_run_cost(output.directory)
    assert document.run_id == "cost-test"
    (record,) = document.spend
    assert record.role == "task_model"
    # The document and the report agree, because the document is projected
    # from the report rather than from a second reading of the rows.
    assert output.report.spend is not None
    assert record.calls == output.report.spend.task_model.calls
    assert record.usd == output.report.spend.task_model.usd


def test_a_report_with_no_provider_rows_publishes_no_cost() -> None:
    """An evaluation that evidenced no call reports nothing, not zero.

    Asserted on the projection directly, because every transport this
    package can run in a test does reach the row-producing path -- and the
    rule that matters is that "no evidence" and "free" stay distinct.
    """
    from whetstone_envs.reporting.projection import project_eval_spend

    class _EmptyStore:
        def get(self, _reference):  # pragma: no cover - never reached
            raise AssertionError("no evidence to dereference")

    assert (
        project_eval_spend(
            store=cast("ObjectStore", _EmptyStore()), results=()
        )
        is None
    )


# --------------------------------------------------------------------------
# The standalone report applies the study's per-task completeness floor
# --------------------------------------------------------------------------


def test_standalone_eval_refuses_a_fully_lost_task(
    tmp_path, monkeypatch
) -> None:
    """``whetstone-eval`` publishes claim-grade numbers, so it takes the floor.

    **Fails-before: published.** The per-task floor lived only in
    ``RoleScorer.evidence_for``, so a study stage refused an evaluation
    that lost a whole task while this path -- the one behind the
    standalone command, and the one a held-out number is finally reported
    from -- projected and published exactly the biased mean the floor
    exists to catch.

    The loss is injected at the evidence rather than at the transport
    because what is under test is that this path *consults* the shared
    helper at all. Whether the helper classifies correctly is settled in
    ``tests/optim/study/test_arms.py``; duplicating that here would test
    the helper twice and the wiring not at all.
    """
    from whetstone_envs.optim.completeness import TaskCompletenessError

    seen: list[str] = []

    def _refuse(_evidence: object, *, purpose: str) -> None:
        seen.append(purpose)
        raise TaskCompletenessError(
            f"{purpose}: 1 of 2 tasks lost every repeat"
        )

    monkeypatch.setattr(
        "whetstone_envs.reporting.execution.require_task_completeness",
        _refuse,
    )
    # Surfaced through the durable-run boundary, which is this path's own
    # contract: the directory already exists, so a failure after it is
    # created is reported with the directory it left behind. The
    # completeness refusal is preserved as the cause rather than flattened.
    with pytest.raises(DurableRunError, match="lost every repeat") as raised:
        run_c19_evaluation(
            C19EvalSpec(
                transport="fake",
                role="official",
                candidates=(
                    CandidateInput(
                        "ceiling", "ceiling", PROBES.ceiling_template
                    ),
                ),
                split_sizes=(1, 1, 0),
                output_dir=tmp_path / "lost",
                run_id="lost-test",
            )
        )
    assert isinstance(raised.value.cause, TaskCompletenessError)
    # Consulted once per candidate, named by the candidate it judged.
    assert seen == ["eval:ceiling"]


def test_standalone_eval_publishes_a_complete_evaluation(tmp_path) -> None:
    """The floor is a floor, not a tax on the path's normal operation."""
    output = run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="official",
            candidates=(
                CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
            ),
            split_sizes=(1, 1, 0),
            output_dir=tmp_path / "complete",
            run_id="complete-test",
        )
    )
    result = output.report.results[0]
    assert isinstance(result, EvalSuccess)
    assert result.score == 1.0
