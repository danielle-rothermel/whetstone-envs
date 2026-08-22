from __future__ import annotations

from typing import cast

import pytest

from whetstone_envs.c19 import PROBES
from whetstone_envs.reporting.execution import (
    C19EvalSpec,
    CandidateInput,
    run_c19_evaluation,
)
from whetstone_envs.reporting.publication import load_eval_report
from whetstone_envs.reporting.schema import EvalRoleName, EvalSuccess


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
