from __future__ import annotations

import pytest

from whetstone_envs.c19 import PROBES
from whetstone_envs.reporting.execution import (
    C19EvalSpec,
    CandidateInput,
    run_c19_evaluation,
)
from whetstone_envs.reporting.publication import load_eval_report
from whetstone_envs.reporting.schema import EvalSuccess


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
