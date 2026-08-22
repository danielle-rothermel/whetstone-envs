from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.core.identity import TypedRef
from whetstone.eval import EvalEvidence
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import OptimResult

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.cli import main
from whetstone_envs.optim.experiment import C19_MUTATION_FIELD
from whetstone_envs.optim.run import C19RunSpec, run_c19_optimizer
from whetstone_envs.reporting.publication import (
    DurableRunError,
    load_trajectory_report,
)

#: MIPROv2 cannot run against a C19 experiment until whetstone-ai stops
#: hardcoding the toy mutation field. ``_materialize_bootstrap_teacher``
#: reads ``payload["user_prompt_template"]`` instead of
#: ``state.control.mutation_field``, so every demo mode raises KeyError on
#: the first bootstrap teacher. Every mode bootstraps, so no mode escapes it.
MIPROV2_UPSTREAM_BLOCKER = (
    "whetstone-ai miprov2/runtime.py:_materialize_bootstrap_teacher "
    "hardcodes the 'user_prompt_template' mutation field"
)


@pytest.mark.parametrize("optimizer", ["copro", "gepa"])
def test_fake_transport_completes(  # noqa: PLR0915
    tmp_path, optimizer: str
) -> None:
    output = tmp_path / f"{optimizer}-run"
    code = main(
        [
            "--family",
            "c19",
            "--optimizer",
            optimizer,
            "--transport",
            "fake",
            "--split-sizes",
            "2,2,0",
            "--run-id",
            f"c19-{optimizer}-e2e",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"
    assert result.proposals
    assert (output / "trajectory-report.json").is_file()
    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    assert all(row.eval_report is not None for row in trajectory.resolutions)

    if optimizer == "gepa":
        from whetstone.optim.gepa.contracts import (
            GepaEffectTranscript,
            GepaEvaluationEffectResult,
        )
        from whetstone.optim.gepa.harness_adapter import (
            GEPA_TERMINAL_ARTIFACT_KEY,
        )
        from whetstone.optim.gepa.result_artifact import GepaRunResultArtifact

        template = result.proposals[0].candidate.record.payload[
            C19_MUTATION_FIELD
        ]
        assert template == PROBES.ceiling_template
        assert template != PROBES.naive_template
        final_step = result.step_results[-1].record
        assert final_step.history_ref is not None
        expected = []
        with open_sqlite(str(output / "runtime.sqlite")) as store:
            history = store.get(final_step.history_ref.reference)
            assert isinstance(history, dict)
            artifact_ref = TypedRef.model_validate(
                history[GEPA_TERMINAL_ARTIFACT_KEY]
            )
            artifact = GepaRunResultArtifact.model_validate_json(
                json.dumps(store.get(artifact_ref.reference))
            )
            transcript = GepaEffectTranscript.model_validate_json(
                json.dumps(store.get(artifact.effect_transcript_ref.reference))
            )
            for entry in transcript.entries:
                if entry.effect_kind != "evaluate":
                    continue
                effect_result = GepaEvaluationEffectResult.model_validate_json(
                    json.dumps(store.get(entry.result_ref.reference))
                )
                assert effect_result.resolution is not None
                resolution = effect_result.resolution
                assert resolution.eval_result_ref is not None
                evidence = EvalEvidence.model_validate_json(
                    json.dumps(store.get(resolution.eval_result_ref.reference))
                )
                expected.append(
                    (
                        int(resolution.optim_eval_request.optim_step_index),
                        entry.invocation_ordinal,
                        resolution.optim_eval_request.eval_request.request_id,
                        resolution.optim_eval_request.eval_request.candidate,
                        evidence.task_hashes,
                    )
                )
        assert expected
        actual = []
        for row in trajectory.resolutions:
            assert row.eval_report is not None
            actual.append(
                (
                    row.step_index,
                    row.resolution_index,
                    row.request_id,
                    row.candidate_ref.content_hash,
                    tuple(task.task_hash for task in row.eval_report.tasks),
                )
            )
        assert tuple(actual) == tuple(
            (
                step_index,
                invocation_ordinal,
                request_id,
                candidate_reference(candidate).record_ref.content_hash,
                task_hashes,
            )
            for (
                step_index,
                invocation_ordinal,
                request_id,
                candidate,
                task_hashes,
            ) in expected
        )
        candidate_counts = Counter(
            candidate_reference(candidate).record_ref.content_hash
            for _step, _ordinal, _request, candidate, _tasks in expected
        )
        assert any(count > 1 for count in candidate_counts.values())
        assert len(trajectory.resolutions) < sum(
            len(step.record.search_evidence) for step in result.step_results
        )
    else:
        assert tuple(step.step_index for step in trajectory.steps) == (0, 1)
        assert tuple(
            (row.step_index, row.resolution_index)
            for row in trajectory.resolutions
        ) == ((0, 0), (0, 1))
        assert any(
            candidate.base_candidate_ref is not None
            for candidate in trajectory.candidates
        )
        assert all(row.gains is None for row in trajectory.resolutions)
        assert all(row.regressions is None for row in trajectory.resolutions)
        assert trajectory.steps[0].budget_delta_consumed == {
            "proposal_calls": 1
        }
        assert trajectory.steps[0].budget_cumulative_consumed == {
            "proposal_calls": 1
        }
        assert trajectory.steps[0].budget_remaining == {"proposal_calls": 0}


def test_projection_failure_preserves_runtime_and_terminal_result(
    tmp_path,
) -> None:
    output = tmp_path / "projection-failure"
    with (
        patch(
            "whetstone_envs.optim.run.project_trajectory_report",
            side_effect=RuntimeError("projection boom"),
        ),
        pytest.raises(DurableRunError, match="projection boom") as captured,
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                split_sizes=(2, 2, 0),
                output_dir=output,
                run_id="c19-projection-failure",
            )
        )

    assert captured.value.directory == output.resolve()
    assert (output / "runtime.sqlite").is_file()
    OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert not (output / "trajectory-report.json").exists()


def test_run_refuses_in_repo_output() -> None:
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=Path("artifacts") / "c19-run",
            )
        )


def test_run_refuses_in_repo_output_when_cwd_is_elsewhere(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=repo_root / "artifacts" / "c19-run",
            )
        )


@pytest.mark.parametrize("demo_mode", ["fewshot", "zeroshot", "ground_only"])
@pytest.mark.xfail(reason=MIPROV2_UPSTREAM_BLOCKER, strict=True)
def test_miprov2_fake_transport_completes(tmp_path, demo_mode: str) -> None:
    """MIPROv2 completes on the fake transport in every demonstration mode.

    Demonstrations reach the candidate through MIPROv2's own composed
    ``### Demonstrations`` section: ``fewshot`` renders the selected demo set
    there, while ``zeroshot`` and ``ground_only`` leave it empty.
    """
    output = tmp_path / f"miprov2-{demo_mode}-run"
    code = main(
        [
            "--family",
            "c19",
            "--optimizer",
            "miprov2",
            "--demo-mode",
            demo_mode,
            "--transport",
            "fake",
            "--split-sizes",
            "2,2,0",
            "--run-id",
            f"c19-miprov2-{demo_mode}-e2e",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"
    # MIPROv2 drives its search through Steps, not SearchEvidence.
    assert all(not step.record.search_evidence for step in result.step_results)
    assert any(step.record.proposer_usage for step in result.step_results)

    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    spend = trajectory.spend
    assert spend is not None
    assert spend.proposer.calls > 0
    assert spend.task_model.calls > 0


def test_runner_rejects_unknown_demo_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported demo mode"):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="miprov2",
                transport="fake",
                output_dir=tmp_path / "bad-demo-mode",
                demo_mode="handful",
            )
        )


def test_runner_rejects_non_positive_num_seeds(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_seeds must be at least 1"):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=tmp_path / "bad-num-seeds",
                num_seeds=0,
            )
        )


@pytest.mark.parametrize("optimizer", ["copro", "gepa"])
def test_fake_transport_reports_run_spend(tmp_path, optimizer: str) -> None:
    """Every optimizer's trajectory report carries the run's spend."""
    output = tmp_path / f"{optimizer}-spend"
    assert (
        main(
            [
                "--optimizer",
                optimizer,
                "--transport",
                "fake",
                "--run-id",
                f"c19-{optimizer}-spend",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    trajectory = load_trajectory_report(output)
    spend = trajectory.spend
    assert spend is not None
    assert spend.task_model.role == "task_model"
    assert spend.proposer.role == "proposer"
    for role in (spend.task_model, spend.proposer):
        assert role.priced_calls + role.unpriced_calls == role.calls
        # A fake transport reports no price, so no total is presented.
        if role.unpriced_calls:
            assert role.usd is None
