from __future__ import annotations

from types import SimpleNamespace

from whetstone_envs.optim import ExactMatchEvalProcedureRunner
from whetstone_envs.scoring import exact_match


def test_exact_match_runner_scores_gold_and_is_zero_arg() -> None:
    runner = ExactMatchEvalProcedureRunner()
    task = SimpleNamespace(task_id="t", gold="1,2")

    score, submission, metadata = runner.run_eval_node(
        node_id="evaluate",
        node_inputs={"provider_generation": "1,2"},
        evaluation_procedure_config_hash="h" * 64,
        task=task,
    )

    assert score == float(exact_match("1,2", "1,2"))
    assert submission == {"text": "1,2"}
    assert metadata == {}


def test_exact_match_runner_normalizes_before_compare() -> None:
    runner = ExactMatchEvalProcedureRunner()
    task = SimpleNamespace(task_id="t", gold="yes")

    score, _, _ = runner.run_eval_node(
        node_id="evaluate",
        node_inputs={"provider_generation": "```\nyes\n```"},
        evaluation_procedure_config_hash="h" * 64,
        task=task,
    )

    assert score == 1.0


def test_exact_match_runner_constructs_with_no_arguments() -> None:
    ExactMatchEvalProcedureRunner()
