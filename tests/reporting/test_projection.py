from __future__ import annotations

from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.optim.contracts import OptimResult

from whetstone_envs.reporting.projection import (
    _comparison,
    _provider_error_projection,
    _TrajectoryEvaluationHistory,
)
from whetstone_envs.reporting.schema import EvalReport, EvalSuccess


def test_provider_error_projection_excludes_credential_bearing_data() -> None:
    raw = ImmutableJsonObject(
        {
            "failure_class": "rate-limit",
            "message": "Authorization: Bearer exposed",
            "api_key": "exposed-key",
            "transport_failure": {
                "recoverability": "rate_limited",
                "code": "api-key-exposed",
                "message": "cookie=exposed-cookie",
                "status_code": 429,
                "containment": None,
                "response_body": {"token": "exposed-token"},
                "metadata": {"headers": {"authorization": "exposed"}},
            },
        }
    )

    projected = _provider_error_projection(raw)

    assert projected is not None
    assert projected.model_dump(mode="json") == {
        "failure_class": "rate-limit",
        "source": "transport_failure",
        "recoverability": "rate_limited",
        "status_code": 429,
        "timeout_containment": None,
    }
    assert "exposed" not in projected.model_dump_json()


def test_comparison_counts_failed_planned_rows_and_keeps_snapshots(
    fake_eval_output,
) -> None:
    base = _single_success_report(fake_eval_output.report, 0)
    improved = base.model_copy(
        update={
            "observations": tuple(
                row.model_copy(update={"score": 1.0})
                for row in base.observations
            )
        }
    )
    failed = base.model_copy(update={"observations": ()})
    assert _comparison(base, failed) == (0, 0, len(base.tasks))

    external_ref = TypedRef(schema_name="root", content_hash="e" * 64)
    base_ref = TypedRef(schema_name="candidate", content_hash="a" * 64)
    child_ref = TypedRef(schema_name="candidate", content_hash="b" * 64)
    history = _TrajectoryEvaluationHistory()
    assert history.compare_then_remember(
        candidate_ref=base_ref,
        base_ref=external_ref,
        report=base,
    ) == (None, None, None)
    first_comparison = history.compare_then_remember(
        candidate_ref=child_ref,
        base_ref=base_ref,
        report=improved,
    )
    assert first_comparison == (len(base.tasks), 0, 0)

    history.compare_then_remember(
        candidate_ref=base_ref,
        base_ref=external_ref,
        report=improved,
    )
    assert history.compare_then_remember(
        candidate_ref=child_ref,
        base_ref=base_ref,
        report=base,
    ) == (0, len(base.tasks), 0)
    assert first_comparison == (len(base.tasks), 0, 0)


def _single_success_report(report: EvalReport, index: int) -> EvalReport:
    candidate = report.candidates[index]
    result = report.results[index]
    assert isinstance(result, EvalSuccess)
    return EvalReport(
        schema_version=report.schema_version,
        run=report.run,
        candidates=(candidate,),
        tasks=report.tasks,
        observations=tuple(
            row
            for row in report.observations
            if row.candidate_name == candidate.name
        ),
        results=(result,),
    )


def test_project_run_spend_reads_the_serialized_cost_report() -> None:
    from whetstone.optim.cost import RoleCost, RunCostReport

    from whetstone_envs.reporting.projection import project_run_spend

    report = RunCostReport(
        task_model=RoleCost(
            calls=4,
            input_tokens=100,
            output_tokens=20,
            priced_calls=0,
            unpriced_calls=4,
            cached_calls=1,
            rows_missing_token_breakdown=4,
        ),
        proposer=RoleCost(
            calls=2,
            input_tokens=30,
            output_tokens=5,
            priced_calls=2,
            unpriced_calls=0,
            usd=0.25,
        ),
    )

    result = OptimResult.model_construct(
        cost=ImmutableJsonObject(report.model_dump(mode="json"))
    )

    spend = project_run_spend(result)

    assert spend is not None
    assert spend.task_model.role == "task_model"
    assert spend.task_model.calls == 4
    assert spend.task_model.cached_calls == 1
    assert spend.task_model.rows_missing_token_breakdown == 4
    # No total when a contributing call carried no price.
    assert spend.task_model.usd is None
    assert spend.proposer.role == "proposer"
    assert spend.proposer.usd == 0.25


def test_project_run_spend_is_absent_without_a_cost_report() -> None:
    from whetstone_envs.reporting.projection import project_run_spend

    result = OptimResult.model_construct(cost=ImmutableJsonObject({}))

    assert project_run_spend(result) is None


def test_role_spend_rejects_a_total_alongside_unpriced_calls() -> None:
    import pytest

    from whetstone_envs.reporting.schema import RoleSpend

    with pytest.raises(ValueError, match="reports no total spend"):
        RoleSpend(
            role="proposer",
            calls=2,
            cached_calls=0,
            input_tokens=0,
            output_tokens=0,
            priced_calls=1,
            unpriced_calls=1,
            rows_missing_token_breakdown=0,
            usd=1.0,
        )


def test_role_spend_requires_priced_and_unpriced_to_exhaust_calls() -> None:
    import pytest

    from whetstone_envs.reporting.schema import RoleSpend

    with pytest.raises(ValueError, match="exhaust billable calls"):
        RoleSpend(
            role="task_model",
            calls=5,
            cached_calls=0,
            input_tokens=0,
            output_tokens=0,
            priced_calls=1,
            unpriced_calls=1,
            rows_missing_token_breakdown=0,
            usd=None,
        )
