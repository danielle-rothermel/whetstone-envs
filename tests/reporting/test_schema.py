from __future__ import annotations

import json

import pytest
from pydantic import JsonValue, ValidationError

from whetstone_envs.reporting.schema import (
    EVAL_REPORT_SCHEMA,
    SPLIT_ROLE_BY_REPORT_ROLE,
    TRAJECTORY_REPORT_SCHEMA,
    EvalFailed,
    EvalReport,
    EvalSuccess,
    ReportRef,
    TrajectoryCandidate,
    TrajectoryReport,
    TrajectoryResolution,
    TrajectoryStep,
)


def test_persisted_schema_literals_are_pinned() -> None:
    assert EVAL_REPORT_SCHEMA == "whetstone_envs.eval_report/v1"
    assert TRAJECTORY_REPORT_SCHEMA == "whetstone_envs.trajectory_report/v1"


def test_report_role_names_and_split_roles_are_pinned() -> None:
    assert SPLIT_ROLE_BY_REPORT_ROLE == {
        "internal": "internal_eval",
        "official": "official",
        "held_out": "held_out",
    }


def test_split_role_mapping_agrees_with_upstream_split_roles() -> None:
    from whetstone.experiment.sampling import SPLIT_ROLES

    assert set(SPLIT_ROLE_BY_REPORT_ROLE.values()) == set(SPLIT_ROLES)


def test_eval_report_is_strict_and_forbids_unknown_fields(
    fake_eval_output,
) -> None:
    payload = fake_eval_output.report.model_dump(mode="json")
    payload["unknown"] = True
    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        EvalReport.model_validate_json(json.dumps(payload))


def test_eval_report_rejects_nonfinite_numbers(fake_eval_output) -> None:
    payload = fake_eval_output.report.model_dump(mode="json")
    payload["observations"][0]["score"] = float("nan")
    with pytest.raises(ValidationError):
        EvalReport.model_validate_json(json.dumps(payload))


def test_eval_report_rejects_nonbinary_c19_score(fake_eval_output) -> None:
    payload = fake_eval_output.report.model_dump(mode="json")
    payload["observations"][0]["score"] = 0.5
    with pytest.raises(ValidationError, match="exact binary score"):
        EvalReport.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("score", 1.0, "normalized exact match"),
        ("normalized_output", "tampered", "prediction normalization"),
    ],
)
def test_eval_report_reconciles_c19_output_and_gold(
    fake_eval_output, field, value, message
) -> None:
    payload = fake_eval_output.report.model_dump(mode="json")
    payload["observations"][0][field] = value
    with pytest.raises(ValidationError, match=message):
        EvalReport.model_validate_json(json.dumps(payload))


def test_eval_report_rejects_candidate_major_order_drift(
    fake_eval_output,
) -> None:
    payload = fake_eval_output.report.model_dump(mode="json")
    payload["observations"][0], payload["observations"][1] = (
        payload["observations"][1],
        payload["observations"][0],
    )
    with pytest.raises(ValidationError, match="candidate-major matrix order"):
        EvalReport.model_validate_json(json.dumps(payload))


def test_trajectory_preserves_repeated_rejected_and_failed_resolutions(
    fake_eval_output,
) -> None:
    completed_report = _single_success_report(fake_eval_output.report, 0)
    candidate = completed_report.candidates[0]
    candidate_ref = candidate.record_ref
    completed_result = completed_report.results[0]
    assert isinstance(completed_result, EvalSuccess)
    result_ref = completed_result.evidence.evidence_ref
    assert result_ref is not None
    failure_result_ref = ReportRef(
        schema_name="whetstone.eval.failure", content_hash="b" * 64
    )
    failed_report = EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
        run=completed_report.run,
        candidates=completed_report.candidates,
        tasks=completed_report.tasks,
        observations=(),
        results=(
            EvalFailed(
                kind="failed",
                candidate_name=candidate.name,
                classification="execution",
                message="failed",
                evidence_ref=failure_result_ref,
                exception_type="RuntimeError",
            ),
        ),
    )
    failure: dict[str, JsonValue] = {
        "classification": "provider",
        "message": "failed",
    }
    report = TrajectoryReport(
        schema_version=TRAJECTORY_REPORT_SCHEMA,
        result_ref=ReportRef(schema_name="optim", content_hash="c" * 64),
        run_id="trajectory-contract",
        mutation_field="prompt_template",
        terminal_status="failed",
        candidates=(
            TrajectoryCandidate(
                first_step=0,
                candidate_id=candidate.candidate_id,
                record_ref=candidate_ref,
                identity_hash=candidate.identity_hash,
                base_ref=ReportRef(schema_name="root", content_hash="e" * 64),
                base_candidate_ref=None,
                payload=candidate.payload,
                mutation_text=candidate.prompt_template,
                dispositions=("proposed", "rejected"),
            ),
        ),
        steps=(
            TrajectoryStep(
                step_index=0,
                status="failed",
                request_candidates=(candidate_ref,),
                proposed_candidates=(candidate_ref,),
                accepted_candidates=(),
                resolution_indexes=(0, 1, 2),
                budget_delta_consumed={"evals": 2},
                budget_cumulative_consumed={"evals": 3},
                budget_remaining={"evals": 0},
                terminal_failure=failure,
            ),
        ),
        resolutions=(
            TrajectoryResolution(
                step_index=0,
                resolution_index=0,
                request_id="repeat-0",
                candidate_ref=candidate_ref,
                outcome="completed",
                classification="measured",
                message="complete",
                eval_result_ref=result_ref,
                reward_ref=completed_result.evidence.reward_ref,
                reward=completed_result.score,
                terminal_failure=None,
                eval_report=completed_report,
                gains=1,
                regressions=0,
                execution_mismatches=0,
            ),
            TrajectoryResolution(
                step_index=0,
                resolution_index=1,
                request_id="repeat-1",
                candidate_ref=candidate_ref,
                outcome="rejected",
                classification="validation",
                message="rejected",
                eval_result_ref=None,
                reward_ref=None,
                reward=None,
                terminal_failure=None,
                eval_report=None,
            ),
            TrajectoryResolution(
                step_index=0,
                resolution_index=2,
                request_id="repeat-2",
                candidate_ref=candidate_ref,
                outcome="failed",
                classification="provider",
                message="failed",
                eval_result_ref=failure_result_ref,
                reward_ref=None,
                reward=None,
                terminal_failure=failure,
                eval_report=failed_report,
            ),
        ),
        terminal_candidate_refs=(),
    )
    reloaded = TrajectoryReport.model_validate_json(report.model_dump_json())
    assert [row.outcome for row in reloaded.resolutions] == [
        "completed",
        "rejected",
        "failed",
    ]
    assert reloaded.steps[0].budget_delta_consumed == {"evals": 2}
    assert reloaded.steps[0].budget_cumulative_consumed == {"evals": 3}

    payload = report.model_dump(mode="json")
    payload["resolutions"][0]["eval_report"] = _single_success_report(
        fake_eval_output.report, 1
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="exact resolution candidate"):
        TrajectoryReport.model_validate_json(json.dumps(payload))


def _single_success_report(report: EvalReport, index: int) -> EvalReport:
    candidate = report.candidates[index]
    result = report.results[index]
    assert isinstance(result, EvalSuccess)
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
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


def test_run_spend_wire_keys_are_pinned() -> None:
    """The spend block's persisted keys are pinned by exact literal."""
    from whetstone_envs.reporting.schema import RoleSpend, RunSpend

    spend = RunSpend(
        schema_version=1,
        task_model=RoleSpend(
            role="task_model",
            calls=3,
            cached_calls=1,
            input_tokens=120,
            output_tokens=40,
            priced_calls=0,
            unpriced_calls=3,
            rows_missing_token_breakdown=3,
            usd=None,
        ),
        proposer=RoleSpend(
            role="proposer",
            calls=2,
            cached_calls=0,
            input_tokens=60,
            output_tokens=10,
            priced_calls=2,
            unpriced_calls=0,
            rows_missing_token_breakdown=0,
            usd=0.125,
        ),
    )

    assert spend.model_dump(mode="json") == {
        "schema_version": 1,
        "task_model": {
            "role": "task_model",
            "calls": 3,
            "cached_calls": 1,
            "input_tokens": 120,
            "output_tokens": 40,
            "priced_calls": 0,
            "unpriced_calls": 3,
            "rows_missing_token_breakdown": 3,
            "usd": None,
        },
        "proposer": {
            "role": "proposer",
            "calls": 2,
            "cached_calls": 0,
            "input_tokens": 60,
            "output_tokens": 10,
            "priced_calls": 2,
            "unpriced_calls": 0,
            "rows_missing_token_breakdown": 0,
            "usd": 0.125,
        },
    }


def test_run_spend_rejects_a_role_filed_under_the_wrong_name() -> None:
    from whetstone_envs.reporting.schema import RoleSpend, RunSpend

    role = RoleSpend(
        role="proposer",
        calls=0,
        cached_calls=0,
        input_tokens=0,
        output_tokens=0,
        priced_calls=0,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=None,
    )
    with pytest.raises(ValidationError, match="task_model role"):
        RunSpend(schema_version=1, task_model=role, proposer=role)


def test_run_spend_schema_version_is_pinned_to_the_supported_value() -> None:
    """FAILS-BEFORE probe: an unknown cost-report version must be rejected."""
    from whetstone.optim.cost import COST_REPORT_SCHEMA_VERSION

    from whetstone_envs.reporting.schema import (
        SPEND_SCHEMA_VERSION,
        RoleSpend,
        RunSpend,
    )

    assert SPEND_SCHEMA_VERSION == 1
    assert SPEND_SCHEMA_VERSION == COST_REPORT_SCHEMA_VERSION

    role = RoleSpend(
        role="task_model",
        calls=0,
        cached_calls=0,
        input_tokens=0,
        output_tokens=0,
        priced_calls=0,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=None,
    )
    proposer = role.model_copy(update={"role": "proposer"})
    payload = RunSpend(
        schema_version=SPEND_SCHEMA_VERSION,
        task_model=role,
        proposer=proposer,
    ).model_dump(mode="json")
    payload["schema_version"] = 999
    with pytest.raises(ValidationError, match="schema_version"):
        RunSpend.model_validate(payload)
