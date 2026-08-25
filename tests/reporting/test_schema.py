from __future__ import annotations

import json

import pytest
from pydantic import JsonValue, ValidationError

from whetstone_envs.probes import normalize
from whetstone_envs.reporting.schema import (
    EVAL_REPORT_SCHEMA,
    SPLIT_ROLE_BY_REPORT_ROLE,
    TRAJECTORY_REPORT_SCHEMA,
    EvalFailed,
    EvalReport,
    EvalSuccess,
    Observation,
    ObservationState,
    ReportRef,
    TrajectoryCandidate,
    TrajectoryReport,
    TrajectoryResolution,
    TrajectoryStep,
)


def test_persisted_schema_literals_are_pinned() -> None:
    assert EVAL_REPORT_SCHEMA == "whetstone_envs.eval_report/v2"
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
        ("score", 1.0, "disagrees with the c19 scorer"),
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


def _invalid_observation(trace_state: str) -> Observation:
    """One blank-generation row, at a given trace state."""
    return Observation(
        candidate_name="naive",
        task_id="t-1",
        task_hash="a" * 64,
        task_index=0,
        seed_index=0,
        rendered_prompt="prompt",
        output_text=None,
        normalized_output=None,
        score=None,
        state=ObservationState.INVALID,
        trace_state=trace_state,  # ty: ignore[invalid-argument-type]
        failure_code="blank-provider-generation",
        finish_reason=None,
        provider_error=None,
        max_budget=None,
        over_budget=None,
        submission_result=None,
        component_trace=(),
    )


def test_an_invalid_observation_requires_an_invalid_trace_state() -> None:
    """**Fails-before: an invalid row's trace had to say ``failed``.**

    whetstone-ai 0.1.14 gave ``ExecutedRowState`` its own ``invalid``
    member for a row that executed and was billed but produced nothing
    the eval contract can score -- a blank generation, which 0.1.14 also
    began scoring as terminally invalid rather than as a node execution
    error. Before that member existed the trace could only spell such a
    row ``failed``, so this reconciliation folded the two states
    together. Under 0.1.14 that fold would reject exactly the rows the
    upstream change made reachable, and ``trace_state``'s closed literal
    would not have admitted the new spelling at all.
    """
    observation = _invalid_observation("invalid")
    assert observation.state is ObservationState.INVALID
    assert observation.trace_state == "invalid"


def test_an_invalid_observation_rejects_a_failed_trace_state() -> None:
    """The fold is gone in both directions, not merely widened.

    Accepting either spelling would leave the report unable to say which
    of the two upstream states a row was actually in, which is the whole
    reason the trace state is carried beside the row state.
    """
    with pytest.raises(ValidationError, match="trace state disagrees"):
        _invalid_observation("failed")


def test_the_trace_states_are_exactly_upstreams_executed_row_states() -> None:
    """Pinned against the upstream enum, so a widening cannot pass quietly.

    ``trace_state`` is a persisted report field restated as a closed
    literal rather than typed as the imported enum, which is what makes a
    new upstream member a deliberate change here. This is the check that
    makes the restatement safe rather than merely duplicated: it fails
    when whetstone-ai grows a row state this package has not considered.
    """
    from typing import get_args

    from whetstone.eval.traces import ExecutedRowState

    field = Observation.model_fields["trace_state"]
    assert set(get_args(field.annotation)) == {
        member.value for member in ExecutedRowState
    }


def test_eval_report_scores_each_family_by_its_own_scorer(
    fake_eval_output,
) -> None:
    """A c18 verdict reply is scored the way c18 scores it.

    C18's ceiling probe asks for reasoning ending in a lone verdict
    line, and :func:`whetstone_envs.c18.score_gold` extracts that
    verdict before comparing. The schema re-derives a reported score to
    check it, so re-deriving it as raw normalized exact match -- a c19
    rule -- made a *correct* c18 answer look like a lie: the row said
    1.0, the schema recomputed 0.0, and the whole report was refused.

    Fails-before: with the check hard-coded to normalized exact match
    this raises ``observation score disagrees``, and the c18 arm could
    not publish a report at all.

    The same payload with the c19 family is the control: c19 has no
    verdict extraction, so there the reasoned reply genuinely does not
    match gold and 1.0 is genuinely wrong.
    """
    payload = fake_eval_output.report.model_dump(mode="json")
    reasoned = "The rules entail the query.\nTrue"
    payload["run"]["family"] = "c18"
    # C18 gold is the bare verdict; the reply reasons its way to it.
    for task in payload["tasks"]:
        task["gold"] = "True"
    for row in payload["observations"]:
        row["output_text"] = reasoned
        row["normalized_output"] = normalize(reasoned)
        row["score"] = 1.0
    # Every row now scores 1.0, so each result's own tally has to say so.
    for result in payload["results"]:
        if result["kind"] != "success":
            continue
        rows = [
            row
            for row in payload["observations"]
            if row["candidate_name"] == result["candidate_name"]
        ]
        result["numerator"] = len(rows)
        result["denominator"] = len(rows)
        result["score"] = 1.0
        for stratum in result["strata"]:
            stratum["numerator"] = stratum["denominator"]
            stratum["score"] = 1.0

    report = EvalReport.model_validate_json(json.dumps(payload))

    assert report.run.family == "c18"
    assert all(row.score == 1.0 for row in report.observations)

    # Control: c19 does not extract a verdict, so the same rows are wrong.
    payload["run"]["family"] = "c19"
    with pytest.raises(ValidationError, match="disagrees with the c19 scorer"):
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


def test_eval_spend_rejects_a_role_filed_under_the_wrong_name() -> None:
    """A standalone evaluation's one row must be the task model's.

    ``EvalSpend`` carries a single ``RoleSpend`` and no proposer, because a
    standalone evaluation runs no optimizer. Nothing in the field's own
    type stops a *proposer* row from being filed there -- the annotation
    admits either literal -- so the validator is what keeps the field's
    name and its contents in agreement. Without this test the check was
    unexercised, and a proposer row filed as the task model would have
    been reported as task-model spend.
    """
    from whetstone_envs.reporting.schema import EvalSpend, RoleSpend

    proposer = RoleSpend(
        role="proposer",
        calls=1,
        cached_calls=0,
        input_tokens=8,
        output_tokens=2,
        priced_calls=1,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.5,
    )
    with pytest.raises(ValidationError, match="task_model role"):
        EvalSpend(schema_version=1, task_model=proposer)


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
