from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from whetstone_envs.c19 import PROBES, Action, C19Fact, C19Scenario, C19Size
from whetstone_envs.c19._info import C19_INFO
from whetstone_envs.reporting.rich_views import (
    compare_buckets,
    render_compare,
    render_failures,
    render_trajectory,
)
from whetstone_envs.reporting.schema import (
    TRAJECTORY_REPORT_SCHEMA,
    EvalFailed,
    EvalReport,
    ObservationState,
    ReportRef,
    TrajectoryCandidate,
    TrajectoryReport,
    TrajectoryResolution,
    TrajectoryStep,
)


def test_c19_info_agrees_with_enums_and_probes() -> None:
    assert {item.name for item in C19_INFO.actions} == {
        item.value for item in Action
    }
    assert {item.name for item in C19_INFO.facts} == {
        item.value for item in C19Fact
    }
    assert {item.name for item in C19_INFO.scenarios} == {
        item.value for item in C19Scenario
    }
    assert {item.name for item in C19_INFO.sizes} == {
        item.name.lower() for item in C19Size
    }
    assert C19_INFO.naive_template == PROBES.naive_template
    assert C19_INFO.ceiling_template == PROBES.ceiling_template


def test_compare_includes_expected_complete_bucket(fake_eval_output) -> None:
    buckets = compare_buckets(fake_eval_output.report, "naive", "ceiling")
    assert [row.bucket for row in buckets] == [
        "ceiling only",
        "ceiling only",
    ]


def test_compare_separates_execution_mismatch(fake_eval_output) -> None:
    observations = list(fake_eval_output.report.observations)
    observations[0] = observations[0].model_copy(
        update={
            "score": None,
            "state": ObservationState.FAILED,
            "trace_state": "failed",
        }
    )
    report = fake_eval_output.report.model_copy(
        update={"observations": tuple(observations)}
    )
    buckets = compare_buckets(report, "naive", "ceiling")
    assert buckets[0].bucket == "execution mismatch"


def test_compare_keeps_rows_when_one_candidate_failed(
    fake_eval_output,
) -> None:
    source = fake_eval_output.report
    payload = source.model_dump(mode="json")
    payload["observations"] = [
        row
        for row in payload["observations"]
        if row["candidate_name"] != "naive"
    ]
    payload["results"][0] = EvalFailed(
        kind="failed",
        candidate_name="naive",
        classification="execution",
        message="candidate [bold]failed[/bold] λ",
        evidence_ref=ReportRef(
            schema_name="eval_failure", content_hash="f" * 64
        ),
        exception_type="SyntheticFailure",
    ).model_dump(mode="json")
    report = EvalReport.model_validate_json(json.dumps(payload))

    rows = compare_buckets(report, "naive", "ceiling")

    assert len(rows) == len(report.tasks) * report.run.repeats
    assert all(row.bucket == "execution mismatch" for row in rows)
    assert all(row.candidate_a is None for row in rows)
    stream = StringIO()
    render_compare(
        Console(file=stream, width=180, color_system=None),
        report,
        "naive",
        "ceiling",
    )
    assert "failed: candidate [bold]failed[/bold] λ" in stream.getvalue()


def test_rich_tables_render_persisted_markup_as_literal(
    fake_eval_output,
) -> None:
    source = fake_eval_output.report
    adversarial = "[/bold] [bold]visible[/bold] λ\nnext"
    observations = list(source.observations)
    observations[0] = observations[0].model_copy(
        update={"normalized_output": adversarial}
    )
    report = source.model_copy(update={"observations": tuple(observations)})
    stream = StringIO()
    console = Console(file=stream, width=180, color_system=None)

    render_failures(console, report)
    render_compare(console, report, "naive", "ceiling")

    rendered = stream.getvalue()
    assert "[/bold]" in rendered
    assert "[bold]visible[/bold]" in rendered
    assert "λ" in rendered
    assert "next" in rendered


def _candidate_report(text: str) -> TrajectoryReport:
    record_ref = ReportRef(schema_name="candidate", content_hash="a" * 64)
    base_ref = ReportRef(schema_name="root", content_hash="b" * 64)
    return TrajectoryReport(
        schema_version=TRAJECTORY_REPORT_SCHEMA,
        result_ref=ReportRef(schema_name="result", content_hash="c" * 64),
        run_id="render",
        mutation_field="prompt_template",
        terminal_status="complete",
        candidates=(
            TrajectoryCandidate(
                first_step=0,
                candidate_id="adversarial",
                record_ref=record_ref,
                identity_hash="d" * 64,
                base_ref=base_ref,
                base_candidate_ref=None,
                payload={"prompt_template": text},
                mutation_text=text,
                dispositions=("proposed", "rejected"),
            ),
        ),
        steps=(),
        resolutions=(),
        terminal_candidate_refs=(),
    )


def test_candidate_rendering_preserves_exact_adversarial_text() -> None:
    text = "  λ [bold] {grid}\n\n" + "x" * 180 + "  \n"
    for width in (36, 120):
        stream = StringIO()
        console = Console(
            file=stream,
            width=width,
            color_system=None,
            force_terminal=False,
        )
        render_trajectory(
            console, _candidate_report(text), show_candidates=True
        )
        rendered = stream.getvalue()
        assert "λ [bold] {grid}" in rendered
        assert "xxx" in rendered
        assert "..." not in rendered
        assert "proposed" in rendered
        assert "rejected" in rendered


def test_default_trajectory_view_wraps_refs_relations_and_budgets() -> None:
    base_ref = ReportRef(schema_name="candidate", content_hash="b" * 64)
    child_ref = ReportRef(schema_name="candidate", content_hash="c" * 64)
    report = TrajectoryReport.model_construct(
        schema_version=TRAJECTORY_REPORT_SCHEMA,
        result_ref=ReportRef(schema_name="result", content_hash="e" * 64),
        run_id="table",
        mutation_field="prompt_template",
        terminal_status="complete",
        candidates=(
            TrajectoryCandidate(
                first_step=0,
                candidate_id="base",
                record_ref=base_ref,
                identity_hash="a" * 64,
                base_ref=ReportRef(schema_name="root", content_hash="f" * 64),
                base_candidate_ref=None,
                payload={"prompt_template": "base"},
                mutation_text="base",
                dispositions=("requested",),
            ),
            TrajectoryCandidate(
                first_step=0,
                candidate_id="child",
                record_ref=child_ref,
                identity_hash="d" * 64,
                base_ref=base_ref,
                base_candidate_ref=base_ref,
                payload={"prompt_template": "child"},
                mutation_text="child",
                dispositions=("proposed", "accepted", "evaluated"),
            ),
        ),
        steps=(
            TrajectoryStep(
                step_index=0,
                status="complete",
                request_candidates=(base_ref,),
                proposed_candidates=(child_ref,),
                accepted_candidates=(child_ref,),
                resolution_indexes=(0,),
                budget_delta_consumed={"evals": 1},
                budget_cumulative_consumed={"evals": 3},
                budget_remaining={"evals": 2},
                terminal_failure=None,
            ),
        ),
        resolutions=(
            TrajectoryResolution.model_construct(
                step_index=0,
                resolution_index=0,
                request_id="table-0",
                candidate_ref=child_ref,
                outcome="completed",
                classification="measured",
                message="complete",
                eval_result_ref=None,
                reward_ref=None,
                reward=1.0,
                terminal_failure=None,
                eval_report=None,
            ),
        ),
        terminal_candidate_refs=(child_ref,),
    )
    for width in (40, 120):
        stream = StringIO()
        console = Console(
            file=stream,
            width=width,
            color_system=None,
            force_terminal=False,
        )

        render_trajectory(console, report, show_candidates=False)

        rendered = stream.getvalue()
        unwrapped = " ".join(
            line.strip(" │") for line in rendered.splitlines()
        )
        assert "Step 0 · complete" in rendered
        assert "Resolution 0 · completed" in rendered
        assert "Budget used this step: evals +1" in rendered
        assert "Cumulative used: evals 3" in rendered
        assert "Remaining: evals 2" in rendered
        assert "Candidate: child" in rendered
        assert "Record: cccccccccc" in rendered
        assert "Base / parent: base" in rendered
        assert "Disposition: proposed → accepted → evaluated" in unwrapped
        assert "Evaluation: measured" in rendered
        assert "Reward: 1.0" in rendered
        assert "Message: complete" in rendered
        assert "dddddddddd" not in rendered
        assert "..." not in rendered
