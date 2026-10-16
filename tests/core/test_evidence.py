import json
from copy import deepcopy
from html import unescape
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pydantic import ValidationError
from rich.console import Console

from whetstone_envs.evidence import (
    EVAL_REPORT_SCHEMA,
    TRAJECTORY_REPORT_SCHEMA,
    CandidateRecord,
    EvalReport,
    EvalRun,
    EvalSuccess,
    Observation,
    ObservationState,
    ReportedEvidence,
    ReportRef,
    TaskRecord,
    account_observations,
    load_eval_report,
    publish_eval_report,
    summarize_categories,
    two_stage_task_mean,
)
from whetstone_envs.reporting.html import render_html_bytes
from whetstone_envs.reporting.rich_views import render_summary, render_task


def report() -> EvalReport:
    tasks = tuple(
        TaskRecord(
            task_id=f"t{index}",
            task_hash=f"h{index}",
            seed=index,
            categories=("free|form label",),
            prompt_inputs={"value": f"input-{index}"},
            gold="scorer-owned",
        )
        for index in range(2)
    )
    rows = tuple(
        Observation(
            candidate_name="draft",
            task_id=task.task_id,
            task_hash=task.task_hash,
            task_index=index,
            seed_index=seed,
            rendered_prompt=f"Reply {task.prompt_inputs['value']}",
            output_text="reply" if index == 0 else None,
            normalized_output="reply" if index == 0 else None,
            score=float(seed == 0) if index == 0 else None,
            state=ObservationState.SCORED
            if index == 0
            else ObservationState.MISSING,
            trace_state="success" if index == 0 else "missing",
            failure_code="",
            finish_reason=None,
            provider_error=None,
            max_budget=None,
            over_budget=None,
            submission_result=None,
            component_trace=(),
        )
        for index, task in enumerate(tasks)
        for seed in range(2)
    )
    accounting = account_observations(rows)
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
        run=EvalRun(
            run_id="fixture",
            family="user.supplied",
            transport="fake",
            model="fake-model",
            role="internal",
            split_sizes=(2, 1, 0),
            repeats=2,
            dataset_revision="fixture/v1",
            graph_hash="graph",
            eval_config_hash="eval",
            package_version="test",
        ),
        candidates=(
            CandidateRecord(
                name="draft",
                candidate_id="draft",
                source="custom",
                record_ref=ReportRef(
                    schema_name="candidate", content_hash="candidate-hash"
                ),
                identity_hash="identity",
                payload={"instruction": "Reply {value}"},
                mutation_field="instruction",
                prompt_template="Reply {value}",
            ),
        ),
        tasks=tasks,
        observations=rows,
        results=(
            EvalSuccess(
                kind="success",
                candidate_name="draft",
                classification="measured",
                message="fixture",
                evidence=ReportedEvidence(
                    evidence_ref=None,
                    outputs_ref=None,
                    traces_ref=None,
                    aggregate_ref=None,
                    reward_ref=None,
                    aggregate_name="mean",
                    aggregate_value=0.5,
                    aggregate_status="partial",
                    row_accounting=accounting,
                ),
                accounting=accounting,
                numerator=1,
                denominator=4,
                score=0.5,
                categories=summarize_categories(tasks, rows),
            ),
        ),
    )


class EvidenceTests(TestCase):
    def test_schema_literals_and_explicit_mutation_field(self) -> None:
        self.assertEqual(EVAL_REPORT_SCHEMA, "whetstone_envs.eval_report/v5")
        self.assertEqual(
            TRAJECTORY_REPORT_SCHEMA, "whetstone_envs.trajectory_report/v4"
        )
        original = report()
        self.assertEqual(original.candidates[0].mutation_field, "instruction")
        self.assertEqual(two_stage_task_mean(original.observations), 0.5)
        assert isinstance(original.results[0], EvalSuccess)
        self.assertEqual(original.results[0].accounting.missing, 2)
        self.assertEqual(
            EvalReport.model_validate_json(original.model_dump_json()),
            original,
        )

    def test_recorded_scores_are_not_regraded(self) -> None:
        data = report().model_dump(mode="json")
        data["tasks"][0]["gold"] = "not the recorded output"
        data["run"]["family"] = "another arbitrary task"
        self.assertEqual(
            EvalReport.model_validate_json(json.dumps(data))
            .observations[0]
            .score,
            1.0,
        )

    def test_structure_accounting_and_binary_bounds_are_enforced(self) -> None:
        data = report().model_dump(mode="json")
        variants = []
        for score in (0.25, float("inf"), True):
            changed = deepcopy(data)
            changed["observations"][0]["score"] = score
            variants.append(changed)
        changed = deepcopy(data)
        changed["observations"].reverse()
        variants.append(changed)
        changed = deepcopy(data)
        changed["observations"][0]["task_hash"] = "foreign"
        variants.append(changed)
        changed = deepcopy(data)
        changed["results"][0]["numerator"] = 0
        variants.append(changed)
        changed = deepcopy(data)
        changed["candidates"][0]["mutation_field"] = "absent"
        variants.append(changed)
        changed = deepcopy(data)
        changed["schema_version"] = "whetstone_envs.eval_report/v4"
        variants.append(changed)
        for changed in variants:
            with (
                self.subTest(changed=changed),
                self.assertRaises(ValidationError),
            ):
                EvalReport.model_validate_json(json.dumps(changed))

    def test_publication_revalidates_and_preserves_existing_report(
        self,
    ) -> None:
        original = report()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = publish_eval_report(root, original)
            saved = path.read_bytes()
            self.assertEqual(load_eval_report(root), original)
            original.candidates[0].payload["instruction"] = "invalid mutation"
            with self.assertRaises(ValidationError):
                publish_eval_report(root, original)
            self.assertEqual(path.read_bytes(), saved)

    def test_generic_terminal_and_inert_html(self) -> None:
        original = report()
        attack = "</script><script>alert('x')</script>&\u2028"
        original.tasks[0].prompt_inputs["value"] = attack
        output = StringIO()
        console = Console(file=output, force_terminal=False, width=100)
        render_summary(console, original)
        render_task(console, original, "t0")
        self.assertIn("free|form label", output.getvalue())
        self.assertIn("value", output.getvalue())
        rendered = render_html_bytes(original, kind="eval")
        self.assertEqual(rendered, render_html_bytes(original, kind="eval"))
        self.assertNotIn(attack.encode(), rendered)
        self.assertIn(b"\\u003c/script>", rendered)
        self.assertNotIn(b"About C19", rendered)
        self.assertIn("connect-src 'none'", unescape(rendered.decode()))
