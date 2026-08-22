from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whetstone_envs.c19._info import C19_INFO, NamedDescription
from whetstone_envs.reporting.schema import (
    EvalReport,
    EvalSuccess,
    Observation,
    ObservationState,
    TrajectoryReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _description_table(title: str, rows: Iterable[NamedDescription]) -> Table:
    table = Table(title=title, box=None, show_header=False)
    table.add_column(style="bold")
    table.add_column()
    for row in rows:
        table.add_row(row.name, row.description)
    return table


def render_c19_info(console: Console, *, show_templates: bool) -> None:
    console.print(Panel(C19_INFO.objective, title="C19"))
    for title, rows in (
        ("Public inputs and private gold", C19_INFO.public_inputs),
        ("Grid sizes", C19_INFO.sizes),
        ("Scenario families", C19_INFO.scenarios),
        ("Actions and no-op rules", C19_INFO.actions),
        ("Grid tokens and coordinates", C19_INFO.tokens),
        ("Facts and exact answer forms", C19_INFO.facts),
    ):
        console.print(_description_table(title, rows))
    console.print(Panel(C19_INFO.scoring, title="Normalization and scoring"))
    console.print(Panel(C19_INFO.pool, title="Default pool and split roles"))
    console.print(
        _description_table("Candidate terminology", C19_INFO.terminology)
    )
    if show_templates:
        console.print(Panel(Text(C19_INFO.naive_template), title="naive"))
        console.print(Panel(Text(C19_INFO.ceiling_template), title="ceiling"))


def _accounting_text(result: EvalSuccess) -> str:
    accounting = result.accounting
    score = "incomplete" if result.score is None else f"{result.score:.3f}"
    return (
        f"{result.numerator}/{result.denominator} score={score} "
        f"scored={accounting.present} failed={accounting.failed} "
        f"missing={accounting.missing} invalid={accounting.invalid}"
    )


def render_summary(console: Console, report: EvalReport) -> None:
    console.print(
        Panel(
            Text(
                f"family={report.run.family} role={report.run.role} "
                f"transport={report.run.transport} "
                f"model={report.run.model}\n"
                f"tasks={len(report.tasks)} repeats={report.run.repeats}"
            ),
            title=Text(report.run.run_id),
        )
    )
    table = Table(title="Candidates")
    table.add_column("candidate")
    table.add_column("result")
    table.add_column("accounting")
    for result in report.results:
        if isinstance(result, EvalSuccess):
            table.add_row(
                Text(result.candidate_name),
                Text(result.kind),
                Text(_accounting_text(result)),
            )
        else:
            table.add_row(
                Text(result.candidate_name),
                Text(result.kind),
                Text(result.message),
            )
    console.print(table)
    strata = Table(title="C19 strata")
    strata.add_column("candidate")
    strata.add_column("stratum")
    strata.add_column("score")
    for result in report.results:
        if not isinstance(result, EvalSuccess):
            continue
        for summary in result.strata:
            value = (
                "incomplete"
                if summary.score is None
                else f"{summary.score:.3f}"
            )
            strata.add_row(
                Text(result.candidate_name),
                Text(summary.stratum),
                Text(f"{summary.numerator}/{summary.denominator} ({value})"),
            )
    console.print(strata)


def _matches(  # noqa: PLR0913
    row: Observation,
    report: EvalReport,
    *,
    candidate: str | None,
    scenario: str | None,
    size: str | None,
    fact: str | None,
) -> bool:
    if candidate is not None and row.candidate_name != candidate:
        return False
    task = report.tasks[row.task_index]
    parts = task.strata[0].split("|")
    return all(
        expected is None or actual == expected
        for actual, expected in zip(parts, (scenario, size, fact), strict=True)
    )


def render_failures(  # noqa: PLR0913
    console: Console,
    report: EvalReport,
    *,
    candidate: str | None = None,
    scenario: str | None = None,
    size: str | None = None,
    fact: str | None = None,
) -> None:
    if candidate is not None and candidate not in {
        item.name for item in report.candidates
    }:
        raise ValueError(f"unknown candidate {candidate!r}")
    table = Table(title="Incorrect and execution-problem rows")
    for column in ("outcome", "candidate", "task", "repeat", "output", "gold"):
        table.add_column(column)
    selected: list[tuple[int, str, Observation]] = []
    for ordinal, row in enumerate(report.observations):
        if not _matches(
            row,
            report,
            candidate=candidate,
            scenario=scenario,
            size=size,
            fact=fact,
        ) or (row.state is ObservationState.SCORED and row.score == 1.0):
            continue
        outcome = (
            "incorrect"
            if row.state is ObservationState.SCORED
            else row.state.value
        )
        selected.append((ordinal, outcome, row))
    outcome_order = {
        name: index
        for index, name in enumerate(
            ("incorrect", "failed", "missing", "invalid")
        )
    }
    for _ordinal, outcome, row in sorted(
        selected,
        key=lambda item: (outcome_order[item[1]], item[0]),
    ):
        table.add_row(
            Text(outcome),
            Text(row.candidate_name),
            Text(row.task_id),
            Text(str(row.seed_index)),
            Text(row.normalized_output or ""),
            Text(report.tasks[row.task_index].gold),
        )
    console.print(table)


def render_task(console: Console, report: EvalReport, task_id: str) -> None:
    tasks = [task for task in report.tasks if task.task_id == task_id]
    if not tasks:
        raise ValueError(f"unknown task ID {task_id!r}")
    task = tasks[0]
    console.print(
        Panel(
            Text(
                f"hash={task.task_hash}\nseed={task.seed}\n"
                f"strata={', '.join(task.strata)}\n"
                + "\n".join(
                    f"{key}:\n{value}"
                    for key, value in task.prompt_inputs.items()
                )
                + f"\ngold:\n{task.gold}"
            ),
            title=Text(task.task_id),
        )
    )
    for row in report.observations:
        if row.task_id != task_id:
            continue
        details = Text(
            f"state={row.state.value} trace_state={row.trace_state} "
            f"score={row.score!r}\n"
            f"failure_code={row.failure_code!r} "
            f"finish_reason={row.finish_reason!r}\n"
            f"provider_error={row.provider_error!r}\n"
            f"max_budget={row.max_budget!r} over_budget={row.over_budget!r}\n"
            f"submission_result={row.submission_result!r}\n\n"
            f"rendered prompt:\n{row.rendered_prompt}\n\n"
            f"raw output:\n{row.output_text or ''}\n\n"
            f"normalized output:\n{row.normalized_output or ''}\n\n"
            f"gold:\n{task.gold}\n\ncomponent trace:\n{row.component_trace!r}"
        )
        console.print(
            Panel(
                details,
                title=Text(f"{row.candidate_name} repeat {row.seed_index}"),
            )
        )


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    bucket: str
    task_id: str
    seed_index: int
    candidate_a: Observation | None
    candidate_b: Observation | None


def compare_buckets(
    report: EvalReport, candidate_a: str, candidate_b: str
) -> tuple[ComparisonRow, ...]:
    by_coordinate = {
        (row.candidate_name, row.task_id, row.seed_index): row
        for row in report.observations
    }
    rows: list[ComparisonRow] = []
    for task in report.tasks:
        for seed_index in range(report.run.repeats):
            a = by_coordinate.get((candidate_a, task.task_id, seed_index))
            b = by_coordinate.get((candidate_b, task.task_id, seed_index))
            if (
                a is None
                or b is None
                or a.state != b.state
                or a.state is not ObservationState.SCORED
            ):
                bucket = "execution mismatch"
            elif a.score == 1.0 and b.score == 1.0:
                bucket = "both correct"
            elif a.score == 0.0 and b.score == 0.0:
                bucket = "both wrong"
            elif a.score == 1.0:
                bucket = f"{candidate_a} only"
            else:
                bucket = f"{candidate_b} only"
            rows.append(
                ComparisonRow(
                    bucket=bucket,
                    task_id=task.task_id,
                    seed_index=seed_index,
                    candidate_a=a,
                    candidate_b=b,
                )
            )
    return tuple(rows)


def _comparison_value(
    report: EvalReport, candidate_name: str, row: Observation | None
) -> str:
    if row is not None:
        return row.normalized_output or row.state.value
    result = next(
        result
        for result in report.results
        if result.candidate_name == candidate_name
    )
    return f"{result.kind}: {result.message}"


def render_compare(
    console: Console, report: EvalReport, candidate_a: str, candidate_b: str
) -> None:
    known = {candidate.name for candidate in report.candidates}
    for candidate in (candidate_a, candidate_b):
        if candidate not in known:
            raise ValueError(f"unknown candidate {candidate!r}")
    rows = compare_buckets(report, candidate_a, candidate_b)
    counts = Counter(row.bucket for row in rows)
    console.print(
        Panel(
            Text(
                "\n".join(
                    f"{bucket}: {count}" for bucket, count in counts.items()
                )
            ),
            title=Text(f"{candidate_a} vs {candidate_b}"),
        )
    )
    table = Table(title="Paired rows")
    for column in ("bucket", "task", "repeat", candidate_a, candidate_b):
        table.add_column(Text(column))
    for row in rows:
        table.add_row(
            Text(row.bucket),
            Text(row.task_id),
            Text(str(row.seed_index)),
            Text(_comparison_value(report, candidate_a, row.candidate_a)),
            Text(_comparison_value(report, candidate_b, row.candidate_b)),
        )
    console.print(table)


def render_trajectory(
    console: Console, report: TrajectoryReport, *, show_candidates: bool
) -> None:
    candidates = {
        candidate.record_ref: candidate for candidate in report.candidates
    }
    resolutions = {
        (row.step_index, row.resolution_index): row
        for row in report.resolutions
    }
    console.print(
        Panel(
            Group(
                _trajectory_line("Status", report.terminal_status),
                _trajectory_line("Mutation field", report.mutation_field),
                _trajectory_line("Steps", str(len(report.steps))),
                _trajectory_line(
                    "Evaluation resolutions", str(len(report.resolutions))
                ),
            ),
            title=Text(f"Optimization trajectory: {report.run_id}"),
        )
    )
    for step in report.steps:
        sections: list[Text | Padding] = [
            _trajectory_line(
                "Budget used this step",
                _format_budget(step.budget_delta_consumed, delta=True),
            ),
            _trajectory_line(
                "Cumulative used",
                _format_budget(step.budget_cumulative_consumed),
            ),
            _trajectory_line(
                "Remaining",
                _format_budget(step.budget_remaining),
            ),
        ]
        if step.terminal_failure is not None:
            sections.append(
                _trajectory_line("Step failure", str(step.terminal_failure))
            )
        if not step.resolution_indexes:
            sections.extend(
                (
                    Text(""),
                    Text(
                        "No evaluation resolutions in this step.",
                        style="dim",
                    ),
                )
            )
        for resolution_index in step.resolution_indexes:
            row = resolutions[(step.step_index, resolution_index)]
            candidate = candidates[row.candidate_ref]
            base_candidate = (
                None
                if candidate.base_candidate_ref is None
                else candidates[candidate.base_candidate_ref]
            )
            base = (
                f"external · {candidate.base_ref.content_hash[:10]}"
                if base_candidate is None
                else base_candidate.candidate_id
            )
            sections.extend(
                (
                    Text(""),
                    Text(
                        f"Resolution {row.resolution_index} · {row.outcome}",
                        style="bold cyan",
                    ),
                    Padding(
                        Group(
                            _trajectory_line(
                                "Candidate", candidate.candidate_id
                            ),
                            _trajectory_line(
                                "Record",
                                candidate.record_ref.content_hash[:10],
                            ),
                            _trajectory_line("Base / parent", base),
                            _trajectory_line(
                                "Disposition",
                                " → ".join(candidate.dispositions),
                            ),
                            _trajectory_line("Evaluation", row.classification),
                            _trajectory_line(
                                "Reward",
                                "not available"
                                if row.reward is None
                                else str(row.reward),
                            ),
                            _trajectory_line("Message", row.message),
                        ),
                        (0, 0, 0, 2),
                    ),
                )
            )
        console.print(
            Panel(
                Group(*sections),
                title=Text(f"Step {step.step_index} · {step.status}"),
                border_style="blue",
            )
        )
    if not show_candidates:
        return
    panels = []
    for candidate in report.candidates:
        base_candidate = (
            None
            if candidate.base_candidate_ref is None
            else candidates[candidate.base_candidate_ref]
        )
        base = (
            f"external · {candidate.base_ref.content_hash}"
            if base_candidate is None
            else (
                f"{base_candidate.candidate_id} · "
                f"{base_candidate.record_ref.content_hash}"
            )
        )
        panels.append(
            Panel(
                Group(
                    _trajectory_line(
                        "First seen", f"step {candidate.first_step}"
                    ),
                    _trajectory_line(
                        "Record", candidate.record_ref.content_hash
                    ),
                    _trajectory_line("Base / parent", base),
                    _trajectory_line(
                        "Disposition", " → ".join(candidate.dispositions)
                    ),
                    Text(""),
                    Text(f"Exact {report.mutation_field}", style="bold cyan"),
                    Text(candidate.mutation_text),
                ),
                title=Text(candidate.candidate_id),
                border_style="cyan",
            )
        )
    console.print(Group(*panels))


def _trajectory_line(label: str, value: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="bold")
    line.append(value)
    return line


def _format_budget(
    values: Mapping[str, object], *, delta: bool = False
) -> str:
    if not values:
        return "none"
    rendered = []
    for key, value in sorted(values.items()):
        prefix = "+" if delta and isinstance(value, (int, float)) else ""
        rendered.append(f"{key} {prefix}{value}")
    return ", ".join(rendered)


__all__ = [
    "ComparisonRow",
    "compare_buckets",
    "render_c19_info",
    "render_compare",
    "render_failures",
    "render_summary",
    "render_task",
    "render_trajectory",
]
