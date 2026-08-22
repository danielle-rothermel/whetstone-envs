from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

_SPLIT_PARTS = 3
CandidateInputKind = Literal["naive", "ceiling", "custom"]


def _split_sizes(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "split sizes must be three comma-separated integers"
        ) from error
    if len(parts) != _SPLIT_PARTS:
        raise argparse.ArgumentTypeError(
            "split sizes must be three comma-separated integers"
        )
    if any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("split sizes must be non-negative")
    return parts


class _CandidateAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, str):
            parser.error(f"{option_string} requires one string value")
        selected = getattr(namespace, self.dest, None)
        if selected is None:
            selected = []
            setattr(namespace, self.dest, selected)
        if option_string == "--candidate":
            if values not in {"naive", "ceiling"}:
                parser.error("--candidate must be naive or ceiling")
            selected.append((values, values, None))
            return
        name, separator, raw_path = values.partition("=")
        if not separator or not name.strip() or not raw_path:
            parser.error("--candidate-file must be NAME=PATH")
        path = Path(raw_path)
        try:
            template = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            parser.error(f"cannot read candidate template {path}: {error}")
        selected.append((name, "custom", template))


def _color_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-color", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and inspect C19 evaluations and optimization trajectories."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info")
    info.add_argument("family", choices=("c19",))
    info.add_argument("--show-templates", action="store_true")
    _color_argument(info)

    run = commands.add_parser("run")
    run.add_argument("--family", choices=("c19",), default="c19")
    run.add_argument(
        "--candidate",
        dest="candidate_inputs",
        action=_CandidateAction,
    )
    run.add_argument(
        "--candidate-file",
        dest="candidate_inputs",
        action=_CandidateAction,
    )
    run.add_argument(
        "--transport", choices=("fake", "openrouter"), default="fake"
    )
    run.add_argument(
        "--role", choices=("internal", "official"), default="internal"
    )
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--split-sizes", type=_split_sizes, default=(20, 20, 0))
    run.add_argument("--model", default="openai/gpt-4.1-nano")
    run.add_argument("--run-id")
    run.add_argument("--output", type=Path)
    _color_argument(run)

    summary = commands.add_parser("summary")
    summary.add_argument("run_dir", type=Path)
    _color_argument(summary)

    failures = commands.add_parser("failures")
    failures.add_argument("run_dir", type=Path)
    failures.add_argument("--candidate")
    failures.add_argument(
        "--scenario", choices=("navigation", "manipulation", "door")
    )
    failures.add_argument("--size", choices=("small", "medium"))
    failures.add_argument(
        "--fact", choices=("coordinate", "heading", "front", "carrying")
    )
    _color_argument(failures)

    task = commands.add_parser("task")
    task.add_argument("run_dir", type=Path)
    task.add_argument("task_id")
    _color_argument(task)

    compare = commands.add_parser("compare")
    compare.add_argument("run_dir", type=Path)
    compare.add_argument("candidate_a")
    compare.add_argument("candidate_b")
    _color_argument(compare)

    trajectory = commands.add_parser("trajectory")
    trajectory.add_argument("run_dir", type=Path)
    trajectory.add_argument("--show-candidates", action="store_true")
    _color_argument(trajectory)
    return parser


def _console(*, no_color: bool):
    from rich.console import Console

    return Console(color_system=None if no_color else "auto")


def _candidate_inputs(raw: list[tuple[str, str, str | None]] | None):
    from whetstone_envs.c19 import PROBES
    from whetstone_envs.reporting.execution import CandidateInput

    if raw is None:
        return ()
    templates = {
        "naive": PROBES.naive_template,
        "ceiling": PROBES.ceiling_template,
    }
    return tuple(
        CandidateInput(
            name=name,
            source=cast("CandidateInputKind", source),
            template=templates[name] if template is None else template,
        )
        for name, source, template in raw
    )


def _dispatch(arguments: argparse.Namespace) -> int:
    console = _console(no_color=arguments.no_color)
    if arguments.command == "info":
        from whetstone_envs.reporting.rich_views import render_c19_info

        render_c19_info(console, show_templates=arguments.show_templates)
        return 0
    if arguments.command == "run":
        from whetstone_envs.reporting.execution import (
            C19EvalSpec,
            run_c19_evaluation,
        )
        from whetstone_envs.reporting.rich_views import render_summary

        output = run_c19_evaluation(
            C19EvalSpec(
                transport=arguments.transport,
                role=arguments.role,
                candidates=_candidate_inputs(arguments.candidate_inputs),
                repeats=arguments.repeats,
                split_sizes=arguments.split_sizes,
                output_dir=arguments.output,
                run_id=arguments.run_id,
                model=arguments.model,
            )
        )
        render_summary(console, output.report)
        console.print(str(output.directory))
        return (
            0
            if all(
                result.kind == "success" for result in output.report.results
            )
            else 1
        )

    from whetstone_envs.reporting.publication import (
        load_eval_report,
        load_trajectory_report,
    )
    from whetstone_envs.reporting.rich_views import (
        render_compare,
        render_failures,
        render_summary,
        render_task,
        render_trajectory,
    )

    if arguments.command == "trajectory":
        render_trajectory(
            console,
            load_trajectory_report(arguments.run_dir),
            show_candidates=arguments.show_candidates,
        )
        return 0
    report = load_eval_report(arguments.run_dir)
    if arguments.command == "summary":
        render_summary(console, report)
    elif arguments.command == "failures":
        render_failures(
            console,
            report,
            candidate=arguments.candidate,
            scenario=arguments.scenario,
            size=arguments.size,
            fact=arguments.fact,
        )
    elif arguments.command == "task":
        render_task(console, report, arguments.task_id)
    elif arguments.command == "compare":
        render_compare(
            console,
            report,
            arguments.candidate_a,
            arguments.candidate_b,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return _dispatch(arguments)
    except Exception as error:  # noqa: BLE001 - CLI translates boundary errors.
        from whetstone_envs.reporting.publication import DurableRunError

        console = _console(no_color=arguments.no_color)
        if isinstance(error, DurableRunError):
            console.print(f"error: {error.cause}", style="bold red")
            console.print(str(error.directory), soft_wrap=True)
            return 2
        console.print(f"error: {error}", style="bold red")
        return 2


__all__ = ["build_parser", "main"]
