from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whetstone_envs.evidence import load_eval_report, load_trajectory_report
from whetstone_envs.reporting.html import (
    publish_eval_html,
    publish_trajectory_html,
)
from whetstone_envs.reporting.rich_views import (
    render_compare,
    render_failures,
    render_summary,
    render_task,
    render_trajectory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect saved evaluation and optimization evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "summary",
        "failures",
        "task",
        "compare",
        "trajectory",
        "html",
        "trajectory-html",
    ):
        subparser = commands.add_parser(command)
        subparser.add_argument("run_dir", type=Path)
        subparser.add_argument("--no-color", action="store_true")
        if command == "failures":
            subparser.add_argument("--candidate")
        elif command == "task":
            subparser.add_argument("task_id")
        elif command == "compare":
            subparser.add_argument("candidate_a")
            subparser.add_argument("candidate_b")
        elif command == "trajectory":
            subparser.add_argument("--show-candidates", action="store_true")
    return parser


def _dispatch(arguments: argparse.Namespace, console: Console) -> int:
    if arguments.command in {"html", "trajectory-html"}:
        publisher = (
            publish_eval_html
            if arguments.command == "html"
            else publish_trajectory_html
        )
        console.print(str(publisher(arguments.run_dir)), soft_wrap=True)
    elif arguments.command == "trajectory":
        render_trajectory(
            console,
            load_trajectory_report(arguments.run_dir),
            show_candidates=arguments.show_candidates,
        )
    else:
        report = load_eval_report(arguments.run_dir)
        if arguments.command == "summary":
            render_summary(console, report)
        elif arguments.command == "failures":
            render_failures(console, report, candidate=arguments.candidate)
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
    console = Console(color_system=None if arguments.no_color else "auto")
    try:
        return _dispatch(arguments, console)
    except Exception as error:  # noqa: BLE001 - CLI translates boundary errors.
        console.print(f"error: {error}", style="bold red")
        return 2


__all__ = ["build_parser", "main"]
