from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

from whetstone_envs.optim.concurrency import (
    DEFAULT_PROVIDER_CONCURRENCY,
    MAX_UNFORCED_PROVIDER_CONCURRENCY,
    PROVIDER_CONCURRENCY_FLAG,
    PROVIDER_CONCURRENCY_FORCE_FLAG,
    resolve_provider_concurrency,
)

_SPLIT_PARTS = 3

#: The reasoning efforts ``--task-reasoning-effort`` accepts, spelled as
#: literals so ``build_parser`` -- and therefore ``whetstone-eval --help``
#: -- imports nothing from the ``optim`` extra. ``_reasoning_effort``
#: asserts this tuple still equals ``dr_providers.ReasoningEffort``'s
#: values, so a vocabulary change fails loudly rather than silently
#: rejecting a valid effort.
REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)
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


def _task_route_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the task route's model and reasoning effort.

    The effort is declared as a plain string and converted to
    ``dr_providers.ReasoningEffort`` in :func:`main`, once a command has
    actually been chosen. Declaring it as the enum would import
    ``dr_providers`` here, and ``dr_providers`` ships with the ``optim``
    extra while ``build_parser`` backs a console script a base install
    must be able to run -- ``whetstone-eval --help`` is exercised by the
    wheel smoke test on exactly that install.

    :data:`REASONING_EFFORTS` is the argparse-visible copy of the
    vocabulary; :func:`_reasoning_effort` checks it against the enum so
    the two cannot drift.
    """
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    parser.add_argument(
        "--task-reasoning-effort",
        default=None,
        choices=REASONING_EFFORTS,
        help=(
            "Reasoning effort for the task route. Omitted sends no "
            "reasoning key and leaves the route on the provider's "
            "default. The proposer route never takes it."
        ),
    )


def _reasoning_effort(value: str | None):
    """Parse a ``--task-reasoning-effort`` string into the provider enum.

    Called from ``main`` rather than as an argparse ``type``, so the
    ``dr_providers`` import stays off the base install's ``--help`` path.
    The membership assertion is what keeps :data:`REASONING_EFFORTS` --
    which exists only so ``--help`` can list the choices without the
    extra -- from drifting away from the enum that owns the vocabulary.
    """
    from dr_providers import ReasoningEffort

    if tuple(member.value for member in ReasoningEffort) != REASONING_EFFORTS:
        raise RuntimeError(
            "REASONING_EFFORTS no longer matches dr_providers' "
            "ReasoningEffort; update the CLI's choices"
        )
    return None if value is None else ReasoningEffort(value)


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
        "--role",
        choices=("internal", "official", "held_out"),
        default="internal",
    )
    run.add_argument("--repeats", type=int, default=1)
    # Declared through the same constants the study CLI uses, so the two
    # surfaces cannot drift on the flag's spelling or its bounds.
    run.add_argument(
        PROVIDER_CONCURRENCY_FLAG,
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
        metavar="N",
        help=(
            "How many task evaluations run against the provider at once. "
            f"Defaults to {DEFAULT_PROVIDER_CONCURRENCY}. Sets both the "
            "evaluation worker pool and the HTTP connection pool. Above "
            f"{MAX_UNFORCED_PROVIDER_CONCURRENCY} it is refused unless "
            f"{PROVIDER_CONCURRENCY_FORCE_FLAG} is also passed."
        ),
    )
    run.add_argument(
        PROVIDER_CONCURRENCY_FORCE_FLAG,
        action="store_true",
        help=(
            "Authorize a provider concurrency above the sanity cap of "
            f"{MAX_UNFORCED_PROVIDER_CONCURRENCY}."
        ),
    )
    run.add_argument("--split-sizes", type=_split_sizes, default=(20, 20, 0))
    _task_route_arguments(run)
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

    html = commands.add_parser("html")
    html.add_argument("run_dir", type=Path)
    _color_argument(html)

    trajectory_html = commands.add_parser("trajectory-html")
    trajectory_html.add_argument("run_dir", type=Path)
    _color_argument(trajectory_html)
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
                task_reasoning_effort=_reasoning_effort(
                    arguments.task_reasoning_effort
                ),
                provider_concurrency=resolve_provider_concurrency(
                    arguments.provider_concurrency,
                    force=arguments.force_provider_concurrency,
                ),
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

    if arguments.command in {"html", "trajectory-html"}:
        from whetstone_envs.reporting.html import (
            publish_eval_html,
            publish_trajectory_html,
        )

        publisher = (
            publish_eval_html
            if arguments.command == "html"
            else publish_trajectory_html
        )
        console.print(
            str(publisher(arguments.run_dir).resolve()), soft_wrap=True
        )
        return 0

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
