from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.c19.generation import (
    DEFAULT_SEED_START,
    MAX_N_PER_CATEGORY,
)
from whetstone_envs.c19.model import parse_command

if TYPE_CHECKING:
    from whetstone_envs.instances import Instance


def render_instance(console: Console, instance: Instance) -> None:
    console.rule(Text(instance.id, style="bold"))
    console.print(
        Text(
            f"Seed: {instance.seed} | "
            f"Categories: {', '.join(instance.categories)}"
        )
    )
    rows = instance.prompt_inputs["grid"].splitlines()
    grid = Table(
        title="Public initial grid (row, column)",
        box=box.SIMPLE,
        show_lines=True,
    )
    grid.add_column("row", style="dim")
    for column in range(len(rows[0]) // 2):
        grid.add_column(str(column), justify="center")
    for index, row in enumerate(rows):
        cells = (
            row[position : position + 2] for position in range(0, len(row), 2)
        )
        grid.add_row(
            str(index),
            *(
                Text(
                    cell,
                    style="bold cyan"
                    if cell in {">>", "VV", "<<", "^^"}
                    else "",
                )
                for cell in cells
            ),
        )
    console.print(grid)
    console.print(
        Text(
            "W: wall | K: key | A: ball | D: closed door | L: locked door\n"
            "Blank: empty | __: open door | >> VV << ^^: agent heading"
        )
    )
    command = instance.prompt_inputs["command"]
    actions = parse_command(command)
    console.print(
        Panel(Text(command), title="Public action script", border_style="cyan")
    )
    console.print(
        Text(
            " → ".join(
                f"{index + 1}. {action.name.lower()}"
                for index, action in enumerate(actions)
            )
        )
    )
    console.print(
        Panel(
            Text(instance.prompt_inputs["question"]),
            title="Public question",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(instance.gold),
            title="Private gold: final-state answer",
            border_style="green",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C19 MiniGrid state-prediction instances."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--n-per-category", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Show both rendered probe prompts.",
    )
    arguments = parser.parse_args(argv)
    if (
        arguments.count < 1
        or not 1 <= arguments.n_per_category <= MAX_N_PER_CATEGORY
    ):
        parser.error(
            "count must be positive and n-per-category must be in "
            f"1..{MAX_N_PER_CATEGORY}"
        )
    instances = generate_pool(
        n_per_category=arguments.n_per_category,
        seed_start=arguments.seed_start,
    ).instance_group.instances
    if arguments.count > len(instances):
        parser.error(
            f"count exceeds the generated pool of {len(instances)} "
            "instances; increase --n-per-category"
        )
    console = Console()
    console.print(Text("C19 | MiniGrid state prediction", style="bold"))
    for instance in instances[: arguments.count]:
        render_instance(console, instance)
        if arguments.probes:
            console.print(
                Panel(Text(PROBES.render_naive(instance)), title="Naive probe")
            )
            console.print(
                Panel(
                    Text(PROBES.render_ceiling(instance)),
                    title="Ceiling probe",
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
