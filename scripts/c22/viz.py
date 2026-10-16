from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whetstone_envs.c22 import PROBES, Preset, generate_pool
from whetstone_envs.c22.constraints import ConstraintStack

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
    console.print(
        Panel(
            Text(instance.prompt_inputs["constraints"]),
            title="Public numbered constraints",
            border_style="cyan",
        )
    )
    stack = ConstraintStack.from_gold(instance.gold)
    table = Table(title="Private gold: scoring specification")
    table.add_column("#")
    table.add_column("Constraint")
    table.add_column("Parameters")
    for index, constraint in enumerate(stack.constraints, start=1):
        parameters = constraint.model_dump(exclude={"kind"})
        table.add_row(
            str(index),
            Text(constraint.kind.replace("_", " ")),
            Text(
                ", ".join(
                    f"{name}={value!r}" for name, value in parameters.items()
                )
                or "—"
            ),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C22 instruction-constraint instances."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--preset", type=Preset, choices=tuple(Preset), default=Preset.DEFAULT
    )
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Show both rendered probe prompts.",
    )
    arguments = parser.parse_args(argv)
    if arguments.count < 1:
        parser.error("count must be positive")
    instances = generate_pool(arguments.preset).instance_group.instances
    if arguments.count > len(instances):
        parser.error(
            f"count exceeds the fixed pool of {len(instances)} instances"
        )
    console = Console()
    console.print(
        Text(
            f"C22 | Instruction constraints | "
            f"Preset: {arguments.preset.value}",
            style="bold",
        )
    )
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
