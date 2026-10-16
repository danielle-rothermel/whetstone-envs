from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from whetstone_envs.c23 import PROBES, generate_pool
from whetstone_envs.c23._pool import DEFAULT_CONFIG

if TYPE_CHECKING:
    from whetstone_envs.c23._domain import CategoryConfiguration
    from whetstone_envs.instances import Instance


def render_instance(
    console: Console, instance: Instance, category: CategoryConfiguration
) -> None:
    console.rule(Text(instance.id, style="bold"))
    console.print(
        Text(
            f"Seed: {instance.seed} | "
            f"Categories: {', '.join(instance.categories)}"
        )
    )
    console.print(
        Text(
            f"Rule family: {category.rule.family.value} | "
            f"Context length: {category.rule.context_length}"
        )
    )
    console.print(
        Panel(
            Text(instance.prompt_inputs["demos_block"]),
            title="Public demonstrations",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(repr(instance.prompt_inputs["query"])),
            title="Public query (quoted string)",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(repr(instance.gold)),
            title="Private gold: expected output (quoted string)",
            border_style="green",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C23 subregular string-induction instances."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--n-per-category", type=int, default=1)
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Show both rendered probe prompts.",
    )
    arguments = parser.parse_args(argv)
    if arguments.count < 1 or arguments.n_per_category < 1:
        parser.error("count and n-per-category must be positive")
    instances = generate_pool(
        n_per_category=arguments.n_per_category
    ).instance_group.instances
    if arguments.count > len(instances):
        parser.error(
            f"count exceeds the generated pool of {len(instances)} "
            "instances; increase --n-per-category"
        )
    console = Console()
    console.print(Text("C23 | Subregular string induction", style="bold"))
    for instance in instances[: arguments.count]:
        category = next(
            c
            for c in DEFAULT_CONFIG.categories
            if c.label in instance.categories
        )
        render_instance(console, instance, category)
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
