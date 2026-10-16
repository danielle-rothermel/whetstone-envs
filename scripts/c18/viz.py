from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from whetstone_envs.c18 import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
    PROBES,
    DepthCategory,
    generate_pool,
)

if TYPE_CHECKING:
    from whetstone_envs.instances import Instance


def render_instance(
    console: Console, instance: Instance, category: DepthCategory
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
            f"Deduction depth: {category.hops} | "
            f"Distractors: {category.distractors.value}"
        )
    )
    console.print(
        Panel(
            Text(instance.prompt_inputs["question"]),
            title="Public facts and rules",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(instance.prompt_inputs["query"]),
            title="Public query",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(instance.gold),
            title="Private gold: entailment verdict",
            border_style="green",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C18 PrOntoQA entailment instances."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--n-per-category", type=int, default=1)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use the hard generation configuration.",
    )
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Show both rendered probe prompts.",
    )
    arguments = parser.parse_args(argv)
    if arguments.count < 1 or arguments.n_per_category < 1:
        parser.error("count and n-per-category must be positive")
    config = HARD_CONFIG if arguments.hard else DEFAULT_CONFIG
    if arguments.seed_start is not None:
        config = replace(config, seed_start=arguments.seed_start)
    instances = generate_pool(
        config, n_per_category=arguments.n_per_category
    ).instance_group.instances
    if arguments.count > len(instances):
        parser.error(
            f"count exceeds the generated pool of {len(instances)} "
            "instances; increase --n-per-category"
        )
    console = Console()
    console.print(Text("C18 | PrOntoQA deductive entailment", style="bold"))
    console.print(
        Text(
            f"Generator: {config.generator_version} | "
            f"Seed start: {config.seed_start}"
        )
    )
    for instance in instances[: arguments.count]:
        category = next(
            c for c in config.categories if c.label in instance.categories
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
