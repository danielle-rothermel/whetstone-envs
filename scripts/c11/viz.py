from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from whetstone_envs.c11 import PROBES, generate_pool
from whetstone_envs.c11.generation import DEFAULT_SEED_START, INPUT_JSON_FIELD

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
    source = instance.prompt_inputs[INPUT_JSON_FIELD]
    console.print(
        Panel(Text(source), title="Public input JSON", border_style="cyan")
    )
    console.print(
        Panel(
            Text(json.dumps(source, ensure_ascii=False)),
            title="Exact input with escapes visible",
        )
    )
    console.print(
        Panel(
            Text(instance.gold),
            title="Private gold: canonical JSON",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Text(json.dumps(instance.gold, ensure_ascii=False)),
            title="Exact gold with escapes visible",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C11 JSON canonicalization instances."
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
    if arguments.count < 1 or arguments.n_per_category < 1:
        parser.error("count and n-per-category must be positive")
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
    console.print(Text("C11 | RFC 8785 JSON canonicalization", style="bold"))
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
