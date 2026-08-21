from __future__ import annotations

import argparse
from pathlib import Path

from whetstone_envs.optim.run import (
    C19RunSpec,
    default_output_dir,
    run_c19_optimizer,
)


def _split_sizes(value: str) -> tuple[int, int, int]:
    try:
        internal, official, held_out = (int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "split sizes must be three comma-separated integers"
        ) from error
    return (internal, official, held_out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run COPRO or GEPA on a named whetstone-envs family.",
    )
    parser.add_argument("--family", choices=("c19",), default="c19")
    parser.add_argument(
        "--optimizer",
        choices=("copro",),
        required=True,
    )
    parser.add_argument(
        "--transport",
        choices=("fake", "openrouter"),
        default="fake",
    )
    parser.add_argument(
        "--split-sizes",
        type=_split_sizes,
        default=(2, 2, 0),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_id = arguments.run_id
    output = arguments.output
    if output is None and run_id is not None:
        output = default_output_dir(run_id)
    path = run_c19_optimizer(
        C19RunSpec(
            optimizer=arguments.optimizer,
            transport=arguments.transport,
            split_sizes=arguments.split_sizes,
            output_dir=output,
            run_id=run_id,
            model=arguments.model,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
