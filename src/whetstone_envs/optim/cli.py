from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Import order is load-bearing: ``whetstone_envs.optim.run`` must come before
# ``optim.gepa`` / ``optim.miprov2``. whetstone-ai <= 0.1.5 has a
# provider <-> eval.drivers import cycle -- ``whetstone.eval.schema`` reaches
# back into a partially initialized ``whetstone.experiment.binding`` for
# ``EvalConfigRef`` -- so importing ``optim.gepa`` or ``optim.miprov2`` first
# raises ImportError. ``optim.run`` imports whetstone's modules in an order
# that resolves the cycle, so importing it first makes the later imports safe.
# Upstream is fixing this for 0.1.6; until the pin moves and this is verified
# gone, do not reorder these imports. ``tests/optim/test_import_order.py``
# cold-imports both CLIs in fresh interpreters so a reorder fails loudly.
from whetstone_envs.optim.run import (
    C19_DEMO_MODES,
    C19_OPTIMIZERS,
    C19_TRANSPORTS,
    C19RunSpec,
    default_output_dir,
    run_c19_optimizer,
)
from whetstone_envs.reporting.publication import DurableRunError


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
        description=(
            "Run COPRO, GEPA, or MIPROv2 on a named whetstone-envs family."
        ),
    )
    parser.add_argument("--family", choices=("c19",), default="c19")
    parser.add_argument(
        "--optimizer",
        choices=C19_OPTIMIZERS,
        required=True,
    )
    parser.add_argument(
        "--demo-mode",
        choices=C19_DEMO_MODES,
        default="fewshot",
        help="MIPROv2 demonstration regime; ignored by COPRO and GEPA.",
    )
    parser.add_argument(
        "--transport",
        choices=C19_TRANSPORTS,
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
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Repeats per task (K_REPEAT).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_id = arguments.run_id
    output = arguments.output
    if output is None and run_id is not None:
        output = default_output_dir(run_id)
    try:
        path = run_c19_optimizer(
            C19RunSpec(
                optimizer=arguments.optimizer,
                transport=arguments.transport,
                split_sizes=arguments.split_sizes,
                output_dir=output,
                run_id=run_id,
                model=arguments.model,
                demo_mode=arguments.demo_mode,
                num_seeds=arguments.num_seeds,
            )
        )
    except DurableRunError as error:
        traceback.print_exception(error.cause)
        print(error.directory, file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
