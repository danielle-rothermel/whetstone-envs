from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from whetstone_envs.optim.run import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
    CODEX_DEFAULT_BINARY,
    CODEX_EVALUATE_CALL_CAP,
    CODEX_REASONING_EFFORTS,
    DEFAULT_COPRO_BREADTH,
    DEFAULT_COPRO_DEPTH,
    DEFAULT_MIPROV2_FULL_EVAL_STEPS,
    DEFAULT_MIPROV2_MINIBATCH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    DEFAULT_SPLIT_SIZES,
    DEMO_MODES,
    MIN_COPRO_BREADTH,
    OPTIMIZERS,
    TRANSPORTS,
    CodexReasoningEffort,
    RunSpec,
    default_output_dir,
    registered_family_ids,
    run_optimizer,
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


def _int_at_least(value: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed < minimum:
        raise argparse.ArgumentTypeError(
            f"expected an integer of at least {minimum}, got {parsed}"
        )
    return parsed


def _copro_breadth(value: str) -> int:
    """COPRO needs at least two drafts per step to have a choice."""
    return _int_at_least(value, minimum=MIN_COPRO_BREADTH)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a number, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive number, got {parsed}"
        )
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {parsed}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run COPRO, GEPA, or MIPROv2 on a named whetstone-envs family."
        ),
    )
    parser.add_argument(
        "--family",
        choices=registered_family_ids(),
        default="c19",
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        required=True,
        help=(
            "The arm to run. 'null-random' is null-A, the "
            "selection-on-noise control: it drives COPRO's search shape "
            "with an uninformative proposer, so it honours "
            "--copro-breadth and --copro-depth and produces the same "
            "evidence a COPRO run does."
        ),
    )
    parser.add_argument(
        "--demo-mode",
        choices=DEMO_MODES,
        default="fewshot",
        help="MIPROv2 demonstration regime; ignored by COPRO and GEPA.",
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="fake",
    )
    parser.add_argument(
        "--split-sizes",
        type=_split_sizes,
        default=DEFAULT_SPLIT_SIZES,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    parser.add_argument(
        "--proposer-model",
        default=None,
        help=(
            "Model for the proposer role. Defaults to --model, which keeps "
            "a single-model run identical to one that never named a "
            "proposer."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=_positive_int,
        default=1,
        help="Repeats per task (K_REPEAT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "This run's algorithmic seed. GEPA and MIPROv2 carry it onto "
            "their controls; COPRO has no control seed, so its runs are "
            "seeded only by the provider SEED control. Omit to keep each "
            "optimizer's own default."
        ),
    )
    parser.add_argument(
        "--n-per-stratum",
        type=_positive_int,
        default=None,
        help=(
            "Instances generated per stratum. Defaults to the family's own "
            "pool size."
        ),
    )
    parser.add_argument(
        "--pool-seed-start",
        type=int,
        default=None,
        help=(
            "First generator seed for the task pool. Defaults to the "
            "family's own start."
        ),
    )
    parser.add_argument(
        "--copro-breadth",
        type=_copro_breadth,
        default=DEFAULT_COPRO_BREADTH,
        help=(
            "COPRO candidates proposed per step; must be at least "
            f"{MIN_COPRO_BREADTH}."
        ),
    )
    parser.add_argument(
        "--copro-depth",
        type=int,
        default=DEFAULT_COPRO_DEPTH,
        help="COPRO search depth; step count is depth + 1.",
    )
    parser.add_argument(
        "--gepa-max-metric-calls",
        type=_positive_int,
        default=None,
        help=(
            "GEPA's paid metric-call ceiling. Defaults to one full pass "
            "over the trainset plus one reflection minibatch."
        ),
    )
    parser.add_argument(
        "--miprov2-minibatch",
        action="store_true",
        default=DEFAULT_MIPROV2_MINIBATCH,
        help=(
            "Evaluate each MIPROv2 trial on a sampled minibatch rather than "
            "the whole validation split."
        ),
    )
    parser.add_argument(
        "--miprov2-minibatch-size",
        type=_positive_int,
        default=None,
        help=(
            "Tasks per minibatched MIPROv2 trial. Defaults to the whole "
            "validation split."
        ),
    )
    parser.add_argument(
        "--miprov2-minibatch-full-eval-steps",
        type=_positive_int,
        default=DEFAULT_MIPROV2_FULL_EVAL_STEPS,
        help=(
            "Trials between full-validation re-evaluations of the MIPROv2 "
            "incumbent."
        ),
    )
    parser.add_argument(
        "--miprov2-num-trials",
        type=_positive_int,
        default=DEFAULT_MIPROV2_NUM_TRIALS,
        help=(
            "MIPROv2 optimization trials. The default is this runner's own "
            "shape; the protocol's auto-light configuration assumes 10."
        ),
    )
    parser.add_argument(
        "--miprov2-num-candidates",
        type=_positive_int,
        default=DEFAULT_MIPROV2_NUM_CANDIDATES,
        help=(
            "MIPROv2 candidates per component. The default is this "
            "runner's own shape; the protocol's auto-light assumes 6."
        ),
    )
    parser.add_argument(
        "--train-size",
        type=_positive_int,
        default=None,
        help=(
            "Tasks from the internal split used as the trainset. Required "
            "for --optimizer miprov2 and gepa, which bootstrap or reflect "
            "on the trainset and score on a disjoint valset; refused on "
            "the optimizers that have no train/val concept."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=_positive_int,
        default=None,
        help=(
            "Tasks from the internal split used as the valset, taken "
            "after the trainset so the two are disjoint. Required "
            "alongside --train-size."
        ),
    )
    parser.add_argument(
        "--codex-capacity",
        type=_positive_int,
        default=None,
        help=(
            "Admitted evaluate-call cap for the Codex arm, which is its "
            f"eval budget. Defaults to {CODEX_EVALUATE_CALL_CAP}."
        ),
    )
    parser.add_argument(
        "--codex-binary",
        default=CODEX_DEFAULT_BINARY,
        help=(
            "The Codex CLI to spawn, resolved on the run PATH. Defaults "
            f"to the real {CODEX_DEFAULT_BINARY!r}."
        ),
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help=(
            "The Codex agent's own model. Defaults to --model, the task "
            "model. The agent's own spend is not on the study's key and "
            "is not priced in the run's cost report."
        ),
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=CODEX_REASONING_EFFORTS,
        default=CodexReasoningEffort.MEDIUM.value,
        help="How hard the Codex agent reasons.",
    )
    parser.add_argument(
        "--codex-wall-seconds",
        type=_positive_float,
        default=None,
        help=(
            "The Codex agent's wall budget in seconds. Defaults to "
            "whetstone-ai's own."
        ),
    )
    parser.add_argument(
        "--allow-real-codex",
        action="store_true",
        help=(
            "Opt in to spawning the real, billed Codex CLI. Half of the "
            "opt-in: the run is still refused unless "
            f"{ALLOW_REAL_CODEX_ENV}={ALLOW_REAL_CODEX_ENV_VALUE} is also "
            "set in the environment. Without both, a Codex run is refused "
            "before any session probe, because the authentication "
            "preflight itself spawns the CLI and costs money."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_id = arguments.run_id
    output = arguments.output
    if output is None and run_id is not None:
        output = default_output_dir(run_id)
    try:
        path = run_optimizer(
            RunSpec(
                optimizer=arguments.optimizer,
                transport=arguments.transport,
                family=arguments.family,
                split_sizes=arguments.split_sizes,
                output_dir=output,
                run_id=run_id,
                model=arguments.model,
                proposer_model=arguments.proposer_model,
                demo_mode=arguments.demo_mode,
                num_seeds=arguments.num_seeds,
                n_per_stratum=arguments.n_per_stratum,
                pool_seed_start=arguments.pool_seed_start,
                seed=arguments.seed,
                copro_breadth=arguments.copro_breadth,
                copro_depth=arguments.copro_depth,
                gepa_max_metric_calls=arguments.gepa_max_metric_calls,
                miprov2_minibatch=arguments.miprov2_minibatch,
                miprov2_minibatch_size=arguments.miprov2_minibatch_size,
                miprov2_minibatch_full_eval_steps=(
                    arguments.miprov2_minibatch_full_eval_steps
                ),
                miprov2_num_trials=arguments.miprov2_num_trials,
                miprov2_num_candidates=arguments.miprov2_num_candidates,
                train_size=arguments.train_size,
                val_size=arguments.val_size,
                codex_capacity=arguments.codex_capacity,
                codex_binary=arguments.codex_binary,
                codex_model=arguments.codex_model,
                codex_reasoning_effort=arguments.codex_reasoning_effort,
                codex_wall_seconds=arguments.codex_wall_seconds,
                allow_real_codex=arguments.allow_real_codex,
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
