from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from whetstone_envs.artifacts import DurableRunError, default_output_dir
from whetstone_envs.evaluation.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    MAX_UNFORCED_PROVIDER_CONCURRENCY,
    PROVIDER_CONCURRENCY_FLAG,
    PROVIDER_CONCURRENCY_FORCE_FLAG,
    resolve_provider_concurrency,
)
from whetstone_envs.evaluation.runtime import TransportKind
from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
    CODEX_DEFAULT_BINARY,
    CODEX_EVALUATE_CALL_CAP,
    CODEX_REASONING_EFFORTS,
    CodexReasoningEffort,
)
from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_FULL_EVAL_STEPS,
    DEFAULT_MIPROV2_MINIBATCH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    DEMO_MODES,
    Miprov2DemoMode,
)
from whetstone_envs.optim.problem import load_problem
from whetstone_envs.optim.run import run_optimizer
from whetstone_envs.optim.spec import (
    DEFAULT_COPRO_BREADTH,
    DEFAULT_COPRO_DEPTH,
    GEPA_DEFAULT_SEED,
    MIPROV2_DEFAULT_SEED,
    CodexSettings,
    CoproSettings,
    GepaSettings,
    Miprov2Settings,
    OptimizerKind,
    OptimizerSettings,
    RunSpec,
    TrainValSizes,
)
from whetstone_envs.optim.split import MIN_COPRO_BREADTH


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
            "Run Codex, COPRO, GEPA, or MIPROv2 on a prepared problem."
        ),
    )
    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="Python file defining an OptimizationProblem named problem.",
    )
    parser.add_argument(
        "--optimizer",
        choices=tuple(OptimizerKind),
        required=True,
        help="Optimizer to run.",
    )
    parser.add_argument(
        "--demo-mode",
        choices=DEMO_MODES,
        default="fewshot",
        help="MIPROv2 demonstration regime; ignored by COPRO and GEPA.",
    )
    parser.add_argument(
        "--transport",
        choices=tuple(TransportKind),
        default="fake",
    )
    parser.add_argument(
        PROVIDER_CONCURRENCY_FLAG,
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
        metavar="N",
        help=(
            "How many task evaluations run against the provider at once. "
            f"Defaults to {DEFAULT_PROVIDER_CONCURRENCY}. Sets both the "
            "evaluation worker pool and the HTTP connection pool; it "
            "changes how long the run takes, never what it computes. "
            f"Above {MAX_UNFORCED_PROVIDER_CONCURRENCY} it is refused "
            f"unless {PROVIDER_CONCURRENCY_FORCE_FLAG} is also passed."
        ),
    )
    parser.add_argument(
        PROVIDER_CONCURRENCY_FORCE_FLAG,
        action="store_true",
        help=(
            "Authorize a provider concurrency above the sanity cap of "
            f"{MAX_UNFORCED_PROVIDER_CONCURRENCY}."
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--proposer-model",
        default=None,
        help=(
            "Proposer model. Defaults to the prepared experiment task route."
        ),
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
        help="MIPROv2 optimization trials.",
    )
    parser.add_argument(
        "--miprov2-num-candidates",
        type=_positive_int,
        default=DEFAULT_MIPROV2_NUM_CANDIDATES,
        help="MIPROv2 candidates per component.",
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
            "The Codex agent's own model. Omit to use the adapter's "
            "default agent model."
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


def _optimizer_settings(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> OptimizerSettings:
    kind = OptimizerKind(arguments.optimizer)
    groups = (
        (OptimizerKind.COPRO, ("copro_breadth", "copro_depth")),
        (OptimizerKind.GEPA, ("gepa_max_metric_calls",)),
        (
            OptimizerKind.MIPROV2,
            (
                "demo_mode",
                "miprov2_minibatch",
                "miprov2_minibatch_size",
                "miprov2_minibatch_full_eval_steps",
                "miprov2_num_trials",
                "miprov2_num_candidates",
            ),
        ),
        (
            OptimizerKind.CODEX,
            (
                "codex_capacity",
                "codex_binary",
                "codex_model",
                "codex_reasoning_effort",
                "codex_wall_seconds",
                "allow_real_codex",
            ),
        ),
    )
    for owner, names in groups:
        if kind is owner:
            continue
        for name in names:
            if getattr(arguments, name) != parser.get_default(name):
                raise ValueError(
                    f"--{name.replace('_', '-')} applies only "
                    f"to --optimizer {owner}"
                )
    if kind in (OptimizerKind.COPRO, OptimizerKind.CODEX):
        if (
            arguments.train_size is not None
            or arguments.val_size is not None
            or arguments.seed is not None
        ):
            raise ValueError(
                "train/val sizes and seed apply only to GEPA or MIPROv2"
            )
        if kind is OptimizerKind.COPRO:
            return CoproSettings(
                breadth=arguments.copro_breadth, depth=arguments.copro_depth
            )
        return CodexSettings(
            capacity=arguments.codex_capacity,
            binary=arguments.codex_binary,
            model=arguments.codex_model,
            reasoning_effort=CodexReasoningEffort(
                arguments.codex_reasoning_effort
            ),
            wall_seconds=arguments.codex_wall_seconds,
            allow_real_codex=arguments.allow_real_codex,
        )
    partition = TrainValSizes(
        train_size=arguments.train_size, val_size=arguments.val_size
    )
    if kind is OptimizerKind.GEPA:
        return GepaSettings(
            partition=partition,
            seed=GEPA_DEFAULT_SEED
            if arguments.seed is None
            else arguments.seed,
            max_metric_calls=arguments.gepa_max_metric_calls,
        )
    return Miprov2Settings(
        partition=partition,
        seed=MIPROV2_DEFAULT_SEED
        if arguments.seed is None
        else arguments.seed,
        demo_mode=Miprov2DemoMode(arguments.demo_mode),
        minibatch=arguments.miprov2_minibatch,
        minibatch_size=arguments.miprov2_minibatch_size,
        minibatch_full_eval_steps=arguments.miprov2_minibatch_full_eval_steps,
        num_trials=arguments.miprov2_num_trials,
        num_candidates=arguments.miprov2_num_candidates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    run_id = arguments.run_id
    output = arguments.output
    if output is None and run_id is not None:
        output = default_output_dir(run_id)
    try:
        path = run_optimizer(
            RunSpec(
                optimizer=_optimizer_settings(arguments, parser),
                transport=TransportKind(arguments.transport),
                provider_concurrency=resolve_provider_concurrency(
                    arguments.provider_concurrency,
                    force=arguments.force_provider_concurrency,
                ),
                output_dir=output,
                run_id=run_id,
                proposer_model=arguments.proposer_model,
            ),
            problem=load_problem(arguments.problem),
            problem_file=arguments.problem.resolve(),
        )
    except (ValueError, TypeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except DurableRunError as error:
        traceback.print_exception(error.cause)
        print(error.directory, file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
