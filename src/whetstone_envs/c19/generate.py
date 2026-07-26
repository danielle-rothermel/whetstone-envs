r"""The seeded c19 generator.

Produces a :class:`~whetstone_envs.core.pool.TaskPool` of grid-world
state-prediction instances, deterministic given ``(GENERATOR_VERSION,
seed_start, n_per_stratum, env_ids, size_levels, command_length)``. Each
instance carries the public ASCII grid and command string in
``prompt_inputs`` and the derived-fact gold in ``gold``.

Ground truth is read from the *live* Minigrid object model
(:func:`whetstone_envs.c19.envs.rollout`) and then cross-checked against
the independent ASCII-only oracle walk
(:func:`whetstone_envs.c19.oracle.derive_fact`) at construction. The gold
frozen into each instance is the value the two independent walks agree
on, so gold and oracle can never silently diverge.

Contamination guard (spec rubric mapping, criterion 8 -- "fresh reserved
seeds, no published instance"). Minigrid ships *fixed registered seeds*
only implicitly (its unit tests and docs use small seeds like 0, 42);
the spec calls for a fresh reserved range no paper has published. This
module:

* reserves a low seed range as "published/example" space and asserts
  every consumed seed sits strictly above it at construction; and
* asserts no generated instance reuses one of the well-known Minigrid
  example seeds (:data:`PUBLISHED_SEEDS`) -- the concrete "never a
  published instance" check for a task built on a standard env whose
  documentation ships worked example seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from whetstone_envs.c19 import envs, oracle
from whetstone_envs.c19.envs import (
    ENV_IDS,
    SIZE_LEVELS,
    applicable_fact_types,
)
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c19-generate-1"

# --- Contamination bounds (rubric criterion 8) ----------------------------
# Minigrid's own docs/tests reproduce grids from small example seeds
# (0, 1, 2, 42, ...). We reserve the entire low range as published/example
# space and draw fresh seeds strictly above it, then additionally assert no
# consumed seed is one of the well-known published example seeds.
RESERVED_SEED_MAX = 100_000
DEFAULT_SEED_START = 1_000_000

# Seeds that appear as worked examples in Minigrid's documentation, README,
# and test suite. The guard asserts the generated pool never reuses one, so
# a c19 instance can never coincide with a published Minigrid example.
PUBLISHED_SEEDS: frozenset[int] = frozenset({0, 1, 2, 3, 42, 1337})

# --- Strata design (spec Section 1) ---------------------------------------
# N per stratum = 15 (spec Section 1, resolvability arithmetic). Split into
# an internal-eval slice (>=2/stratum, criterion 5), an official slice, and
# a held-out slice. TaskPool.split groups complete strata combinations before
# drawing its disjoint subsets. 3 + 6 + 6 = 15.
DEFAULT_INTERNAL_EVAL_PER_STRATUM = 3
DEFAULT_OFFICIAL_PER_STRATUM = 6
DEFAULT_HELD_OUT_PER_STRATUM = 6
DEFAULT_N_PER_STRATUM = (
    DEFAULT_INTERNAL_EVAL_PER_STRATUM
    + DEFAULT_OFFICIAL_PER_STRATUM
    + DEFAULT_HELD_OUT_PER_STRATUM
)

# Command length: short enough that discriminating difficulty comes from
# the conventions, not deep multi-step execution (spec Section 3 outcome
# (b): "keep execution depth shallow -- short command sequences").
DEFAULT_COMMAND_LENGTH = 8


def _build_one(
    env_id: str,
    size: str,
    fact_type: str,
    seed: int,
    *,
    command_length: int,
) -> Instance:
    """Build one instance and cross-check its gold against the oracle.

    The live Minigrid walk supplies ground truth; the independent
    ASCII-only oracle walk must reproduce it, or construction fails
    (rubric criterion 2/8: gold is never a re-derivation the oracle
    cannot independently confirm).
    """
    roll = envs.rollout(env_id, size, seed, command_length=command_length)
    live_gold = roll.facts[fact_type]
    oracle_gold = oracle.derive_fact(roll.grid_ascii, roll.command, fact_type)
    if live_gold != oracle_gold:
        msg = (
            f"gold disagreement for {env_id}|{size}|{fact_type} seed {seed}: "
            f"live Minigrid={live_gold!r} vs oracle={oracle_gold!r}"
        )
        raise AssertionError(msg)
    return make_instance(
        id=f"c19-{env_id}-{size}-{fact_type}-{seed}",
        seed=seed,
        strata=f"{env_id}|{size}|{fact_type}",
        prompt_inputs={
            "grid": roll.grid_ascii,
            "command": roll.command,
            "fact_type": fact_type,
        },
        gold=live_gold,
    )


def _assert_fresh_seeds(seeds: Sequence[int]) -> None:
    """Assert every seed is fresh: above the reserved range and unpublished.

    This is the construction-time fresh-seed / contamination assertion the
    spec requires (rubric criterion 8), not a comment.
    """
    for seed in seeds:
        if seed <= RESERVED_SEED_MAX:
            msg = (
                f"seed {seed} is not strictly above the reserved published "
                f"range ceiling {RESERVED_SEED_MAX} -- contamination guard "
                f"(rubric criterion 8)"
            )
            raise AssertionError(msg)
        if seed in PUBLISHED_SEEDS:
            msg = (
                f"seed {seed} is a published Minigrid example seed -- "
                f"contamination guard (rubric criterion 8)"
            )
            raise AssertionError(msg)


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    env_ids: Sequence[str] = ENV_IDS,
    size_levels: Sequence[str] = SIZE_LEVELS,
    command_length: int = DEFAULT_COMMAND_LENGTH,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c19 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Instances per stratum. Default 15 (spec Section 1); owner-open.
    env_ids:
        Env ids to include (default all four spec Section 1 envs).
    size_levels:
        Size levels to include (default ``small`` and ``medium``).
    command_length:
        Length of every sampled command string (spec Section 3: kept
        short so difficulty is convention-driven, not execution-depth).
    seed_start:
        First fresh seed; seeds are consumed contiguously and asserted
        fresh (above the reserved range and unpublished).

    Deterministic given the arguments: each stratum consumes a contiguous
    seed sub-range in a fixed env/size/fact order, and each instance's
    grid and command are pure functions of its seed (Minigrid threads the
    seed through ``reset``; the command RNG is seeded identically). The
    strata are then interleaved round-robin. Role stratification is handled
    independently by :meth:`~whetstone_envs.core.pool.TaskPool.split`, which
    groups full strata combinations before drawing its disjoint subsets.
    """
    stratum_specs: list[tuple[str, str, str]] = [
        (env_id, size, fact)
        for env_id in env_ids
        for size in size_levels
        for fact in applicable_fact_types(env_id)
    ]
    per_stratum: list[list[Instance]] = []
    consumed_seeds: list[int] = []
    next_seed = seed_start
    for env_id, size, fact in stratum_specs:
        block: list[Instance] = []
        for _ in range(n_per_stratum):
            seed = next_seed
            next_seed += 1
            consumed_seeds.append(seed)
            block.append(
                _build_one(
                    env_id,
                    size,
                    fact,
                    seed,
                    command_length=command_length,
                ),
            )
        per_stratum.append(block)

    _assert_fresh_seeds(consumed_seeds)

    # Interleave: row 0 = one instance from each stratum, then row 1, ...
    instances: list[Instance] = [
        block[row]
        for row in range(n_per_stratum)
        for block in per_stratum
    ]
    return TaskPool(instances)


def default_split_sizes(
    pool: TaskPool,
    *,
    internal_eval_per_stratum: int = DEFAULT_INTERNAL_EVAL_PER_STRATUM,
    official_per_stratum: int = DEFAULT_OFFICIAL_PER_STRATUM,
    held_out_per_stratum: int = DEFAULT_HELD_OUT_PER_STRATUM,
) -> tuple[int, int, int]:
    """Return ``(internal_eval_n, official_n, held_out_n)`` for a pool.

    Scales the per-stratum split sizes (spec Section 1: internal-eval
    >=2/stratum kept at 3, then 6 official, 6 held-out) by the number of
    strata present. :meth:`~whetstone_envs.core.pool.TaskPool.split`
    groups complete strata combinations and draws them round-robin, so these
    role sizes yield disjoint, stratum-balanced internal-eval, official, and
    held-out subsets: 3/stratum, 6/stratum, and 6/stratum respectively.
    """
    n_strata = len(pool.strata)
    return (
        internal_eval_per_stratum * n_strata,
        official_per_stratum * n_strata,
        held_out_per_stratum * n_strata,
    )


def build_manifest(pool: TaskPool, seed_start: int) -> Manifest:
    """Derive the default-config :class:`Manifest` for ``pool``.

    The seed range recorded is ``[seed_start, seed_start + len(pool))`` --
    dense, since no seeds are skipped, so it pins the exact fresh window.
    """
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(seed_start, seed_start + len(pool)),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate the default pool and write its manifest.

    Every owner-open numeric is a flag with the spec's proposed default,
    so changing N-per-stratum, the env/size sets, the command length, or
    the seed start never requires an env-code edit (PLAN "Config
    surface").
    """
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c19-generate",
        description="Generate the c19 Minigrid state-prediction pool.",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=DEFAULT_N_PER_STRATUM,
        help=f"instances per stratum (spec Sec 1: {DEFAULT_N_PER_STRATUM})",
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=list(ENV_IDS),
        help=f"env ids to include (default: {' '.join(ENV_IDS)})",
    )
    parser.add_argument(
        "--size-levels",
        nargs="+",
        default=list(SIZE_LEVELS),
        help=f"size levels to include (default: {' '.join(SIZE_LEVELS)})",
    )
    parser.add_argument(
        "--command-length",
        type=int,
        default=DEFAULT_COMMAND_LENGTH,
        help=f"command string length (default: {DEFAULT_COMMAND_LENGTH})",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help=f"first fresh seed (default: {DEFAULT_SEED_START})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
        help="where to write the default-config manifest JSON",
    )
    args = parser.parse_args(argv)

    pool = generate_pool(
        n_per_stratum=args.n_per_stratum,
        env_ids=args.env_ids,
        size_levels=args.size_levels,
        command_length=args.command_length,
        seed_start=args.seed_start,
    )
    manifest = build_manifest(pool, args.seed_start)
    manifest.write(args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
