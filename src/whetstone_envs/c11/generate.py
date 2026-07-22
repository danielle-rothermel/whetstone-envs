"""The seeded, adversarial c11 generator.

Produces a :class:`~whetstone_envs.core.pool.TaskPool` of JSON
canonicalization instances, deterministic given ``(GENERATOR_VERSION,
seed_start, n_per_stratum, strata)``. Each instance is a messy input JSON
string (the model's ``{input}``) whose gold is that string's RFC 8785
canonical form, computed by the independent oracle so gold and oracle can
never disagree.

Adversarial construction (spec Section 1): a candidate messy string is
kept only if its canonical form actually *differs* from it, so every
instance genuinely exercises its stratum's JCS sub-rule; a string that is
already canonical is rejected and the next seed tried. Determinism holds
because rejection is a pure function of the seeded RNG stream: the same
config always walks the same accept/reject decisions.

Contamination guard (spec rubric mapping, criterion 8 -- "fresh seeds
only, never published instances"):

* the fresh seed range is asserted to sit strictly above a reserved
  published range at construction; and
* no generated instance's canonical gold may equal any RFC 8785
  *published test vector* (Section 3.2.3 / Appendix B), asserted at
  construction -- the concrete "never a published instance" check for a
  task adjacent to a standard that ships worked examples.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import rfc8785

from whetstone_envs.c11 import oracle
from whetstone_envs.c11.strata import BUILDERS, STRATA
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c11-generate-1"

# --- Contamination bounds (rubric criterion 8) ----------------------------
# c11's pool is synthetic and infinite; there is no published *seed* set to
# collide with, but RFC 8785 itself ships worked test vectors. We reserve a
# low range as "published/example" space and start fresh seeds strictly
# above it, mirroring the c22 convention, then additionally assert no gold
# reproduces a published RFC 8785 test vector (see PUBLISHED_VECTORS).
RESERVED_SEED_MAX = 100_000
DEFAULT_SEED_START = 1_000_000

# RFC 8785 published canonical outputs (Section 3.2.3 French key-ordering
# example and the Appendix B number list). Regenerated through the oracle
# in the test suite; asserted absent from every generated gold so the pool
# can never coincide with a standard-published instance.
PUBLISHED_VECTORS: frozenset[str] = frozenset(
    {
        (
            '{"peach":"This sorting order","péché":"is wrong according to '
            'French","pêche":"but canonicalization MUST","sin":"ignore '
            'locale"}'
        ),
        "[333333333.3333333,1e+30,4.5,0.002,1e-27]",
    },
)

# --- Strata design (spec Section 1) ---------------------------------------
# Per stratum: 2 internal-eval + 40 official + 40 held-out = 82. The
# internal-eval slice is kept *disjoint* (not a labeled subset) so the
# generic PoolSplit disjointness assertion holds; spec Section 7.4 leaves
# disjoint-vs-subset to the owner, and disjoint is the safer default.
# 82 x 5 strata = 410 pinned instances: 10 internal-eval + 200 official +
# 200 held-out (spec Section 1 / 7.3 proposed 200 + 200, plus the >=2/
# stratum internal-eval slice criterion 5 names).
DEFAULT_OFFICIAL_PER_STRATUM = 40
DEFAULT_HELD_OUT_PER_STRATUM = 40
DEFAULT_INTERNAL_EVAL_PER_STRATUM = 2
DEFAULT_N_PER_STRATUM = (
    DEFAULT_INTERNAL_EVAL_PER_STRATUM
    + DEFAULT_OFFICIAL_PER_STRATUM
    + DEFAULT_HELD_OUT_PER_STRATUM
)

# A generous per-stratum attempt budget: the adversarial predicate rejects
# only already-canonical strings, which are rare given the messy separators,
# so this ceiling is never approached in practice.
_MAX_ATTEMPTS_PER_INSTANCE = 100


def _build_one(stratum: str, seed: int) -> Instance | None:
    """Try to build one adversarial instance for ``(stratum, seed)``.

    Returns ``None`` if the seed produces an input that is already
    canonical (so it fails to exercise the sub-rule) or one the oracle
    cannot canonicalize; the caller advances to the next seed. The
    canonical gold is produced by the oracle, never recomputed here, so
    the oracle stays the sole source of truth.
    """
    rng = random.Random(seed)
    input_json = BUILDERS[stratum](rng)
    try:
        gold = oracle.canonicalize(input_json)
    except (json.JSONDecodeError, rfc8785.CanonicalizationError):
        # A seed whose input the oracle cannot canonicalize (e.g. an int
        # beyond the IEEE-754 safe domain) is skipped, not minted.
        return None
    # Adversarial: messy and canonical must differ, else the instance is
    # trivial for this stratum (spec Section 1).
    if input_json == gold:
        return None
    return make_instance(
        id=f"c11-{stratum}-{seed}",
        seed=seed,
        strata=stratum,
        prompt_inputs={"input": input_json},
        gold=gold,
    )


def _assert_fresh_seeds(seeds: Sequence[int]) -> None:
    """Assert every seed sits strictly above the reserved published range."""
    for seed in seeds:
        if seed <= RESERVED_SEED_MAX:
            msg = (
                f"seed {seed} is not strictly above the reserved published "
                f"range ceiling {RESERVED_SEED_MAX} -- contamination guard "
                f"(rubric criterion 8)"
            )
            raise AssertionError(msg)


def _assert_no_published_vectors(instances: Sequence[Instance]) -> None:
    """Assert no generated gold reproduces a published RFC 8785 vector."""
    for inst in instances:
        if inst.gold in PUBLISHED_VECTORS:
            msg = (
                f"instance {inst.id!r} reproduces a published RFC 8785 test "
                f"vector -- contamination guard (rubric criterion 8)"
            )
            raise AssertionError(msg)


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    strata: Sequence[str] = STRATA,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c11 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Instances per stratum (official + held-out combined). Default 80
        = 40 official + 40 held-out (spec Section 1 / 7.3 proposed N per
        stratum); owner-adjustable.
    strata:
        The strata to populate, in order. Default is all five spec
        Section 1 strata; a subset is permitted (spec Section 3 outcome
        (b) biases toward S1/S2, dropping S3).
    seed_start:
        First fresh seed. Seeds are consumed contiguously and asserted to
        sit strictly above the reserved published range.

    Deterministic given the arguments: each stratum consumes its own
    contiguous seed sub-range in order, skipping any seed whose
    adversarial predicate fails, so the same config always yields
    byte-identical instances. The strata are then **interleaved**
    round-robin (one instance per stratum per row) so that a contiguous
    :meth:`~whetstone_envs.core.pool.TaskPool.split` slice is
    stratum-balanced -- the internal-eval front slice then carries
    >=2/stratum without any per-candidate split logic (spec Section 1).
    """
    per_stratum: list[list[Instance]] = []
    consumed_seeds: list[int] = []
    next_seed = seed_start
    for stratum in strata:
        kept: list[Instance] = []
        attempts = 0
        while len(kept) < n_per_stratum:
            if attempts >= n_per_stratum * _MAX_ATTEMPTS_PER_INSTANCE:
                msg = (
                    f"exhausted seed budget building stratum {stratum!r}: "
                    f"kept {len(kept)} of {n_per_stratum} after "
                    f"{attempts} seeds"
                )
                raise RuntimeError(msg)
            seed = next_seed
            next_seed += 1
            attempts += 1
            inst = _build_one(stratum, seed)
            if inst is None:
                continue
            consumed_seeds.append(seed)
            kept.append(inst)
        per_stratum.append(kept)

    # Interleave: row 0 = one instance from each stratum, then row 1, ...
    instances: list[Instance] = [
        block[row] for row in range(n_per_stratum) for block in per_stratum
    ]

    _assert_fresh_seeds(consumed_seeds)
    _assert_no_published_vectors(instances)
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
    >=2/stratum, 40 official/stratum, 40 held-out/stratum) by the number
    of strata present. Because the pool is interleaved round-robin, a
    contiguous front slice of ``k * n_strata`` instances is exactly ``k``
    per stratum, so the three disjoint slices are each stratum-balanced:
    the internal-eval front slice carries >=2/stratum, official the next
    40/stratum, held-out the final 40/stratum.
    """
    n_strata = len(pool.strata)
    return (
        internal_eval_per_stratum * n_strata,
        official_per_stratum * n_strata,
        held_out_per_stratum * n_strata,
    )


def build_manifest(pool: TaskPool, seed_start: int) -> Manifest:
    """Derive the default-config :class:`Manifest` for ``pool``.

    The seed range recorded is ``[seed_start, seed_start + len(pool))``.
    Because some seeds are skipped by the adversarial predicate, the range
    is an inclusive-of-consumed span, not a dense one; the content hash is
    what pins reproducibility, and the range documents the fresh window.
    """
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(seed_start, seed_start + len(pool)),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate the default pool and write its manifest.

    Every owner-open numeric is a flag with the spec's proposed default,
    so changing N-per-stratum, the strata set, or the seed start never
    requires an env-code edit (PLAN "Config surface").
    """
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c11-generate",
        description="Generate the c11 JSON-canonicalization pool + manifest.",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=DEFAULT_N_PER_STRATUM,
        help=(
            "instances per stratum, official+held-out combined "
            f"(spec Section 1 proposed: {DEFAULT_N_PER_STRATUM})"
        ),
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        default=list(STRATA),
        help=f"strata to populate (default: {' '.join(STRATA)})",
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
        strata=args.strata,
        seed_start=args.seed_start,
    )
    manifest = build_manifest(pool, args.seed_start)
    manifest.write(args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
