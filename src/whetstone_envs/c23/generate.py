r"""The seeded c23 generator (subregular ISL/OSL rule induction).

Produces a :class:`~whetstone_envs.core.pool.TaskPool` of single-rule
ISL/OSL induction instances by reseeding the *vendored + patched*
InductionBench generator (see :mod:`whetstone_envs.c23.upstream`), one fresh
seed per stratum. Each instance carries the public ``demos_block`` (the
``IN -> OUT`` demonstration lines) and ``query`` input in ``prompt_inputs``,
and the oracle-derived transformed string in ``gold``.

Strata (spec Section 1): the primary difficulty axis is ``k`` (context
window), the secondary axis is the transducer ``type``; vocab is held fixed
at |Sigma|=4 and ``number_of_rules`` is fixed at 1 (the single-rule
constraint). Four strata:

* **S1** -- ISL, k=2 (easiest)
* **S2** -- L-OSL, k=2
* **S3** -- R-OSL, k=2
* **S4** -- ISL, k=3 (moderate; the k ladder stops at 3, not 4, so the
  hardest stratum stays inside the partial-success band -- spec Section 1.2)

Ground truth is cross-checked at construction: each instance's frozen
``gold`` is re-derived by the independent oracle
(:func:`whetstone_envs.c23.oracle.apply_to_query`, which reuses the vendored
``apply_*_rule`` transducers unmodified) and asserted equal to the boundary's
stored gold. A generation bug that desynchronized the two would fail the
build here.

Two construction-time contamination assertions encode the spec's fixed
constraints (rubric criterion 8) as checks, not comments:

* **fresh seed range** -- every consumed seed is asserted strictly above a
  reserved range and never the upstream module-global default seed (``0``),
  so a c23 instance can never coincide with a published InductionBench
  instance (which the paper generated under ``random.seed(0)``).
* **single-rule only** -- ``number_of_rules`` is pinned to 1; any other value
  fails at construction (the baseline's fixed constraint).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from whetstone_envs.c23 import oracle, prompts, upstream
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c23-generate-1"

# --- Fixed constraints (spec "Fixed constraints") -------------------------
# Single-rule instances only; vocab held small and fixed at |Sigma|=4.
FIXED_NUMBER_OF_RULES = 1
FIXED_VOCAB_SIZE = 4

# --- Strata design (spec Section 1) ---------------------------------------
# (label, rule_type, k). k is the primary axis; type is secondary; vocab is
# fixed. The k ladder stops at 3 (S4) so the hardest stratum stays reachable.
Stratum = tuple[str, str, int]
DEFAULT_STRATA: tuple[Stratum, ...] = (
    ("S1", upstream.ISL, 2),
    ("S2", upstream.L_OSL, 2),
    ("S3", upstream.R_OSL, 2),
    ("S4", upstream.ISL, 3),
)

# N per stratum. Spec Section 1.3 proposes 50 (the pooled-decision floor;
# 4 strata x 50 = 200 instances). Kept modest by default so the test suite
# stays fast; owner-open (raise to 100-200 for per-stratum resolution).
DEFAULT_N_PER_STRATUM = 50

# Demonstrations shown per instance. The vendored characteristic sample is
# large (>100 pairs); the spec's few-shot block shows a handful, so we
# down-sample to a small demo set (rule-revealing pairs preferred). Owner-open.
DEFAULT_N_DEMOS = 6

# The InductionBench sample-size multiplier (vendored ``--sample_size_times``);
# only affects the size of the internal characteristic sample we down-sample
# from, never the frozen demo count. Left at the upstream default.
DEFAULT_SAMPLE_SIZE_TIMES = 10

# Longest held-out query string drawn (chars). Kept short so the query stays
# a bounded, cheap exact-match target (spec Section 5 cost bound).
DEFAULT_MAX_QUERY_LEN = 8

# --- Contamination bounds (rubric criterion 8) ----------------------------
# Published InductionBench instances were generated under the module-global
# ``random.seed(0)`` (upstream inference.py / __main__). We refuse seed 0
# exactly, reserve the whole low range as published/example space, and draw
# fresh seeds strictly above it -- one per stratum.
PUBLISHED_SEED = 0
RESERVED_SEED_MAX = 100_000_000
DEFAULT_SEED_START = 555_000_000

# Split sizes per stratum (disjoint internal-eval / official / held-out).
# TaskPool.split groups complete strata combinations and draws them
# round-robin. Owner-open.
DEFAULT_INTERNAL_EVAL_PER_STRATUM = 10
DEFAULT_OFFICIAL_PER_STRATUM = 20
DEFAULT_HELD_OUT_PER_STRATUM = 20


def _assert_single_rule(number_of_rules: int) -> None:
    """Assert the single-rule fixed constraint (spec Fixed constraints)."""
    if number_of_rules != FIXED_NUMBER_OF_RULES:
        msg = (
            f"c23 baseline is single-rule only (number_of_rules="
            f"{FIXED_NUMBER_OF_RULES}); the composition ladder is out of "
            f"scope for this baseline, got {number_of_rules}"
        )
        raise AssertionError(msg)


def _assert_fresh_seeds(seeds: Sequence[int]) -> None:
    """Assert every consumed seed is fresh (rubric criterion 8).

    Fresh means strictly above the reserved published range and never the
    upstream default seed (``0``) behind published InductionBench instances.
    A construction-time assertion, not a comment.
    """
    for seed in seeds:
        if seed == PUBLISHED_SEED:
            msg = (
                f"seed {seed} is the upstream default seed (random.seed(0)) "
                f"behind published InductionBench instances -- contamination "
                f"guard (rubric criterion 8)"
            )
            raise AssertionError(msg)
        if seed <= RESERVED_SEED_MAX:
            msg = (
                f"seed {seed} is not strictly above the reserved published "
                f"range ceiling {RESERVED_SEED_MAX} -- contamination guard "
                f"(rubric criterion 8)"
            )
            raise AssertionError(msg)


def _build_stratum(
    label: str,
    rule_type: str,
    k: int,
    seed: int,
    n_per_stratum: int,
    *,
    n_demos: int,
    sample_size_times: int,
    max_query_len: int,
) -> list[Instance]:
    """Generate + oracle-cross-check one stratum's instances.

    Reseeds the vendored generator once at ``seed`` for ``n_per_stratum``
    single-rule instances of this ``(type, k)``, then re-derives each gold
    with the independent oracle (reusing the vendored transducer unmodified)
    and asserts agreement -- the self-consistency check the repos review
    spot-checked, made a hard build gate here.
    """
    raws = upstream.generate_raw(
        rule_type=rule_type,
        k=k,
        vocab_size=FIXED_VOCAB_SIZE,
        seed=seed,
        num_instances=n_per_stratum,
        sample_size_times=sample_size_times,
        max_query_len=max_query_len,
        n_demos=n_demos,
    )
    if len(raws) != n_per_stratum:
        msg = (
            f"stratum {label} seed {seed}: vendored generator produced "
            f"{len(raws)} instances, expected {n_per_stratum}"
        )
        raise AssertionError(msg)
    instances: list[Instance] = []
    for idx, raw in enumerate(raws):
        derived = oracle.apply_to_query(
            raw.rule_type,
            raw.k,
            raw.rule,
            raw.query,
        )
        if derived != raw.gold:
            msg = (
                f"gold disagreement at {label} seed {seed} #{idx}: boundary "
                f"gold={raw.gold!r} vs independent-oracle re-application "
                f"{derived!r} for query {raw.query!r} rule {dict(raw.rule)!r}"
            )
            raise AssertionError(msg)
        demos_block = prompts.render_demos_block(dict(raw.demos))
        instances.append(
            make_instance(
                id=f"c23-{label}-{seed}-{idx:04d}",
                seed=seed,
                strata=label,
                prompt_inputs={
                    "demos_block": demos_block,
                    "query": raw.query,
                },
                gold=raw.gold,
            ),
        )
    return instances


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    strata: Sequence[Stratum] = DEFAULT_STRATA,
    n_demos: int = DEFAULT_N_DEMOS,
    sample_size_times: int = DEFAULT_SAMPLE_SIZE_TIMES,
    max_query_len: int = DEFAULT_MAX_QUERY_LEN,
    number_of_rules: int = FIXED_NUMBER_OF_RULES,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c23 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Instances per stratum (default 50; spec Section 1.3).
    strata:
        ``(label, rule_type, k)`` triples (default S1..S4; spec Section 1).
    n_demos:
        Demonstrations shown per instance (default 6; owner-open).
    sample_size_times / max_query_len:
        Vendored sample multiplier and the longest held-out query length.
    number_of_rules:
        Pinned to 1; any other value fails the single-rule assertion.
    seed_start:
        First fresh seed; one seed is consumed per stratum, contiguously,
        and asserted fresh.

    Deterministic given the arguments: each stratum consumes one fixed seed
    and the vendored generator is byte-reproducible under a fixed seed after
    the four vendor patches (the ``sorted(...)`` determinism fix), verified
    across runs under a randomized ``PYTHONHASHSEED``. The deterministic
    generation order interleaves strata round-robin. Role stratification is
    handled independently by :meth:`~whetstone_envs.core.pool.TaskPool.split`,
    which groups complete strata combinations before drawing its disjoint
    subsets round-robin.
    """
    _assert_single_rule(number_of_rules)

    consumed_seeds = [seed_start + i for i in range(len(strata))]
    _assert_fresh_seeds(consumed_seeds)

    per_stratum: list[list[Instance]] = []
    for (label, rule_type, k), seed in zip(
        strata,
        consumed_seeds,
        strict=True,
    ):
        per_stratum.append(
            _build_stratum(
                label,
                rule_type,
                k,
                seed,
                n_per_stratum,
                n_demos=n_demos,
                sample_size_times=sample_size_times,
                max_query_len=max_query_len,
            ),
        )

    # Interleave: row 0 = one instance from each stratum, then row 1, ...
    instances: list[Instance] = [
        block[row] for row in range(n_per_stratum) for block in per_stratum
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

    Scales the per-stratum split sizes by the number of strata.
    :meth:`~whetstone_envs.core.pool.TaskPool.split` groups complete strata
    combinations and draws them round-robin, yielding disjoint,
    stratum-balanced subsets for these role sizes.
    """
    n_strata = len(pool.strata)
    return (
        internal_eval_per_stratum * n_strata,
        official_per_stratum * n_strata,
        held_out_per_stratum * n_strata,
    )


def build_manifest(
    pool: TaskPool,
    seed_start: int,
    n_strata: int,
) -> Manifest:
    """Derive the default-config :class:`Manifest` for ``pool``.

    The seed range recorded is ``[seed_start, seed_start + n_strata)`` --
    one fresh seed per stratum, the exact fresh window consumed.
    """
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(seed_start, seed_start + n_strata),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate the default pool and write its manifest.

    Every owner-open numeric is a flag with the spec's proposed default, so
    changing N-per-stratum, the demo count, the query length, or the seed
    start never requires an env-code edit (PLAN "Config surface").
    """
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c23-generate",
        description="Generate the c23 subregular ISL/OSL induction pool.",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=DEFAULT_N_PER_STRATUM,
        help=f"instances per stratum (default {DEFAULT_N_PER_STRATUM})",
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=DEFAULT_N_DEMOS,
        help=f"demonstrations per instance (default {DEFAULT_N_DEMOS})",
    )
    parser.add_argument(
        "--max-query-len",
        type=int,
        default=DEFAULT_MAX_QUERY_LEN,
        help=f"longest held-out query (default {DEFAULT_MAX_QUERY_LEN})",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help=f"first fresh seed (default {DEFAULT_SEED_START})",
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
        n_demos=args.n_demos,
        max_query_len=args.max_query_len,
        seed_start=args.seed_start,
    )
    manifest = build_manifest(pool, args.seed_start, len(DEFAULT_STRATA))
    manifest.write(args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
