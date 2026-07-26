r"""The seeded c18 generator (PrOntoQA deductive entailment).

Produces a :class:`~whetstone_envs.core.pool.TaskPool` of depth-binned
deductive-entailment instances by reseeding the *vendored*
``asaparov/prontoqa`` generator (see
:mod:`whetstone_envs.c18.upstream`), one fresh ``--seed`` per depth
stratum. Each instance carries the public ``question`` (facts + rules)
and ``query`` in ``prompt_inputs`` and the ``True``/``False`` entailment
label in ``gold``.

Ground truth is cross-checked at construction. The generator's stored
``answer`` is *definitional* -- a 50 % negation flag, not an independent
prover verdict (repos review red flag #5) -- so this module re-derives
the label with the from-scratch forward-chaining oracle
(:func:`whetstone_envs.c18.oracle.entailment_label`) from the public text
alone and asserts the two agree. A generation-soundness bug that the
label alone would hide fails the build here (spec Open Decision O8: full
fixpoint check on 100 % of instances).

Two construction-time contamination assertions encode the spec's fixed
constraints (rubric criterion 8) as checks, not comments:

* **fixed nonce ontology** -- generation is pinned to
  ``--ontology fictional`` (nonce ``wumpus``/``yumpus`` symbols carrying
  no real-world prior), asserted before any subprocess runs; the
  ``true``/``false`` real-world ontologies are refused.
* **fresh seed range** -- every consumed ``--seed`` is asserted strictly
  above a reserved range and never equal to the upstream default seed
  (the seed behind every *published* PrOntoQA instance), so a c18
  instance can never coincide with a released one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from whetstone_envs.c18 import oracle, upstream
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c18-generate-1"

# --- Fixed nonce ontology (spec fixed constraints; rubric criterion 8) ----
# Generation is pinned to the fictional (nonce) ontology so surface symbols
# carry no real-world prior. The real-world true/false ontologies are
# refused at construction -- they would create a ceiling effect and are a
# contamination risk (the spec excludes them).
FIXED_ONTOLOGY = "fictional"

# --- Distractors (spec Open Decision O1) ----------------------------------
# O1 default: distractors ON (`relevant`) -- required to avoid the trivial
# property-string-match shortcut that would collapse all headroom. An
# owner may flip this via the `distractors` argument / CLI flag.
DEFAULT_DISTRACTORS = "relevant"

# --- Contamination bounds (rubric criterion 8) ----------------------------
# The upstream default seed (62471893) sits behind every *published*
# PrOntoQA instance. We reserve the whole low range as published/example
# space, draw fresh seeds strictly above it, and additionally refuse the
# upstream default exactly.
RESERVED_SEED_MAX = 100_000_000
DEFAULT_SEED_START = 1_000_000_000
PUBLISHED_SEED = upstream.UPSTREAM_DEFAULT_SEED

# --- Strata design (spec Section 1) ---------------------------------------
# One stratification axis: hop depth. Four levels D1, D2, D3, D5 (D4/D0
# dropped to bound N; spec Section 1 + Open Decision O2). Ontology type is
# held constant (fictional), not a stratum axis.
DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 3, 5)

# N per depth stratum. Spec Section 1 proposes 150 (cheap-iteration floor;
# 400 is the publication-grade target, Open Decision O3). Kept small by
# default so the wrapped-subprocess test suite stays fast; owner-open.
DEFAULT_N_PER_STRATUM = 30

# Split sizes per stratum (spec Section 1: disjoint internal-eval subset of
# >=10-20 tasks + a held-out official split). Scaled to the smaller default
# N here (internal-eval kept >=2/stratum, criterion 5); owner-open.
DEFAULT_INTERNAL_EVAL_PER_STRATUM = 6
DEFAULT_OFFICIAL_PER_STRATUM = 12
DEFAULT_HELD_OUT_PER_STRATUM = 12


def depth_label(hops: int) -> str:
    """The stratum label for a hop-depth level (``D1``, ``D2``, ...)."""
    return f"D{hops}"


def _assert_fixed_ontology(ontology: str) -> None:
    """Assert generation uses the fixed nonce ontology (criterion 8).

    A construction-time check, not a comment: the real-world
    ``true``/``false`` ontologies are refused so c18 can never be
    generated over contaminating real-world symbols.
    """
    if ontology != FIXED_ONTOLOGY:
        msg = (
            f"c18 requires the fixed nonce ontology {FIXED_ONTOLOGY!r} "
            f"(contamination guard, rubric criterion 8), got "
            f"{ontology!r}"
        )
        raise AssertionError(msg)


def _assert_fresh_seeds(seeds: Sequence[int]) -> None:
    """Assert every consumed seed is fresh (criterion 8).

    Fresh means strictly above the reserved published range and never the
    upstream default seed behind published PrOntoQA instances. A
    construction-time assertion, not a comment.
    """
    for seed in seeds:
        if seed == PUBLISHED_SEED:
            msg = (
                f"seed {seed} is the upstream default seed behind published "
                f"PrOntoQA instances -- contamination guard (criterion 8)"
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
    hops: int,
    seed: int,
    n_per_stratum: int,
    *,
    distractors: str,
) -> list[Instance]:
    """Generate + oracle-cross-check one depth stratum's instances.

    Reseeds the vendored generator once at ``seed`` for ``n_per_stratum``
    instances at this hop depth, then re-derives each label with the
    independent forward-chaining oracle and asserts agreement -- the
    verifier-soundness check the definitional label cannot provide alone.
    """
    raws = upstream.generate_raw(
        hops=hops,
        seed=seed,
        num_trials=n_per_stratum,
        ontology=FIXED_ONTOLOGY,
        distractors=distractors,
    )
    if len(raws) != n_per_stratum:
        msg = (
            f"depth D{hops} seed {seed}: vendored generator produced "
            f"{len(raws)} instances, expected {n_per_stratum}"
        )
        raise AssertionError(msg)
    instances: list[Instance] = []
    for idx, raw in enumerate(raws):
        derived = oracle.entailment_label(raw.question, raw.query)
        if derived != raw.answer:
            msg = (
                f"gold disagreement at D{hops} seed {seed} #{idx}: "
                f"generator answer={raw.answer!r} vs forward-chaining "
                f"oracle={derived!r} for query {raw.query!r}"
            )
            raise AssertionError(msg)
        instances.append(
            make_instance(
                id=f"c18-D{hops}-{seed}-{idx:04d}",
                seed=seed,
                strata=depth_label(hops),
                prompt_inputs={
                    "question": raw.question,
                    "query": raw.query,
                },
                gold=raw.answer,
            ),
        )
    return instances


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    distractors: str = DEFAULT_DISTRACTORS,
    ontology: str = FIXED_ONTOLOGY,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c18 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Instances per depth stratum. Default 30 (a fast wrapped-subprocess
        default; spec Section 1 proposes 150 / Open Decision O3).
    depths:
        Hop-depth levels to include (default D1, D2, D3, D5; spec Section
        1 / Open Decision O2).
    distractors:
        Native distractor knob (spec Open Decision O1; default
        ``relevant`` = ON to avoid the trivial shortcut).
    ontology:
        Pinned to ``fictional``; any other value fails the construction
        assertion (contamination guard).
    seed_start:
        First fresh seed; one seed is consumed per depth stratum,
        contiguously, and asserted fresh.

    Deterministic given the arguments: each depth stratum consumes one
    fixed seed and the vendored generator is byte-reproducible under a
    fixed seed (repos review; verified by regenerating twice). The strata
    are then interleaved round-robin so a contiguous
    :meth:`~whetstone_envs.core.pool.TaskPool.split` slice is
    depth-balanced.
    """
    _assert_fixed_ontology(ontology)

    consumed_seeds = [seed_start + i for i in range(len(depths))]
    _assert_fresh_seeds(consumed_seeds)

    per_stratum: list[list[Instance]] = []
    for depth, seed in zip(depths, consumed_seeds, strict=True):
        per_stratum.append(
            _build_stratum(
                depth,
                seed,
                n_per_stratum,
                distractors=distractors,
            ),
        )

    # Interleave: row 0 = one instance from each depth, then row 1, ...
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

    Scales the per-stratum split sizes by the number of depth strata. The
    interleaved layout makes each contiguous front slice of
    ``k * n_strata`` instances exactly ``k`` per stratum, so all three
    disjoint slices are depth-balanced.
    """
    n_strata = len(pool.strata)
    return (
        internal_eval_per_stratum * n_strata,
        official_per_stratum * n_strata,
        held_out_per_stratum * n_strata,
    )


def build_manifest(pool: TaskPool, seed_start: int, n_depths: int) -> Manifest:
    """Derive the default-config :class:`Manifest` for ``pool``.

    The seed range recorded is ``[seed_start, seed_start + n_depths)`` --
    one fresh seed per depth stratum, the exact fresh window consumed.
    """
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(seed_start, seed_start + n_depths),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate the default pool and write its manifest.

    Every owner-open numeric is a flag with the spec's proposed default,
    so changing N-per-stratum, the depth set, the distractor knob (O1), or
    the seed start never requires an env-code edit (PLAN "Config
    surface").
    """
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c18-generate",
        description="Generate the c18 PrOntoQA deductive-entailment pool.",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=DEFAULT_N_PER_STRATUM,
        help=f"instances per depth stratum (default {DEFAULT_N_PER_STRATUM})",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="hop-depth levels (spec Sec 1 default: 1 2 3 5)",
    )
    parser.add_argument(
        "--distractors",
        choices=("none", "relevant", "irrelevant"),
        default=DEFAULT_DISTRACTORS,
        help="native distractor knob (spec O1 default: relevant = ON)",
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
        depths=args.depths,
        distractors=args.distractors,
        seed_start=args.seed_start,
    )
    manifest = build_manifest(pool, args.seed_start, len(args.depths))
    manifest.write(args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
