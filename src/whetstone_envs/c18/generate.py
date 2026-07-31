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
from dataclasses import dataclass
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
    distractors_by_depth: dict[int, str] | None = None,
    ontology: str = FIXED_ONTOLOGY,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c18 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Positive integer number of instances per depth stratum. Default 30
        (a fast wrapped-subprocess default; spec Section 1 proposes 150 /
        Open Decision O3).
    depths:
        Nonempty sequence of distinct hop-depth levels to include (default
        D1, D2, D3, D5; spec Section 1 / Open Decision O2).
    distractors:
        Native distractor knob (spec Open Decision O1; default
        ``relevant`` = ON to avoid the trivial shortcut). Applied uniformly
        unless a depth appears in ``distractors_by_depth``.
    distractors_by_depth:
        Optional per-depth override of ``distractors``. Needed for the hard
        preset: the upstream fictional ontology can sustain relevant
        distractors only up to ~5 hops (deeper chains exhaust the nonce
        concept/property vocabulary and never generate), so a deep-chain
        variant sets distractors ``none`` at those depths where they are
        infeasible and keeps them ``relevant`` where the generator can
        produce them. A depth absent from the map uses ``distractors``.
    ontology:
        Pinned to ``fictional``; any other value fails the construction
        assertion (contamination guard).
    seed_start:
        First fresh seed; one seed is consumed per depth stratum,
        contiguously, and asserted fresh.

    Deterministic given the arguments: each depth stratum consumes one
    fixed seed and the vendored generator is byte-reproducible under a
    fixed seed (repos review; verified by regenerating twice). Instances
    are ordered by deterministic depth-interleaving.
    """
    if (
        not isinstance(n_per_stratum, int)
        or isinstance(n_per_stratum, bool)
        or n_per_stratum <= 0
    ):
        msg = (
            f"n_per_stratum must be a positive integer, got {n_per_stratum!r}"
        )
        raise ValueError(msg)
    if not depths:
        msg = "c18 requires at least one depth stratum"
        raise ValueError(msg)

    _assert_fixed_ontology(ontology)
    if len(set(depths)) != len(depths):
        msg = f"c18 requires distinct depth strata, got {tuple(depths)!r}"
        raise ValueError(msg)

    consumed_seeds = [seed_start + i for i in range(len(depths))]
    _assert_fresh_seeds(consumed_seeds)

    by_depth = distractors_by_depth or {}
    per_stratum: list[list[Instance]] = []
    for depth, seed in zip(depths, consumed_seeds, strict=True):
        per_stratum.append(
            _build_stratum(
                depth,
                seed,
                n_per_stratum,
                distractors=by_depth.get(depth, distractors),
            ),
        )

    # Deterministic depth-interleaving: row 0 has one instance per depth.
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

    Scales the per-stratum split sizes by the number of depth strata.
    :meth:`~whetstone_envs.core.pool.TaskPool.split` groups complete strata
    combinations and draws them round-robin, so all three disjoint slices
    are depth-balanced.
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


# --- Hard-mode preset -----------------------------------------------------
# The hardest configuration of the *upstream PrOntoQA* suite along the two
# axes this env already exposes -- deduction depth and distractors -- with
# NO hidden-information change (no Unknown label, no OOD rule type, no
# constraint-puzzle stratum). It pushes hop depth well past the base pool's
# D5 ceiling (D5, D8, D10). Both axes are native `generate_pool` arguments,
# so the preset is config, not a code fork.
#
# UPSTREAM CEILING (measured, root-caused). The pinned vendored generator
# cannot produce chains deeper than ~5 hops with distractors ON: relevant
# distractors add sibling ontology branches drawn from the SAME fixed 17-name
# fictional concept pool (run_experiment.py line ~348) plus a bounded set of
# property families, and a 6+-hop main chain PLUS those distractor branches
# exhausts that vocabulary, so `generate_question` rejects (returns None)
# indefinitely. Empirically: D5+relevant succeeds (~0.4% accept, sub-second
# for N=3); D6/D8/D10+relevant produce ZERO instances in >100k attempts each
# (~590k rejected attempts across four independent probes). `irrelevant`
# distractors are worse still (a fully disjoint parallel ontology). Distractors
# OFF, by contrast, generate D6/D8/D10 in milliseconds.
#
# Consequently the hard preset keeps distractors ON at the DEEPEST depth where
# the generator can honor them (D5) and drops them to `none` at D8/D10, where
# they are upstream-infeasible. At those depths the 8- and 10-hop chain length
# is itself the dominant hardness lever (far past base c18's D5 ceiling), so
# the trivial property-string-match shortcut a distractor guards against is
# already blunted by chain length. Reaching D8/D10 WITH distractors would
# require vendoring a larger concept ontology -- out of scope this pass
# (depth + distractors only, no ontology change). See
# HARD_DISTRACTORS_BY_DEPTH.
#
# The forward-chaining fixpoint ORACLE is UNCHANGED: it is a pure function of
# the public question + query text and closes to a least fixpoint regardless
# of chain length or how many distractor rules pad the theory (its design
# property; verified by hand-traced D8, D10-with-distractor, and
# distractor-not-entailed fixtures in the oracle tests). Nothing in oracle.py
# special-cases depth or distractor count.
HARD_DEPTHS: tuple[int, ...] = (5, 8, 10)

# Per-depth distractor policy (see the ceiling note above): distractors ON
# (`relevant`) at D5 where the generator honors them, `none` at D8/D10 where
# they are upstream-infeasible. A depth absent here falls back to the preset's
# base `distractors`. This is the one forced deviation from a uniform
# distractors-ON hard variant, dictated by the pinned upstream's vocabulary.
HARD_DISTRACTORS_BY_DEPTH: dict[int, str] = {
    5: "relevant",
    8: "none",
    10: "none",
}

# N per depth stratum for the hard variant (config-overridable at the call
# site via `HARD_PRESET.generate(n_per_stratum=...)`). 20 is the committed
# default -- large enough for a meaningful split, small enough that a full
# regenerate stays tractable.
HARD_N_PER_STRATUM = 20

# Fresh seed start for the hard variant. Chosen strictly disjoint from BOTH
# the published PrOntoQA space (reserved <= RESERVED_SEED_MAX, plus the
# upstream default seed) AND the base c18 pool's fresh window
# (DEFAULT_SEED_START == 1_000_000_000, which consumes one seed per default
# depth). 2_000_000_000 sits an order of magnitude above the base start, so
# a hard instance can never reuse a base-pool or published seed.
HARD_SEED_START = 2_000_000_000

# Split sizes per stratum for the hard variant: internal 2 / official 6 /
# held_out 12 = 20 per stratum with no unused instances, giving whole-pool
# totals internal 6 / official 18 / held_out 36 across the three depths.
HARD_INTERNAL_EVAL_PER_STRATUM = 2
HARD_OFFICIAL_PER_STRATUM = 6
HARD_HELD_OUT_PER_STRATUM = 12


@dataclass(frozen=True, slots=True)
class Preset:
    """A named, self-describing c18 generation configuration.

    A preset pins the two difficulty axes this env exposes -- the depth
    stratum levels and the distractor knob -- plus a disjoint fresh seed
    start, so a whole pool variant is expressible as config rather than a
    code fork. :func:`generate_pool` and :func:`build_manifest` already
    accept these axes; a preset just bundles a proposed default N with them
    under a stable name (recorded on the manifest's ``generator_version``).

    Parameters
    ----------
    name:
        Stable identity for the preset (folded into its manifest version).
    depths:
        The hop-depth stratum levels for this variant.
    distractors:
        The native distractor knob applied to any depth not overridden by
        ``distractors_by_depth`` (``relevant`` = ON).
    seed_start:
        First fresh seed; one seed is consumed per depth stratum,
        contiguously, and asserted fresh + disjoint from the base pool.
    n_per_stratum:
        Proposed default instances per depth stratum (overridable at the
        call site).
    distractors_by_depth:
        Optional per-depth distractor override, threaded straight into
        :func:`generate_pool`. The hard preset uses it to keep distractors
        ON at the deepest depth the upstream generator can honor (D5) and
        drop them to ``none`` at the deeper strata where they are
        upstream-infeasible (see the module ceiling note).
    """

    name: str
    depths: tuple[int, ...]
    distractors: str
    seed_start: int
    n_per_stratum: int = HARD_N_PER_STRATUM
    distractors_by_depth: dict[int, str] | None = None

    def generate(self, *, n_per_stratum: int | None = None) -> TaskPool:
        """Generate this preset's pool (``n_per_stratum`` overridable)."""
        return generate_pool(
            n_per_stratum=(
                self.n_per_stratum if n_per_stratum is None else n_per_stratum
            ),
            depths=self.depths,
            distractors=self.distractors,
            distractors_by_depth=self.distractors_by_depth,
            seed_start=self.seed_start,
        )

    def build_manifest(self, pool: TaskPool) -> Manifest:
        """Derive this preset's manifest (name recorded as generator id)."""
        return Manifest.from_pool(
            pool,
            generator_version=f"{GENERATOR_VERSION}+{self.name}",
            seed_range=(self.seed_start, self.seed_start + len(self.depths)),
        )

    def default_split_sizes(self, pool: TaskPool) -> tuple[int, int, int]:
        """Return ``(internal_eval_n, official_n, held_out_n)`` for ``pool``.

        Scales the hard-variant split proportions (2 / 6 / 12 at the
        default N=20) to the pool's actual uniform per-stratum count, then
        multiplies by the number of depth strata. This keeps an
        ``n_per_stratum`` override partitionable and depth-balanced. C18
        generated pools carry exactly one depth label per instance; this
        method is not a general apportioner for multi-label pools.
        """
        stratum_counts = pool.stratum_counts()
        if not stratum_counts:
            msg = "cannot derive preset split sizes from an empty pool"
            raise ValueError(msg)
        unique_counts = set(stratum_counts.values())
        if len(unique_counts) != 1:
            msg = (
                "preset split sizing requires uniform stratum counts, got "
                f"{stratum_counts!r}"
            )
            raise ValueError(msg)

        n_per_stratum = unique_counts.pop()
        hard_total = (
            HARD_INTERNAL_EVAL_PER_STRATUM
            + HARD_OFFICIAL_PER_STRATUM
            + HARD_HELD_OUT_PER_STRATUM
        )
        internal_per_stratum = (
            n_per_stratum * HARD_INTERNAL_EVAL_PER_STRATUM // hard_total
        )
        official_boundary = (
            n_per_stratum
            * (HARD_INTERNAL_EVAL_PER_STRATUM + HARD_OFFICIAL_PER_STRATUM)
            // hard_total
        )
        official_per_stratum = official_boundary - internal_per_stratum
        held_out_per_stratum = n_per_stratum - official_boundary
        n_strata = len(pool.strata)
        return (
            internal_per_stratum * n_strata,
            official_per_stratum * n_strata,
            held_out_per_stratum * n_strata,
        )


HARD_PRESET = Preset(
    name="hard",
    depths=HARD_DEPTHS,
    distractors=DEFAULT_DISTRACTORS,
    seed_start=HARD_SEED_START,
    n_per_stratum=HARD_N_PER_STRATUM,
    distractors_by_depth=HARD_DISTRACTORS_BY_DEPTH,
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
        default=None,
        help=(
            "instances per depth stratum (default: the base pool's "
            f"{DEFAULT_N_PER_STRATUM}, or the preset's own default under "
            "--preset)"
        ),
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
        "--preset",
        choices=["hard"],
        default=None,
        help=(
            "named generation preset (overrides the axis flags and writes "
            "the preset's own manifest by default). 'hard' = the hardest "
            "upstream PrOntoQA configuration along this env's axes: depths "
            "(5, 8, 10), with relevant distractors at D5 and none at D8/D10; "
            "no hidden-information change."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "where to write the manifest JSON (default: manifest.json, or "
            "manifest_<preset>.json when --preset is given)"
        ),
    )
    args = parser.parse_args(argv)

    if args.preset == "hard":
        pool = HARD_PRESET.generate(n_per_stratum=args.n_per_stratum)
        manifest = HARD_PRESET.build_manifest(pool)
        manifest_path = args.manifest or Path(__file__).with_name(
            "manifest_hard.json",
        )
    else:
        pool = generate_pool(
            n_per_stratum=(
                DEFAULT_N_PER_STRATUM
                if args.n_per_stratum is None
                else args.n_per_stratum
            ),
            depths=args.depths,
            distractors=args.distractors,
            seed_start=args.seed_start,
        )
        manifest = build_manifest(pool, args.seed_start, len(args.depths))
        manifest_path = args.manifest or Path(__file__).with_name(
            "manifest.json",
        )
    manifest.write(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
