"""The seeded stacking generator for c22.

Produces a :class:`~whetstone_envs.core.pool.TaskPool` of stacked
IFEval-constraint instances, deterministic given
``(generator_version, seed_range)``. Each instance is a trivial base
micro-task plus 3-5 composed constraints drawn from the atom pools in
:mod:`whetstone_envs.c22.atoms`, honoring IFEval's
``INSTRUCTION_CONFLICTS`` so no two stacked atoms contradict.

Determinism discipline (spec Section 7, item 8 -- the shipped
``config.seed`` is a decoy):

* every value we need to reproduce is drawn from a per-instance
  ``random.Random(seed)`` and passed into ``build_description`` as an
  explicit kwarg (see :mod:`whetstone_envs.c22.atoms`);
* immediately before each ``build_description`` we also seed the
  module-global ``random`` from the same instance RNG, so any atom that
  still reaches for the global RNG internally is reproducible too. No
  vendored line is edited to achieve this.

Contamination guard (spec Section 6 / rubric criterion 8): the fresh
seed range is asserted at construction to lie strictly above -- and to
never intersect -- the published IFEval dataset's integer key range.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from instruction_following_eval import instructions_registry

from whetstone_envs.c22.atoms import (
    EASY_POOL,
    HARD_POOL,
    Atom,
    atom_for,
)
from whetstone_envs.c22.spec import ConstraintSpec
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c22-generate-1"

# --- Contamination bounds (rubric criterion 8) ----------------------------
# The published google-research IFEval dataset (input_data.jsonl, 541 rows,
# vendored-commit 37ffb72) uses integer keys in the inclusive range below.
# Our fresh seeds start strictly above the ceiling and are asserted to
# never intersect the published range.
PUBLISHED_KEY_MIN = 13
PUBLISHED_KEY_MAX = 3757
DEFAULT_SEED_START = 1_000_000

# --- Strata design (spec Section 1) ---------------------------------------
CONSTRAINT_COUNTS: tuple[int, ...] = (3, 4, 5)
MIX_EASY = "easy"
MIX_MIXED = "mixed"
# The 'hard' mix (hardest configuration of the original IFEval suite, no
# hidden-information change): a 'hard' stack always includes ALL hard-pool
# atoms that do not conflict, then fills any remaining count from the easy
# pool (hard-first, conflict-aware). It differs from 'mixed' -- which
# seeds a single hard atom and fills randomly from the combined pool -- by
# maximizing hard-atom density rather than merely guaranteeing one.
MIX_HARD = "hard"

# Trivial base micro-tasks (spec: "produce a short answer or micro-
# description"). Kept generic so the constraint stack, not the base task,
# is the difficulty.
BASE_TASKS: tuple[str, ...] = (
    "Name a color.",
    "Name an animal.",
    "Describe the weather in a few words.",
    "Name a fruit.",
    "Name a musical instrument.",
    "Describe a season in a few words.",
    "Name a country.",
    "Name a kind of tree.",
)


def stratum_label(n_constraints: int, mix: str) -> str:
    """The stratum label for a (constraint-count, atom-mix) cell."""
    return f"n{n_constraints}_{mix}"


@dataclass(frozen=True, slots=True)
class Preset:
    """A named, self-describing generation configuration.

    A preset pins the generation *axes* (the constraint-count levels, the
    atom-mix levels, and the fresh seed start) so a whole pool variant is
    expressible as config rather than a code fork. ``generate_pool`` and
    ``build_manifest`` already accept these axes as arguments; a preset
    just bundles a proposed default N with them under a stable name.

    Parameters
    ----------
    name:
        Stable identity for the preset (recorded on its manifest).
    constraint_counts:
        The constraint-count axis levels for this variant.
    mixes:
        The atom-mix axis levels for this variant.
    seed_start:
        First fresh seed. Chosen disjoint from both the published IFEval
        dataset keys and every other preset's range so instances from
        different variants never collide.
    n_per_stratum:
        Proposed default instances per stratum (config-overridable at the
        call site).
    """

    name: str
    constraint_counts: tuple[int, ...]
    mixes: tuple[str, ...]
    seed_start: int
    n_per_stratum: int = 20

    def generate(self, *, n_per_stratum: int | None = None) -> TaskPool:
        """Generate this preset's pool (``n_per_stratum`` overridable)."""
        return generate_pool(
            n_per_stratum=(
                self.n_per_stratum if n_per_stratum is None else n_per_stratum
            ),
            constraint_counts=self.constraint_counts,
            mixes=self.mixes,
            seed_start=self.seed_start,
        )

    def build_manifest(self, pool: TaskPool) -> Manifest:
        """Derive this preset's manifest (name recorded as generator id)."""
        return Manifest.from_pool(
            pool,
            generator_version=f"{GENERATOR_VERSION}+{self.name}",
            seed_range=(self.seed_start, self.seed_start + len(pool)),
        )


# --- Hard-mode preset -----------------------------------------------------
# The hardest configuration of the original IFEval suite (no hidden-info
# change): 3 strata via counts x mixes = {(3, 6, 8)} x {hard}. At n=3 the
# 'hard' mix places all 3 hard atoms and needs no easy fill (pure-hard); at
# n=6 / n=8 it is 3 hard + 3 / 5 easy fill. The seed start is disjoint from
# both the published IFEval key range and the base c22 pool seeds
# (1_000_000..) so hard-variant instances never collide with either.
HARD_SEED_START = 2_000_000
HARD_PRESET = Preset(
    name="hard",
    constraint_counts=(3, 6, 8),
    mixes=(MIX_HARD,),
    seed_start=HARD_SEED_START,
    n_per_stratum=20,
)


def _conflicts(chosen: Sequence[str], candidate: str) -> bool:
    """True if ``candidate`` conflicts with any already-chosen atom id.

    ``INSTRUCTION_CONFLICTS`` is symmetric and includes each id's own
    self-conflict, so an id already present blocks a duplicate too.
    """
    table = instructions_registry.INSTRUCTION_CONFLICTS
    cand_conf = table.get(candidate, set())
    if any(c in cand_conf for c in chosen):
        return True
    return any(candidate in table.get(c, set()) for c in chosen)


def _sample_atom_ids(
    rng: random.Random,
    n_constraints: int,
    mix: str,
) -> list[str]:
    """Sample ``n_constraints`` distinct, non-conflicting atom ids.

    ``easy`` draws entirely from the easy pool; ``mixed`` guarantees at
    least one hard-pool atom, filling the rest from the combined pool;
    ``hard`` places ALL non-conflicting hard-pool atoms first, then fills
    any remaining count from the easy pool (hard-first). Selection honors
    ``INSTRUCTION_CONFLICTS`` so no stacked pair contradicts; an infeasible
    request raises loudly rather than silently returning a short stack.
    """
    easy_ids = [a.instruction_id for a in EASY_POOL]
    hard_ids = [a.instruction_id for a in HARD_POOL]

    chosen: list[str] = []
    if mix == MIX_HARD:
        # Hard-first: place every hard-pool atom that does not conflict
        # with the ones already chosen, in a shuffled order, then fill any
        # remaining count from the easy pool. This maximizes hard-atom
        # density -- a 'hard' stack contains all hard atoms that co-exist.
        hard_order = list(hard_ids)
        rng.shuffle(hard_order)
        for cand in hard_order:
            if len(chosen) >= n_constraints:
                break
            if cand in chosen or _conflicts(chosen, cand):
                continue
            chosen.append(cand)
        pool = list(easy_ids)
    elif mix == MIX_MIXED:
        # Seed the stack with one hard atom so "mixed" always includes a
        # hard-pool constraint.
        chosen.append(rng.choice(hard_ids))
        pool = easy_ids + hard_ids
    elif mix == MIX_EASY:
        pool = list(easy_ids)
    else:  # pragma: no cover - guarded by caller
        msg = f"unknown atom mix: {mix!r}"
        raise ValueError(msg)

    candidates = list(pool)
    rng.shuffle(candidates)
    for cand in candidates:
        if len(chosen) >= n_constraints:
            break
        if cand in chosen:
            continue
        if _conflicts(chosen, cand):
            continue
        chosen.append(cand)

    if len(chosen) < n_constraints:
        msg = (
            f"could not assemble {n_constraints} non-conflicting "
            f"{mix!r} atoms (got {len(chosen)}): pool too small for the "
            f"requested stack depth"
        )
        raise ValueError(msg)
    return chosen


def _make_instance(
    seed: int,
    n_constraints: int,
    mix: str,
) -> Instance:
    """Construct one pinned :class:`Instance` for a (seed, cell)."""
    rng = random.Random(seed)
    base_task = rng.choice(BASE_TASKS)
    atom_ids = _sample_atom_ids(rng, n_constraints, mix)

    descriptions: list[str] = []
    kwargs_list: list[dict[str, object]] = []
    for atom_id in atom_ids:
        atom: Atom = atom_for(atom_id)
        kwargs = atom.derive_kwargs(rng)
        # Seed the module-global RNG too, so any value the vendored
        # build_description samples internally (for a field we did not
        # pass explicitly) is still reproducible. We derive a fresh
        # sub-seed per atom from the instance RNG to avoid cross-atom
        # coupling.
        random.seed(rng.random())
        cls = instructions_registry.INSTRUCTION_DICT[atom_id]
        instruction = cls(atom_id)
        desc = instruction.build_description(**kwargs)
        # Re-read the checker's own resolved args so serialized kwargs
        # match exactly what the oracle will reconstruct (e.g. sorted
        # keyword lists), independent of what we passed in.
        resolved = instruction.get_instruction_args() or {}
        descriptions.append(desc)
        kwargs_list.append(dict(resolved))

    spec = ConstraintSpec(
        base_task=base_task,
        constraint_descriptions=tuple(descriptions),
        instruction_id_list=tuple(atom_ids),
        kwargs_list=tuple(kwargs_list),
    )
    return make_instance(
        id=f"c22-{seed}",
        seed=seed,
        strata=stratum_label(n_constraints, mix),
        prompt_inputs={"constraints_block": spec.constraints_block()},
        gold=spec.to_gold(),
    )


def _assert_fresh_seeds(seeds: Sequence[int]) -> None:
    """Assert no generated seed collides with a published dataset key.

    Encodes rubric criterion 8: the fresh range must sit strictly above
    the published IFEval key ceiling and never intersect its range.
    """
    for seed in seeds:
        if PUBLISHED_KEY_MIN <= seed <= PUBLISHED_KEY_MAX:
            msg = (
                f"seed {seed} intersects the published IFEval key range "
                f"[{PUBLISHED_KEY_MIN}, {PUBLISHED_KEY_MAX}] -- "
                f"contamination guard (rubric criterion 8)"
            )
            raise AssertionError(msg)
        if seed <= PUBLISHED_KEY_MAX:
            msg = (
                f"seed {seed} is not strictly above the published IFEval "
                f"key ceiling {PUBLISHED_KEY_MAX} -- contamination guard"
            )
            raise AssertionError(msg)


def generate_pool(
    *,
    n_per_stratum: int = 20,
    constraint_counts: Sequence[int] = CONSTRAINT_COUNTS,
    mixes: Sequence[str] = (MIX_EASY, MIX_MIXED),
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the pinned c22 :class:`TaskPool`.

    Parameters
    ----------
    n_per_stratum:
        Instances per (constraint-count x atom-mix) stratum. Default 20
        (spec Section 1's proposed N; owner-adjustable per Section 7.2).
    constraint_counts:
        The constraint-count axis levels. Default ``(3, 4, 5)``.
    mixes:
        The atom-mix axis levels. Default ``("easy", "mixed")``.
    seed_start:
        First fresh seed. Seeds are assigned contiguously from here, one
        per instance, and asserted disjoint from the published dataset.

    The pool is deterministic given ``(GENERATOR_VERSION, seed_start,
    n_per_stratum, constraint_counts, mixes)``: seeds are assigned in a
    fixed cell order, and every sampled value flows from a per-instance
    ``random.Random(seed)``.
    """
    instances: list[Instance] = []
    seeds: list[int] = []
    next_seed = seed_start
    for n_constraints in constraint_counts:
        for mix in mixes:
            for _ in range(n_per_stratum):
                seeds.append(next_seed)
                instances.append(
                    _make_instance(next_seed, n_constraints, mix),
                )
                next_seed += 1

    _assert_fresh_seeds(seeds)
    return TaskPool(instances)


def build_manifest(pool: TaskPool, seed_start: int) -> Manifest:
    """Derive the default-config :class:`Manifest` for ``pool``.

    The seed range recorded is ``[seed_start, seed_start + len(pool))``,
    the contiguous fresh range the pool was generated from.
    """
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(seed_start, seed_start + len(pool)),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate the default pool and write its manifest.

    Every owner-open numeric is a flag with the spec's proposed default,
    so changing N-per-stratum, the axes, or the seed start never requires
    an env-code edit (PLAN "Config surface").
    """
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c22-generate",
        description="Generate the c22 stacked-IFEval pool and manifest.",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=20,
        help="instances per stratum (spec Section 1 proposed: 20)",
    )
    parser.add_argument(
        "--constraint-counts",
        type=int,
        nargs="+",
        default=list(CONSTRAINT_COUNTS),
        help="constraint-count axis levels (default: 3 4 5)",
    )
    parser.add_argument(
        "--mixes",
        nargs="+",
        default=[MIX_EASY, MIX_MIXED],
        help="atom-mix axis levels (default: easy mixed)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="first fresh seed (default: 1_000_000)",
    )
    parser.add_argument(
        "--preset",
        choices=["hard"],
        default=None,
        help=(
            "named generation preset (overrides the axis flags and writes "
            "the preset's own manifest by default). 'hard' = the hardest "
            "IFEval configuration: counts (3, 6, 8) x mix 'hard'."
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
            n_per_stratum=args.n_per_stratum,
            constraint_counts=args.constraint_counts,
            mixes=args.mixes,
            seed_start=args.seed_start,
        )
        manifest = build_manifest(pool, args.seed_start)
        manifest_path = args.manifest or Path(__file__).with_name(
            "manifest.json",
        )
    manifest.write(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
