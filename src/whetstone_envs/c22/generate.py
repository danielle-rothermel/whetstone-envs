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
* supported atoms receive complete kwargs, so neither description
  building nor scoring reaches for the module-global RNG.

Contamination guard (spec Section 6 / rubric criterion 8): the fresh
seed range is asserted at construction to lie strictly above -- and to
never intersect -- the published IFEval dataset's integer key range.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Annotated

import typer

from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.atoms import (
    EASY_POOL,
    HARD_POOL,
    Atom,
    atom_for,
)
from whetstone_envs.c22.spec import ConstraintSpec, compatibility_error
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.core.instance import Instance

GENERATOR_VERSION = "c22-generate-2"
MAX_COMPATIBILITY_ATTEMPTS = 100

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
    rng: Random,
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
    rng = Random(seed)
    last_conflict: str | None = None
    for _attempt in range(1, MAX_COMPATIBILITY_ATTEMPTS + 1):
        base_task = rng.choice(BASE_TASKS)
        atom_ids = _sample_atom_ids(rng, n_constraints, mix)

        descriptions: list[str] = []
        kwargs_list: list[dict[str, object]] = []
        for atom_id in atom_ids:
            atom: Atom = atom_for(atom_id)
            kwargs = atom.derive_kwargs(rng)
            cls = instructions_registry.INSTRUCTION_DICT[atom_id]
            instruction = cls(atom_id)
            desc = instruction.build_description(**kwargs)
            # Persist the checker's normalized arguments, including sorted
            # keyword lists, so the oracle reconstructs the same state.
            resolved = instruction.get_instruction_args() or {}
            descriptions.append(desc)
            kwargs_list.append(dict(resolved))

        last_conflict = compatibility_error(atom_ids, kwargs_list)
        if last_conflict is not None:
            continue

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

    msg = (
        "could not generate a semantically compatible C22 instance "
        f"for seed={seed}, n_constraints={n_constraints}, mix={mix!r} "
        f"after {MAX_COMPATIBILITY_ATTEMPTS} attempts; "
        f"last conflict: {last_conflict}"
    )
    raise RuntimeError(msg)


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validate_generation_config(
    *,
    n_per_stratum: object,
    constraint_counts: Sequence[object],
    mixes: Sequence[object],
    seed_start: object,
) -> tuple[int, tuple[int, ...], tuple[str, ...], int]:
    """Validate and normalize every public generation input up front."""
    valid_n = _require_positive_int(
        n_per_stratum,
        name="n_per_stratum",
    )
    if not constraint_counts:
        msg = "constraint_counts must not be empty"
        raise ValueError(msg)
    valid_counts = tuple(
        _require_positive_int(value, name="constraint_counts item")
        for value in constraint_counts
    )
    if not mixes:
        msg = "mixes must not be empty"
        raise ValueError(msg)
    valid_mixes: list[str] = []
    allowed_mixes = {MIX_EASY, MIX_MIXED, MIX_HARD}
    for mix in mixes:
        if not isinstance(mix, str) or mix not in allowed_mixes:
            msg = (
                f"unknown atom mix {mix!r}; expected one of "
                f"{sorted(allowed_mixes)!r}"
            )
            raise ValueError(msg)
        valid_mixes.append(mix)
    valid_seed_start = _require_positive_int(seed_start, name="seed_start")
    if valid_seed_start <= PUBLISHED_KEY_MAX:
        msg = (
            f"seed_start must be above the published IFEval key ceiling "
            f"{PUBLISHED_KEY_MAX}; got {valid_seed_start}"
        )
        raise ValueError(msg)
    return (
        valid_n,
        valid_counts,
        tuple(valid_mixes),
        valid_seed_start,
    )


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
    (
        valid_n,
        valid_counts,
        valid_mixes,
        valid_seed_start,
    ) = _validate_generation_config(
        n_per_stratum=n_per_stratum,
        constraint_counts=constraint_counts,
        mixes=mixes,
        seed_start=seed_start,
    )

    instances: list[Instance] = []
    next_seed = valid_seed_start
    for n_constraints in valid_counts:
        for mix in valid_mixes:
            for _ in range(valid_n):
                instances.append(
                    _make_instance(next_seed, n_constraints, mix),
                )
                next_seed += 1

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


app = typer.Typer(
    add_completion=False,
    help="Generate the C22 stacked-IFEval pool and manifest.",
)


@app.command()
def main(  # noqa: PLR0913
    n_per_stratum: Annotated[
        int,
        typer.Option(help="Instances per stratum."),
    ] = 20,
    constraint_counts: Annotated[
        list[int] | None,
        typer.Option(
            help="Constraint-count axis level; repeat for multiple levels.",
        ),
    ] = None,
    mixes: Annotated[
        list[str] | None,
        typer.Option(help="Atom mix; repeat for multiple levels."),
    ] = None,
    seed_start: Annotated[
        int,
        typer.Option(help="First fresh seed."),
    ] = DEFAULT_SEED_START,
    preset: Annotated[
        str | None,
        typer.Option(help="Named preset; currently only 'hard'."),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option("--manifest", help="Manifest output path."),
    ] = None,
) -> None:
    """Regenerate a validated C22 pool manifest."""
    selected_counts = (
        CONSTRAINT_COUNTS
        if constraint_counts is None
        else tuple(constraint_counts)
    )
    selected_mixes = (
        (MIX_EASY, MIX_MIXED) if mixes is None else tuple(mixes)
    )
    if preset not in {None, "hard"}:
        msg = f"unknown preset {preset!r}; expected 'hard'"
        raise typer.BadParameter(msg, param_hint="--preset")

    try:
        if preset == "hard":
            pool = HARD_PRESET.generate(n_per_stratum=n_per_stratum)
        else:
            pool = generate_pool(
                n_per_stratum=n_per_stratum,
                constraint_counts=selected_counts,
                mixes=selected_mixes,
                seed_start=seed_start,
            )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    if preset == "hard":
        generated_manifest = HARD_PRESET.build_manifest(pool)
        output_path = manifest_path or Path(__file__).with_name(
            "manifest_hard.json",
        )
    else:
        generated_manifest = build_manifest(pool, seed_start)
        output_path = manifest_path or Path(__file__).with_name(
            "manifest.json",
        )
    generated_manifest.write(output_path)


if __name__ == "__main__":  # pragma: no cover
    app()
