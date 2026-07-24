"""Generator checks for c19: determinism, contamination, strata, manifest.

These are the no-LLM-call blocking checks from the PLAN's Verification
checklist A that concern the pool itself. Oracle correctness lives in its
own hand-built-fixture file (``test_oracle.py``), never here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c19 import generate, oracle
from whetstone_envs.c19.envs import ENV_IDS, SIZE_LEVELS, strata_labels
from whetstone_envs.c19.generate import (
    DEFAULT_HELD_OUT_PER_STRATUM,
    DEFAULT_INTERNAL_EVAL_PER_STRATUM,
    DEFAULT_N_PER_STRATUM,
    DEFAULT_OFFICIAL_PER_STRATUM,
    DEFAULT_SEED_START,
    GENERATOR_VERSION,
    PUBLISHED_SEEDS,
    RESERVED_SEED_MAX,
    build_manifest,
    default_split_sizes,
    generate_pool,
)
from whetstone_envs.core.manifest import Manifest, content_hash

if TYPE_CHECKING:
    from whetstone_envs.core.instance import Instance
    from whetstone_envs.core.pool import TaskPool

_MANIFEST_PATH = Path(generate.__file__).with_name("manifest.json")

def _fast_pool(
    *,
    n_per_stratum: int = 2,
    command_length: int = generate.DEFAULT_COMMAND_LENGTH,
    seed_start: int = generate.DEFAULT_SEED_START,
) -> TaskPool:
    """Generate a small single-env pool for the cheap generator tests.

    A single-env (Fetch) small slice keeps the fast tests cheap while
    still exercising every code path (Fetch has all four fact types,
    including carrying).
    """
    return generate_pool(
        n_per_stratum=n_per_stratum,
        env_ids=("Fetch",),
        size_levels=("small",),
        command_length=command_length,
        seed_start=seed_start,
    )


def test_regenerating_twice_is_byte_identical() -> None:
    # Determinism (checklist A): the same config regenerates a
    # content-hash-identical pool.
    a = _fast_pool()
    b = _fast_pool()
    assert content_hash(a) == content_hash(b)
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.id for i in a.instances] == [i.id for i in b.instances]
    assert [i.prompt_inputs["grid"] for i in a.instances] == [
        i.prompt_inputs["grid"] for i in b.instances
    ]
    assert [i.prompt_inputs["command"] for i in a.instances] == [
        i.prompt_inputs["command"] for i in b.instances
    ]


def test_committed_manifest_matches_regenerated_default_pool() -> None:
    # The frozen default-config manifest must still describe a freshly
    # generated default pool (the regeneration diff check).
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == GENERATOR_VERSION


def test_strata_coverage_matches_manifest_counts() -> None:
    # Strata coverage (checklist A): every stratum carries the declared N.
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert pool.stratum_counts() == frozen.stratum_counts
    assert pool.stratum_counts() == dict.fromkeys(
        strata_labels(),
        DEFAULT_N_PER_STRATUM,
    )


def test_default_pool_is_the_spec_proposed_shape() -> None:
    # Spec Section 1: 26 strata x N=15 = 390 instances.
    pool = generate_pool()
    assert len(strata_labels()) == 26
    assert DEFAULT_N_PER_STRATUM == 15
    assert len(pool) == 390


def test_default_n_is_the_spec_proposed_split() -> None:
    # 3 internal-eval + 6 official + 6 held-out per stratum = 15.
    assert DEFAULT_N_PER_STRATUM == (
        DEFAULT_INTERNAL_EVAL_PER_STRATUM
        + DEFAULT_OFFICIAL_PER_STRATUM
        + DEFAULT_HELD_OUT_PER_STRATUM
    )
    assert DEFAULT_INTERNAL_EVAL_PER_STRATUM >= 2


def test_default_split_is_disjoint_and_stratum_balanced() -> None:
    # TaskPool.split groups full strata combinations before assigning the
    # three disjoint role subsets.
    pool = generate_pool()
    ie, off, ho = default_split_sizes(pool)
    n_strata = len(pool.strata)
    assert (ie, off, ho) == (3 * n_strata, 6 * n_strata, 6 * n_strata)
    split = pool.split(ie, off, ho)

    def counts(subset: tuple[Instance, ...]) -> dict[str, int]:
        out: dict[str, int] = {}
        for inst in subset:
            out[inst.strata[0]] = out.get(inst.strata[0], 0) + 1
        return out

    assert set(counts(split.internal_eval).values()) == {3}
    assert set(counts(split.official).values()) == {6}
    assert set(counts(split.held_out).values()) == {6}


def test_n_per_stratum_is_configurable() -> None:
    # Owner-open numeric: N is a constructor arg, not hardcoded.
    pool = generate_pool(n_per_stratum=1)
    assert set(pool.stratum_counts().values()) == {1}
    assert len(pool) == len(strata_labels())


def test_env_and_size_subsets_are_configurable() -> None:
    # Owner-open: env ids and size levels are constructor args.
    pool = generate_pool(
        n_per_stratum=2,
        env_ids=("Empty-Random",),
        size_levels=("small",),
    )
    # Empty-Random has coordinate, heading, front (no carrying) -> 3 strata.
    assert len(pool.strata) == 3
    assert all(s.startswith("Empty-Random|small|") for s in pool.strata)


def test_command_length_is_configurable() -> None:
    pool = _fast_pool(command_length=3)
    for inst in pool.instances:
        assert len(inst.prompt_inputs["command"]) == 3


def test_carrying_is_fetch_only() -> None:
    # Spec Section 1 applicability matrix: carrying-flag is Fetch-only.
    pool = generate_pool()
    carrying_strata = [s for s in pool.strata if s.endswith("|carrying")]
    assert carrying_strata
    assert all(s.startswith("Fetch|") for s in carrying_strata)
    for env_id in ENV_IDS:
        if env_id != "Fetch":
            assert not any(
                s.startswith(f"{env_id}|") and s.endswith("|carrying")
                for s in pool.strata
            )


def test_instances_carry_only_public_fields() -> None:
    # prompt_inputs exposes only the grid, command, and fact type; the
    # derived-fact gold is the separate oracle-checkable field.
    pool = _fast_pool()
    for inst in pool.instances:
        assert set(inst.prompt_inputs) == {"grid", "command", "fact_type"}


def test_gold_agrees_with_independent_oracle_walk() -> None:
    # Every instance's frozen gold equals the independent ASCII-only oracle
    # walk of its public grid + command (the cross-check the generator
    # asserts, re-verified here from the pool's public fields alone).
    pool = generate_pool(n_per_stratum=1)
    for inst in pool.instances:
        derived = oracle.derive_fact(
            inst.prompt_inputs["grid"],
            inst.prompt_inputs["command"],
            inst.prompt_inputs["fact_type"],
        )
        assert derived == inst.gold, inst.id
        assert (
            oracle.score(
                inst.gold,
                inst.prompt_inputs["grid"],
                inst.prompt_inputs["command"],
                inst.prompt_inputs["fact_type"],
            )
            == 1
        )


def test_contamination_guard_seeds_are_fresh() -> None:
    # Contamination guard (checklist A / rubric 8): every seed sits strictly
    # above the reserved published range and is not a published example seed.
    pool = generate_pool(n_per_stratum=2)
    for inst in pool.instances:
        assert inst.seed > RESERVED_SEED_MAX
        assert inst.seed not in PUBLISHED_SEEDS


def test_contamination_guard_rejects_a_reserved_seed_range() -> None:
    # Pointing the generator into the reserved range must fire the assertion
    # at construction rather than silently proceed.
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=1)


def test_contamination_guard_rejects_a_published_seed() -> None:
    # Starting exactly on a published example seed that is also below the
    # reserved ceiling fires the guard. A published seed inside the fresh
    # range would fire the published-seed branch; both are asserted.
    assert 42 in PUBLISHED_SEEDS
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=42)


def test_default_seed_start_is_above_reserved_ceiling() -> None:
    assert DEFAULT_SEED_START > RESERVED_SEED_MAX


def test_manifest_seed_range_spans_the_pool() -> None:
    pool = generate_pool(n_per_stratum=1)
    manifest = build_manifest(pool, DEFAULT_SEED_START)
    start, end = manifest.seed_range
    assert start == DEFAULT_SEED_START
    assert end == DEFAULT_SEED_START + len(pool)


def test_all_default_strata_labels_are_present() -> None:
    pool = generate_pool()
    assert set(pool.strata) == set(strata_labels())
    # Sanity: labels reference only the default envs and sizes.
    for label in pool.strata:
        env_id, size, _fact = label.split("|")
        assert env_id in ENV_IDS
        assert size in SIZE_LEVELS
