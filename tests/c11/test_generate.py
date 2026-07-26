"""Generator checks for c11: determinism, contamination, strata, manifest.

These are the no-LLM-call blocking checks from the PLAN's Verification
checklist A that concern the pool itself. Oracle correctness lives in its
own hand-built-fixture file (``test_oracle.py``), never here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c11 import generate, oracle
from whetstone_envs.c11.generate import (
    DEFAULT_HELD_OUT_PER_STRATUM,
    DEFAULT_INTERNAL_EVAL_PER_STRATUM,
    DEFAULT_N_PER_STRATUM,
    DEFAULT_OFFICIAL_PER_STRATUM,
    DEFAULT_SEED_START,
    GENERATOR_VERSION,
    PUBLISHED_VECTORS,
    RESERVED_SEED_MAX,
    build_manifest,
    default_split_sizes,
    generate_pool,
)
from whetstone_envs.c11.strata import STRATA
from whetstone_envs.core.manifest import Manifest, content_hash

if TYPE_CHECKING:
    from whetstone_envs.core.instance import Instance

_MANIFEST_PATH = Path(generate.__file__).with_name("manifest.json")


def test_regenerating_twice_is_byte_identical() -> None:
    # Determinism (checklist A): the same config regenerates a
    # content-hash-identical pool.
    a = generate_pool(n_per_stratum=6)
    b = generate_pool(n_per_stratum=6)
    assert content_hash(a) == content_hash(b)
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.id for i in a.instances] == [i.id for i in b.instances]
    assert [i.prompt_inputs["input"] for i in a.instances] == [
        i.prompt_inputs["input"] for i in b.instances
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
        STRATA,
        DEFAULT_N_PER_STRATUM,
    )
    assert len(pool) == DEFAULT_N_PER_STRATUM * len(STRATA)


def test_default_n_is_the_spec_proposed_split() -> None:
    # 2 internal-eval + 40 official + 40 held-out per stratum (spec Sec 1).
    assert DEFAULT_N_PER_STRATUM == (
        DEFAULT_INTERNAL_EVAL_PER_STRATUM
        + DEFAULT_OFFICIAL_PER_STRATUM
        + DEFAULT_HELD_OUT_PER_STRATUM
    )
    assert DEFAULT_OFFICIAL_PER_STRATUM == 40
    assert DEFAULT_HELD_OUT_PER_STRATUM == 40


def test_default_split_is_disjoint_and_stratum_balanced() -> None:
    # TaskPool.split groups full strata combinations before assigning the
    # three disjoint role subsets (spec Section 1: >=2/stratum
    # internal-eval, 40/stratum official, 40/stratum held-out).
    pool = generate_pool()
    ie, off, ho = default_split_sizes(pool)
    assert (ie, off, ho) == (10, 200, 200)
    split = pool.split(ie, off, ho)

    def counts(subset: tuple[Instance, ...]) -> dict[str, int]:
        out: dict[str, int] = {}
        for inst in subset:
            out[inst.strata[0]] = out.get(inst.strata[0], 0) + 1
        return out

    assert counts(split.internal_eval) == dict.fromkeys(STRATA, 2)
    assert counts(split.official) == dict.fromkeys(STRATA, 40)
    assert counts(split.held_out) == dict.fromkeys(STRATA, 40)


def test_n_per_stratum_is_configurable() -> None:
    # Owner-open numeric: N is a constructor arg, not hardcoded.
    pool = generate_pool(n_per_stratum=3)
    assert len(pool) == 3 * len(STRATA)
    assert set(pool.stratum_counts().values()) == {3}


def test_strata_subset_is_configurable() -> None:
    # Spec Section 3 outcome (b): the owner may bias toward S1/S2 and drop
    # S3. A strata subset is a constructor arg.
    pool = generate_pool(n_per_stratum=4, strata=("S1_flat", "S2_keysort"))
    assert set(pool.strata) == {"S1_flat", "S2_keysort"}
    assert len(pool) == 8


def test_every_instance_is_adversarial_and_oracle_consistent() -> None:
    # Each instance's messy input differs from its canonical gold (the
    # adversarial predicate) and the oracle scores that gold as 1 -- so
    # gold is genuinely the canonical form of the public input.
    pool = generate_pool(n_per_stratum=6)
    for inst in pool.instances:
        messy = inst.prompt_inputs["input"]
        assert messy != inst.gold, f"{inst.id} is already canonical"
        assert oracle.score(inst.gold, messy) == 1
        assert oracle.canonicalize(messy) == inst.gold


def test_instances_carry_only_the_public_input_field() -> None:
    # prompt_inputs exposes only the messy JSON; gold is the separate
    # oracle-checkable field. Nothing oracle-only leaks into prompt_inputs.
    pool = generate_pool(n_per_stratum=2)
    for inst in pool.instances:
        assert set(inst.prompt_inputs) == {"input"}


def test_contamination_guard_seeds_are_above_reserved_range() -> None:
    # Contamination guard (checklist A / rubric 8): every seed sits
    # strictly above the reserved published-range ceiling.
    pool = generate_pool(n_per_stratum=6)
    for inst in pool.instances:
        assert inst.seed > RESERVED_SEED_MAX


def test_contamination_guard_rejects_a_reserved_seed_range() -> None:
    # Pointing the generator into the reserved range must fire the
    # assertion at construction rather than silently proceed.
    with pytest.raises(AssertionError, match="contamination guard"):
        generate_pool(n_per_stratum=1, seed_start=1)


def test_no_generated_gold_reproduces_a_published_vector() -> None:
    # Contamination guard (rubric 8, "never published instances"): no
    # generated canonical gold may equal an RFC 8785 published test vector.
    pool = generate_pool(n_per_stratum=8)
    published = set(PUBLISHED_VECTORS)
    for inst in pool.instances:
        assert inst.gold not in published


def test_published_vectors_are_real_rfc8785_outputs() -> None:
    # The guard is only meaningful if PUBLISHED_VECTORS actually are what
    # rfc8785 emits for the RFC's own examples -- regenerate them here so a
    # stale literal can't silently make the guard vacuous.
    french_in = {
        "peach": "This sorting order",
        "péché": "is wrong according to French",
        "pêche": "but canonicalization MUST",
        "sin": "ignore locale",
    }
    numbers_in = [
        333333333.33333329,
        1e30,
        4.5,
        2e-3,
        0.000000000000000000000000001,
    ]
    import json as _json

    french_out = oracle.canonicalize(_json.dumps(french_in))
    numbers_out = oracle.canonicalize(_json.dumps(numbers_in))
    assert french_out in PUBLISHED_VECTORS
    assert numbers_out in PUBLISHED_VECTORS


def test_default_seed_start_is_above_reserved_ceiling() -> None:
    assert DEFAULT_SEED_START > RESERVED_SEED_MAX


def test_manifest_seed_range_spans_the_pool() -> None:
    pool = generate_pool(n_per_stratum=4)
    manifest = build_manifest(pool, DEFAULT_SEED_START)
    start, end = manifest.seed_range
    assert start == DEFAULT_SEED_START
    assert end == DEFAULT_SEED_START + len(pool)
