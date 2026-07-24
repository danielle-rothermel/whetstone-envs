"""Generator checks for c22: determinism, contamination, strata, manifest.

These are the no-LLM-call blocking checks from the PLAN's Verification
checklist A that concern the pool itself. Oracle correctness lives in its
own hand-built-fixture file (``test_oracle.py``), never here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whetstone_envs.c22 import generate
from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.generate import (
    DEFAULT_SEED_START,
    GENERATOR_VERSION,
    PUBLISHED_KEY_MAX,
    PUBLISHED_KEY_MIN,
    build_manifest,
    generate_pool,
)
from whetstone_envs.c22.spec import ConstraintSpec, compatibility_error
from whetstone_envs.core.manifest import Manifest, content_hash

_MANIFEST_PATH = Path(generate.__file__).with_name("manifest.json")


def test_regenerating_twice_is_byte_identical() -> None:
    # Determinism (checklist A): the same config regenerates a
    # content-hash-identical pool.
    a = generate_pool(n_per_stratum=5)
    b = generate_pool(n_per_stratum=5)
    assert content_hash(a) == content_hash(b)
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.id for i in a.instances] == [i.id for i in b.instances]


def test_committed_manifest_matches_regenerated_default_pool() -> None:
    # The frozen default-config manifest must still describe a freshly
    # generated default pool (the regeneration diff check).
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == GENERATOR_VERSION


def test_strata_coverage_matches_manifest_counts() -> None:
    # Strata coverage (checklist A): every stratum carries the declared
    # N, not just the right total.
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert pool.stratum_counts() == frozen.stratum_counts
    assert pool.stratum_counts() == {
        "n3_easy": 20,
        "n3_mixed": 20,
        "n4_easy": 20,
        "n4_mixed": 20,
        "n5_easy": 20,
        "n5_mixed": 20,
    }
    assert len(pool) == 120


def test_n_per_stratum_is_configurable() -> None:
    # Owner-open numeric: N is a constructor arg, not hardcoded.
    pool = generate_pool(n_per_stratum=3)
    assert len(pool) == 18
    assert set(pool.stratum_counts().values()) == {3}


def test_constraint_counts_match_stratum_label() -> None:
    # Each instance's stack has exactly the atom count its stratum names.
    pool = generate_pool(n_per_stratum=2)
    for inst in pool.instances:
        (label,) = inst.strata
        n = int(label[1])  # "n3_easy" -> 3
        spec = ConstraintSpec.from_gold(inst.gold)
        assert len(spec.instruction_id_list) == n
        assert len(spec.constraint_descriptions) == n
        assert len(spec.kwargs_list) == n


def test_mixed_strata_include_a_hard_atom() -> None:
    hard_ids = {a.instruction_id for a in generate.HARD_POOL}
    easy_ids = {a.instruction_id for a in generate.EASY_POOL}
    pool = generate_pool()
    for inst in pool.instances:
        (label,) = inst.strata
        spec = ConstraintSpec.from_gold(inst.gold)
        ids = set(spec.instruction_id_list)
        if label.endswith("_mixed"):
            assert ids & hard_ids, f"{inst.id} mixed but no hard atom"
        else:  # easy-skewed: every atom is from the easy pool
            assert ids <= easy_ids, f"{inst.id} easy but has non-easy atom"


def test_stacked_atoms_are_distinct_and_non_conflicting() -> None:
    conflicts = instructions_registry.INSTRUCTION_CONFLICTS
    pool = generate_pool()
    for inst in pool.instances:
        spec = ConstraintSpec.from_gold(inst.gold)
        ids = list(spec.instruction_id_list)
        assert len(ids) == len(set(ids)), f"{inst.id} has duplicate atoms"
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                assert b not in conflicts.get(a, set()), (
                    f"{inst.id}: {a} conflicts with {b}"
                )
        assert compatibility_error(ids, spec.kwargs_list) is None


def test_contamination_guard_seeds_are_above_published_range() -> None:
    # Contamination guard (checklist A / rubric 8): every seed is above
    # the published IFEval key ceiling and never inside its range.
    pool = generate_pool(n_per_stratum=5)
    for inst in pool.instances:
        assert inst.seed > PUBLISHED_KEY_MAX
        assert not (PUBLISHED_KEY_MIN <= inst.seed <= PUBLISHED_KEY_MAX)


def test_contamination_guard_rejects_a_published_seed_range() -> None:
    # If someone points the generator at the published range, the
    # assertion must fire at construction rather than silently proceed.
    with pytest.raises(ValueError, match="published IFEval key ceiling"):
        generate_pool(n_per_stratum=1, seed_start=PUBLISHED_KEY_MIN)


def test_default_seed_start_is_the_documented_ceiling() -> None:
    assert DEFAULT_SEED_START > PUBLISHED_KEY_MAX


def test_manifest_seed_range_spans_the_pool() -> None:
    pool = generate_pool(n_per_stratum=4)
    manifest = build_manifest(pool, DEFAULT_SEED_START)
    start, end = manifest.seed_range
    assert start == DEFAULT_SEED_START
    assert end == DEFAULT_SEED_START + len(pool)


def test_prompt_inputs_carry_only_the_public_block() -> None:
    # The instance's prompt_inputs must expose the constraints block and
    # nothing oracle-only (ids/kwargs live in gold, not prompt_inputs).
    pool = generate_pool(n_per_stratum=1)
    for inst in pool.instances:
        assert set(inst.prompt_inputs) == {"constraints_block"}
        # gold is valid JSON carrying the oracle-checkable stack.
        parsed = json.loads(inst.gold)
        assert "instruction_id_list" in parsed
        assert "kwargs_list" in parsed
        # ...and none of that leaks into the shown block.
        block = inst.prompt_inputs["constraints_block"]
        for atom_id in parsed["instruction_id_list"]:
            assert atom_id not in block
