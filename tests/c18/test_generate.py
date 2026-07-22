"""Generator checks for c18: determinism, contamination, strata, manifest.

These are the no-LLM-call blocking checks from the PLAN's Verification
checklist A that concern the pool itself. Oracle correctness lives in its
own hand-traced-fixture file (``test_oracle.py``), never here.

Each pool is produced by reseeding the vendored PrOntoQA generator (one
subprocess per depth), so tests use a small N and a single depth where a
full four-depth pool is not needed, to stay fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c18 import generate, oracle
from whetstone_envs.c18.generate import (
    DEFAULT_DEPTHS,
    DEFAULT_DISTRACTORS,
    DEFAULT_HELD_OUT_PER_STRATUM,
    DEFAULT_INTERNAL_EVAL_PER_STRATUM,
    DEFAULT_N_PER_STRATUM,
    DEFAULT_OFFICIAL_PER_STRATUM,
    DEFAULT_SEED_START,
    FIXED_ONTOLOGY,
    GENERATOR_VERSION,
    PUBLISHED_SEED,
    RESERVED_SEED_MAX,
    build_manifest,
    default_split_sizes,
    depth_label,
    generate_pool,
)
from whetstone_envs.core.manifest import Manifest, content_hash

if TYPE_CHECKING:
    from whetstone_envs.core.instance import Instance
    from whetstone_envs.core.pool import TaskPool

_MANIFEST_PATH = Path(generate.__file__).with_name("manifest.json")


def _fast_pool(
    *,
    n_per_stratum: int = 3,
    depths: tuple[int, ...] = (2,),
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """A small single-depth pool for the cheap generator tests."""
    return generate_pool(
        n_per_stratum=n_per_stratum,
        depths=depths,
        seed_start=seed_start,
    )


def test_regenerating_twice_is_byte_identical() -> None:
    # Determinism (checklist A): the same config regenerates a
    # content-hash-identical pool (the vendored generator is byte-repro
    # under a fixed seed).
    a = _fast_pool()
    b = _fast_pool()
    assert content_hash(a) == content_hash(b)
    assert [i.id for i in a.instances] == [i.id for i in b.instances]
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.prompt_inputs["question"] for i in a.instances] == [
        i.prompt_inputs["question"] for i in b.instances
    ]
    assert [i.prompt_inputs["query"] for i in a.instances] == [
        i.prompt_inputs["query"] for i in b.instances
    ]


def test_committed_manifest_matches_regenerated_default_pool() -> None:
    # The frozen default-config manifest must still describe a freshly
    # generated default pool (the regeneration diff check).
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == GENERATOR_VERSION


def test_strata_coverage_matches_manifest_counts() -> None:
    # Strata coverage (checklist A): every depth stratum carries the
    # declared N; the manifest agrees.
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert pool.stratum_counts() == frozen.stratum_counts
    assert pool.stratum_counts() == {
        depth_label(d): DEFAULT_N_PER_STRATUM for d in DEFAULT_DEPTHS
    }


def test_default_pool_is_the_spec_proposed_depth_shape() -> None:
    # Spec Section 1: the depth axis is D1, D2, D3, D5.
    assert DEFAULT_DEPTHS == (1, 2, 3, 5)
    pool = generate_pool()
    assert set(pool.strata) == {"D1", "D2", "D3", "D5"}
    assert len(pool) == DEFAULT_N_PER_STRATUM * len(DEFAULT_DEPTHS)


def test_default_split_is_disjoint_and_depth_balanced() -> None:
    # The interleaved layout makes each contiguous split slice carry the
    # same per-depth count; PoolSplit asserts disjointness at construction.
    assert DEFAULT_N_PER_STRATUM == (
        DEFAULT_INTERNAL_EVAL_PER_STRATUM
        + DEFAULT_OFFICIAL_PER_STRATUM
        + DEFAULT_HELD_OUT_PER_STRATUM
    )
    assert DEFAULT_INTERNAL_EVAL_PER_STRATUM >= 2
    pool = generate_pool()
    ie, off, ho = default_split_sizes(pool)
    n_strata = len(pool.strata)
    assert (ie, off, ho) == (
        DEFAULT_INTERNAL_EVAL_PER_STRATUM * n_strata,
        DEFAULT_OFFICIAL_PER_STRATUM * n_strata,
        DEFAULT_HELD_OUT_PER_STRATUM * n_strata,
    )
    split = pool.split(ie, off, ho)

    def counts(subset: tuple[Instance, ...]) -> set[int]:
        out: dict[str, int] = {}
        for inst in subset:
            out[inst.strata[0]] = out.get(inst.strata[0], 0) + 1
        return set(out.values())

    assert counts(split.internal_eval) == {DEFAULT_INTERNAL_EVAL_PER_STRATUM}
    assert counts(split.official) == {DEFAULT_OFFICIAL_PER_STRATUM}
    assert counts(split.held_out) == {DEFAULT_HELD_OUT_PER_STRATUM}


def test_n_per_stratum_is_configurable() -> None:
    # Owner-open numeric (spec O3): N is a constructor arg, not hardcoded.
    pool = _fast_pool(n_per_stratum=2)
    assert set(pool.stratum_counts().values()) == {2}


def test_depths_are_configurable() -> None:
    # Owner-open (spec O2): the depth set is a constructor arg.
    pool = generate_pool(n_per_stratum=2, depths=(1, 3))
    assert set(pool.strata) == {"D1", "D3"}


def test_instances_carry_only_public_fields() -> None:
    # prompt_inputs exposes only the question and query; the entailment
    # label is the separate oracle-checkable gold field.
    pool = _fast_pool()
    for inst in pool.instances:
        assert set(inst.prompt_inputs) == {"question", "query"}
        assert inst.gold in {"True", "False"}


def test_gold_agrees_with_independent_forward_chaining_oracle() -> None:
    # Every instance's frozen gold equals the independent forward-chaining
    # oracle's re-derivation from its public question + query -- the check
    # that catches a definitional label a generation bug would hide. This
    # re-verifies (from the pool's public fields alone) the cross-check the
    # generator asserts at construction.
    pool = generate_pool(n_per_stratum=4)
    for inst in pool.instances:
        derived = oracle.entailment_label(
            inst.prompt_inputs["question"],
            inst.prompt_inputs["query"],
        )
        assert derived == inst.gold, inst.id
        assert (
            oracle.score(
                inst.gold,
                inst.prompt_inputs["question"],
                inst.prompt_inputs["query"],
            )
            == 1
        )


def test_pool_has_both_true_and_false_labels() -> None:
    # A degenerate all-one-label pool would make the task trivial; the
    # native 50% negation flag should yield a mix at the default N.
    pool = generate_pool(n_per_stratum=8, depths=(2,))
    golds = {inst.gold for inst in pool.instances}
    assert golds == {"True", "False"}


def test_contamination_guard_ontology_is_fixed_fictional() -> None:
    # Fixed nonce ontology assertion (checklist A / rubric 8): a non-nonce
    # ontology fails at construction rather than silently proceeding.
    assert FIXED_ONTOLOGY == "fictional"
    with pytest.raises(AssertionError, match="fixed nonce ontology"):
        generate_pool(n_per_stratum=2, depths=(1,), ontology="true")


def test_contamination_guard_seeds_are_fresh() -> None:
    # Fresh-seed assertion (checklist A / rubric 8): every consumed seed is
    # strictly above the reserved range and never the published default.
    pool = _fast_pool()
    for inst in pool.instances:
        assert inst.seed > RESERVED_SEED_MAX
        assert inst.seed != PUBLISHED_SEED


def test_contamination_guard_rejects_a_reserved_seed_range() -> None:
    # Pointing the generator into the reserved range fires the assertion at
    # construction, before any subprocess runs.
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=1)


def test_contamination_guard_rejects_the_published_default_seed() -> None:
    # Starting exactly on the upstream default seed (behind every published
    # PrOntoQA instance) fires the guard.
    assert PUBLISHED_SEED <= RESERVED_SEED_MAX
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=PUBLISHED_SEED)


def test_default_seed_start_is_above_reserved_ceiling() -> None:
    assert DEFAULT_SEED_START > RESERVED_SEED_MAX


def test_distractors_default_is_on() -> None:
    # Spec Open Decision O1 default: distractors ON (relevant).
    assert DEFAULT_DISTRACTORS == "relevant"


def test_manifest_seed_range_spans_one_seed_per_depth() -> None:
    pool = generate_pool(n_per_stratum=2, depths=(1, 2))
    manifest = build_manifest(pool, DEFAULT_SEED_START, 2)
    start, end = manifest.seed_range
    assert start == DEFAULT_SEED_START
    assert end == DEFAULT_SEED_START + 2
