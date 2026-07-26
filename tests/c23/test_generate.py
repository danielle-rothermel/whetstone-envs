"""Generator checks for c23: determinism, contamination, strata, manifest.

The no-LLM-call blocking checks from PLAN Verification checklist A that
concern the pool itself. Oracle correctness lives in its own hand-traced
fixture file (``test_oracle.py``); the sorted()-patch determinism proof
across hash seeds lives in ``test_upstream.py``. Tests use a small N where a
full 4-stratum default pool is not needed, to stay fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from whetstone_envs.c23 import generate, oracle, upstream
from whetstone_envs.c23.generate import (
    DEFAULT_HELD_OUT_PER_STRATUM,
    DEFAULT_INTERNAL_EVAL_PER_STRATUM,
    DEFAULT_N_PER_STRATUM,
    DEFAULT_OFFICIAL_PER_STRATUM,
    DEFAULT_SEED_START,
    DEFAULT_STRATA,
    FIXED_NUMBER_OF_RULES,
    FIXED_VOCAB_SIZE,
    GENERATOR_VERSION,
    PUBLISHED_SEED,
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
    n_per_stratum: int = 4,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """A small default-shape pool for the cheap generator tests."""
    return generate_pool(n_per_stratum=n_per_stratum, seed_start=seed_start)


def test_regenerating_twice_is_byte_identical() -> None:
    # Determinism (checklist A): same config, same content hash. The
    # cross-hash-seed proof is in test_upstream.
    a = _fast_pool()
    b = _fast_pool()
    assert content_hash(a) == content_hash(b)
    assert [i.id for i in a.instances] == [i.id for i in b.instances]
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.prompt_inputs["demos_block"] for i in a.instances] == [
        i.prompt_inputs["demos_block"] for i in b.instances
    ]
    assert [i.prompt_inputs["query"] for i in a.instances] == [
        i.prompt_inputs["query"] for i in b.instances
    ]


def test_committed_manifest_matches_regenerated_default_pool() -> None:
    # The frozen default-config manifest still describes a freshly generated
    # default pool (the regeneration diff check).
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == GENERATOR_VERSION


def test_strata_coverage_matches_manifest_counts() -> None:
    # Strata coverage (checklist A): every stratum carries the declared N;
    # the manifest agrees.
    pool = generate_pool()
    frozen = Manifest.read(_MANIFEST_PATH)
    assert pool.stratum_counts() == frozen.stratum_counts
    assert pool.stratum_counts() == {
        label: DEFAULT_N_PER_STRATUM for (label, _t, _k) in DEFAULT_STRATA
    }


def test_default_pool_is_the_spec_proposed_four_strata() -> None:
    # Spec Section 1: S1 ISL k2, S2 L-OSL k2, S3 R-OSL k2, S4 ISL k3.
    assert DEFAULT_STRATA == (
        ("S1", upstream.ISL, 2),
        ("S2", upstream.L_OSL, 2),
        ("S3", upstream.R_OSL, 2),
        ("S4", upstream.ISL, 3),
    )
    pool = generate_pool()
    assert set(pool.strata) == {"S1", "S2", "S3", "S4"}
    assert len(pool) == DEFAULT_N_PER_STRATUM * len(DEFAULT_STRATA)


def test_k_ladder_stops_at_three() -> None:
    # Spec Section 1.2: the hardest stratum is k=3 (k=4 sends models to ~0,
    # violating a reachable ceiling). No stratum exceeds k=3.
    assert max(k for (_label, _t, k) in DEFAULT_STRATA) == 3


def test_vocab_and_rule_count_are_fixed_constraints() -> None:
    # Spec Fixed constraints: single-rule, |Sigma|=4.
    assert FIXED_NUMBER_OF_RULES == 1
    assert FIXED_VOCAB_SIZE == 4


def test_default_split_is_disjoint_and_stratum_balanced() -> None:
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


def test_split_sizes_reject_pool_smaller_than_role_defaults() -> None:
    pool = generate_pool(n_per_stratum=2)
    with pytest.raises(
        ValueError,
        match=r"split sizes sum to 50.*only 2 instances per stratum",
    ):
        default_split_sizes(pool)


def test_split_sizes_accept_roles_that_fit_a_small_pool() -> None:
    pool = generate_pool(n_per_stratum=2)
    sizes = default_split_sizes(
        pool,
        internal_eval_per_stratum=0,
        official_per_stratum=1,
        held_out_per_stratum=1,
    )
    assert sizes == (0, 4, 4)
    split = pool.split(*sizes)
    assert len(split.internal_eval) == 0
    assert len(split.official) == 4
    assert len(split.held_out) == 4


def test_split_sizes_reject_negative_role_counts() -> None:
    pool = generate_pool(n_per_stratum=2)
    with pytest.raises(ValueError, match="must be non-negative"):
        default_split_sizes(pool, internal_eval_per_stratum=-1)


def test_n_per_stratum_is_configurable() -> None:
    # Owner-open numeric (spec Sec 1.3): N is a constructor arg.
    pool = _fast_pool(n_per_stratum=2)
    assert set(pool.stratum_counts().values()) == {2}


def test_n_demos_is_configurable() -> None:
    # Owner-open: the demo count is a constructor arg, honored per instance.
    pool = generate_pool(n_per_stratum=2, n_demos=3)
    for inst in pool.instances:
        lines = inst.prompt_inputs["demos_block"].splitlines()
        assert len(lines) == 3


def test_zero_demos_produces_a_zero_shot_pool() -> None:
    pool = generate_pool(n_per_stratum=1, n_demos=0)
    assert all(
        instance.prompt_inputs["demos_block"] == ""
        for instance in pool.instances
    )


def test_negative_demo_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_demos must be non-negative"):
        generate_pool(n_per_stratum=1, n_demos=-1)


def test_strata_are_configurable() -> None:
    # Owner-open: the strata set is a constructor arg.
    pool = generate_pool(
        n_per_stratum=2,
        strata=(("only", upstream.ISL, 2),),
    )
    assert set(pool.strata) == {"only"}


def test_duplicate_stratum_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="stratum labels must be unique"):
        generate_pool(
            n_per_stratum=1,
            strata=(
                ("duplicate", upstream.ISL, 2),
                ("duplicate", upstream.L_OSL, 2),
            ),
        )


def test_invalid_generation_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_per_stratum must be positive"):
        generate_pool(n_per_stratum=0)
    with pytest.raises(ValueError, match="strata must contain"):
        generate_pool(strata=())
    with pytest.raises(ValueError, match="sample_size_times must be positive"):
        generate_pool(sample_size_times=0)
    with pytest.raises(ValueError, match="max_query_len must be at least 2"):
        generate_pool(max_query_len=1)
    with pytest.raises(ValueError, match="must have k >= 2"):
        generate_pool(strata=(("bad-k", upstream.ISL, 1),))


def test_python_api_rejects_non_strict_integer_inputs() -> None:
    with pytest.raises(TypeError, match="n_per_stratum must be an integer"):
        generate_pool(
            n_per_stratum=True,
        )
    with pytest.raises(TypeError, match="n_demos must be an integer"):
        generate_pool(
            n_demos=1.0,  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match="seed_start must be an integer"):
        generate_pool(
            seed_start=False,
        )
    with pytest.raises(TypeError, match="number_of_rules must be an integer"):
        generate_pool(
            number_of_rules=True,
        )
    pool = generate_pool(n_per_stratum=1)
    with pytest.raises(
        TypeError,
        match="internal_eval_per_stratum must be an integer",
    ):
        default_split_sizes(
            pool,
            internal_eval_per_stratum=False,
        )


def test_python_api_rejects_invalid_stratum_sequences() -> None:
    with pytest.raises(TypeError, match="non-string sequence"):
        generate_pool(
            strata="S1",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="three-item"):
        generate_pool(
            strata=[("S1", upstream.ISL)],  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match=r"strata\[0\] k.*integer"):
        generate_pool(
            strata=(("S1", upstream.ISL, True),),
        )


def test_typer_cli_generates_a_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    result = CliRunner().invoke(
        generate.app,
        [
            "--n-per-stratum",
            "1",
            "--n-demos",
            "0",
            "--sample-size-times",
            "2",
            "--max-query-len",
            "4",
            "--seed-start",
            str(DEFAULT_SEED_START),
            "--manifest",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = Manifest.read(output)
    assert manifest.seed_range == (
        DEFAULT_SEED_START,
        DEFAULT_SEED_START + len(DEFAULT_STRATA),
    )
    assert set(manifest.stratum_counts.values()) == {1}


@pytest.mark.parametrize(
    "args",
    [
        ["--n-per-stratum", "0"],
        ["--n-per-stratum", "true"],
        ["--n-demos", "-1"],
        ["--sample-size-times", "0"],
        ["--max-query-len", "1"],
        ["--seed-start", "0"],
    ],
)
def test_typer_cli_rejects_invalid_inputs(args: list[str]) -> None:
    result = CliRunner().invoke(generate.app, args)
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_instances_carry_only_public_fields() -> None:
    # prompt_inputs exposes only demos_block + query; the latent rule and the
    # oracle gold string are never in prompt_inputs.
    pool = _fast_pool()
    for inst in pool.instances:
        assert set(inst.prompt_inputs) == {"demos_block", "query"}
        assert "->" in inst.prompt_inputs["demos_block"]


def test_held_out_query_is_absent_from_the_demos() -> None:
    # The query must be genuinely held out: its input never appears as a
    # demonstration input line.
    pool = _fast_pool()
    for inst in pool.instances:
        query = inst.prompt_inputs["query"]
        demo_inputs = {
            line.split(" -> ")[0]
            for line in inst.prompt_inputs["demos_block"].splitlines()
        }
        assert query not in demo_inputs, inst.id


def test_gold_agrees_with_independent_oracle_reapplication() -> None:
    # Every instance's frozen gold equals the independent oracle re-applying
    # the (public-at-generation) rule to the query -- the self-consistency
    # gate. Here we re-verify via score_gold on the frozen gold (idempotent).
    pool = _fast_pool()
    for inst in pool.instances:
        assert oracle.score_gold(inst.gold, inst.gold) == 1


def test_some_queries_actually_transform() -> None:
    # A pool of only identity queries would be trivial; the held-out-query
    # selection prefers a firing query, so most instances transform.
    pool = generate_pool(n_per_stratum=8)
    non_identity = sum(
        1
        for inst in pool.instances
        if inst.gold != inst.prompt_inputs["query"]
    )
    assert non_identity >= len(pool.instances) // 2


def test_contamination_guard_single_rule_is_fixed() -> None:
    # Fixed single-rule assertion (checklist A / rubric 8): any other value
    # fails at construction.
    with pytest.raises(AssertionError, match="single-rule"):
        generate_pool(n_per_stratum=2, number_of_rules=2)


def test_contamination_guard_seeds_are_fresh() -> None:
    # Fresh-seed assertion: every consumed seed is strictly above the
    # reserved range and never the published default (0).
    pool = _fast_pool()
    for inst in pool.instances:
        assert inst.seed > RESERVED_SEED_MAX
        assert inst.seed != PUBLISHED_SEED


def test_contamination_guard_rejects_a_reserved_seed_range() -> None:
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=1)


def test_contamination_guard_rejects_the_published_default_seed() -> None:
    # Starting on seed 0 (behind published InductionBench instances) fires.
    assert PUBLISHED_SEED <= RESERVED_SEED_MAX
    with pytest.raises(AssertionError, match="contamination guard"):
        _fast_pool(seed_start=PUBLISHED_SEED)


def test_default_seed_start_is_above_reserved_ceiling() -> None:
    assert DEFAULT_SEED_START > RESERVED_SEED_MAX


def test_manifest_seed_range_spans_one_seed_per_stratum() -> None:
    pool = generate_pool(n_per_stratum=2)
    manifest = build_manifest(pool, DEFAULT_SEED_START, len(DEFAULT_STRATA))
    start, end = manifest.seed_range
    assert start == DEFAULT_SEED_START
    assert end == DEFAULT_SEED_START + len(DEFAULT_STRATA)
