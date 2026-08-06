from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c22 import Preset, generate_pool, load_manifest
from whetstone_envs.c22.constraints import (
    HARD_CONSTRAINT_KINDS,
    ConstraintStack,
)
from whetstone_envs.c22.generation import (
    DEFAULT_SEED_START,
    HARD_SEED_START,
    PUBLISHED_KEY_MAX,
    _config_for,
    _generate_pool,
)
from whetstone_envs.manifests import content_hash

if TYPE_CHECKING:
    from whetstone_envs.pools import TaskPool


@pytest.fixture(scope="module")
def default_pool() -> TaskPool:
    return generate_pool()


@pytest.fixture(scope="module")
def hard_pool() -> TaskPool:
    return generate_pool(Preset.HARD)


def test_default_pool_matches_its_canonical_manifest(
    default_pool: TaskPool,
) -> None:
    assert len(default_pool) == 120
    assert default_pool.stratum_counts() == {
        "n3_easy": 20,
        "n3_mixed": 20,
        "n4_easy": 20,
        "n4_mixed": 20,
        "n5_easy": 20,
        "n5_mixed": 20,
    }
    manifest = load_manifest()
    assert manifest.matches_pool(default_pool)
    assert (
        manifest.generator_version
        == _config_for(Preset.DEFAULT).generator_version
    )


def test_hard_pool_matches_its_canonical_manifest(
    hard_pool: TaskPool,
) -> None:
    assert len(hard_pool) == 60
    assert hard_pool.stratum_counts() == {
        "n3_hard": 20,
        "n6_hard": 20,
        "n8_hard": 20,
    }
    manifest = load_manifest(Preset.HARD)
    assert manifest.matches_pool(hard_pool)
    assert (
        manifest.generator_version
        == _config_for(Preset.HARD).generator_version
    )


@pytest.mark.parametrize("preset", [Preset.DEFAULT, Preset.HARD])
def test_generation_is_deterministic_and_preserves_global_random_state(
    preset: Preset,
) -> None:
    before = random.getstate()
    first = _generate_pool(preset, instances_per_stratum=2)
    second = _generate_pool(preset, instances_per_stratum=2)
    assert random.getstate() == before
    assert content_hash(first) == content_hash(second)
    assert first.instances == second.instances


def test_generated_gold_matches_each_stratum_and_stays_private() -> None:
    pool = _generate_pool(Preset.DEFAULT, instances_per_stratum=2)
    for instance in pool.instances:
        stack = ConstraintStack.from_gold(instance.gold)
        count = int(instance.strata[0].split("_", maxsplit=1)[0][1:])
        assert len(stack.constraints) == count
        assert set(instance.prompt_inputs) == {"constraints"}


def test_hard_instances_contain_every_hard_constraint() -> None:
    pool = _generate_pool(Preset.HARD, instances_per_stratum=2)
    for instance in pool.instances:
        stack = ConstraintStack.from_gold(instance.gold)
        kinds = {constraint.kind for constraint in stack.constraints}
        assert kinds >= HARD_CONSTRAINT_KINDS


def test_preset_seed_ranges_are_fresh_and_disjoint(
    default_pool: TaskPool,
    hard_pool: TaskPool,
) -> None:
    assert min(instance.seed for instance in default_pool.instances) == (
        DEFAULT_SEED_START
    )
    assert min(instance.seed for instance in hard_pool.instances) == (
        HARD_SEED_START
    )
    assert DEFAULT_SEED_START > PUBLISHED_KEY_MAX
    assert {instance.id for instance in default_pool.instances}.isdisjoint(
        instance.id for instance in hard_pool.instances
    )


def test_public_generation_rejects_untyped_presets() -> None:
    with pytest.raises(TypeError):
        generate_pool("default")  # ty: ignore[invalid-argument-type]
