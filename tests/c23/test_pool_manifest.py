from pathlib import Path

import pytest

from whetstone_envs.c23 import (
    GENERATOR_VERSION,
    default_split_sizes,
    generate_pool,
)
from whetstone_envs.manifests import Manifest

MANIFEST_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "whetstone_envs"
    / "c23"
    / "manifest.json"
)


def test_small_pool_rejects_default_split_capacity() -> None:
    with pytest.raises(ValueError, match="requires 50 instances"):
        default_split_sizes(generate_pool(n_per_stratum=1))


def test_default_pool_split_and_committed_manifest() -> None:
    pool = generate_pool()

    assert default_split_sizes(pool) == (40, 80, 80)
    split = pool.split(*default_split_sizes(pool))
    for destination in (
        split.internal_eval,
        split.official,
        split.held_out,
    ):
        counts = {
            label: sum(label in instance.strata for instance in destination)
            for label in pool.strata
        }
        assert len(set(counts.values())) == 1

    regenerated = Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(555_000_000, 555_000_004),
    )
    assert Manifest.read(MANIFEST_PATH) == regenerated
