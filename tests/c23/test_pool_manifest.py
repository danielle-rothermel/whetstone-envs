from pathlib import Path

import pytest

from whetstone_envs.c23 import (
    GENERATOR_VERSION,
    default_split_sizes,
)
from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import Manifest
from whetstone_envs.pools import TaskPool

MANIFEST_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "whetstone_envs"
    / "c23"
    / "manifest.json"
)


def _pool_with_strata(
    strata_by_instance: tuple[str | tuple[str, ...], ...],
) -> TaskPool:
    return TaskPool(
        make_instance(
            id=f"task-{index}",
            seed=index,
            strata=strata,
            prompt_inputs={"index": str(index)},
        )
        for index, strata in enumerate(strata_by_instance)
    )


@pytest.mark.parametrize(
    "strata_by_instance",
    [
        ("unexpected",),
        (("S1", "S2"), "S3", "S4"),
        ("S1", "S2", "S3", "S4", "S4"),
        ("S1", "S2", "S3", "S4"),
    ],
    ids=[
        "unexpected-stratum",
        "multi-stratum-instance",
        "imbalanced",
        "insufficient-capacity",
    ],
)
def test_default_split_sizes_rejects_incompatible_pools(
    strata_by_instance: tuple[str | tuple[str, ...], ...],
) -> None:
    with pytest.raises(ValueError):
        default_split_sizes(_pool_with_strata(strata_by_instance))


def test_default_pool_split_and_committed_manifest(
    c23_default_pool: TaskPool,
) -> None:
    assert default_split_sizes(c23_default_pool) == (40, 80, 80)
    split = c23_default_pool.split(*default_split_sizes(c23_default_pool))
    for destination in (
        split.internal_eval,
        split.official,
        split.held_out,
    ):
        counts = {
            label: sum(label in instance.strata for instance in destination)
            for label in c23_default_pool.strata
        }
        assert len(set(counts.values())) == 1

    regenerated = Manifest.from_pool(
        c23_default_pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(555_000_000, 555_000_004),
    )
    assert Manifest.read(MANIFEST_PATH) == regenerated
