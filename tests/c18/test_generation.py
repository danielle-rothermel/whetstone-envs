from __future__ import annotations

from importlib import resources

import pytest

import whetstone_envs.c18.generation as generation_impl
from whetstone_envs.c18 import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
    GenerationConfig,
    build_manifest,
    default_split_sizes,
    generate_pool,
)
from whetstone_envs.c18.upstream import RawInstance
from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import Manifest
from whetstone_envs.pools import TaskPool


def _frozen_manifest(name: str) -> Manifest:
    resource = resources.files("whetstone_envs.c18").joinpath(
        "resources",
        name,
    )
    with resources.as_file(resource) as path:
        return Manifest.read(path)


@pytest.mark.parametrize(
    ("config", "resource_name"),
    [
        (DEFAULT_CONFIG, "default.manifest.json"),
        (HARD_CONFIG, "hard.manifest.json"),
    ],
)
@pytest.mark.integration
def test_committed_manifest_matches_full_regeneration(
    config: GenerationConfig,
    resource_name: str,
) -> None:
    pool = generate_pool(config)
    frozen = _frozen_manifest(resource_name)
    assert frozen.matches_pool(pool)
    assert frozen == build_manifest(pool, config)


def test_small_pool_uses_one_seed_per_depth_and_interleaves() -> None:
    pool = generate_pool(n_per_stratum=2)
    expected = tuple(
        (stratum.label, DEFAULT_CONFIG.seed_start + stratum_index)
        for _ in range(2)
        for stratum_index, stratum in enumerate(DEFAULT_CONFIG.strata)
    )
    assert (
        tuple(
            (instance.strata[0], instance.seed) for instance in pool.instances
        )
        == expected
    )
    assert all(
        set(instance.prompt_inputs) == {"question", "query"}
        and instance.gold in {"True", "False"}
        for instance in pool.instances
    )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_override_is_rejected_before_generation(
    count: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**_kwargs: object) -> tuple[RawInstance, ...]:
        raise AssertionError("upstream generation must not start")

    monkeypatch.setattr(
        generation_impl.upstream, "generate_raw", fail_if_called
    )
    with pytest.raises((TypeError, ValueError)):
        generate_pool(
            n_per_stratum=count,  # ty: ignore[invalid-argument-type]
        )


def test_oracle_disagreement_stops_pool_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RawInstance(
        question="Sally is a brimpus. Every brimpus is sour.",
        query="True or false: Sally is sour.",
        answer="False",
    )
    monkeypatch.setattr(
        generation_impl.upstream,
        "generate_raw",
        lambda **_kwargs: (row,),
    )
    with pytest.raises(AssertionError):
        generate_pool(n_per_stratum=1)


def test_default_split_sizes_scale_uniform_strata() -> None:
    pool = TaskPool(
        make_instance(
            id=f"{stratum.label}-{index}",
            seed=DEFAULT_CONFIG.seed_start + stratum_index,
            strata=stratum.label,
            prompt_inputs={"token": f"{stratum.label}-{index}"},
        )
        for index in range(10)
        for stratum_index, stratum in enumerate(DEFAULT_CONFIG.strata)
    )
    assert default_split_sizes(pool) == (8, 16, 16)


@pytest.mark.parametrize(
    "stratum_counts",
    [
        {"D1": 1, "D2": 1, "D3": 1, "D4": 1},
        {"D1": 2, "D2": 2, "D3": 2, "D5": 1},
    ],
)
def test_default_split_sizes_rejects_incompatible_pools(
    stratum_counts: dict[str, int],
) -> None:
    pool = TaskPool(
        make_instance(
            id=f"{label}-{index}",
            seed=DEFAULT_CONFIG.seed_start + index,
            strata=label,
            prompt_inputs={"token": f"{label}-{index}"},
        )
        for label, count in stratum_counts.items()
        for index in range(count)
    )
    with pytest.raises(ValueError):
        default_split_sizes(pool)
