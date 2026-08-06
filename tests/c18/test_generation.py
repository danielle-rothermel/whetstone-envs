from __future__ import annotations

from importlib import resources

import pytest

from whetstone_envs.c18 import generation
from whetstone_envs.c18.config import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
    GenerationConfig,
)
from whetstone_envs.c18.upstream import RawInstance
from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import Manifest, content_hash
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
def test_committed_manifest_matches_full_regeneration(
    config: GenerationConfig,
    resource_name: str,
) -> None:
    pool = generation.generate_pool(config)
    frozen = _frozen_manifest(resource_name)
    assert frozen.matches_pool(pool)
    assert frozen == generation.build_manifest(pool, config)


def test_small_pool_is_deterministic_and_depth_interleaved() -> None:
    first = generation.generate_pool(n_per_stratum=2)
    second = generation.generate_pool(n_per_stratum=2)
    assert content_hash(first) == content_hash(second)
    assert tuple(instance.strata[0] for instance in first.instances) == (
        "D1",
        "D2",
        "D3",
        "D5",
        "D1",
        "D2",
        "D3",
        "D5",
    )
    assert all(
        set(instance.prompt_inputs) == {"question", "query"}
        and instance.gold in {"True", "False"}
        for instance in first.instances
    )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_override_is_rejected_before_generation(
    count: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**_kwargs: object) -> tuple[RawInstance, ...]:
        raise AssertionError("upstream generation must not start")

    monkeypatch.setattr(generation.upstream, "generate_raw", fail_if_called)
    with pytest.raises((TypeError, ValueError), match="n_per_stratum"):
        generation.generate_pool(
            n_per_stratum=count,  # ty: ignore[invalid-argument-type]
        )


def test_oracle_disagreement_stops_pool_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RawInstance(
        question="Sally is a brimpus. Every brimpus is sour.",
        query="True or false: Sally is sour.",
        answer="False",
        hops=1,
    )
    monkeypatch.setattr(
        generation.upstream,
        "generate_raw",
        lambda **_kwargs: (row,),
    )
    with pytest.raises(AssertionError, match="oracle disagreement"):
        generation.generate_pool(n_per_stratum=1)


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
    assert generation.default_split_sizes(pool) == (8, 16, 16)
