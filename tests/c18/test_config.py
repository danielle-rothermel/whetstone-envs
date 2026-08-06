from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from whetstone_envs.c18.config import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
    RESERVED_SEED_MAX,
    DepthStratum,
    DistractorMode,
    GenerationConfig,
    SplitPlan,
)


def test_default_configuration_is_the_published_pool_shape() -> None:
    assert tuple(stratum.hops for stratum in DEFAULT_CONFIG.strata) == (
        1,
        2,
        3,
        5,
    )
    assert {stratum.distractors for stratum in DEFAULT_CONFIG.strata} == {
        DistractorMode.RELEVANT,
    }
    assert DEFAULT_CONFIG.n_per_stratum == 30
    assert DEFAULT_CONFIG.split == SplitPlan(6, 12, 12)
    assert DEFAULT_CONFIG.seed_start > RESERVED_SEED_MAX


def test_hard_configuration_is_deep_and_explicit() -> None:
    assert tuple(
        (stratum.hops, stratum.distractors) for stratum in HARD_CONFIG.strata
    ) == (
        (5, DistractorMode.RELEVANT),
        (8, DistractorMode.NONE),
        (10, DistractorMode.NONE),
    )
    assert HARD_CONFIG.n_per_stratum == 20
    assert HARD_CONFIG.split == SplitPlan(2, 6, 12)
    assert HARD_CONFIG.seed_start > DEFAULT_CONFIG.seed_range[1]


def test_configuration_is_deeply_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.seed_start = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        DEFAULT_CONFIG.strata[0] = DepthStratum(4, DistractorMode.NONE)  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("count", [0, -1])
def test_configuration_rejects_nonpositive_counts(count: int) -> None:
    with pytest.raises(ValueError, match="n_per_stratum must be positive"):
        GenerationConfig(
            generator_version="test",
            seed_start=1_000_000_000,
            n_per_stratum=count,
            strata=(DepthStratum(1, DistractorMode.NONE),),
            split=SplitPlan(1, 0, 0),
        )


def test_configuration_rejects_empty_and_duplicate_strata() -> None:
    common = {
        "generator_version": "test",
        "seed_start": 1_000_000_000,
        "n_per_stratum": 1,
        "split": SplitPlan(1, 0, 0),
    }
    with pytest.raises(ValueError, match="at least one depth stratum"):
        GenerationConfig(strata=(), **common)
    repeated = DepthStratum(1, DistractorMode.NONE)
    with pytest.raises(ValueError, match="distinct depth strata"):
        GenerationConfig(strata=(repeated, repeated), **common)


def test_split_plan_scales_without_losing_instances() -> None:
    assert SplitPlan(6, 12, 12).scale(10) == (2, 4, 4)
    assert SplitPlan(2, 6, 12).scale(10) == (1, 3, 6)
