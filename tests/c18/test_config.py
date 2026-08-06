from __future__ import annotations

import pytest

from whetstone_envs.c18 import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
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


def test_only_operational_distractor_modes_are_exposed() -> None:
    assert tuple(DistractorMode) == (
        DistractorMode.NONE,
        DistractorMode.RELEVANT,
    )
    with pytest.raises(ValueError):
        DistractorMode("irrelevant")


@pytest.mark.parametrize("count", [0, -1])
def test_configuration_rejects_nonpositive_counts(count: int) -> None:
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        GenerationConfig(strata=(), **common)
    repeated = DepthStratum(1, DistractorMode.NONE)
    with pytest.raises(ValueError):
        GenerationConfig(strata=(repeated, repeated), **common)


@pytest.mark.parametrize(
    ("hops", "distractors", "error"),
    [
        (0, DistractorMode.NONE, ValueError),
        (True, DistractorMode.NONE, TypeError),
        (1, "none", TypeError),
    ],
)
def test_depth_stratum_rejects_invalid_values(
    hops: object,
    distractors: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        DepthStratum(
            hops,  # ty: ignore[invalid-argument-type]
            distractors,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("counts", "error"),
    [
        ((-1, 1, 1), ValueError),
        ((0, 0, 0), ValueError),
        ((True, 0, 0), TypeError),
    ],
)
def test_split_plan_rejects_invalid_values(
    counts: tuple[object, object, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        SplitPlan(*counts)  # ty: ignore[invalid-argument-type]


def test_configuration_rejects_an_unusable_seed_range() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(
            generator_version="test",
            seed_start=(1 << 32) - 1,
            n_per_stratum=1,
            strata=(
                DepthStratum(1, DistractorMode.NONE),
                DepthStratum(2, DistractorMode.NONE),
            ),
            split=SplitPlan(1, 0, 0),
        )


def test_split_plan_scales_without_losing_instances() -> None:
    assert SplitPlan(6, 12, 12).scale(10) == (2, 4, 4)
    assert SplitPlan(2, 6, 12).scale(10) == (1, 3, 6)
    assert SplitPlan(1, 1, 1).scale(5) == (1, 2, 2)
