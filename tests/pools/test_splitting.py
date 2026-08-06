"""Tests for deterministic, disjoint task-pool splitting."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.pools import PoolSplit, TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


def test_split_is_stratified_and_independent_of_global_layout(
    synthetic_instance: Callable[..., Instance],
) -> None:
    easy = [synthetic_instance(i, "easy") for i in range(3)]
    hard = [synthetic_instance(i, "hard") for i in range(3, 6)]
    blocked_pool = TaskPool([*easy, *hard])
    interleaved_pool = TaskPool(
        [easy[0], hard[0], easy[1], hard[1], easy[2], hard[2]]
    )

    blocked = blocked_pool.split(2, 2, 2)
    interleaved = interleaved_pool.split(2, 2, 2)
    expected_ids = {inst.id for inst in blocked_pool.instances}

    blocked_roles: dict[str, set[str]] = {}
    interleaved_roles: dict[str, set[str]] = {}
    for role in ("internal_eval", "official", "held_out"):
        blocked_subset = getattr(blocked, role)
        interleaved_subset = getattr(interleaved, role)
        assert len(blocked_subset) == 2
        assert len(interleaved_subset) == 2
        assert {inst.strata[0] for inst in blocked_subset} == {
            "easy",
            "hard",
        }
        assert {inst.strata[0] for inst in interleaved_subset} == {
            "easy",
            "hard",
        }
        blocked_roles[role] = {inst.id for inst in blocked_subset}
        interleaved_roles[role] = {inst.id for inst in interleaved_subset}

    assert set().union(*blocked_roles.values()) == expected_ids
    assert sum(map(len, blocked_roles.values())) == len(expected_ids)
    assert set().union(*interleaved_roles.values()) == expected_ids
    assert sum(map(len, interleaved_roles.values())) == len(expected_ids)
    assert blocked_roles == interleaved_roles


def test_split_allows_leaving_instances_unassigned(
    synthetic_instance: Callable[..., Instance],
) -> None:
    alpha = [synthetic_instance(index, "alpha") for index in range(5)]
    beta = [synthetic_instance(index, "beta") for index in range(5, 10)]
    pool = TaskPool([*alpha, *beta])

    split = pool.split(2, 1, 1)
    assigned = {
        instance.id
        for subset in (
            split.internal_eval,
            split.official,
            split.held_out,
        )
        for instance in subset
    }

    assert assigned == {
        alpha[0].id,
        alpha[1].id,
        beta[0].id,
        beta[1].id,
    }


def test_split_distributes_scarce_combinations_across_destinations(
    synthetic_instance: Callable[..., Instance],
) -> None:
    alpha = [
        synthetic_instance(index, ("shared", "alpha")) for index in range(5)
    ]
    beta = [
        synthetic_instance(index, ("shared", "beta")) for index in range(5, 8)
    ]

    split = TaskPool([*alpha, *beta]).split(2, 2, 4)

    assert {instance.strata for instance in split.internal_eval} == {
        ("shared", "alpha"),
        ("shared", "beta"),
    }
    assert {instance.strata for instance in split.official} == {
        ("shared", "alpha"),
        ("shared", "beta"),
    }
    assert {instance.strata for instance in split.held_out} == {
        ("shared", "alpha"),
        ("shared", "beta"),
    }


def test_split_coverage_respects_destination_capacity(
    synthetic_instance: Callable[..., Instance],
) -> None:
    combinations = [
        [
            synthetic_instance(index * 2 + offset, ("shared", label))
            for offset in range(2)
        ]
        for index, label in enumerate(("alpha", "beta", "gamma"))
    ]
    pool = TaskPool(
        instance for combination in combinations for instance in combination
    )

    split = pool.split(1, 1, 4)

    assert len(split.internal_eval) == 1
    assert len(split.official) == 1
    assert len(split.held_out) == 4
    assert {instance.strata for instance in split.held_out} == {
        ("shared", "alpha"),
        ("shared", "beta"),
        ("shared", "gamma"),
    }


def test_split_balances_combinations_within_each_destination(
    synthetic_instance: Callable[..., Instance],
) -> None:
    combinations = [
        [synthetic_instance(index * 9 + offset, label) for offset in range(9)]
        for index, label in enumerate(("alpha", "beta"))
    ]
    split = TaskPool(
        instance for combination in combinations for instance in combination
    ).split(4, 7, 7)

    assert Counter(instance.strata for instance in split.internal_eval) == {
        ("alpha",): 2,
        ("beta",): 2,
    }
    assert Counter(instance.strata for instance in split.official) == {
        ("alpha",): 4,
        ("beta",): 3,
    }
    assert Counter(instance.strata for instance in split.held_out) == {
        ("alpha",): 3,
        ("beta",): 4,
    }


def test_split_balances_five_uniform_strata_at_c11_default_scale(
    synthetic_instance: Callable[..., Instance],
) -> None:
    labels = ("S0", "S1", "S2", "S3", "S4")
    combinations = [
        [
            synthetic_instance(index * 82 + offset, label)
            for offset in range(82)
        ]
        for index, label in enumerate(labels)
    ]
    split = TaskPool(
        instance for combination in combinations for instance in combination
    ).split(10, 200, 200)

    assert Counter(instance.strata for instance in split.internal_eval) == {
        (label,): 2 for label in labels
    }
    assert Counter(instance.strata for instance in split.official) == {
        (label,): 40 for label in labels
    }
    assert Counter(instance.strata for instance in split.held_out) == {
        (label,): 40 for label in labels
    }


def test_split_preserves_pool_order_with_interleaved_combinations(
    synthetic_instance: Callable[..., Instance],
) -> None:
    instances = [
        synthetic_instance(0, ("shared", "alpha")),
        synthetic_instance(1, ("shared", "beta")),
        synthetic_instance(2, ("shared", "alpha")),
        synthetic_instance(3, ("shared", "beta")),
        synthetic_instance(4, ("shared", "alpha")),
        synthetic_instance(5, ("shared", "beta")),
    ]
    pool = TaskPool(instances)

    split = pool.split(3, 2, 1)
    positions = {
        instance.id: index for index, instance in enumerate(instances)
    }

    for subset in (
        split.internal_eval,
        split.official,
        split.held_out,
    ):
        assert [positions[instance.id] for instance in subset] == sorted(
            positions[instance.id] for instance in subset
        )


def test_split_oversize_rejected(two_stratum_pool: TaskPool) -> None:
    with pytest.raises(ValueError, match="sum to"):
        two_stratum_pool.split(3, 3, 3)


def test_split_negative_size_rejected(two_stratum_pool: TaskPool) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        two_stratum_pool.split(-1, 0, 0)


def test_pool_split_asserts_no_overlap(
    synthetic_instance: Callable[..., Instance],
) -> None:
    shared = synthetic_instance(0, "easy")
    other = synthetic_instance(1, "easy")
    with pytest.raises(AssertionError, match="not disjoint"):
        PoolSplit(
            internal_eval=(shared,),
            official=(shared, other),
            held_out=(),
        )
