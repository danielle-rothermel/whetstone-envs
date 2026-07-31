"""Tests for the :class:`TaskPool` container and its disjoint split."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.pool import (
    PoolSplit,
    TaskPool,
    public_prompt_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.core.instance import Instance


def test_stratum_counts_match_membership(two_stratum_pool: TaskPool) -> None:
    assert len(two_stratum_pool) == 6
    assert two_stratum_pool.stratum_counts() == {"easy": 3, "hard": 3}
    assert two_stratum_pool.strata == ("easy", "hard")


def test_in_stratum_returns_members(two_stratum_pool: TaskPool) -> None:
    easy = two_stratum_pool.in_stratum("easy")
    assert [i.id for i in easy] == ["easy-0", "easy-1", "easy-2"]
    assert two_stratum_pool.in_stratum("absent") == ()


def test_duplicate_id_rejected(
    synthetic_instance: Callable[..., Instance],
) -> None:
    dup = synthetic_instance(0, "easy")
    with pytest.raises(ValueError, match="duplicate instance id"):
        TaskPool([dup, dup])


def test_public_prompt_identity_uses_only_sorted_prompt_inputs() -> None:
    first = make_instance(
        id="first",
        seed=1,
        strata="alpha",
        prompt_inputs={"z": "last", "a": "first"},
        gold="first-gold",
    )
    second = make_instance(
        id="second",
        seed=2,
        strata="beta",
        prompt_inputs={"a": "first", "z": "last"},
        gold="second-gold",
    )

    expected = (("a", "first"), ("z", "last"))
    assert public_prompt_identity(first) == expected
    assert public_prompt_identity(second) == expected


@pytest.mark.parametrize(
    ("first_gold", "second_gold"),
    [("same", "same"), ("first", "second")],
    ids=["same-gold", "different-gold"],
)
def test_duplicate_public_prompt_identity_rejected(
    first_gold: str,
    second_gold: str,
) -> None:
    first = make_instance(
        id="first",
        seed=1,
        strata="alpha",
        prompt_inputs={"question": "Q", "context": "C"},
        gold=first_gold,
    )
    second = make_instance(
        id="second",
        seed=2,
        strata="beta",
        prompt_inputs={"context": "C", "question": "Q"},
        gold=second_gold,
    )

    with pytest.raises(ValueError, match="duplicate public prompt identity"):
        TaskPool([first, second])


def test_empty_pool_is_valid() -> None:
    assert TaskPool([]).instances == ()


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
