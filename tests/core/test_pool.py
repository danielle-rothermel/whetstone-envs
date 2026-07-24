"""Tests for the :class:`TaskPool` container and its disjoint split."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.core.pool import PoolSplit, TaskPool

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
    two_stratum_pool: TaskPool,
) -> None:
    split = two_stratum_pool.split(1, 1, 1)
    assigned = (
        len(split.internal_eval) + len(split.official) + len(split.held_out)
    )
    assert assigned == 3


def test_split_redistributes_quota_across_full_strata_combinations(
    synthetic_instance: Callable[..., Instance],
) -> None:
    alpha = [synthetic_instance(0, ("shared", "alpha"))]
    beta = [synthetic_instance(1, ("shared", "beta"))]
    gamma = [
        synthetic_instance(index, ("shared", "gamma")) for index in range(2, 7)
    ]

    split = TaskPool([*alpha, *beta, *gamma]).split(4, 2, 1)

    assert split.internal_eval == (
        alpha[0],
        beta[0],
        gamma[0],
        gamma[1],
    )
    assert split.official == (gamma[2], gamma[3])
    assert split.held_out == (gamma[4],)


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
