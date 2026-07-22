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


def test_split_is_contiguous_and_disjoint(
    two_stratum_pool: TaskPool,
) -> None:
    split = two_stratum_pool.split(2, 2, 2)
    assert [i.id for i in split.internal_eval] == ["easy-0", "easy-1"]
    assert [i.id for i in split.official] == ["easy-2", "hard-3"]
    assert [i.id for i in split.held_out] == ["hard-4", "hard-5"]

    ids = (
        {i.id for i in split.internal_eval}
        | {i.id for i in split.official}
        | {i.id for i in split.held_out}
    )
    assert len(ids) == 6


def test_split_allows_leaving_instances_unassigned(
    two_stratum_pool: TaskPool,
) -> None:
    split = two_stratum_pool.split(1, 1, 1)
    assigned = (
        len(split.internal_eval) + len(split.official) + len(split.held_out)
    )
    assert assigned == 3


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
