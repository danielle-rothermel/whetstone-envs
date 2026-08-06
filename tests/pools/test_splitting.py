from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.pools import PoolSplit, TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


def _ids_by_destination(
    split: PoolSplit,
) -> tuple[set[str], set[str], set[str]]:
    return (
        {instance.id for instance in split.internal_eval},
        {instance.id for instance in split.official},
        {instance.id for instance in split.held_out},
    )


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

    blocked_subsets = (
        blocked.internal_eval,
        blocked.official,
        blocked.held_out,
    )
    interleaved_subsets = (
        interleaved.internal_eval,
        interleaved.official,
        interleaved.held_out,
    )
    for blocked_subset, interleaved_subset in zip(
        blocked_subsets,
        interleaved_subsets,
        strict=True,
    ):
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

    blocked_ids = _ids_by_destination(blocked)
    interleaved_ids = _ids_by_destination(interleaved)
    assert set().union(*blocked_ids) == expected_ids
    assert sum(map(len, blocked_ids)) == len(expected_ids)
    assert set().union(*interleaved_ids) == expected_ids
    assert sum(map(len, interleaved_ids)) == len(expected_ids)
    assert blocked_ids == interleaved_ids


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

    assert len(split.internal_eval) == 2
    assert len(split.official) == 2
    assert len(split.held_out) == 4
    assert Counter(instance.strata for instance in split.internal_eval) == {
        ("shared", "alpha"): 1,
        ("shared", "beta"): 1,
    }
    assert Counter(instance.strata for instance in split.official) == {
        ("shared", "alpha"): 1,
        ("shared", "beta"): 1,
    }
    assert Counter(instance.strata for instance in split.held_out) == {
        ("shared", "alpha"): 3,
        ("shared", "beta"): 1,
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

    subsets = (split.internal_eval, split.official, split.held_out)
    assert tuple(map(len, subsets)) == (4, 7, 7)
    assert Counter(
        instance.strata for subset in subsets for instance in subset
    ) == {("alpha",): 9, ("beta",): 9}
    for subset in subsets:
        counts = Counter(instance.strata for instance in subset)
        assert set(counts) == {("alpha",), ("beta",)}
        assert max(counts.values()) - min(counts.values()) <= 1


@pytest.mark.parametrize(
    "combination_order",
    [("alpha", "beta"), ("beta", "alpha")],
)
def test_split_balances_after_scarce_destination_coverage(
    synthetic_instance: Callable[..., Instance],
    combination_order: tuple[str, str],
) -> None:
    counts = {"alpha": 2, "beta": 3}
    instances = [
        synthetic_instance(index, label)
        for index, label in enumerate(
            label for label in combination_order for _ in range(counts[label])
        )
    ]

    split = TaskPool(instances).split(0, 1, 4)

    assert len(split.internal_eval) == 0
    assert len(split.official) == 1
    assert len(split.held_out) == 4
    assert Counter(instance.strata for instance in split.held_out) == {
        ("alpha",): 2,
        ("beta",): 2,
    }


def test_split_finds_global_balance_across_three_way_exchange(
    synthetic_instance: Callable[..., Instance],
) -> None:
    counts = {"alpha": 3, "beta": 2, "gamma": 4}
    instances = [
        synthetic_instance(index, label)
        for index, label in enumerate(
            label for label, count in counts.items() for _ in range(count)
        )
    ]

    split = TaskPool(instances).split(2, 1, 6)

    assert [
        Counter(instance.strata for instance in subset)
        for subset in (split.internal_eval, split.official, split.held_out)
    ] == [
        Counter({("alpha",): 1, ("gamma",): 1}),
        Counter({("gamma",): 1}),
        Counter({("alpha",): 2, ("beta",): 2, ("gamma",): 2}),
    ]


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
    assert (
        len(split.internal_eval),
        len(split.official),
        len(split.held_out),
    ) == (3, 2, 1)
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
    with pytest.raises(ValueError):
        two_stratum_pool.split(3, 3, 3)


@pytest.mark.parametrize("role", range(3))
@pytest.mark.parametrize(
    ("invalid_size", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        (0.5, TypeError),
    ],
)
def test_split_invalid_size_rejected(
    two_stratum_pool: TaskPool,
    role: int,
    invalid_size: object,
    error: type[Exception],
) -> None:
    sizes: list[object] = [0, 0, 0]
    sizes[role] = invalid_size

    with pytest.raises(error):
        two_stratum_pool.split(*sizes)  # ty: ignore[invalid-argument-type]


def test_empty_pool_splits_into_empty_destinations() -> None:
    assert TaskPool([]).split(0, 0, 0) == PoolSplit((), (), ())


def test_pool_split_asserts_no_overlap(
    synthetic_instance: Callable[..., Instance],
) -> None:
    shared = synthetic_instance(0, "easy")
    other = synthetic_instance(1, "easy")
    with pytest.raises(AssertionError):
        PoolSplit(
            internal_eval=(shared,),
            official=(shared, other),
            held_out=(),
        )
