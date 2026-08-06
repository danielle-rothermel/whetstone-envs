from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


def test_stratum_counts_match_membership(two_stratum_pool: TaskPool) -> None:
    assert len(two_stratum_pool) == 6
    assert two_stratum_pool.stratum_counts() == {"easy": 3, "hard": 3}
    assert two_stratum_pool.strata == ("easy", "hard")


def test_in_stratum_returns_members(two_stratum_pool: TaskPool) -> None:
    easy = two_stratum_pool.in_stratum("easy")
    assert [instance.id for instance in easy] == [
        "easy-0",
        "easy-1",
        "easy-2",
    ]
    assert two_stratum_pool.in_stratum("absent") == ()


def test_multi_label_membership_counts_and_first_seen_order(
    synthetic_instance: Callable[..., Instance],
) -> None:
    first = synthetic_instance(0, ("shared", "alpha"))
    second = synthetic_instance(1, ("beta", "shared"))
    third = synthetic_instance(2, ("alpha", "beta"))

    pool = TaskPool([first, second, third])

    assert pool.strata == ("shared", "alpha", "beta")
    assert pool.stratum_counts() == {"shared": 2, "alpha": 2, "beta": 2}
    assert pool.in_stratum("shared") == (first, second)
    assert pool.in_stratum("alpha") == (first, third)
    assert pool.in_stratum("beta") == (second, third)


def test_duplicate_id_rejected(
    synthetic_instance: Callable[..., Instance],
) -> None:
    first = synthetic_instance(0, "easy")
    second = synthetic_instance(0, "easy")
    assert first is not second

    with pytest.raises(ValueError, match="duplicate instance id"):
        TaskPool([first, second])


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
