from typing import cast

import pytest

from whetstone_envs.scoring import Observation, Outcome, scored


@pytest.mark.parametrize("score", [2, True, 1.0])
def test_scored_requires_binary_integer(score: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        scored("t", 0, cast("int", score))


def test_non_scored_must_not_carry_score() -> None:
    with pytest.raises(ValueError):
        Observation("t", 0, Outcome.FAILED, 1)


def test_observation_requires_outcome_member_even_without_score() -> None:
    with pytest.raises(TypeError):
        Observation("t", 0, cast("Outcome", "failed"), None)


@pytest.mark.parametrize("repeat_id", [True, 1.0, "1"])
def test_observation_requires_integer_repeat_id(repeat_id: object) -> None:
    with pytest.raises(TypeError):
        scored("t", cast("int", repeat_id), 1)
