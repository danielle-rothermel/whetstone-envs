from typing import cast

import pytest

from whetstone_envs.scoring import Observation, Outcome, scored


def test_scored_requires_binary_score() -> None:
    with pytest.raises(ValueError, match="score 0 or 1"):
        scored("t", 0, 2)


def test_non_scored_must_not_carry_score() -> None:
    with pytest.raises(ValueError, match="must not"):
        Observation("t", 0, Outcome.FAILED, 1)


def test_observation_requires_outcome_member_even_without_score() -> None:
    with pytest.raises(TypeError, match="Outcome"):
        Observation("t", 0, cast("Outcome", "failed"), None)
