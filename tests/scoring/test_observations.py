"""Tests for repeat-level scoring observations."""

import pytest

from whetstone_envs.scoring import Observation, Outcome, scored


def test_scored_requires_binary_score() -> None:
    with pytest.raises(ValueError, match="score 0 or 1"):
        scored("t", 0, 2)


def test_non_scored_must_not_carry_score() -> None:
    with pytest.raises(ValueError, match="must not"):
        Observation("t", 0, Outcome.FAILED, 1)
