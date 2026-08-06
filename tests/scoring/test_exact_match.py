"""Tests for binary exact-match scoring."""

import pytest

from whetstone_envs.scoring import exact_match


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("yes", "yes", 1),
        ("  yes  ", "yes", 1),
        ("```\nyes\n```", "yes", 1),
        ("yes", "no", 0),
        ("Yes", "yes", 0),
    ],
)
def test_exact_match(prediction: str, gold: str, expected: int) -> None:
    result = exact_match(prediction, gold)
    assert result == expected
    assert result in (0, 1)
