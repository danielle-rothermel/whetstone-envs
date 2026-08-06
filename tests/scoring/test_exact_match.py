import pytest

from whetstone_envs.scoring import exact_match


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("yes", "yes", 1),
        ("yes", "no", 0),
        ("Yes", "yes", 0),
        ("yes", "```\nyes\n```", 1),
    ],
)
def test_exact_match(prediction: str, gold: str, expected: int) -> None:
    assert exact_match(prediction, gold) == expected
