from whetstone_envs.c23._domain import Demonstration
from whetstone_envs.c23._selection import _bounded_cover, _greedy_cover


def test_bounded_cover_recovers_when_greedy_choice_fails() -> None:
    masks = (0b0011, 0b0101, 0b1010)
    coverage = tuple(
        (Demonstration(str(index), ""), mask)
        for index, mask in enumerate(masks)
    )

    assert _greedy_cover(coverage, 0b1111, 2) is None
    assert _bounded_cover(masks, 0b1111, 2) == (1, 2)


def test_bounded_cover_does_not_prune_lower_index_companion() -> None:
    masks = (0b010, 0b101)

    assert _bounded_cover(masks, 0b111, 2) == (1, 0)
