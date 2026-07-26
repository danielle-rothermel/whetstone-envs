"""Oracle correctness on HAND-BUILT fixtures (checklist A).

Every grid and command below is hand-constructed and hand-traced -- none
is generator-produced -- so this file catches an oracle that is silently
a re-derivation of generator internals rather than a true independent
object-model walk (rubric criteria 2, 8; PLAN Verification checklist A).

Each fixture's expected fact is worked out by hand in the comment beside
it. Positions are stated as ``(row, col)`` with row 0 at the top, matching
the oracle's answer format.
"""

from __future__ import annotations

import pytest

from whetstone_envs.c19 import oracle
from whetstone_envs.c19.oracle import (
    OracleError,
    derive_fact,
    score,
    score_gold,
)

# A 5x5 grid, empty interior, agent at (row1, col1) facing east.
_EMPTY_EAST = "\n".join(
    [
        "WGWGWGWGWG",
        "WG>>    WG",
        "WG      WG",
        "WG      WG",
        "WGWGWGWGWG",
    ],
)

# A grid with a wall directly ahead of an east-facing agent at (1, 1).
_WALL_AHEAD = "\n".join(
    [
        "WGWGWGWGWG",
        "WG>>WG  WG",
        "WG      WG",
        "WGWGWGWGWG",
    ],
)

# A key (KY) directly ahead of an east-facing agent at (1, 1).
_KEY_AHEAD = "\n".join(
    [
        "WGWGWGWGWG",
        "WG>>KY  WG",
        "WG      WG",
        "WGWGWGWGWG",
    ],
)

# A goal (GG) directly ahead of an east-facing agent at (1, 1).
_GOAL_AHEAD = "\n".join(["WGWGWGWG", "WG>>GGWG", "WGWGWGWG"])

# A ball (AR = red ball) directly ahead of an east-facing agent at (1, 1).
_BALL_AHEAD = "\n".join(["WGWGWGWG", "WG>>ARWG", "WGWGWGWG"])

# A north-facing agent at (1, 1) with the top wall directly ahead.
_NORTH_AT_TOP = "\n".join(["WGWGWGWG", "WG^^  WG", "WG    WG", "WGWGWGWG"])


def test_coordinate_two_steps_east() -> None:
    # FF: east col1->col2->col3, row unchanged -> (1, 3).
    assert derive_fact(_EMPTY_EAST, "FF", "coordinate") == "1,3"


def test_heading_after_turns() -> None:
    # R turns east->south; still south after moving forward.
    assert derive_fact(_EMPTY_EAST, "RFF", "heading") == "S"
    assert derive_fact(_EMPTY_EAST, "RFF", "coordinate") == "3,1"


def test_left_turn_heading_order() -> None:
    # L from east -> north (spec E->N->W->S->E).
    assert derive_fact(_EMPTY_EAST, "L", "heading") == "N"
    # Two lefts from east -> west.
    assert derive_fact(_EMPTY_EAST, "LL", "heading") == "W"


def test_forward_blocked_by_wall_stays_put() -> None:
    # Wall directly ahead: the agent does not move (no pass-through).
    assert derive_fact(_WALL_AHEAD, "F", "coordinate") == "1,1"
    assert derive_fact(_WALL_AHEAD, "F", "front") == "wall"


def test_forward_blocked_by_edge_stays_put() -> None:
    # North-facing at row 1 with the top border wall ahead: stays at (1, 1).
    assert derive_fact(_NORTH_AT_TOP, "F", "coordinate") == "1,1"


def test_front_reports_wall_key_goal_ball_empty() -> None:
    assert derive_fact(_WALL_AHEAD, "", "front") == "wall"
    assert derive_fact(_KEY_AHEAD, "", "front") == "key"
    assert derive_fact(_GOAL_AHEAD, "", "front") == "goal"
    assert derive_fact(_BALL_AHEAD, "", "front") == "ball"
    # After picking the key up, the vacated cell reads as empty.
    assert derive_fact(_KEY_AHEAD, "P", "front") == "empty"


def test_pickup_sets_carrying_and_vacates_cell() -> None:
    assert derive_fact(_KEY_AHEAD, "", "carrying") == "no"
    assert derive_fact(_KEY_AHEAD, "P", "carrying") == "yes"
    # The picked cell is now empty, so F moves the agent into it -> (1, 2).
    assert derive_fact(_KEY_AHEAD, "PF", "coordinate") == "1,2"


def test_drop_clears_carrying() -> None:
    # P picks the key, F steps onto the vacated cell, D drops into the
    # now-empty cell ahead -> no longer carrying.
    assert derive_fact(_KEY_AHEAD, "PF", "carrying") == "yes"
    assert derive_fact(_KEY_AHEAD, "PFD", "carrying") == "no"


def test_forward_onto_goal_overlaps() -> None:
    # A goal is overlappable: the agent steps onto it.
    assert derive_fact(_GOAL_AHEAD, "F", "coordinate") == "1,2"


def test_ball_is_pickupable() -> None:
    assert derive_fact(_BALL_AHEAD, "P", "carrying") == "yes"


def test_toggle_is_a_noop_for_position() -> None:
    assert derive_fact(_GOAL_AHEAD, "T", "coordinate") == "1,1"
    assert derive_fact(_GOAL_AHEAD, "T", "heading") == "E"


def test_full_hand_traced_sequence() -> None:
    # Hand trace on _EMPTY_EAST from (1,1) east:
    #   F -> (1,2) east; R -> (1,2) south; F -> (2,2) south;
    #   F -> (3,2) south; L -> (3,2) east.
    grid = _EMPTY_EAST
    assert derive_fact(grid, "FRFFL", "coordinate") == "3,2"
    assert derive_fact(grid, "FRFFL", "heading") == "E"


def test_score_exact_match_and_normalization() -> None:
    grid, command, fact = _EMPTY_EAST, "FF", "coordinate"
    assert score("1,3", grid, command, fact) == 1
    # Surrounding whitespace / a wrapping code fence is normalized away.
    assert score("  1,3\n", grid, command, fact) == 1
    assert score("```\n1,3\n```", grid, command, fact) == 1
    assert score("2,3", grid, command, fact) == 0


def test_score_unparsable_input_is_zero_not_raise() -> None:
    # A malformed grid scores 0 (a model response is graded, not trusted).
    assert score("1,3", "not a grid", "FF", "coordinate") == 0


def test_score_gold_matches_frozen_gold() -> None:
    assert score_gold("E", "E") == 1
    assert score_gold(" E ", "E") == 1
    assert score_gold("W", "E") == 0


def test_derive_fact_rejects_unknown_fact_type() -> None:
    with pytest.raises(OracleError, match="unknown fact type"):
        derive_fact(_EMPTY_EAST, "F", "not-a-fact")


def test_parser_rejects_non_vanilla_command() -> None:
    with pytest.raises(OracleError, match="non-vanilla action"):
        derive_fact(_EMPTY_EAST, "FJF", "coordinate")


def test_parser_rejects_grid_without_agent() -> None:
    no_agent = "\n".join(["WGWGWG", "WG  WG", "WGWGWG"])
    with pytest.raises(OracleError, match="no agent"):
        derive_fact(no_agent, "F", "coordinate")


def test_valid_commands_are_the_six_vanilla_actions() -> None:
    assert set(oracle.VALID_COMMANDS) == {"L", "R", "F", "P", "D", "T"}
