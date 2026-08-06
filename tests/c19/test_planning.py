from __future__ import annotations

import pytest
from minigrid.core.grid import Grid

from whetstone_envs.c19 import planning
from whetstone_envs.c19._minigrid import MiniGridState, run_script
from whetstone_envs.c19.model import MAX_COMMAND_LENGTH, Heading
from whetstone_envs.c19.planning import (
    PlanningError,
    route_to_pose,
)


def _empty_state() -> MiniGridState:
    grid = Grid(5, 5)
    grid.wall_rect(0, 0, 5, 5)
    return MiniGridState(
        grid=grid,
        agent_position=(1, 3),
        agent_direction=0,
    )


def test_route_search_finds_an_exact_pose_without_blocked_steps() -> None:
    state = _empty_state()

    command = route_to_pose(
        state,
        target_position=(3, 1),
        target_direction=3,
        context="test route",
    )
    result = run_script(state, command)

    assert (result.agent_column, result.agent_row) == (3, 1)
    assert result.heading is Heading.NORTH
    assert len(command) <= MAX_COMMAND_LENGTH


def test_route_search_stops_at_its_expansion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planning, "MAX_SEARCH_EXPANSIONS", 0)

    with pytest.raises(PlanningError):
        route_to_pose(
            _empty_state(),
            target_position=(3, 1),
            target_direction=3,
            context="family=navigation size=small seed=7",
        )


def test_route_search_rejects_a_blocked_target() -> None:
    with pytest.raises(PlanningError):
        route_to_pose(
            _empty_state(),
            target_position=(0, 0),
            target_direction=0,
            context="family=door size=small seed=9",
        )
