from __future__ import annotations

import pytest
from minigrid.core.grid import Grid

from whetstone_envs.c19 import planning
from whetstone_envs.c19._minigrid import MiniGridState, run_script
from whetstone_envs.c19.model import MAX_COMMAND_LENGTH
from whetstone_envs.c19.planning import (
    MAX_SEARCH_EXPANSIONS,
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
    assert result.heading.value == "N"
    assert len(command) <= MAX_COMMAND_LENGTH


def test_route_search_budget_is_explicit_and_fails_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_SEARCH_EXPANSIONS == 512
    monkeypatch.setattr(planning, "MAX_SEARCH_EXPANSIONS", 0)

    with pytest.raises(
        PlanningError,
        match=r"family=navigation size=small seed=7.*0 expansions",
    ):
        route_to_pose(
            _empty_state(),
            target_position=(3, 1),
            target_direction=3,
            context="family=navigation size=small seed=7",
        )


def test_route_search_rejects_a_blocked_target_with_context() -> None:
    with pytest.raises(
        PlanningError,
        match=r"family=door size=small seed=9.*blocked",
    ):
        route_to_pose(
            _empty_state(),
            target_position=(0, 0),
            target_direction=0,
            context="family=door size=small seed=9",
        )
