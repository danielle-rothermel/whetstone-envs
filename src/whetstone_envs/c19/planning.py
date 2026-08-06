from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from whetstone_envs.c19.model import Action

if TYPE_CHECKING:
    from whetstone_envs.c19._minigrid import MiniGridState

MAX_SEARCH_EXPANSIONS = 512

_ROUTE_ACTIONS: tuple[Action, ...] = (
    Action.LEFT,
    Action.RIGHT,
    Action.FORWARD,
)
_DIRECTION_TO_VECTOR: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class PlanningError(RuntimeError):
    """A bounded constructive route could not be produced."""


@dataclass(frozen=True, slots=True)
class _Pose:
    x: int
    y: int
    direction: int


def _next_pose(state: MiniGridState, pose: _Pose, action: Action) -> _Pose:
    if action is Action.LEFT:
        return _Pose(pose.x, pose.y, (pose.direction - 1) % 4)
    if action is Action.RIGHT:
        return _Pose(pose.x, pose.y, (pose.direction + 1) % 4)

    delta_x, delta_y = _DIRECTION_TO_VECTOR[pose.direction]
    next_x = pose.x + delta_x
    next_y = pose.y + delta_y
    if not (
        0 <= next_x < state.grid.width and 0 <= next_y < state.grid.height
    ):
        return pose
    cell = state.grid.get(next_x, next_y)
    if cell is not None and not cell.can_overlap():
        return pose
    return _Pose(next_x, next_y, pose.direction)


def route_to_pose(
    state: MiniGridState,
    *,
    target_position: tuple[int, int],
    target_direction: int,
    context: str,
) -> str:
    """Find a shortest L/R/F route through one fixed family layout.

    The search state is the complete movement-relevant pose. Scenario
    builders call this only during phases where the grid is fixed, so object
    and carrying state cannot distinguish two otherwise equal search nodes.
    """
    if not 0 <= target_direction < len(_DIRECTION_TO_VECTOR):
        msg = f"{context}: target direction {target_direction!r} is invalid"
        raise ValueError(msg)
    target_x, target_y = target_position
    if not (
        0 <= target_x < state.grid.width and 0 <= target_y < state.grid.height
    ):
        msg = f"{context}: target position {target_position!r} is outside grid"
        raise ValueError(msg)
    target_cell = state.grid.get(target_x, target_y)
    if target_cell is not None and not target_cell.can_overlap():
        msg = f"{context}: target position {target_position!r} is blocked"
        raise PlanningError(msg)

    start = _Pose(*state.agent_position, state.agent_direction)
    target = _Pose(target_x, target_y, target_direction)
    frontier = deque([start])
    previous: dict[_Pose, tuple[_Pose, Action] | None] = {start: None}
    expansions = 0

    while frontier:
        current = frontier.popleft()
        if current == target:
            actions: list[Action] = []
            while (edge := previous[current]) is not None:
                current, action = edge
                actions.append(action)
            actions.reverse()
            return "".join(action.value for action in actions)

        expansions += 1
        if expansions > MAX_SEARCH_EXPANSIONS:
            msg = (
                f"{context}: route search exceeded "
                f"{MAX_SEARCH_EXPANSIONS} expansions"
            )
            raise PlanningError(msg)
        for action in _ROUTE_ACTIONS:
            candidate = _next_pose(state, current, action)
            if candidate == current or candidate in previous:
                continue
            previous[candidate] = (current, action)
            frontier.append(candidate)

    msg = (
        f"{context}: no route from {state.agent_position!r}/"
        f"{state.agent_direction} to {target_position!r}/{target_direction}"
    )
    raise PlanningError(msg)
