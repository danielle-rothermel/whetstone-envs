from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, WorldObj
from minigrid.minigrid_env import MiniGridEnv

from whetstone_envs.c19.model import (
    Action,
    CellSnapshot,
    Color,
    DoorState,
    Heading,
    ObjectKind,
    ObjectSnapshot,
    WorldSnapshot,
    parse_command,
)

_DIRECTION_TO_HEADING: tuple[Heading, ...] = (
    Heading.EAST,
    Heading.SOUTH,
    Heading.WEST,
    Heading.NORTH,
)
_DIRECTION_TO_VECTOR: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


@dataclass(slots=True)
class MiniGridState:
    """Mutable transition state over a concrete MiniGrid grid."""

    grid: Grid
    agent_position: tuple[int, int]
    agent_direction: int
    carrying: WorldObj | None = None


def _snapshot_object(world_object: WorldObj) -> ObjectSnapshot:
    try:
        kind = ObjectKind(world_object.type)
    except ValueError as error:
        msg = f"unsupported MiniGrid object type {world_object.type!r}"
        raise ValueError(msg) from error
    try:
        color = Color(world_object.color)
    except ValueError as error:
        msg = f"unsupported MiniGrid object color {world_object.color!r}"
        raise ValueError(msg) from error

    if kind is not ObjectKind.DOOR:
        return ObjectSnapshot(kind=kind, color=color)
    if not isinstance(world_object, Door):
        msg = "MiniGrid door objects must use Door"
        raise TypeError(msg)
    if world_object.is_open and world_object.is_locked:
        msg = "MiniGrid doors cannot be both open and locked"
        raise ValueError(msg)
    if world_object.is_open:
        door_state = DoorState.OPEN
    elif world_object.is_locked:
        door_state = DoorState.LOCKED
    else:
        door_state = DoorState.CLOSED
    return ObjectSnapshot(kind=kind, color=color, door_state=door_state)


def snapshot(state: MiniGridState) -> WorldSnapshot:
    """Read the complete supported state in row-major order."""
    x, y = state.agent_position
    if not (0 <= x < state.grid.width and 0 <= y < state.grid.height):
        msg = f"agent position {(x, y)!r} is outside the grid"
        raise ValueError(msg)
    if not 0 <= state.agent_direction < len(_DIRECTION_TO_HEADING):
        msg = f"agent direction {state.agent_direction!r} is invalid"
        raise ValueError(msg)

    cells = tuple(
        CellSnapshot(
            row=row,
            column=column,
            object=(
                None
                if (world_object := state.grid.get(column, row)) is None
                else _snapshot_object(world_object)
            ),
        )
        for row in range(state.grid.height)
        for column in range(state.grid.width)
    )
    carrying = (
        None if state.carrying is None else _snapshot_object(state.carrying)
    )
    return WorldSnapshot(
        width=state.grid.width,
        height=state.grid.height,
        agent_row=y,
        agent_column=x,
        heading=_DIRECTION_TO_HEADING[state.agent_direction],
        carrying=carrying,
        cells=cells,
    )


def clone_state(state: MiniGridState) -> MiniGridState:
    """Return an independent mutable copy of one supported live state."""
    return MiniGridState(
        grid=state.grid.copy(),
        agent_position=state.agent_position,
        agent_direction=state.agent_direction,
        carrying=deepcopy(state.carrying),
    )


@dataclass(slots=True)
class _PprintView:
    """The exact attributes MiniGrid 3.1.0's pprint_grid reads."""

    grid: Grid
    agent_pos: tuple[int, int]
    agent_dir: int


def pprint_grid(state: MiniGridState) -> str:
    """Render with MiniGrid 3.1.0's authoritative pprint encoding."""
    snapshot(state)
    view = _PprintView(
        grid=state.grid,
        agent_pos=state.agent_position,
        agent_dir=state.agent_direction,
    )
    # pprint_grid is deliberately duck-typed by MiniGrid and reads only the
    # three fields above. Calling the pinned implementation avoids a second
    # locally maintained ASCII encoding.
    return MiniGridEnv.pprint_grid(cast("MiniGridEnv", view))


def _front_position(state: MiniGridState) -> tuple[int, int]:
    delta_x, delta_y = _DIRECTION_TO_VECTOR[state.agent_direction]
    x, y = state.agent_position
    return x + delta_x, y + delta_y


def _in_bounds(state: MiniGridState, position: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < state.grid.width and 0 <= y < state.grid.height


def _apply_action(state: MiniGridState, action: Action) -> None:
    if action is Action.LEFT:
        state.agent_direction = (state.agent_direction - 1) % 4
        return
    if action is Action.RIGHT:
        state.agent_direction = (state.agent_direction + 1) % 4
        return

    front_position = _front_position(state)
    if not _in_bounds(state, front_position):
        return
    front = state.grid.get(*front_position)

    if action is Action.FORWARD:
        if front is None or front.can_overlap():
            state.agent_position = front_position
    elif action is Action.PICKUP:
        if front is not None and front.can_pickup() and state.carrying is None:
            state.carrying = front
            state.carrying.cur_pos = (-1, -1)
            state.grid.set(*front_position, None)
    elif action is Action.DROP:
        if front is None and state.carrying is not None:
            state.grid.set(*front_position, state.carrying)
            state.carrying.cur_pos = front_position
            state.carrying = None
    elif action is Action.TOGGLE and front is not None:
        front.toggle(cast("MiniGridEnv", state), front_position)


def run_script(state: MiniGridState, command: str) -> WorldSnapshot:
    """Apply every action directly to state, then return its full snapshot."""
    actions = parse_command(command)
    snapshot(state)
    for action in actions:
        _apply_action(state, action)
    return snapshot(state)


def trace_script(
    state: MiniGridState,
    command: str,
) -> tuple[WorldSnapshot, ...]:
    """Return the initial state followed by every nonempty prefix state."""
    actions = parse_command(command)
    trace = [snapshot(state)]
    for action in actions:
        _apply_action(state, action)
        trace.append(snapshot(state))
    return tuple(trace)
