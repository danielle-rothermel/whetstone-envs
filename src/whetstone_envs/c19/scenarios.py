from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, IntEnum, StrEnum, verify

from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Door, Key, WorldObj

from whetstone_envs.c19._minigrid import (
    MiniGridState,
    clone_state,
    run_script,
)
from whetstone_envs.c19.model import MAX_COMMAND_LENGTH, Color, parse_command
from whetstone_envs.c19.planning import PlanningError, route_to_pose


@verify(UNIQUE)
class C19Scenario(StrEnum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    DOOR = "door"


@verify(UNIQUE)
class C19Size(IntEnum):
    SMALL = 5
    MEDIUM = 8


SCENARIO_ORDER: tuple[C19Scenario, ...] = (
    C19Scenario.NAVIGATION,
    C19Scenario.MANIPULATION,
    C19Scenario.DOOR,
)
SIZE_ORDER: tuple[C19Size, ...] = (
    C19Size.SMALL,
    C19Size.MEDIUM,
)
_OBJECT_COLORS: tuple[Color, ...] = (
    Color.BLUE,
    Color.PURPLE,
    Color.RED,
    Color.YELLOW,
)
_DIRECTION_TO_VECTOR: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)
_QUARTER_TURNS = 4
_REFLECTION_START = 4


@dataclass(slots=True)
class BuiltScenario:
    state: MiniGridState
    command: str


def _context(
    scenario: C19Scenario,
    size: C19Size,
    seed: int,
) -> str:
    return f"family={scenario.value} size={size.name.lower()} seed={seed}"


def _empty_state(
    size: C19Size,
    *,
    agent_position: tuple[int, int],
    agent_direction: int,
) -> MiniGridState:
    side = int(size)
    grid = Grid(side, side)
    grid.wall_rect(0, 0, side, side)
    return MiniGridState(
        grid=grid,
        agent_position=agent_position,
        agent_direction=agent_direction,
    )


def _put(grid: Grid, position: tuple[int, int], obj: WorldObj) -> None:
    obj.init_pos = position
    obj.cur_pos = position
    grid.set(*position, obj)


def _poses(size: C19Size) -> tuple[tuple[tuple[int, int], int], ...]:
    side = int(size)
    return tuple(
        ((x, y), direction)
        for y in range(1, side - 1)
        for x in range(1, side - 1)
        for direction in (0, 1, 2, 3)
    )


def _directed_interior_edges(
    size: C19Size,
) -> tuple[tuple[tuple[int, int], int], ...]:
    side = int(size)
    edges: list[tuple[tuple[int, int], int]] = []
    for y in range(1, side - 1):
        for x in range(1, side - 1):
            for direction, (delta_x, delta_y) in enumerate(
                _DIRECTION_TO_VECTOR,
            ):
                front_x = x + delta_x
                front_y = y + delta_y
                if 1 <= front_x < side - 1 and 1 <= front_y < side - 1:
                    edges.append(((x, y), direction))
    return tuple(edges)


def _wall_target(
    size: C19Size,
    selector: int,
) -> tuple[tuple[int, int], int]:
    last = int(size) - 2
    return (
        ((1, 1), 3),
        ((last, 1), 0),
        ((last, last), 1),
        ((1, last), 2),
    )[selector % 4]


def _bounded_command(command: str, *, context: str) -> str:
    try:
        parse_command(command)
    except (TypeError, ValueError) as error:
        msg = f"{context}: invalid constructive command: {error}"
        raise PlanningError(msg) from error
    if len(command) > MAX_COMMAND_LENGTH:
        msg = (
            f"{context}: constructive command has {len(command)} actions, "
            f"limit is {MAX_COMMAND_LENGTH}"
        )
        raise PlanningError(msg)
    return command


def _navigation(size: C19Size, seed: int) -> BuiltScenario:
    poses = _poses(size)
    position, direction = poses[seed % len(poses)]
    state = _empty_state(
        size,
        agent_position=position,
        agent_direction=direction,
    )
    target_position, target_direction = _wall_target(
        size,
        seed // len(poses),
    )
    context = _context(C19Scenario.NAVIGATION, size, seed)
    route = route_to_pose(
        state,
        target_position=target_position,
        target_direction=target_direction,
        context=context,
    )
    # F is blocked at the wall. R then enables a successful tangential move;
    # L faces the wall again, making the final F blocked for a second direct
    # movement-precondition witness.
    command = _bounded_command(f"{route}FRFLF", context=context)
    return BuiltScenario(state=state, command=command)


def _manipulation(
    size: C19Size,
    seed: int,
    *,
    carrying: bool,
) -> BuiltScenario:
    edges = _directed_interior_edges(size)
    variant = seed % (len(edges) * len(_OBJECT_COLORS))
    edge_index, color_index = divmod(variant, len(_OBJECT_COLORS))
    position, direction = edges[edge_index]
    state = _empty_state(
        size,
        agent_position=position,
        agent_direction=direction,
    )
    delta_x, delta_y = _DIRECTION_TO_VECTOR[direction]
    object_position = (position[0] + delta_x, position[1] + delta_y)
    _put(state.grid, object_position, Ball(_OBJECT_COLORS[color_index].value))

    context = _context(C19Scenario.MANIPULATION, size, seed)
    planning_state = clone_state(state)
    run_script(planning_state, "P")
    if planning_state.carrying is None:
        msg = f"{context}: planned initial pickup did not succeed"
        raise PlanningError(msg)
    target_position, target_direction = _wall_target(
        size,
        seed // (len(edges) * len(_OBJECT_COLORS)),
    )
    route = route_to_pose(
        planning_state,
        target_position=target_position,
        target_direction=target_direction,
        context=context,
    )
    suffix = "FRFLF" if carrying else "FRFLFLLD"
    command = _bounded_command(f"P{route}{suffix}", context=context)
    return BuiltScenario(state=state, command=command)


def _transform_point(
    point: tuple[int, int],
    *,
    side: int,
    transform: int,
) -> tuple[int, int]:
    x, y = point
    if transform >= _REFLECTION_START:
        x = side - 1 - x
    for _ in range(transform % _QUARTER_TURNS):
        x, y = side - 1 - y, x
    return x, y


def _transform_direction(direction: int, *, transform: int) -> int:
    if transform >= _REFLECTION_START:
        direction = (2, 1, 0, 3)[direction]
    return (direction + transform % _QUARTER_TURNS) % _QUARTER_TURNS


def _door(
    size: C19Size,
    seed: int,
    *,
    carrying: bool,
) -> BuiltScenario:
    variant = seed % 128
    transform = variant % 8
    locked_color = _OBJECT_COLORS[(variant // 8) % len(_OBJECT_COLORS)]
    closed_color = _OBJECT_COLORS[(variant // 32) % len(_OBJECT_COLORS)]
    side = int(size)

    position = _transform_point((1, 1), side=side, transform=transform)
    direction = _transform_direction(0, transform=transform)
    state = _empty_state(
        size,
        agent_position=position,
        agent_direction=direction,
    )
    key_position = _transform_point(
        (2, 1),
        side=side,
        transform=transform,
    )
    locked_position = _transform_point(
        (2, 2),
        side=side,
        transform=transform,
    )
    closed_position = _transform_point(
        (2, 3),
        side=side,
        transform=transform,
    )
    _put(state.grid, key_position, Key(locked_color.value))
    _put(
        state.grid,
        locked_position,
        Door(locked_color.value, is_locked=True),
    )
    _put(state.grid, closed_position, Door(closed_color.value))

    # The base path picks up the matching key, traverses a locked door, then
    # traverses a closed unlocked door.
    command = "PFRTFTF"
    if transform >= _REFLECTION_START:
        command = command.translate(str.maketrans({"L": "R", "R": "L"}))

    planning_state = clone_state(state)
    run_script(planning_state, command)
    target_position = _transform_point(
        (side - 2, side - 2),
        side=side,
        transform=transform,
    )
    target_direction = _transform_direction(1, transform=transform)
    context = _context(C19Scenario.DOOR, size, seed)
    route = route_to_pose(
        planning_state,
        target_position=target_position,
        target_direction=target_direction,
        context=context,
    )
    movement_suffix = "FRFLF"
    if transform >= _REFLECTION_START:
        movement_suffix = movement_suffix.translate(
            str.maketrans({"L": "R", "R": "L"}),
        )
    command = f"{command}{route}{movement_suffix}"
    if not carrying:
        # Face the empty cell just traversed and finish with a successful
        # drop. The reflected construction reverses handedness.
        face_empty = "R" if transform >= _REFLECTION_START else "L"
        command = f"{command}{face_empty}D"
    return BuiltScenario(
        state=state,
        command=_bounded_command(command, context=context),
    )


def build_scenario(
    scenario: C19Scenario,
    size: C19Size,
    seed: int,
    *,
    carrying: bool | None,
) -> BuiltScenario:
    """Build one deterministic custom C19 world and constructive command."""
    if not isinstance(scenario, C19Scenario):
        msg = f"scenario must be a C19Scenario, got {scenario!r}"
        raise TypeError(msg)
    if not isinstance(size, C19Size):
        msg = f"size must be a C19Size, got {size!r}"
        raise TypeError(msg)
    if type(seed) is not int:
        msg = "seed must be an int"
        raise TypeError(msg)
    if scenario is C19Scenario.NAVIGATION:
        if carrying is not None:
            msg = "navigation scenarios do not have carrying outcomes"
            raise ValueError(msg)
        return _navigation(size, seed)
    if type(carrying) is not bool:
        msg = f"{scenario.value} scenarios require a carrying outcome"
        raise TypeError(msg)
    if scenario is C19Scenario.MANIPULATION:
        return _manipulation(size, seed, carrying=carrying)
    return _door(size, seed, carrying=carrying)
