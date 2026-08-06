from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.c19.model import (
    Action,
    C19Fact,
    CellSnapshot,
    Color,
    DoorState,
    Heading,
    ObjectKind,
    ObjectSnapshot,
    WorldSnapshot,
    parse_command,
)

_HEADING_TO_VECTOR: dict[Heading, tuple[int, int]] = {
    Heading.EAST: (0, 1),
    Heading.SOUTH: (1, 0),
    Heading.WEST: (0, -1),
    Heading.NORTH: (-1, 0),
}
_LEFT_HEADING: dict[Heading, Heading] = {
    Heading.EAST: Heading.NORTH,
    Heading.SOUTH: Heading.EAST,
    Heading.WEST: Heading.SOUTH,
    Heading.NORTH: Heading.WEST,
}
_RIGHT_HEADING: dict[Heading, Heading] = {
    Heading.EAST: Heading.SOUTH,
    Heading.SOUTH: Heading.WEST,
    Heading.WEST: Heading.NORTH,
    Heading.NORTH: Heading.EAST,
}
_AGENT_TOKENS: dict[str, Heading] = {
    ">>": Heading.EAST,
    "VV": Heading.SOUTH,
    "<<": Heading.WEST,
    "^^": Heading.NORTH,
}
_COLOR_INITIALS: dict[str, Color] = {
    "B": Color.BLUE,
    "P": Color.PURPLE,
    "R": Color.RED,
    "Y": Color.YELLOW,
}
_EMPTY_CELL = "  "
_OPEN_DOOR_CELL = "__"


class OracleError(ValueError):
    """The public grid text cannot identify one supported C19 world."""


@dataclass(slots=True)
class _OracleState:
    width: int
    height: int
    agent_row: int
    agent_column: int
    heading: Heading
    cells: list[list[ObjectSnapshot | None]]
    carrying: ObjectSnapshot | None = None


def _color(token: str, kind: ObjectKind) -> Color:
    initial = token[1]
    if initial == "G":
        # MiniGrid pprint_grid collapses green and grey to G. The supported
        # generated subset uses grey walls and green movable/door objects.
        return Color.GREY if kind is ObjectKind.WALL else Color.GREEN
    try:
        return _COLOR_INITIALS[initial]
    except KeyError as error:
        msg = f"unrecognized color in grid token {token!r}"
        raise OracleError(msg) from error


def _parse_object(cell_text: str) -> ObjectSnapshot:
    if cell_text == _OPEN_DOOR_CELL:
        msg = (
            "initial open-door token '__' does not preserve a concrete "
            "door color"
        )
        raise OracleError(msg)

    prefix = cell_text[0]
    if prefix == "W":
        kind = ObjectKind.WALL
        door_state = None
    elif prefix == "K":
        kind = ObjectKind.KEY
        door_state = None
    elif prefix == "A":
        kind = ObjectKind.BALL
        door_state = None
    elif prefix == "D":
        kind = ObjectKind.DOOR
        door_state = DoorState.CLOSED
    elif prefix == "L":
        kind = ObjectKind.DOOR
        door_state = DoorState.LOCKED
    else:
        msg = f"unrecognized grid token {cell_text!r}"
        raise OracleError(msg)
    return ObjectSnapshot(
        kind=kind,
        color=_color(cell_text, kind),
        door_state=door_state,
    )


def _parse_grid(grid_text: str) -> _OracleState:
    if not isinstance(grid_text, str):
        msg = "grid text must be a string"
        raise TypeError(msg)
    rows = grid_text.split("\n")
    if not rows or not rows[0]:
        msg = "grid must contain at least one complete row"
        raise OracleError(msg)
    if len(rows[0]) % 2:
        msg = "grid rows must contain complete two-character tokens"
        raise OracleError(msg)
    row_width = len(rows[0])
    if any(len(row) != row_width for row in rows):
        msg = "grid rows must have equal width"
        raise OracleError(msg)

    agent: tuple[int, int, Heading] | None = None
    cells: list[list[ObjectSnapshot | None]] = []
    for row_index, row in enumerate(rows):
        parsed_row: list[ObjectSnapshot | None] = []
        for column_index in range(row_width // 2):
            cell_text = row[2 * column_index : 2 * column_index + 2]
            if cell_text == _EMPTY_CELL:
                parsed_row.append(None)
            elif cell_text in _AGENT_TOKENS:
                if agent is not None:
                    msg = "grid must contain exactly one agent"
                    raise OracleError(msg)
                agent = (
                    row_index,
                    column_index,
                    _AGENT_TOKENS[cell_text],
                )
                parsed_row.append(None)
            else:
                parsed_row.append(_parse_object(cell_text))
        cells.append(parsed_row)
    if agent is None:
        msg = "grid must contain exactly one agent"
        raise OracleError(msg)
    perimeter = (
        cells[0]
        + cells[-1]
        + [row[0] for row in cells[1:-1]]
        + [row[-1] for row in cells[1:-1]]
    )
    if any(
        world_object is None or world_object.kind is not ObjectKind.WALL
        for world_object in perimeter
    ):
        msg = "grid perimeter must contain only walls"
        raise OracleError(msg)
    agent_row, agent_column, heading = agent
    return _OracleState(
        width=row_width // 2,
        height=len(rows),
        agent_row=agent_row,
        agent_column=agent_column,
        heading=heading,
        cells=cells,
    )


def _front_position(state: _OracleState) -> tuple[int, int]:
    row_delta, column_delta = _HEADING_TO_VECTOR[state.heading]
    return (
        state.agent_row + row_delta,
        state.agent_column + column_delta,
    )


def _in_bounds(state: _OracleState, row: int, column: int) -> bool:
    return 0 <= row < state.height and 0 <= column < state.width


def _toggle(state: _OracleState, row: int, column: int) -> None:
    front = state.cells[row][column]
    if front is None or front.kind is not ObjectKind.DOOR:
        return
    if front.door_state is DoorState.LOCKED:
        carrying_matching_key = (
            state.carrying is not None
            and state.carrying.kind is ObjectKind.KEY
            and state.carrying.color is front.color
        )
        if carrying_matching_key:
            state.cells[row][column] = ObjectSnapshot(
                kind=ObjectKind.DOOR,
                color=front.color,
                door_state=DoorState.OPEN,
            )
    else:
        next_state = (
            DoorState.CLOSED
            if front.door_state is DoorState.OPEN
            else DoorState.OPEN
        )
        state.cells[row][column] = ObjectSnapshot(
            kind=ObjectKind.DOOR,
            color=front.color,
            door_state=next_state,
        )


def _apply_action(state: _OracleState, action: Action) -> None:
    if action is Action.LEFT:
        state.heading = _LEFT_HEADING[state.heading]
        return
    if action is Action.RIGHT:
        state.heading = _RIGHT_HEADING[state.heading]
        return

    front_row, front_column = _front_position(state)
    if not _in_bounds(state, front_row, front_column):
        return
    front = state.cells[front_row][front_column]

    if action is Action.FORWARD:
        can_overlap = front is None or (
            front.kind is ObjectKind.DOOR
            and front.door_state is DoorState.OPEN
        )
        if can_overlap:
            state.agent_row = front_row
            state.agent_column = front_column
    elif action is Action.PICKUP:
        pickup_kind = front is not None and front.kind in {
            ObjectKind.KEY,
            ObjectKind.BALL,
        }
        if pickup_kind and state.carrying is None:
            state.carrying = front
            state.cells[front_row][front_column] = None
    elif action is Action.DROP:
        if front is None and state.carrying is not None:
            state.cells[front_row][front_column] = state.carrying
            state.carrying = None
    elif action is Action.TOGGLE:
        _toggle(state, front_row, front_column)


def _snapshot(state: _OracleState) -> WorldSnapshot:
    return WorldSnapshot(
        width=state.width,
        height=state.height,
        agent_row=state.agent_row,
        agent_column=state.agent_column,
        heading=state.heading,
        carrying=state.carrying,
        cells=tuple(
            CellSnapshot(row=row, column=column, object=world_object)
            for row in range(state.height)
            for column, world_object in enumerate(state.cells[row])
        ),
    )


def simulate(grid_text: str, command: str) -> WorldSnapshot:
    """Independently apply a complete command to one public ASCII world."""
    actions = parse_command(command)
    state = _parse_grid(grid_text)
    for action in actions:
        _apply_action(state, action)
    return _snapshot(state)


def derive_fact(grid_text: str, command: str, fact: C19Fact) -> str:
    """Derive one answer from supported answer-relevant final state."""
    if not isinstance(fact, C19Fact):
        msg = f"fact must be a C19Fact, got {fact!r}"
        raise TypeError(msg)
    final = simulate(grid_text, command)
    if fact is C19Fact.COORDINATE:
        return f"{final.agent_row},{final.agent_column}"
    if fact is C19Fact.HEADING:
        return final.heading.value
    if fact is C19Fact.CARRYING:
        return "yes" if final.carrying is not None else "no"

    row_delta, column_delta = _HEADING_TO_VECTOR[final.heading]
    front_row = final.agent_row + row_delta
    front_column = final.agent_column + column_delta
    if not (0 <= front_row < final.height and 0 <= front_column < final.width):
        return ObjectKind.WALL.value
    cell_index = front_row * final.width + front_column
    front = final.cells[cell_index].object
    return "empty" if front is None else front.kind.value
