from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class Action(StrEnum):
    # Validate membership directly; never iterate to build command payloads.
    LEFT = "L"
    RIGHT = "R"
    FORWARD = "F"
    PICKUP = "P"
    DROP = "D"
    TOGGLE = "T"


ACTION_ALPHABET: tuple[Action, ...] = (
    Action.LEFT,
    Action.RIGHT,
    Action.FORWARD,
    Action.PICKUP,
    Action.DROP,
    Action.TOGGLE,
)
MAX_COMMAND_LENGTH = 32


@verify(UNIQUE)
class C19Fact(StrEnum):
    # Validate membership directly; never iterate to build prompt payloads.
    COORDINATE = "coordinate"
    HEADING = "heading"
    FRONT = "front"
    CARRYING = "carrying"


@verify(UNIQUE)
class Heading(StrEnum):
    # Values are the answers exposed by heading questions.
    EAST = "E"
    SOUTH = "S"
    WEST = "W"
    NORTH = "N"


@verify(UNIQUE)
class ObjectKind(StrEnum):
    # Validate membership directly; never iterate to build world payloads.
    WALL = "wall"
    KEY = "key"
    BALL = "ball"
    DOOR = "door"


@verify(UNIQUE)
class Color(StrEnum):
    # These are MiniGrid 3.1.0's complete concrete object colors.
    BLUE = "blue"
    GREEN = "green"
    GREY = "grey"
    PURPLE = "purple"
    RED = "red"
    YELLOW = "yellow"


@verify(UNIQUE)
class DoorState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    LOCKED = "locked"


@dataclass(frozen=True, slots=True)
class ObjectSnapshot:
    kind: ObjectKind
    color: Color
    door_state: DoorState | None = None

    def __post_init__(self) -> None:
        if self.kind is ObjectKind.DOOR:
            if self.door_state is None:
                msg = "door snapshots must include door state"
                raise ValueError(msg)
        elif self.door_state is not None:
            msg = "only door snapshots may include door state"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CellSnapshot:
    row: int
    column: int
    object: ObjectSnapshot | None


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    width: int
    height: int
    agent_row: int
    agent_column: int
    heading: Heading
    carrying: ObjectSnapshot | None
    cells: tuple[CellSnapshot, ...]


def parse_command(command: str) -> tuple[Action, ...]:
    """Parse one complete, bounded C19 command without normalization."""
    if not isinstance(command, str):
        msg = "command must be a string"
        raise TypeError(msg)
    if not command:
        msg = "command must contain at least one action"
        raise ValueError(msg)
    if len(command) > MAX_COMMAND_LENGTH:
        msg = f"command must contain at most {MAX_COMMAND_LENGTH} actions"
        raise ValueError(msg)

    actions: list[Action] = []
    for index, character in enumerate(command):
        try:
            actions.append(Action(character))
        except ValueError as error:
            msg = (
                f"command contains unsupported action {character!r} "
                f"at index {index}"
            )
            raise ValueError(msg) from error
    return tuple(actions)
