import pytest

from whetstone_envs.c19.model import (
    Action,
    Color,
    DoorState,
    ObjectKind,
    ObjectSnapshot,
    parse_command,
)


@pytest.mark.parametrize(
    "command",
    ["", "f", " F", "F ", "F?", "F\n", "L" * 33],
)
def test_command_parser_rejects_malformed_commands(command: str) -> None:
    with pytest.raises(ValueError):
        parse_command(command)


def test_command_parser_preserves_the_complete_ordered_alphabet() -> None:
    assert parse_command("LRFPDT") == (
        Action.LEFT,
        Action.RIGHT,
        Action.FORWARD,
        Action.PICKUP,
        Action.DROP,
        Action.TOGGLE,
    )


@pytest.mark.parametrize(
    ("kind", "door_state"),
    [
        (ObjectKind.DOOR, None),
        (ObjectKind.KEY, DoorState.OPEN),
    ],
)
def test_object_snapshot_requires_door_state_only_for_doors(
    kind: ObjectKind,
    door_state: DoorState | None,
) -> None:
    with pytest.raises(ValueError):
        ObjectSnapshot(kind=kind, color=Color.BLUE, door_state=door_state)
