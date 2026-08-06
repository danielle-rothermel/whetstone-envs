from __future__ import annotations

import pytest

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
from whetstone_envs.c19.oracle import OracleError, derive_fact, simulate

SIMPLE_WORLD = "WGWGWGWGWG\nWG>>    WG\nWGWGWGWGWG"


@pytest.mark.parametrize(
    ("command", "expected_row", "expected_column", "expected_heading"),
    [
        ("L", 1, 1, Heading.NORTH),
        ("R", 1, 1, Heading.SOUTH),
        ("F", 1, 2, Heading.EAST),
        ("FFFFR", 1, 3, Heading.SOUTH),
    ],
)
def test_rotation_forward_and_blocked_forward(
    command: str,
    expected_row: int,
    expected_column: int,
    expected_heading: Heading,
) -> None:
    result = simulate(SIMPLE_WORLD, command)

    assert result.agent_row == expected_row
    assert result.agent_column == expected_column
    assert result.heading is expected_heading


@pytest.mark.parametrize(
    ("token", "command", "expected_kind"),
    [
        ("KB", "P", ObjectKind.KEY),
        ("AR", "P", ObjectKind.BALL),
    ],
)
def test_pickup_removes_supported_object_and_fills_hands(
    token: str,
    command: str,
    expected_kind: ObjectKind,
) -> None:
    world = "\n".join(("WGWGWGWG", f"WG>>{token}WG", "WGWGWGWG"))

    result = simulate(world, command)

    assert result.carrying is not None
    assert result.carrying.kind is expected_kind
    assert result.cells[1 * result.width + 2].object is None


def test_pickup_with_full_hands_is_a_noop() -> None:
    world = "WGWGWGWGWG\nWG>>KBARWG\nWGWGWGWGWG"

    result = simulate(world, "PFP")

    assert result.carrying == ObjectSnapshot(ObjectKind.KEY, Color.BLUE)
    assert result.cells[1 * result.width + 3].object == ObjectSnapshot(
        ObjectKind.BALL,
        Color.RED,
    )


def test_drop_requires_an_exactly_empty_cell() -> None:
    empty_world = "WGWGWGWG\nWG>>KBWG\nWG  WGWG\nWGWGWGWG"
    occupied_world = "WGWGWGWG\nWG>>KBWG\nWGWGWGWG\nWGWGWGWG"

    dropped = simulate(empty_world, "PRD")
    blocked = simulate(occupied_world, "PRD")

    assert dropped.carrying is None
    assert dropped.cells[2 * dropped.width + 1].object == ObjectSnapshot(
        ObjectKind.KEY,
        Color.BLUE,
    )
    assert blocked.carrying == ObjectSnapshot(ObjectKind.KEY, Color.BLUE)


def test_closed_door_can_open_move_and_close_again() -> None:
    world = "WGWGWGWGWG\nWG>>DB  WG\nWGWGWGWGWG"

    opened = simulate(world, "T")
    crossed = simulate(world, "TF")
    closed = simulate(world, "TFRRFRRT")

    assert opened.cells[1 * opened.width + 2].object == ObjectSnapshot(
        ObjectKind.DOOR,
        Color.BLUE,
        DoorState.OPEN,
    )
    assert (crossed.agent_row, crossed.agent_column) == (1, 2)
    assert closed.cells[1 * closed.width + 2].object == ObjectSnapshot(
        ObjectKind.DOOR,
        Color.BLUE,
        DoorState.CLOSED,
    )


@pytest.mark.parametrize(
    ("key_cell", "expected_key_color", "expected_state"),
    [
        ("KB", Color.BLUE, DoorState.OPEN),
        ("KR", Color.RED, DoorState.LOCKED),
    ],
)
def test_locked_door_requires_a_matching_concrete_key_color(
    key_cell: str,
    expected_key_color: Color,
    expected_state: DoorState,
) -> None:
    world = "\n".join(
        (
            "WGWGWGWGWGWG",
            f"WG>>{key_cell}LB  WG",
            "WGWGWGWGWGWG",
        ),
    )

    result = simulate(world, "PFT")

    assert result.cells[1 * result.width + 3].object == ObjectSnapshot(
        ObjectKind.DOOR,
        Color.BLUE,
        expected_state,
    )
    assert result.carrying == ObjectSnapshot(
        ObjectKind.KEY,
        expected_key_color,
    )


@pytest.mark.parametrize(
    "command",
    ["", "f", " F", "F ", "F?", "F\n", "L" * 33],
)
def test_command_parser_is_strict(command: str) -> None:
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


def test_command_parser_reports_invalid_character_and_index() -> None:
    with pytest.raises(
        ValueError,
        match=r"unsupported action '\?' at index 2",
    ):
        parse_command("LF?")


@pytest.mark.parametrize(
    "grid_text",
    [
        "",
        ">",
        ">>\n    ",
        ">>ZZ",
        ">>WG\nWG",
        ">>__",
        "> KB",
    ],
)
def test_grid_parser_requires_complete_unambiguous_tokens(
    grid_text: str,
) -> None:
    with pytest.raises(OracleError):
        simulate(grid_text, "L")


def test_grid_parser_rejects_initial_open_door_without_color() -> None:
    world = "WGWGWGWG\nWG>>__WG\nWGWGWGWG"

    with pytest.raises(
        OracleError,
        match="does not preserve a concrete door color",
    ):
        simulate(world, "F")


@pytest.mark.parametrize(
    "grid_text",
    [
        "WGWGWG\nWG  WG\nWGWGWG",
        "WGWGWG\nWG>>WG\nWG>>WG",
    ],
)
def test_grid_parser_requires_exactly_one_agent(grid_text: str) -> None:
    with pytest.raises(OracleError, match="exactly one agent"):
        simulate(grid_text, "L")


@pytest.mark.parametrize(
    "grid_text",
    [
        "  WGWG\nWG>>WG\nWGWGWG",
        "WGWGWG\n  >>WG\nWGWGWG",
        "WG>>WG\nWG  WG\nWGWGWG",
    ],
)
def test_grid_parser_requires_a_wall_perimeter(grid_text: str) -> None:
    with pytest.raises(OracleError, match="perimeter"):
        simulate(grid_text, "L")


def test_whole_state_snapshot_preserves_all_semantics() -> None:
    world = "WGWGWGWGWG\nWG>>KRLBWG\nWG  AP  WG\nWGWGWGWGWG"

    result = simulate(world, "PFT")

    wall = ObjectSnapshot(ObjectKind.WALL, Color.GREY)
    red_key = ObjectSnapshot(ObjectKind.KEY, Color.RED)
    locked_blue_door = ObjectSnapshot(
        ObjectKind.DOOR,
        Color.BLUE,
        DoorState.LOCKED,
    )
    purple_ball = ObjectSnapshot(ObjectKind.BALL, Color.PURPLE)
    expected_rows = (
        (wall, wall, wall, wall, wall),
        (wall, None, None, locked_blue_door, wall),
        (wall, None, purple_ball, None, wall),
        (wall, wall, wall, wall, wall),
    )
    assert result == WorldSnapshot(
        width=5,
        height=4,
        agent_row=1,
        agent_column=2,
        heading=Heading.EAST,
        carrying=red_key,
        cells=tuple(
            CellSnapshot(row=row, column=column, object=world_object)
            for row, objects in enumerate(expected_rows)
            for column, world_object in enumerate(objects)
        ),
    )


@pytest.mark.parametrize(
    ("fact", "expected"),
    [
        (C19Fact.COORDINATE, "1,2"),
        (C19Fact.HEADING, "E"),
        (C19Fact.CARRYING, "no"),
        (C19Fact.FRONT, "empty"),
    ],
)
def test_derive_fact_reads_only_the_complete_final_snapshot(
    fact: C19Fact,
    expected: str,
) -> None:
    assert derive_fact(SIMPLE_WORLD, "F", fact) == expected
