from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Door, Key, Wall

from whetstone_envs.c19._minigrid import (
    MiniGridState,
    run_script,
)
from whetstone_envs.c19.model import (
    CellSnapshot,
    Color,
    DoorState,
    Heading,
    ObjectKind,
    ObjectSnapshot,
    WorldSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from minigrid.core.world_object import WorldObj


def _state(
    *,
    front: Callable[[], WorldObj] | None = None,
    carrying: Callable[[], WorldObj] | None = None,
) -> MiniGridState:
    grid = Grid(4, 3)
    if front is not None:
        grid.set(2, 1, front())
    return MiniGridState(
        grid=grid,
        agent_position=(1, 1),
        agent_direction=0,
        carrying=None if carrying is None else carrying(),
    )


@pytest.mark.parametrize(
    (
        "command",
        "front",
        "carrying",
        "expected_position",
        "expected_direction",
        "expected_front_kind",
        "expected_carrying_kind",
    ),
    [
        ("L", None, None, (1, 1), 3, None, None),
        ("R", None, None, (1, 1), 1, None, None),
        ("F", None, None, (2, 1), 0, None, None),
        ("F", Wall, None, (1, 1), 0, ObjectKind.WALL, None),
        ("P", Key, None, (1, 1), 0, None, ObjectKind.KEY),
        ("P", Ball, None, (1, 1), 0, None, ObjectKind.BALL),
        (
            "P",
            Key,
            Ball,
            (1, 1),
            0,
            ObjectKind.KEY,
            ObjectKind.BALL,
        ),
        ("P", Wall, None, (1, 1), 0, ObjectKind.WALL, None),
        ("D", None, Key, (1, 1), 0, ObjectKind.KEY, None),
        ("D", Wall, Key, (1, 1), 0, ObjectKind.WALL, ObjectKind.KEY),
        ("D", None, None, (1, 1), 0, None, None),
        ("T", Wall, None, (1, 1), 0, ObjectKind.WALL, None),
    ],
)
def test_action_effects_and_failed_preconditions(  # noqa: PLR0913
    command: str,
    front: Callable[[], WorldObj] | None,
    carrying: Callable[[], WorldObj] | None,
    expected_position: tuple[int, int],
    expected_direction: int,
    expected_front_kind: ObjectKind | None,
    expected_carrying_kind: ObjectKind | None,
) -> None:
    state = _state(front=front, carrying=carrying)

    result = run_script(state, command)

    assert state.agent_position == expected_position
    assert state.agent_direction == expected_direction
    front_snapshot = result.cells[1 * result.width + 2].object
    assert (
        None if front_snapshot is None else front_snapshot.kind
    ) is expected_front_kind
    assert (
        None if result.carrying is None else result.carrying.kind
    ) is expected_carrying_kind


@pytest.mark.parametrize(
    ("door", "carrying", "expected_state"),
    [
        (
            lambda: Door("blue"),
            None,
            DoorState.OPEN,
        ),
        (
            lambda: Door("blue", is_open=True),
            None,
            DoorState.CLOSED,
        ),
        (
            lambda: Door("blue", is_locked=True),
            lambda: Key("blue"),
            DoorState.OPEN,
        ),
        (
            lambda: Door("blue", is_locked=True),
            lambda: Key("red"),
            DoorState.LOCKED,
        ),
        (
            lambda: Door("blue", is_locked=True),
            None,
            DoorState.LOCKED,
        ),
    ],
)
def test_toggle_uses_minigrid_door_semantics(
    door: Callable[[], WorldObj],
    carrying: Callable[[], WorldObj] | None,
    expected_state: DoorState,
) -> None:
    state = _state(front=door, carrying=carrying)

    result = run_script(state, "T")

    front = result.cells[1 * result.width + 2].object
    assert front is not None
    assert front.door_state is expected_state


def test_forward_overlaps_only_an_open_door() -> None:
    open_state = _state(front=lambda: Door("yellow", is_open=True))
    closed_state = _state(front=lambda: Door("yellow"))

    open_result = run_script(open_state, "F")
    closed_result = run_script(closed_state, "F")

    assert (open_result.agent_row, open_result.agent_column) == (1, 2)
    assert (closed_result.agent_row, closed_result.agent_column) == (1, 1)


def test_run_script_validates_the_whole_command_before_mutation() -> None:
    state = _state()

    with pytest.raises(ValueError):
        run_script(state, "F?")

    assert state.agent_position == (1, 1)
    assert state.agent_direction == 0


def test_run_script_applies_actions_after_blocked_forward_moves() -> None:
    state = _state(front=Wall)

    result = run_script(state, "FFFFR")

    assert (result.agent_row, result.agent_column) == (1, 1)
    assert result.heading is Heading.SOUTH


def test_transition_path_never_calls_environment_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minigrid.minigrid_env import MiniGridEnv

    def unexpected_step(*_args: object, **_kwargs: object) -> None:
        pytest.fail("MiniGridEnv.step must not drive C19 transitions")

    monkeypatch.setattr(MiniGridEnv, "step", unexpected_step)
    state = _state()

    result = run_script(state, "FRF")

    assert (result.agent_row, result.agent_column) == (2, 2)


def test_supported_answer_relevant_physical_state_preserves_semantics() -> (
    None
):
    grid = Grid(4, 3)
    grid.set(0, 0, Wall())
    grid.set(3, 0, Door("purple", is_locked=True))
    grid.set(2, 1, Key("red"))
    state = MiniGridState(
        grid=grid,
        agent_position=(1, 1),
        agent_direction=0,
    )

    result = run_script(state, "PPRD")

    wall = ObjectSnapshot(ObjectKind.WALL, Color.GREY)
    locked_door = ObjectSnapshot(
        ObjectKind.DOOR,
        Color.PURPLE,
        DoorState.LOCKED,
    )
    red_key = ObjectSnapshot(ObjectKind.KEY, Color.RED)
    expected_cells = tuple(
        CellSnapshot(row=row, column=column, object=world_object)
        for row, objects in enumerate(
            (
                (wall, None, None, locked_door),
                (None, None, None, None),
                (None, red_key, None, None),
            ),
        )
        for column, world_object in enumerate(objects)
    )
    assert result == WorldSnapshot(
        width=4,
        height=3,
        agent_row=1,
        agent_column=1,
        heading=Heading.SOUTH,
        carrying=None,
        cells=expected_cells,
    )
