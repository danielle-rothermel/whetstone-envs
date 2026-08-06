from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from whetstone_envs.c19._minigrid import (
    clone_state,
    pprint_grid,
    snapshot,
    trace_script,
)
from whetstone_envs.c19.model import (
    MAX_COMMAND_LENGTH,
    Color,
    DoorState,
    ObjectKind,
)
from whetstone_envs.c19.oracle import simulate
from whetstone_envs.c19.scenarios import (
    SCENARIO_ORDER,
    SIZE_ORDER,
    C19Scenario,
    C19Size,
    build_scenario,
)


@pytest.mark.parametrize("scenario", SCENARIO_ORDER)
@pytest.mark.parametrize("size", SIZE_ORDER)
def test_custom_scenarios_have_exact_sizes_and_supported_initial_objects(
    scenario: C19Scenario,
    size: C19Size,
) -> None:
    carrying = None if scenario is C19Scenario.NAVIGATION else True
    built = build_scenario(scenario, size, 1_000_000, carrying=carrying)
    initial = snapshot(built.state)

    assert (initial.width, initial.height) == (int(size), int(size))
    assert initial.carrying is None
    for cell in initial.cells:
        obj = cell.object
        if obj is None:
            continue
        assert obj.kind in {
            ObjectKind.WALL,
            ObjectKind.KEY,
            ObjectKind.BALL,
            ObjectKind.DOOR,
        }
        if obj.kind is ObjectKind.WALL:
            assert obj.color is Color.GREY
        else:
            assert obj.color in {
                Color.BLUE,
                Color.PURPLE,
                Color.RED,
                Color.YELLOW,
            }
        assert obj.door_state is not DoorState.OPEN

    for row in range(initial.height):
        for column in range(initial.width):
            if row in {0, initial.height - 1} or column in {
                0,
                initial.width - 1,
            }:
                cell = initial.cells[row * initial.width + column]
                assert cell.object is not None
                assert cell.object.kind is ObjectKind.WALL


@pytest.mark.parametrize("scenario", SCENARIO_ORDER)
@pytest.mark.parametrize("size", SIZE_ORDER)
def test_scenario_builds_are_deterministic_and_prefix_exact(
    scenario: C19Scenario,
    size: C19Size,
) -> None:
    carrying = None if scenario is C19Scenario.NAVIGATION else False
    first = build_scenario(scenario, size, 1_000_007, carrying=carrying)
    second = build_scenario(scenario, size, 1_000_007, carrying=carrying)

    assert pprint_grid(first.state) == pprint_grid(second.state)
    assert first.command == second.command
    live = trace_script(clone_state(first.state), first.command)
    for prefix_length, live_snapshot in enumerate(live[1:], start=1):
        assert live_snapshot == simulate(
            pprint_grid(first.state),
            first.command[:prefix_length],
        )


@pytest.mark.parametrize("scenario", SCENARIO_ORDER)
@pytest.mark.parametrize("size", SIZE_ORDER)
def test_every_command_is_bounded_and_not_all_noop(
    scenario: C19Scenario,
    size: C19Size,
) -> None:
    carrying = None if scenario is C19Scenario.NAVIGATION else True
    built = build_scenario(scenario, size, 1_000_003, carrying=carrying)
    trace = trace_script(clone_state(built.state), built.command)

    assert 1 <= len(built.command) <= MAX_COMMAND_LENGTH
    assert any(before != after for before, after in pairwise(trace))


@pytest.mark.parametrize(
    "scenario",
    [C19Scenario.MANIPULATION, C19Scenario.DOOR],
)
@pytest.mark.parametrize("size", SIZE_ORDER)
def test_carrying_outcome_is_constructed_not_labeled(
    scenario: C19Scenario,
    size: C19Size,
) -> None:
    outcomes: list[bool] = []
    for index in range(16):
        expected = index % 2 == 0
        built = build_scenario(
            scenario,
            size,
            1_000_000 + index,
            carrying=expected,
        )
        final = trace_script(clone_state(built.state), built.command)[-1]
        outcomes.append(final.carrying is not None)
        if not expected:
            assert built.command.endswith("D")
    assert outcomes == [index % 2 == 0 for index in range(16)]
    assert Counter(outcomes) == {True: 8, False: 8}


def test_scenario_boundary_validation() -> None:
    with pytest.raises(TypeError):
        build_scenario(  # type: ignore[arg-type]
            "navigation",  # ty: ignore[invalid-argument-type]
            C19Size.SMALL,
            1,
            carrying=None,
        )
    with pytest.raises(TypeError):
        build_scenario(  # type: ignore[arg-type]
            C19Scenario.NAVIGATION,
            5,  # ty: ignore[invalid-argument-type]
            1,
            carrying=None,
        )
    with pytest.raises(ValueError):
        build_scenario(
            C19Scenario.NAVIGATION,
            C19Size.SMALL,
            1,
            carrying=False,
        )
