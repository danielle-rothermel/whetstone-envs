from __future__ import annotations

from string import Formatter
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c19.probes import (
    CEILING_TEMPLATE,
    NAIVE_TEMPLATE,
    PROBES,
)
from whetstone_envs.instances import Instance, make_instance

if TYPE_CHECKING:
    from collections.abc import Callable

GRID = """WGWGWGWG
WG>>KRWG
WG  DBWG
WGWGWGWG"""
COMMAND = "PRTFF"
QUESTION = "What is the agent's final coordinate?"


def _instance(*, prompt_inputs: dict[str, str] | None = None) -> Instance:
    return make_instance(
        id="PRIVATE-ID",
        seed=987_654,
        strata="PRIVATE-STRATUM",
        prompt_inputs=(
            {
                "grid": GRID,
                "command": COMMAND,
                "question": QUESTION,
            }
            if prompt_inputs is None
            else prompt_inputs
        ),
        gold="PRIVATE-GOLD",
    )


def test_naive_rendering_is_exact() -> None:
    assert (
        PROBES.render_naive(_instance())
        == """Grid:
WGWGWGWG
WG>>KRWG
WG  DBWG
WGWGWGWG

Actions: PRTFF

What is the agent's final coordinate?
Return only the answer."""
    )


def test_ceiling_rendering_is_exact() -> None:
    expected = """Simulate the complete action script on this grid.

The grid uses zero-based row,column coordinates. Row 0 is at the top, and
column 0 is at the left. Rows increase southward; columns increase eastward.
Every initial-grid cell is exactly two characters:
- two spaces ("  ") are an empty cell;
- >>, VV, <<, and ^^ are the agent facing E, S, W, and N respectively;
- WG is a grey wall. K*, A*, D*, and L* are a key, ball, closed door, and
  locked door respectively, where * is B for blue, P for purple, R for red,
  or Y for yellow. Color matters only when a key is used on a locked door.
  During the script, __ is an open door. Initial grids do not contain __
  because it does not preserve the door's color.

The agent starts with empty hands. Apply every uppercase action in the action
script in order, from left to right:
- L turns left in place: E->N->W->S->E.
- R turns right in place: E->S->W->N->E.
- F moves forward one cell only when the cell in front is exactly empty or is
  an open door. Walls, keys, balls, and closed or locked doors block movement.
- P picks up a pickup-capable key or ball in front only when the agent's hands
  are empty. The object leaves its cell and becomes the carried object.
- D drops the carried object into the cell in front only when that cell is
  exactly empty. The object enters that cell and the agent's hands become
  empty. An open-door cell is not empty.
- T toggles a door in front. It opens a closed door and closes an open door. A
  locked door opens and unlocks only when the agent carries a key of the same
  color; the key remains carried.
When an action's preconditions are not met, that action is a no-op. Apply the
entire script. There is no reward, mission, or episode termination.

Answer forms:
- coordinate: row,col
- heading: one of {E,S,W,N}
- front: one of {empty,wall,key,ball,door}; outside the grid is
  wall, and every door state is door
- carrying: {yes,no}

Grid:
WGWGWGWG
WG>>KRWG
WG  DBWG
WGWGWGWG

Actions: PRTFF

Question: What is the agent's final coordinate?
Return only the answer."""

    assert PROBES.render_ceiling(_instance()) == expected


@pytest.mark.parametrize("template", [NAIVE_TEMPLATE, CEILING_TEMPLATE])
def test_templates_reference_only_declared_public_fields(
    template: str,
) -> None:
    referenced = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }

    assert referenced == {"grid", "command", "question"}


@pytest.mark.parametrize(
    "render",
    [PROBES.render_naive, PROBES.render_ceiling],
)
@pytest.mark.parametrize("missing", ["grid", "command", "question"])
def test_missing_public_inputs_propagate(
    render: Callable[[Instance], str],
    missing: str,
) -> None:
    inputs = {
        "grid": GRID,
        "command": COMMAND,
        "question": QUESTION,
    }
    del inputs[missing]

    with pytest.raises(KeyError, match=missing):
        render(_instance(prompt_inputs=inputs))


@pytest.mark.parametrize(
    "render",
    [PROBES.render_naive, PROBES.render_ceiling],
)
def test_rendering_does_not_expose_private_instance_fields(
    render: Callable[[Instance], str],
) -> None:
    rendered = render(_instance())

    assert "PRIVATE-ID" not in rendered
    assert "987654" not in rendered
    assert "PRIVATE-STRATUM" not in rendered
    assert "PRIVATE-GOLD" not in rendered


def test_ceiling_covers_every_action_and_state_transition_fact() -> None:
    rendered = PROBES.render_ceiling(_instance())
    required_facts = (
        "Apply every uppercase action",
        "in order, from left to right",
        "starts with empty hands",
        "- L turns left in place",
        "- R turns right in place",
        "- F moves forward one cell only",
        "exactly empty or is\n  an open door",
        "- P picks up a pickup-capable key or ball in front only",
        "hands\n  are empty",
        "- D drops the carried object",
        "cell is\n  exactly empty",
        "- T toggles a door in front",
        "key of the same\n  color",
        "key remains carried",
        "preconditions are not met",
        "action is a no-op",
        "There is no reward, mission, or episode termination",
    )

    for fact in required_facts:
        assert fact in rendered


def test_ceiling_covers_exact_supported_glyphs_and_answers() -> None:
    rendered = PROBES.render_ceiling(_instance())
    required_facts = (
        'two spaces ("  ") are an empty cell',
        ">>, VV, <<, and ^^",
        "WG is a grey wall",
        "K*, A*, D*, and L* are a key, ball, closed door, and\n  locked door",
        "During the script, __ is an open door",
        "Initial grids do not contain __",
        "* is B for blue, P for purple, R for red,\n  or Y for yellow",
        "coordinate: row,col",
        "heading: one of {E,S,W,N}",
        "front: one of {empty,wall,key,ball,door}",
        "carrying: {yes,no}",
        "Return only the answer.",
    )

    for fact in required_facts:
        assert fact in rendered


def test_ceiling_does_not_claim_unsupported_grid_objects() -> None:
    rendered = PROBES.render_ceiling(_instance()).lower()

    for unsupported in ("box", "goal", "lava", "floor"):
        assert unsupported not in rendered
