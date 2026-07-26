r"""The two c19 probe prompts.

Probe (a), the naive prompt, follows spec Section 2.1. Probe (b), the
ceiling prompt, makes the installed Minigrid glyph and action semantics
explicit. Both use the fact-specific question/answer line for the requested
fact type. This module encodes each prompt's fixed body plus the four
fact-line variants, with only the instance's public fields substituted:

* ``{GRID}`` -- the ``pprint_grid`` ASCII from ``prompt_inputs['grid']``.
* ``{COMMAND}`` -- the vanilla command string from
  ``prompt_inputs['command']``.
* the fact-specific question line -- selected by
  ``prompt_inputs['fact_type']``.

No gold/oracle-only field is ever interpolated; the fact *type* is a
public input (the model is told which fact to predict), but the fact
*value* (the gold) never appears in either prompt. A static test asserts
no prompt contains the gold answer, and a byte-for-byte render test pins
both rendered prompts against a fixed fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.core.probes import ProbePair

if TYPE_CHECKING:
    from whetstone_envs.core.instance import Instance

# The four fact types this task predicts (mirrors envs.FACT_TYPES; kept
# local so prompts has no import-time dependency on Minigrid).
_FACT_TYPES: tuple[str, ...] = ("coordinate", "heading", "carrying", "front")

# --- Naive probe (spec Section 2.1) ---------------------------------------
# The fact-line is the final sentence; variants are from the Section 2.1
# "Fact-line variants" note. {GRID} and {COMMAND} are the only slots.
_NAIVE_HEAD = """Here is a grid and a sequence of moves for the robot.

{GRID}

Moves: {COMMAND}

"""

_NAIVE_FACT_LINE: dict[str, str] = {
    "coordinate": (
        "Where does the robot end up? Answer with just the final coordinate."
    ),
    "heading": (
        "Which direction is the robot facing at the end? Answer with one "
        "letter."
    ),
    "carrying": (
        "Is the robot carrying an object at the end? Answer yes or no."
    ),
    "front": (
        "What is directly in front of the robot at the end? Answer with one "
        "word."
    ),
}

# --- Ceiling probe (spec Section 2.2) -------------------------------------
# The fixed rule body down to (but not including) the QUESTION line, then
# the fact-specific QUESTION-line variants.
_CEILING_HEAD = """You are simulating a robot on a 2D grid. Follow these rules EXACTLY.

COORDINATES: cells are written (row, col). Row 0 is the TOP row; col 0 is the LEFTMOST
column. Rows increase downward, columns increase rightward.

GLYPHS in the grid below (two characters per cell):
  - two space characters "  " are an empty floor cell the robot may enter.
  - a wall glyph starts with "W" (e.g. "WG"); the robot CANNOT enter or pass through it.
  - a two-letter tile or object glyph (e.g. "KY" = yellow key, "AR" = red ball,
    "BG" = green box, "GG" = green goal) identifies the cell contents.
  - the robot is shown by a doubled direction arrow: ">>" faces right (east),
    "<<" faces left (west), "^^" faces up (north / toward row 0), "VV" faces down (south).

HEADINGS: the robot has a facing direction, one of E (east/right), W (west/left),
N (north/up), S (south/down).

MOVES (apply in order, left to right):
  - L = turn left 90 degrees in place (does not move): E->N->W->S->E.
  - R = turn right 90 degrees in place: E->S->W->N->E.
  - F = step ONE cell forward in the current facing direction if the cell ahead is empty
    floor, a goal, or lava. A wall, the grid edge, or a solid object such as a key, ball,
    box, or closed door blocks movement; the robot stays put and never wraps around.
  - P = pick up a pickup-able key, ball, or box in the cell directly ahead, if any, and only
    if not already carrying something; the robot then carries it.
  - D = drop the carried object into the cell directly ahead, if that cell is empty.
  - T = toggle the object directly ahead if it supports toggling. The object types in these
    grids do not, so T does not change the grid, position, heading, or carried object.

Work step by step: track the robot's (row, col) position and its heading after EACH move,
then report only the final answer.

{GRID}

Moves: {COMMAND}

"""

_CEILING_QUESTION_LINE: dict[str, str] = {
    "coordinate": (
        "QUESTION: What is the robot's final coordinate? Answer on the last "
        "line as: row,col"
    ),
    "heading": (
        "QUESTION: Which direction is the robot facing at the end? Answer on "
        "the last line as one of: E W N S"
    ),
    "carrying": (
        "QUESTION: Is the robot carrying an object at the end? Answer on the "
        "last line as: yes or no"
    ),
    "front": (
        "QUESTION: What is directly in front of the robot at the end? Answer "
        "on the last line with one word (e.g. wall, empty, key, ball, goal)."
    ),
}


def _render(
    template_head: str,
    fact_lines: dict[str, str],
    instance: Instance,
) -> str:
    """Render one probe: substitute public fields, append the fact line."""
    inputs = dict(instance.prompt_inputs)
    fact_type = inputs["fact_type"]
    if fact_type not in fact_lines:
        msg = f"no probe fact-line for fact type {fact_type!r}"
        raise KeyError(msg)
    body = template_head.replace("{GRID}", inputs["grid"]).replace(
        "{COMMAND}",
        inputs["command"],
    )
    return body + fact_lines[fact_type]


def render_naive(instance: Instance) -> str:
    """Render the naive probe (spec Section 2.1) for ``instance``."""
    return _render(_NAIVE_HEAD, _NAIVE_FACT_LINE, instance)


def render_ceiling(instance: Instance) -> str:
    """Render the ceiling probe (spec Section 2.2) for ``instance``."""
    return _render(_CEILING_HEAD, _CEILING_QUESTION_LINE, instance)


def _probe_render(template: str, instance: Instance) -> str:
    """Dispatch the ProbePair render by which template is passed.

    ``ProbePair`` stores the two templates as strings; c19's templates
    are fact-type-parameterized, so the single ``render`` callable picks
    the naive vs ceiling body by identity of the template it is handed.
    """
    if template is NAIVE_TEMPLATE:
        return render_naive(instance)
    if template is CEILING_TEMPLATE:
        return render_ceiling(instance)
    msg = "unknown c19 probe template"
    raise KeyError(msg)


# The ``ProbePair`` templates are sentinels naming which probe to render;
# the fact-line body is assembled by the renderer above (c19 prompts are
# parameterized by fact type, not a single format string).
NAIVE_TEMPLATE = "c19-naive"
CEILING_TEMPLATE = "c19-ceiling"

PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=_probe_render,
)
"""The naive/ceiling probe pair for c19 (spec Section 2), rendered by
:func:`render_naive` / :func:`render_ceiling` -- both substitute only the
public ``grid``/``command``/``fact_type`` inputs, never gold."""
