from __future__ import annotations

from whetstone_envs.probes import ProbePair

NAIVE_TEMPLATE = """Grid:
{grid}

Actions: {command}

{question}
Return only the answer."""


CEILING_TEMPLATE = """Simulate the complete action script on this grid.

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
- heading: one of {{E,S,W,N}}
- front: one of {{empty,wall,key,ball,door}}; outside the grid is
  wall, and every door state is door
- carrying: {{yes,no}}

Grid:
{grid}

Actions: {command}

Question: {question}
Return only the answer."""


PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
)


__all__ = ["PROBES"]
