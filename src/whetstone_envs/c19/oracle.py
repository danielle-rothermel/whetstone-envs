r"""The independent derived-fact oracle for c19.

A pure function of an instance's **public** fields -- the rendered ASCII
grid (``pprint_grid`` output) and the vanilla command string -- and
nothing else. It parses that grid into an object model, walks the
command string under standard Minigrid dynamics, and reads one derived
fact off the final state (spec Section 1: "read directly from the
serialized grid and agent state by independent oracle glue").

Independence (rubric criteria 2, 8, 11): the oracle never consults the
generator's internal ``MiniGridEnv`` instance or its RNG. It reconstructs
the world purely from the ASCII the model itself is shown, so it cannot
silently become a re-derivation of how the generator built the instance.
The generator computes the same facts from the *live* Minigrid env and
asserts agreement at construction, so the two independent walks are
cross-checked, but the oracle here is the sole source of truth for
scoring a model response.

The step semantics mirror ``MiniGridEnv.step`` exactly (verbatim
behaviour, not a reinterpretation):

* ``L`` / ``R`` -- rotate in place; heading order matches
  :data:`_DIR_TO_VEC` (0=E, 1=S, 2=W, 3=N).
* ``F`` -- step one cell forward iff the cell ahead is empty floor, a
  goal, or lava (the ``can_overlap`` objects); a wall, key, ball, box,
  closed door, or the grid edge blocks the move (agent stays put). No
  wrap, no pass-through.
* ``P`` -- pick up the object directly ahead iff it is pickup-able
  (key/ball/box) and the agent carries nothing; the cell becomes empty.
* ``D`` -- drop the carried object into the cell ahead iff that cell is
  empty; the agent then carries nothing.
* ``T`` -- toggle: never changes position, heading, or carry for the
  object kinds this task uses, so it is a no-op here.
"""

from __future__ import annotations

from whetstone_envs.core.probes import normalize

# --- Object-model glyph tables (Minigrid pprint_grid conventions) ---------
# ``pprint_grid`` renders every cell as two characters: an object-type
# initial plus a colour initial (or a doubled agent-direction arrow, or two
# spaces for empty floor). These tables are the *reading* half of that same
# convention, kept here so the oracle depends only on the public glyphs.
_AGENT_GLYPHS: dict[str, int] = {">": 0, "V": 1, "<": 2, "^": 3}
"""Doubled arrow -> heading index (0=E, 1=S, 2=W, 3=N)."""

# Object-type initial (first char of a cell) -> canonical object name.
_TYPE_INITIALS: dict[str, str] = {
    "W": "wall",
    "F": "floor",
    "K": "key",
    "A": "ball",
    "B": "box",
    "G": "goal",
    "V": "lava",
    "D": "door",
    "L": "door",  # locked-door glyph in pprint_grid
}

# Heading index -> single-letter answer and (dcol, drow) step vector. The
# vectors match Minigrid's DIR_TO_VEC exactly: x is the column, y is the
# row, and row increases downward (south).
_DIR_TO_LETTER: tuple[str, ...] = ("E", "S", "W", "N")
_DIR_TO_VEC: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

# Object kinds the agent may walk onto (Minigrid ``can_overlap``): empty
# floor, an explicit floor tile, a goal, or lava.
_OVERLAPPABLE: frozenset[str] = frozenset({"empty", "floor", "goal", "lava"})
# Object kinds the agent may pick up (Minigrid ``can_pickup``).
_PICKUPABLE: frozenset[str] = frozenset({"key", "ball", "box"})

_CELL_WIDTH = 2

VALID_COMMANDS: frozenset[str] = frozenset("LRFPDT")
"""The six vanilla Minigrid actions, as single letters (spec Section 2)."""


class OracleError(ValueError):
    """Raised when the public grid or command cannot be parsed/walked."""


def _cell_type(cell: str) -> str:
    """Map one 2-char grid cell to its object name (``empty`` for floor).

    An agent glyph is treated as the empty floor it stands on -- the
    parser records the agent separately, so what remains under it is
    floor.
    """
    if cell == "  ":
        return "empty"
    head = cell[0]
    if head in _AGENT_GLYPHS:
        return "empty"
    name = _TYPE_INITIALS.get(head)
    if name is None:
        msg = f"unrecognized grid cell glyph {cell!r}"
        raise OracleError(msg)
    return name


class _World:
    """The parsed object model: a type grid plus the agent's pose.

    Positions are ``(col, row)`` with ``row`` increasing downward, to
    match Minigrid's ``(x, y)`` convention exactly so the oracle's walk
    reproduces ``MiniGridEnv.step``.
    """

    def __init__(self, grid_ascii: str) -> None:
        rows = grid_ascii.split("\n")
        if not rows or not rows[0]:
            msg = "empty grid"
            raise OracleError(msg)
        width = len(rows[0]) // _CELL_WIDTH
        if any(len(r) != width * _CELL_WIDTH for r in rows):
            msg = "ragged grid: rows differ in width"
            raise OracleError(msg)
        self.width = width
        self.height = len(rows)
        self.cells: list[list[str]] = []
        agent: tuple[int, int] | None = None
        agent_dir: int | None = None
        for row_idx, row in enumerate(rows):
            cell_types: list[str] = []
            for col_idx in range(width):
                cell = row[col_idx * _CELL_WIDTH : (col_idx + 1) * _CELL_WIDTH]
                head = cell[0]
                if head in _AGENT_GLYPHS:
                    if agent is not None:
                        msg = "grid has more than one agent glyph"
                        raise OracleError(msg)
                    agent = (col_idx, row_idx)
                    agent_dir = _AGENT_GLYPHS[head]
                cell_types.append(_cell_type(cell))
            self.cells.append(cell_types)
        if agent is None or agent_dir is None:
            msg = "grid has no agent glyph"
            raise OracleError(msg)
        self.agent: tuple[int, int] = agent
        self.agent_dir: int = agent_dir
        self.carrying: str | None = None

    def _type_at(self, col: int, row: int) -> str:
        if not (0 <= col < self.width and 0 <= row < self.height):
            return "edge"
        return self.cells[row][col]

    def _front(self) -> tuple[int, int]:
        dcol, drow = _DIR_TO_VEC[self.agent_dir]
        col, row = self.agent
        return col + dcol, row + drow

    def front_name(self) -> str:
        """Return the object name directly ahead (``wall`` off the edge)."""
        fcol, frow = self._front()
        ahead = self._type_at(fcol, frow)
        return "wall" if ahead == "edge" else ahead

    def step(self, action: str) -> None:
        """Apply one vanilla command letter, mirroring Minigrid's step."""
        if action == "L":
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == "R":
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == "F":
            fcol, frow = self._front()
            if self._type_at(fcol, frow) in _OVERLAPPABLE:
                self.agent = (fcol, frow)
        elif action == "P":
            fcol, frow = self._front()
            ahead = self._type_at(fcol, frow)
            if ahead in _PICKUPABLE and self.carrying is None:
                self.carrying = ahead
                self.cells[frow][fcol] = "empty"
        elif action == "D":
            fcol, frow = self._front()
            ahead_empty = self._type_at(fcol, frow) == "empty"
            if ahead_empty and self.carrying is not None:
                self.cells[frow][fcol] = self.carrying
                self.carrying = None
        elif action == "T":
            # Toggle is a no-op for this task's object kinds (no doors are
            # placed with lock/open state that changes position or carry).
            pass
        else:
            msg = f"unrecognized command letter {action!r}"
            raise OracleError(msg)

    def walk(self, command: str) -> None:
        for action in command:
            self.step(action)


def _parse_command(command: str) -> str:
    """Validate and return the command string (letters only, no spaces)."""
    cleaned = command.strip()
    for ch in cleaned:
        if ch not in VALID_COMMANDS:
            msg = f"command has non-vanilla action {ch!r}"
            raise OracleError(msg)
    return cleaned


def derive_fact(grid_ascii: str, command: str, fact_type: str) -> str:
    """Walk ``command`` on ``grid_ascii`` and return the ``fact_type`` gold.

    Pure function of the two public fields plus which fact is asked. This
    is the ground-truth definition the generator freezes as
    ``Instance.gold`` and the value :func:`score` compares a model
    response against.

    ``fact_type`` is one of ``coordinate`` (``"row,col"``), ``heading``
    (one of ``E S W N``), ``carrying`` (``yes``/``no``), or ``front``
    (the object name directly ahead: ``wall``/``empty``/``key``/...).
    """
    world = _World(grid_ascii)
    world.walk(_parse_command(command))
    col, row = world.agent
    if fact_type == "coordinate":
        return f"{row},{col}"
    if fact_type == "heading":
        return _DIR_TO_LETTER[world.agent_dir]
    if fact_type == "carrying":
        return "yes" if world.carrying is not None else "no"
    if fact_type == "front":
        return world.front_name()
    msg = f"unknown fact type {fact_type!r}"
    raise OracleError(msg)


def score(
    prediction: str,
    grid_ascii: str,
    command: str,
    fact_type: str,
) -> int:
    """Return 1 iff ``prediction`` matches the derived fact, else 0.

    The model's ``prediction`` and the freshly derived gold are both
    passed through the shared
    :func:`whetstone_envs.core.probes.normalize` (strip surrounding
    whitespace / one code fence) before an exact-match compare -- no
    partial credit (rubric criterion 2). A grid or command the oracle
    cannot parse scores ``0`` rather than raising: a model response is
    graded, not trusted.
    """
    try:
        gold = derive_fact(grid_ascii, command, fact_type)
    except OracleError:
        return 0
    return int(normalize(prediction) == normalize(gold))


def score_gold(prediction: str, gold: str) -> int:
    """Return the 0/1 score of ``prediction`` against a frozen ``gold``.

    The pool-facing entry point mirroring the other candidates'
    ``score_gold``: given an instance's already-derived ``gold`` string
    and a model response, return 0 or 1 by normalized exact match. Use
    :func:`score` instead when scoring straight from the public grid.
    """
    return int(normalize(prediction) == normalize(gold))
