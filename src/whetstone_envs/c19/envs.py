r"""Minigrid env construction, strata, and the live seeded rollout.

This is c19's real runtime binding to Minigrid (spec Section 1 / the
PLAN's "env instantiation, seeded rollout execution, object-model
introspection for the oracle"). It owns three things:

* the mapping from the four spec env ids (``Fetch``, ``SimpleCrossing``,
  ``FourRooms``, ``Empty-Random``) and two size levels to a concrete
  Minigrid ``MiniGridEnv`` constructor -- built directly rather than via
  ``gym.make`` so the size level is a real constructor arg for every env;
* the fact-type applicability matrix (spec Section 1: carrying-flag is
  Fetch-only under vanilla dynamics); and
* :func:`rollout`, which instantiates a seeded env, renders its ASCII
  grid, samples a seeded vanilla command string, and walks the *live*
  env to read the ground-truth final pose/carry off Minigrid's own
  object model.

The generator then cross-checks that live ground truth against the
independent ASCII-only oracle walk (:mod:`whetstone_envs.c19.oracle`),
so a disagreement between the two independent object-model walks fails
construction rather than shipping a wrong gold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from minigrid.core.world_object import Wall
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.fetch import FetchEnv
from minigrid.envs.fourrooms import FourRoomsEnv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from minigrid.minigrid_env import MiniGridEnv

# --- Env ids and size levels (spec Section 1) -----------------------------
ENV_IDS: tuple[str, ...] = (
    "Fetch",
    "SimpleCrossing",
    "FourRooms",
    "Empty-Random",
)
SIZE_LEVELS: tuple[str, ...] = ("small", "medium")

# Concrete grid side length per size level. FourRooms has a single native
# size (19) and ignores this; the other three take ``size`` directly, so
# ``small``/``medium`` are real env-native variants (spec Section 1: "small
# ~5x5 and medium ~8x8, env-native size variants where available").
# SimpleCrossing requires an odd side length, so it uses 5/9 rather than
# 5/8 for its two levels.
_SIZE_TO_SIDE: dict[str, int] = {"small": 5, "medium": 8}
_SIZE_TO_SIDE_ODD: dict[str, int] = {"small": 5, "medium": 9}

# Fetch object count scales with grid size so a small grid is not crowded.
_FETCH_NUM_OBJS: dict[str, int] = {"small": 2, "medium": 3}

# --- Fact-type applicability matrix (spec Section 1) -----------------------
FACT_TYPES: tuple[str, ...] = (
    "coordinate",
    "heading",
    "carrying",
    "front",
)

# Carrying-flag needs pickup-able objects plus a command that can pick one
# up; only Fetch supplies that under vanilla dynamics (spec Section 1). The
# other three facts apply to every env.
_FACT_APPLICABILITY: dict[str, frozenset[str]] = {
    "coordinate": frozenset(ENV_IDS),
    "heading": frozenset(ENV_IDS),
    "front": frozenset(ENV_IDS),
    "carrying": frozenset({"Fetch"}),
}


def applicable_fact_types(env_id: str) -> tuple[str, ...]:
    """Return the fact types meaningful for ``env_id`` (spec Section 1)."""
    if env_id not in ENV_IDS:
        msg = f"unknown env id {env_id!r}"
        raise ValueError(msg)
    return tuple(
        fact for fact in FACT_TYPES if env_id in _FACT_APPLICABILITY[fact]
    )


def strata_labels(
    *,
    env_ids: Sequence[str] = ENV_IDS,
    size_levels: Sequence[str] = SIZE_LEVELS,
) -> tuple[str, ...]:
    """Return every ``env|size|fact`` stratum label, in a stable order.

    The crossing is env x size x applicable-fact, so carrying-flag
    contributes only its Fetch strata (spec Section 1: 26 strata under
    the default 4 envs x 2 sizes).
    """
    return tuple(
        f"{env_id}|{size}|{fact}"
        for env_id in env_ids
        for size in size_levels
        for fact in applicable_fact_types(env_id)
    )


def _make_env(env_id: str, size: str) -> MiniGridEnv:
    """Construct the Minigrid env for ``(env_id, size)`` (unseeded).

    Built via the concrete env classes rather than ``gym.make`` so the
    size level maps to a real constructor arg; the caller resets with a
    seed to lay out the grid.
    """
    side = _SIZE_TO_SIDE[size]
    if env_id == "Fetch":
        return FetchEnv(size=side, numObjs=_FETCH_NUM_OBJS[size])
    if env_id == "SimpleCrossing":
        return CrossingEnv(
            size=_SIZE_TO_SIDE_ODD[size],
            num_crossings=1,
            obstacle_type=Wall,
        )
    if env_id == "Empty-Random":
        return EmptyEnv(size=side, agent_start_pos=None)
    if env_id == "FourRooms":
        # FourRooms has a single native 19x19 size; the size level varies
        # only the stratum label, layout variety comes from the seed.
        return FourRoomsEnv()
    msg = f"unknown env id {env_id!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Rollout:
    """The public, oracle-checkable result of one seeded rollout.

    Parameters
    ----------
    grid_ascii:
        The initial grid rendered by Minigrid's ``pprint_grid`` -- the
        exact ASCII the probe prompt shows the model.
    command:
        The vanilla command string (letters ``L R F P D T``).
    facts:
        The ground-truth derived fact per fact type, read from the live
        Minigrid object model after walking ``command``.
    """

    grid_ascii: str
    command: str
    facts: dict[str, str]


_DIR_TO_LETTER: tuple[str, ...] = ("E", "S", "W", "N")
_TYPE_TO_NAME: dict[str, str] = {
    "wall": "wall",
    "floor": "floor",
    "key": "key",
    "ball": "ball",
    "box": "box",
    "goal": "goal",
    "lava": "lava",
    "door": "door",
}


def _live_front_name(env: MiniGridEnv) -> str:
    """Read the object name directly ahead of the agent from the env."""
    fwd = env.front_pos
    if not (0 <= fwd[0] < env.grid.width and 0 <= fwd[1] < env.grid.height):
        return "wall"
    cell = env.grid.get(int(fwd[0]), int(fwd[1]))
    if cell is None:
        return "empty"
    return _TYPE_TO_NAME.get(cell.type, cell.type)


def _live_facts(env: MiniGridEnv) -> dict[str, str]:
    """Read all derived facts off the live Minigrid object model."""
    col, row = (int(env.agent_pos[0]), int(env.agent_pos[1]))
    return {
        "coordinate": f"{row},{col}",
        "heading": _DIR_TO_LETTER[int(env.agent_dir)],
        "carrying": "yes" if env.carrying is not None else "no",
        "front": _live_front_name(env),
    }


_ACTION_LETTER_TO_INT: dict[str, int] = {
    "L": 0,  # Actions.left
    "R": 1,  # Actions.right
    "F": 2,  # Actions.forward
    "P": 3,  # Actions.pickup
    "D": 4,  # Actions.drop
    "T": 5,  # Actions.toggle
}


def sample_command(rng: random.Random, length: int) -> str:
    """Sample a seeded vanilla command string of ``length`` letters.

    Draws from the six standard actions ``L R F P D T`` (spec Section 2).
    Weighted toward movement/turn so a short command still relocates the
    agent, while pickup/drop/toggle appear often enough to exercise the
    carrying-flag and what-is-in-front facts.
    """
    letters = ("L", "R", "F", "F", "F", "P", "D", "T")
    return "".join(rng.choice(letters) for _ in range(length))


def rollout(
    env_id: str,
    size: str,
    seed: int,
    *,
    command_length: int,
) -> Rollout:
    """Instantiate a seeded env, sample a command, and read ground truth.

    The env is reset with ``seed`` (Minigrid threads it through
    ``self.np_random``, so the same seed reproduces a byte-identical
    layout). The initial ASCII grid is captured *before* stepping; the
    command is sampled from an independent ``random.Random(seed)`` stream
    so it, too, is deterministic given the seed. The live env is then
    walked and every derived fact read from its object model.
    """
    env = _make_env(env_id, size)
    env.reset(seed=seed)
    grid_ascii = env.pprint_grid()
    cmd_rng = random.Random(seed)
    command = sample_command(cmd_rng, command_length)
    for action in command:
        env.step(_ACTION_LETTER_TO_INT[action])
    facts = _live_facts(env)
    return Rollout(grid_ascii=grid_ascii, command=command, facts=facts)
