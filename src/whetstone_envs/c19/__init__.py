"""c19 -- Minigrid grid-world state prediction.

The task: given a small ASCII grid (Farama-Foundation Minigrid's
``pprint_grid`` output) and a short vanilla command string, predict one
derived fact about the robot's final state -- its coordinate, heading,
carrying-flag, or what-is-in-front -- scored 0/1 by exact match against
an independent oracle. This is the vanilla-dynamics *baseline* (spec
"Headroom Before Complexity"): standard Minigrid actions and semantics,
measuring naive-vs-ceiling headroom before any invented-command layer.

Strata cross env id x grid size x fact type (spec Section 1): the four
seed-threaded stochastic-layout envs (Fetch, SimpleCrossing, FourRooms,
Empty-Random), two size levels, and the four derived facts (carrying is
Fetch-only under vanilla dynamics).

Minigrid is a real runtime dependency, not just a checker library: the
generator instantiates seeded envs, renders their ASCII, and walks the
live object model for ground truth. The oracle then reproduces that walk
independently from the public ASCII alone, and construction asserts the
two agree.

Public surface:

* :mod:`whetstone_envs.c19.generate` -- the seeded generator (produces a
  :class:`~whetstone_envs.core.pool.TaskPool`).
* :mod:`whetstone_envs.c19.oracle` -- the independent derived-fact
  oracle, a pure function of an instance's public grid + command.
* :mod:`whetstone_envs.c19.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec (Section 2).
* :mod:`whetstone_envs.c19.envs` -- the Minigrid env construction,
  strata, and live seeded rollout (the runtime binding to Minigrid).
"""

from __future__ import annotations
