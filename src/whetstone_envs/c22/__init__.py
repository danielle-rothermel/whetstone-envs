"""c22 -- stacked IFEval instruction-following constraints.

A deliberately trivial base micro-task gated by a stack of 3-5 composed,
deterministically-checkable constraints, scored strict all-pass (0/1).
The constraint checkers are a pinned Google Research IFEval snapshot,
vendored under :mod:`whetstone_envs.c22._vendor` for generation and
scoring. Its imports are package-relative, so C22 is isolated from any
top-level package with the upstream name.

Public surface:

* :mod:`whetstone_envs.c22.generate` -- the seeded stacking generator
  (produces a :class:`~whetstone_envs.core.pool.TaskPool`).
* :mod:`whetstone_envs.c22.oracle` -- the strict all-pass oracle, a pure
  function of an instance's public fields.
* :mod:`whetstone_envs.c22.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec.
"""
