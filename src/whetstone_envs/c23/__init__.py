"""c23 -- subregular ISL/OSL rule induction (InductionBench-style).

The task: infer a single latent subregular string-transform rule from a
handful of input->output demonstrations, then apply that rule to one
held-out query string; scored 0/1 by exact match against an independent
oracle. This is the reseed-only *baseline* (spec "Headroom Before
Complexity"): the existing InductionBench single-rule ISL/OSL shape,
measuring naive-vs-ceiling headroom before any composition-ladder
complexity is built.

Strata (spec Section 1): the primary axis is ``k`` (the context window),
the secondary axis is the transducer ``type`` (ISL / L-OSL / R-OSL); vocab
is held fixed (|Sigma|=4) and ``number_of_rules`` is fixed at 1 under the
single-rule constraint. Four strata: S1 (ISL k=2), S2 (L-OSL k=2), S3
(R-OSL k=2), S4 (ISL k=3).

The instance pool is regenerated from the *vendored + patched*
InductionBench generator (see ``_vendor/inductionbench/PROVENANCE.md`` and
its ``VENDORED_DIFF.patch``); the oracle reuses the vendored
``apply_ISL_rule`` / ``apply_L_OSL_rule`` / ``apply_R_OSL_rule``
transducers **unmodified**, re-applied to the held-out query.

The vendored tree is imported only through its package-qualified path.
Its internal imports are package-relative, so importing c23 neither changes
``sys.path`` nor claims generic top-level module names such as ``config`` or
``utils`` in the embedding process.

Public surface:

* :mod:`whetstone_envs.c23.generate` -- the seeded generator (produces a
  :class:`~whetstone_envs.core.pool.TaskPool`), with the fresh-seed and
  fixed-single-rule contamination assertions at construction.
* :mod:`whetstone_envs.c23.oracle` -- the independent oracle, reusing the
  vendored ``apply_*_rule`` transducers unmodified, re-applied to the
  held-out query.
* :mod:`whetstone_envs.c23.prompts` -- the naive/ceiling probe pair,
  with character-level task conventions in the ceiling prompt.
* :mod:`whetstone_envs.c23.upstream` -- the boundary around the vendored
  generator + oracle (sets ``config.vocab`` under a lock; marshals args).
"""

from __future__ import annotations
