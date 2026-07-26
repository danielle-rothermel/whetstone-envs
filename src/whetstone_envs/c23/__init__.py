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

Importing this package installs the one side effect the vendored tree
needs: it prepends the ``inductionbench`` vendor directory to ``sys.path``
so the upstream modules' *bare* internal imports (``import config``,
``from utils import ...``) resolve against the vendored copy without
editing a single upstream line -- the same convention c22 uses for its
IFEval vendor. Doing this in the package ``__init__`` means the path is set
before any submodule's own ``import synthetic_data_generation`` runs.

Public surface:

* :mod:`whetstone_envs.c23.generate` -- the seeded generator (produces a
  :class:`~whetstone_envs.core.pool.TaskPool`), with the fresh-seed and
  fixed-single-rule contamination assertions at construction.
* :mod:`whetstone_envs.c23.oracle` -- the independent oracle, reusing the
  vendored ``apply_*_rule`` transducers unmodified, re-applied to the
  held-out query.
* :mod:`whetstone_envs.c23.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec (Section 2).
* :mod:`whetstone_envs.c23.upstream` -- the boundary around the vendored
  generator + oracle (sets ``config.vocab`` under a lock; marshals args).
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(
    Path(__file__).resolve().parent / "_vendor" / "inductionbench",
)
if _VENDOR_DIR not in sys.path:
    # Prepend so the vendored ``synthetic_data_generation`` / ``config`` /
    # ``utils`` modules win over any like-named module elsewhere on the
    # path (upstream imports them by bare name).
    sys.path.insert(0, _VENDOR_DIR)
