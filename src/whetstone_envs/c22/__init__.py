"""c22 -- stacked IFEval instruction-following constraints.

A deliberately trivial base micro-task gated by a stack of 3-5 composed,
deterministically-checkable constraints, scored strict all-pass (0/1).
The constraint checkers are Google Research's IFEval library, reused
verbatim (vendored under :mod:`whetstone_envs.c22._vendor`) for both
generation-side constraint selection and the scoring oracle.

Importing this package installs the one side effect the vendored tree
needs: it prepends the vendor directory to ``sys.path`` so the upstream
package's *absolute* internal imports
(``from instruction_following_eval import instructions_util``) resolve
against the vendored copy without editing a single upstream line. Doing
this in the package ``__init__`` -- rather than in each submodule --
means the path is set before any submodule's own
``import instruction_following_eval`` runs, regardless of import
ordering the linter may enforce.

Public surface:

* :mod:`whetstone_envs.c22.generate` -- the seeded stacking generator
  (produces a :class:`~whetstone_envs.core.pool.TaskPool`).
* :mod:`whetstone_envs.c22.oracle` -- the strict all-pass oracle, a pure
  function of an instance's public fields.
* :mod:`whetstone_envs.c22.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "_vendor")
if _VENDOR_DIR not in sys.path:
    # Prepend so the vendored ``instruction_following_eval`` wins over any
    # like-named package elsewhere on the path.
    sys.path.insert(0, _VENDOR_DIR)
