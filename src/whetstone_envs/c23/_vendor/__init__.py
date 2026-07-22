"""Vendored third-party code for the c23 candidate.

This subpackage carries a copy of Wenyueh's InductionBench
(``inductive_reasoning_benchmark``) subregular ISL/OSL generator +
oracle, from upstream commit ``e0b8392``. Unlike c22's IFEval vendor
(reused byte-for-byte), the InductionBench active path does **not** import
or run as shipped, so this copy is applied as a *reviewable series of
named patches* -- each a separate commit -- on top of the byte-for-byte
originals. The delta versus upstream is captured in
``inductionbench/VENDORED_DIFF.patch`` so the whole change is auditable in
one file.

The upstream modules use *bare* internal imports (``import config``,
``from utils import ...``). The parent package :mod:`whetstone_envs.c23`
prepends the ``inductionbench`` directory to ``sys.path`` at import time so
those bare imports resolve against the vendored copy.

Provenance, license, and the full patch list live in
``inductionbench/PROVENANCE.md`` and ``inductionbench/LICENSE``.
"""

from __future__ import annotations
