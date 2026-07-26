"""Vendored third-party code for the c23 candidate.

This subpackage carries a copy of Wenyueh's InductionBench
(``inductive_reasoning_benchmark``) subregular ISL/OSL generator +
oracle, from upstream commit ``e0b8392``. Unlike c22's IFEval vendor
(reused byte-for-byte), the InductionBench active path does **not** import
or run as shipped, so this copy carries a reviewable series of named
modifications on top of the byte-for-byte originals. The delta versus
upstream is captured in
``inductionbench/VENDORED_DIFF.patch`` so the whole change is auditable in
one file.

The vendored modules use package-relative internal imports so they remain
isolated from generic top-level module names in the embedding process.

Provenance, license, and the full patch list live in
``inductionbench/PROVENANCE.md`` and ``inductionbench/LICENSE``.
"""

from __future__ import annotations
