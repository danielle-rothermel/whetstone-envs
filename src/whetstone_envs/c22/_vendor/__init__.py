"""Vendored third-party code for the c22 candidate.

This subpackage carries a byte-for-byte copy of Google Research's IFEval
constraint-checker library (``instruction_following_eval``), reused
unmodified as both the generation-side constraint sampler and the
scoring oracle per the c22 baseline spec.

The upstream package uses *absolute* internal imports of the form
``from instruction_following_eval import instructions_util``. The parent
package :mod:`whetstone_envs.c22` prepends this directory to ``sys.path``
at import time so those imports resolve against the vendored copy without
editing a single upstream line.

Provenance and license live in
``instruction_following_eval/PROVENANCE.md`` and
``instruction_following_eval/LICENSE``.
"""
