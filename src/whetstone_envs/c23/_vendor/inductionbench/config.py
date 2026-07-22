"""Config stub for the vendored InductionBench generator.

**Patch note (added by this vendor; not present upstream).** The upstream
``standard_benchmark/synthetic_data_generation.py`` does ``import config``
(upstream line 4) and reads ``config.vocab`` throughout, but the upstream
repository ships **no** ``config.py`` -- ``import config`` raises
``ModuleNotFoundError`` as cloned (repos review red flag: "Missing
``config`` module"). Upstream relied on ``config`` existing purely as a
namespace object to hang a module-global ``config.vocab`` list on, mutated
once per run in ``inference.py`` / ``standard_run.py`` (neither of which we
vendor).

This stub restores exactly that namespace so the vendored generator
imports. ``vocab`` starts empty; the c23 boundary
(:mod:`whetstone_envs.c23.upstream`) sets it explicitly per generation --
under a lock, restored after use -- so the module-global is never a hidden
cross-call coupling in our code path.

This file is the "config stub" patch named in PLAN.md. It adds no
generator logic: it is a single mutable module attribute.
"""

from __future__ import annotations

# The generation alphabet. Populated per-run by the c23 boundary
# (``whetstone_envs.c23.upstream``); empty by default so an accidental
# import without a set-up alphabet fails loudly rather than silently
# generating over the wrong vocabulary.
vocab: list[str] = []
