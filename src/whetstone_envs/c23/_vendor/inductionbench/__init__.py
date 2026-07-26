"""Package marker for the vendored InductionBench generator + oracle.

Upstream imports these modules by *bare* name (``import config``,
``from utils import ...``); the parent :mod:`whetstone_envs.c23` package
puts this directory on ``sys.path`` so those bare imports resolve against
the vendored copy. See ``PROVENANCE.md`` for the vendored commit, the
license, and the full list of applied patches.
"""

from __future__ import annotations
