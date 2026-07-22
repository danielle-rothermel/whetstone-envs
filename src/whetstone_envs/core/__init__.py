"""Shared harness for whetstone-envs quick-test task families.

These five modules are the model-call-free parts of the outer shape
every candidate shares -- generate a pinned instance pool, render one of
two probe prompts per instance, score 0/1 against an independent oracle,
and aggregate repeat -> task -> stratum -> overall. They are built once
here (PR 0) and reused by every candidate rather than duplicated:

* :mod:`~whetstone_envs.core.instance` -- the frozen :class:`Instance`.
* :mod:`~whetstone_envs.core.probes` -- :class:`ProbePair` plus the
  shared :func:`normalize` step.
* :mod:`~whetstone_envs.core.scoring` -- :func:`exact_match` and the
  strata-crossing aggregation ladder.
* :mod:`~whetstone_envs.core.pool` -- :class:`TaskPool` and its disjoint
  :meth:`~whetstone_envs.core.pool.TaskPool.split`.
* :mod:`~whetstone_envs.core.manifest` -- the diffable JSON
  :class:`Manifest`.
"""

from __future__ import annotations

from whetstone_envs.core.instance import Instance, make_instance
from whetstone_envs.core.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    content_hash,
)
from whetstone_envs.core.pool import PoolSplit, TaskPool
from whetstone_envs.core.probes import (
    ProbePair,
    normalize,
    render_with_prompt_inputs,
)
from whetstone_envs.core.scoring import (
    Aggregate,
    Observation,
    Outcome,
    aggregate,
    aggregate_overall,
    aggregate_stratum,
    aggregate_task,
    exact_match,
    failed,
    missing,
    scored,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Aggregate",
    "Instance",
    "Manifest",
    "Observation",
    "Outcome",
    "PoolSplit",
    "ProbePair",
    "TaskPool",
    "aggregate",
    "aggregate_overall",
    "aggregate_stratum",
    "aggregate_task",
    "content_hash",
    "exact_match",
    "failed",
    "make_instance",
    "missing",
    "normalize",
    "render_with_prompt_inputs",
    "scored",
]
