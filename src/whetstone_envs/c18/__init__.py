"""c18 -- PrOntoQA deductive entailment (True/False).

The task: given a set of facts and universally-quantified if-then rules
over fictional nonce predicates ("every wumpus is a yumpus") plus a single
query statement, predict whether the query is entailed -- ``True`` or
``False`` -- scored 0/1 by exact match against an independent oracle. This
is the reseed-only *baseline* (spec "Headroom Before Complexity"): the
existing PrOntoQA task shape, measuring naive-vs-ceiling headroom before
any added-complexity axis (Unknown label, constraint-puzzle stratum, soft
rules) is built.

Strata: one axis, hop depth (D1, D2, D3, D5; spec Section 1), via the
native ``--min-hops/--max-hops`` loop. Ontology type is held constant to
``fictional`` (nonce symbols) for contamination resistance.

The instance pool is reseeded from the *vendored* ``asaparov/prontoqa``
generator (a subprocess boundary; see ``_vendor/prontoqa/PROVENANCE.md``);
the generator's stored label is definitional, so an independent
from-scratch forward-chaining fixpoint oracle re-derives the label from
the public text and construction asserts the two agree.

Public surface:

* :mod:`whetstone_envs.c18.generate` -- the seeded generator (produces a
  :class:`~whetstone_envs.core.pool.TaskPool`), with the fixed-nonce and
  fresh-seed contamination assertions at construction.
* :mod:`whetstone_envs.c18.oracle` -- the independent forward-chaining
  fixpoint oracle, a pure function of an instance's public question +
  query.
* :mod:`whetstone_envs.c18.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec (Section 2).
* :mod:`whetstone_envs.c18.upstream` -- the subprocess/import boundary
  around the vendored PrOntoQA generator.
"""

from __future__ import annotations
