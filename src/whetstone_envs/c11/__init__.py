"""c11 -- JSON canonicalization (RFC 8785 / JCS).

The task: normalize a messy input JSON value to its RFC 8785 (JCS)
canonical string, scored 0/1 by whole-string exact match against an
independent oracle. This is the plain-JCS *baseline* (spec "Headroom
Before Complexity"): the canonical form follows the published standard,
not yet the candidate's invented house rules, so the oracle can reuse
trailofbits ``rfc8785`` strictly unmodified.

The stratum axis is schema-content shape -- which JCS sub-rule an
instance stresses (key sort, number canonicalization, escaping, or a
mixed combination) -- because the capability synthesis says schema
content swings the score more than model choice (spec Section 1).

Public surface:

* :mod:`whetstone_envs.c11.generate` -- the seeded, adversarial
  generator (produces a :class:`~whetstone_envs.core.pool.TaskPool`).
* :mod:`whetstone_envs.c11.oracle` -- the canonicalization oracle, a
  pure function of an instance's public input string, delegating to
  ``rfc8785.dumps`` unmodified.
* :mod:`whetstone_envs.c11.prompts` -- the naive/ceiling probe pair,
  verbatim from the baseline spec (Section 2), with the ceiling prompt's
  worked-example outputs regenerated through the real oracle (spec
  Section 7.5).
"""

from __future__ import annotations
