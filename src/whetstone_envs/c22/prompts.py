r"""The two c22 probe prompts, verbatim from the baseline spec (Section 2).

Both prompts receive the same generated instance and interpolate exactly
one field -- the base-task-plus-constraints block
(``{constraints_block}``, the spec's
``{BASE_TASK_AND_CONCATENATED_CONSTRAINTS}`` slot). Neither prompt
interpolates any gold/oracle-only field: the constraint stack in the
reseed-only baseline is fully stated (spec Section 2.2, "nothing is
hidden"), so the block the naive prompt shows is the same text the
ceiling prompt enumerates.

* Probe (a) -- deliberately minimal naive prompt (spec Section 2.1).
* Probe (b) -- best-effort ceiling prompt with all standard conventions
  stated (spec Section 2.2).

The templates below are copied byte-for-byte from the spec's drafted
``<pre><code>`` blocks, with the single templated slot renamed from
``{BASE_TASK_AND_CONCATENATED_CONSTRAINTS}`` to ``{constraints_block}``
to match the instance's ``prompt_inputs`` key. The byte-for-byte render
test in ``tests/c22/test_prompts.py`` pins them against a fixed fixture.
"""

from __future__ import annotations

from whetstone_envs.core.probes import ProbePair, render_with_prompt_inputs

# Probe (a): the entire prompt is the raw generated instance text, a
# blank line, and "Answer:" -- no restatement, enumeration, or hygiene.
NAIVE_TEMPLATE = """{constraints_block}

Answer:"""

# Probe (b): states every standard convention of the reseed-only task --
# strict all-pass semantics, output-only hygiene, per-atom-type hints,
# and a self-verify pass.
CEILING_TEMPLATE = """You must produce a response that satisfies EVERY constraint below. The response
is scored 1 only if all constraints pass a deterministic checker; missing even
one constraint scores 0. There is no partial credit.

Constraints (all must hold simultaneously):
{constraints_block}

Follow these conventions of this task exactly:
- Every constraint is stated explicitly above. Nothing is hidden; you do not need
  to infer any unstated rule. Enumerate each constraint and satisfy it directly.
- Output ONLY the answer text itself. Do not add a preamble, explanation,
  restatement, label, quotation marks, or trailing commentary — extra text can
  itself violate a length, word-count, casing, or forbidden-token constraint.
- Do NOT use markdown, bold, bullets, or headers unless a constraint explicitly
  requires them; stray formatting characters count against exact-match checks.
- For any word-count or length constraint, count exactly and match the stated
  number precisely (exact means exact, not "about").
- For any forbidden-letter or forbidden-word constraint, scan your whole answer
  and confirm the letter/word appears nowhere.
- For any casing, start-token, or end-token constraint, verify the first/last
  characters or tokens literally match what is required.
- Before finalizing, silently check each constraint one more time; if any fails,
  revise until all pass. Then output only the final answer.

Answer:"""


PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=render_with_prompt_inputs,
)
"""The naive/ceiling probe pair for c22, rendered via the shared core
:func:`~whetstone_envs.core.probes.render_with_prompt_inputs` (which
formats against ``prompt_inputs`` only, so a template can never
interpolate gold/oracle-only state)."""
