r"""The two c23 character-transformation probe prompts.

Probe (a), the naive floor prompt, is the spec Section 2.1 form: the
demonstration pairs as ``IN -> OUT`` lines, then the query input followed by
``->`` and nothing else -- "the entire prompt, no system prompt beyond the
provider default." Probe (b), the best-effort ceiling prompt, states every
standard *convention* of the ISL/OSL task family (single deterministic rule,
individual-character units, bounded adjacent-character context, "fits all
demos / simplest rule" induction, no-CoT, strict ``Output:`` format) but
**never** the specific latent rule of any instance. The model still has to
induce the rule from the demos.

Both prompts interpolate exactly two instance fields -- the pre-rendered
``demos_block`` (``IN -> OUT`` lines) and the ``query`` input string -- both
public. The latent rule (the oracle-only field) is never passed to the
renderer, so a static leak test over the rendered prompt confirms neither
probe reveals it, and a byte-for-byte render test pins both against a fixed
fixture (guards against template drift).
"""

from __future__ import annotations

from whetstone_envs.core.probes import ProbePair, render_with_prompt_inputs

# --- Probe (a): deliberately naive prompt (spec Section 2.1) ---------------
# The demonstration pairs as "IN -> OUT" lines, a blank line, then the query
# input followed by "-> " (trailing space) and nothing else: no task framing,
# statement that a hidden rule exists, or output-format cue.
NAIVE_TEMPLATE = """{demos_block}

{query} -> """

# --- Probe (b): best-effort ceiling prompt (spec Section 2.2) --------------
# States the conventions and inductive bias of the task family without
# revealing any instance's latent rule. The "Output:" line is the extraction
# contract the scorer's "text after the last Output: line" step consumes.
CEILING_TEMPLATE = """SYSTEM:
You are solving a hidden-rule string-transformation puzzle. Each puzzle has one fixed,
deterministic transformation rule that maps an input string to an output string. The rule
operates on individual characters and their nearby character context — it never uses outside
knowledge, randomness, or context beyond the examples. Each input is one sequence of
characters; whitespace does not define the transformation units. The same rule was applied
to every example below. Your job: infer that single rule from the examples, then apply exactly
the same rule to the final query.

Rules of the format:
- Read all the demonstration pairs. Each is written "INPUT -> OUTPUT".
- Each input character may be preserved, replaced, or deleted according to a short,
  contiguous context of adjacent characters. Depending on the rule family, that context
  comes from the input or the partially produced output and is evaluated left-to-right or
  right-to-left.
- Determine the exact rule that makes ALL demonstrations correct simultaneously. If more than
  one rule fits the demonstrations, choose the simplest one that fits every pair.
- Do not explain your reasoning. Output only the transformed string for the query.
- Preserve spacing and casing exactly as the rule dictates. Do not add quotes, punctuation,
  or commentary.

USER:
Demonstrations:
{demos_block}

Query:
{query} ->

Output only the transformed string, on a single line prefixed with "Output:"."""


PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=render_with_prompt_inputs,
)
"""The naive/ceiling probe pair for c23 (spec Section 2), rendered via the
shared :func:`~whetstone_envs.core.probes.render_with_prompt_inputs` (which
formats against ``prompt_inputs`` only, so a template can never interpolate
the oracle-only latent rule)."""


def render_demos_block(demos: dict[str, str]) -> str:
    """Render a demo mapping as newline-joined ``IN -> OUT`` lines.

    Pairs are emitted in sorted-input order so the block is deterministic
    and independent of dict iteration order -- the same discipline the
    manifest content hash relies on.
    """
    return "\n".join(f"{i} -> {o}" for i, o in sorted(demos.items()))


def render_naive(demos_block: str, query: str) -> str:
    """Render the naive probe (spec Section 2.1) for the given fields."""
    return NAIVE_TEMPLATE.format(demos_block=demos_block, query=query)


def render_ceiling(demos_block: str, query: str) -> str:
    """Render the ceiling probe (spec Section 2.2) for the given fields."""
    return CEILING_TEMPLATE.format(demos_block=demos_block, query=query)
