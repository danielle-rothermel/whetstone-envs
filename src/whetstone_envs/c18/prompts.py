r"""The two c18 probe prompts, verbatim from the baseline spec (Section 2).

Probe (a), the naive prompt, is copied byte-for-byte from spec Section
2.1; probe (b), the best-effort ceiling prompt, from spec Section 2.2.
Both take the generator's native ``question`` (facts + rules) and
``query`` fields -- the instance's public ``prompt_inputs`` -- and nothing
else. The entailment label (the gold) never appears in either prompt; a
static test asserts no rendered prompt contains the gold token, and a
byte-for-byte render test pins both rendered prompts against a fixed
fixture (guards against template drift).

The templates are ``str.format``-style with ``{question}`` and ``{query}``
fields, rendered through the shared
:func:`whetstone_envs.core.probes.render_with_prompt_inputs`, which
formats only against ``instance.prompt_inputs`` so a gold field can never
be interpolated even by accident.
"""

from __future__ import annotations

from whetstone_envs.core.probes import ProbePair, render_with_prompt_inputs

# --- Probe (a): deliberately naive prompt (spec Section 2.1) ---------------
# Byte-for-byte from the spec. No chain-of-thought, no closed-world
# convention, no output-format discipline beyond "True or False."
NAIVE_TEMPLATE = """{question}

{query}

Answer True or False."""

# --- Probe (b): best-effort ceiling prompt (spec Section 2.2) --------------
# Byte-for-byte from the spec. Each clause maps to a literature-supported
# lever (rely-only-on-given, closed-world True/False convention, reason
# step by step, one-word final line). The em dash in "no real-world
# meaning" is the spec's own U+2014.
CEILING_TEMPLATE = """You are a careful deductive reasoner. You are given a set of facts and if-then
rules, followed by a single query statement. Every predicate is fictional and
carries no real-world meaning — rely ONLY on the facts and rules given, never on
outside knowledge or surface plausibility.

Determine whether the query statement is entailed by the facts and rules under
the closed-world assumption: a statement is True if it can be derived by
chaining the given rules from the given facts, and False otherwise. Apply the
rules step by step, following each "every X is a Y" / "X are (not) Z" rule in
order, until you reach the queried property.

Facts and rules:
{question}

Query:
{query}

Reason step by step through the chain of rules, then end your reply with exactly
one word on its own final line: either
True
or
False"""


PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=render_with_prompt_inputs,
)
"""The naive/ceiling probe pair for c18 (spec Section 2), rendered via the
shared ``render_with_prompt_inputs`` against only the public
``question``/``query`` inputs -- never the gold label."""


def render_naive(question: str, query: str) -> str:
    """Render the naive probe (spec Section 2.1) for the given fields."""
    return NAIVE_TEMPLATE.format(question=question, query=query)


def render_ceiling(question: str, query: str) -> str:
    """Render the ceiling probe (spec Section 2.2) for the given fields."""
    return CEILING_TEMPLATE.format(question=question, query=query)
