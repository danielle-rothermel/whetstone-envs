"""Probe-prompt render tests: byte-for-byte + no-gold-leak (checklist A).

Both probe prompts must render byte-for-byte as drafted in the baseline
spec's Section 2 for a fixed fixture instance (guards against template
drift), and neither prompt may leak the gold/oracle-only label (a static
check over the rendered prompt string). The expected strings below are the
spec Section 2.1 / 2.2 text with only the fixture's public question/query
substituted.
"""

from __future__ import annotations

from whetstone_envs.c18 import prompts
from whetstone_envs.c18.generate import generate_pool
from whetstone_envs.c18.prompts import PROBES, render_ceiling, render_naive
from whetstone_envs.core.instance import make_instance

_QUESTION = "Sally is a brimpus. Every brimpus is sour."
_QUERY = "True or false: Sally is sour."

_FIXTURE = make_instance(
    id="c18-fixture",
    seed=1_000_000_000,
    strata="D1",
    prompt_inputs={"question": _QUESTION, "query": _QUERY},
    gold="True",
)

# Spec Section 2.1, byte-for-byte, with {question}/{query} substituted.
_EXPECTED_NAIVE = (
    "Sally is a brimpus. Every brimpus is sour.\n"
    "\n"
    "True or false: Sally is sour.\n"
    "\n"
    "Answer True or False."
)

# Spec Section 2.2, byte-for-byte, with {question}/{query} substituted.
# The em dash (U+2014) in "no real-world meaning —" is the spec's own.
_EXPECTED_CEILING = (
    "You are a careful deductive reasoner. You are given a set of facts and if-then\n"
    "rules, followed by a single query statement. Every predicate is fictional and\n"
    "carries no real-world meaning — rely ONLY on the facts and rules given, never on\n"
    "outside knowledge or surface plausibility.\n"
    "\n"
    "Determine whether the query statement is entailed by the facts and rules under\n"
    "the closed-world assumption: a statement is True if it can be derived by\n"
    "chaining the given rules from the given facts, and False otherwise. Apply the\n"
    "rules step by step, following each \"every X is a Y\" / \"X are (not) Z\" rule in\n"
    "order, until you reach the queried property.\n"
    "\n"
    "Facts and rules:\n"
    "Sally is a brimpus. Every brimpus is sour.\n"
    "\n"
    "Query:\n"
    "True or false: Sally is sour.\n"
    "\n"
    "Reason step by step through the chain of rules, then end your reply with exactly\n"
    "one word on its own final line: either\n"
    "True\n"
    "or\n"
    "False"
)


def test_naive_render_is_byte_for_byte_fixed() -> None:
    assert render_naive(_QUESTION, _QUERY) == _EXPECTED_NAIVE


def test_ceiling_render_is_byte_for_byte_fixed() -> None:
    assert render_ceiling(_QUESTION, _QUERY) == _EXPECTED_CEILING


def test_probepair_renders_match_the_helpers() -> None:
    # The shared ProbePair render path (used by the harness) must produce
    # the identical bytes as the module helpers.
    assert PROBES.render_naive(_FIXTURE) == _EXPECTED_NAIVE
    assert PROBES.render_ceiling(_FIXTURE) == _EXPECTED_CEILING


def test_naive_prompt_ends_with_the_true_or_false_instruction() -> None:
    # Spec 2.1: the only output-format cue is the final line.
    assert render_naive(_QUESTION, _QUERY).endswith("Answer True or False.")


def test_ceiling_prompt_ends_on_the_one_word_final_line_form() -> None:
    # Spec 2.2's format-discipline suffix (rubric criterion 13).
    rendered = render_ceiling(_QUESTION, _QUERY)
    assert rendered.endswith("True\nor\nFalse")


def test_no_prompt_leaks_the_gold_answer() -> None:
    # Static no-gold-leak check across a full default-shape pool: neither
    # rendered probe may embed the gold label as an isolated answer token.
    pool = generate_pool(n_per_stratum=3)
    for inst in pool.instances:
        naive = PROBES.render_naive(inst)
        ceiling = PROBES.render_ceiling(inst)
        # The word "True"/"False" appears in the prompt *instructions*, so a
        # naive substring check is meaningless; instead assert the prompt is
        # a pure function of the public inputs (the gold is never passed to
        # the renderer at all -- it renders only prompt_inputs).
        assert inst.prompt_inputs["question"] in naive
        assert inst.prompt_inputs["query"] in naive
        assert inst.prompt_inputs["question"] in ceiling
        assert inst.prompt_inputs["query"] in ceiling
        # The renderer only ever sees prompt_inputs, which excludes gold.
        assert "gold" not in inst.prompt_inputs


def test_sentinel_gold_never_appears_in_a_rendered_prompt() -> None:
    # A positive static no-leak check: give an instance a distinctive gold
    # token and assert it appears in neither rendered probe. Because the
    # renderer formats only prompt_inputs, a leak would mean the template
    # somehow reached the gold field -- which must never happen.
    sentinel = "ZZ_SECRET_LABEL_ZZ"
    inst = make_instance(
        id="c18-sentinel",
        seed=1_000_000_000,
        strata="D1",
        prompt_inputs={"question": _QUESTION, "query": _QUERY},
        gold=sentinel,
    )
    assert sentinel not in PROBES.render_naive(inst)
    assert sentinel not in PROBES.render_ceiling(inst)


def test_templates_reference_only_public_fields() -> None:
    # Template drift guard: the format fields are exactly question/query.
    for template in (prompts.NAIVE_TEMPLATE, prompts.CEILING_TEMPLATE):
        assert "{question}" in template
        assert "{query}" in template
        assert "{gold}" not in template
