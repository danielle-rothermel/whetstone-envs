"""Probe-prompt render tests: byte-for-byte + latent-rule leak (checklist A).

Both probes must render byte-for-byte as drafted in the spec Section 2 for a
fixed fixture (guards against template drift), and neither probe may leak the
*latent rule* -- the spec's "legitimate ceiling, not cheating" note: the
ceiling states the task conventions but never the specific rule, so a static
check confirms the rule's context/output tokens do not appear as an explicit
rule statement in the rendered ceiling prompt.
"""

from __future__ import annotations

from whetstone_envs.c23.generate import generate_pool
from whetstone_envs.c23.prompts import (
    CEILING_TEMPLATE,
    NAIVE_TEMPLATE,
    PROBES,
    render_ceiling,
    render_demos_block,
    render_naive,
)
from whetstone_envs.core.instance import make_instance

# A fixed fixture: three hand-written demos and a query, all character
# strings over {a,b,c,d} (the |Sigma|=4 alphabet).
_DEMOS = {"acb": "acd", "bcb": "bcd", "aa": "aa"}
_QUERY = "cacb"
_DEMOS_BLOCK = "aa -> aa\nacb -> acd\nbcb -> bcd"  # sorted-input order

_FIXTURE = make_instance(
    id="c23-fixture",
    seed=555_000_000,
    strata="S1",
    prompt_inputs={"demos_block": _DEMOS_BLOCK, "query": _QUERY},
    gold="cacd",
)

# Spec Section 2.1, verbatim, with the fixture's demos + query substituted.
_EXPECTED_NAIVE = "aa -> aa\nacb -> acd\nbcb -> bcd\n\ncacb -> "

# Spec Section 2.2, byte-for-byte, with the fixture substituted. The em
# dashes are the spec's own U+2014.
_EXPECTED_CEILING = (
    "SYSTEM:\n"
    "You are solving a hidden-rule string-transformation puzzle. Each puzzle has one fixed,\n"
    "deterministic transformation rule that maps an input string to an output string. The rule\n"
    "depends only on the tokens in the input (their identity, their length, their position, and\n"
    "the characters at the ends of each token) — it never uses outside knowledge, randomness, or\n"
    "context beyond the examples. Tokens are the whitespace-separated words. The same rule was\n"
    "applied to every example below. Your job: infer that single rule from the examples, then\n"
    "apply exactly the same rule to the final query.\n"
    "\n"
    "Rules of the format:\n"
    '- Read all the demonstration pairs. Each is written "INPUT -> OUTPUT".\n'
    "- The transformation is a length- and position- and suffix/prefix-sensitive edit over the\n"
    "  tokens: tokens may be duplicated, reordered, re-cased, or rewritten based on their local\n"
    "  context (the characters immediately around each token position, up to a small window).\n"
    "- Determine the exact rule that makes ALL demonstrations correct simultaneously. If more than\n"
    "  one rule fits the demonstrations, choose the simplest one that fits every pair.\n"
    "- Do not explain your reasoning. Output only the transformed string for the query.\n"
    "- Preserve spacing and casing exactly as the rule dictates. Do not add quotes, punctuation,\n"
    "  or commentary.\n"
    "\n"
    "USER:\n"
    "Demonstrations:\n"
    "aa -> aa\nacb -> acd\nbcb -> bcd\n"
    "\n"
    "Query:\n"
    "cacb ->\n"
    "\n"
    'Output only the transformed string, on a single line prefixed with "Output:".'
)


def test_render_demos_block_is_sorted_in_out_lines() -> None:
    assert render_demos_block(_DEMOS) == _DEMOS_BLOCK


def test_naive_render_is_byte_for_byte_fixed() -> None:
    assert render_naive(_DEMOS_BLOCK, _QUERY) == _EXPECTED_NAIVE


def test_ceiling_render_is_byte_for_byte_fixed() -> None:
    assert render_ceiling(_DEMOS_BLOCK, _QUERY) == _EXPECTED_CEILING


def test_probepair_renders_match_the_helpers() -> None:
    assert PROBES.render_naive(_FIXTURE) == _EXPECTED_NAIVE
    assert PROBES.render_ceiling(_FIXTURE) == _EXPECTED_CEILING


def test_naive_prompt_ends_with_the_open_query_arrow() -> None:
    # Spec 2.1: the query input then "-> " and nothing else.
    assert render_naive(_DEMOS_BLOCK, _QUERY).endswith("cacb -> ")


def test_ceiling_prompt_ends_on_the_output_extraction_contract() -> None:
    # Spec 2.2's strict output format (feeds the "text after last Output:").
    assert render_ceiling(_DEMOS_BLOCK, _QUERY).endswith(
        'prefixed with "Output:".',
    )


def test_ceiling_states_conventions_but_not_a_concrete_rule() -> None:
    # The "legitimate ceiling, not cheating" note (spec 2.2): the ceiling
    # states the family's conventions (single deterministic rule, tokens =
    # whitespace words, suffix/positional sensitivity, fits-all/simplest,
    # strict Output:) but reveals no specific latent rule.
    rendered = render_ceiling(_DEMOS_BLOCK, _QUERY)
    for convention in (
        "one fixed",
        "whitespace-separated words",
        "suffix/prefix-sensitive",
        "simplest",
        "Output:",
    ):
        assert convention in rendered
    # It must NOT contain an explicit rule statement of the fixture's latent
    # rule (e.g. "cb -> d"): only the demos' own IN -> OUT lines appear.
    assert "cb -> d" not in rendered


def test_latent_rule_is_never_passed_to_the_renderer() -> None:
    # Static leak test: give an instance a distinctive sentinel gold and a
    # sentinel latent-rule token; neither may appear in either rendered probe
    # because the renderer only ever sees prompt_inputs (demos_block, query).
    sentinel = "ZZ_LATENT_RULE_ZZ"
    inst = make_instance(
        id="c23-sentinel",
        seed=555_000_000,
        strata="S1",
        prompt_inputs={"demos_block": _DEMOS_BLOCK, "query": _QUERY},
        gold=sentinel,
    )
    assert sentinel not in PROBES.render_naive(inst)
    assert sentinel not in PROBES.render_ceiling(inst)


def test_no_prompt_leaks_the_gold_over_a_real_pool() -> None:
    # Over a real default-shape pool: the gold string is never embedded in a
    # rendered prompt (the renderer formats only public prompt_inputs).
    pool = generate_pool(n_per_stratum=3)
    for inst in pool.instances:
        naive = PROBES.render_naive(inst)
        ceiling = PROBES.render_ceiling(inst)
        assert inst.prompt_inputs["query"] in naive
        assert inst.prompt_inputs["demos_block"] in ceiling
        assert "gold" not in inst.prompt_inputs


def test_templates_reference_only_public_fields() -> None:
    for template in (NAIVE_TEMPLATE, CEILING_TEMPLATE):
        assert "{demos_block}" in template
        assert "{query}" in template
        assert "{gold}" not in template
        assert "{rule}" not in template
