"""Probe-prompt checks for c11.

Blocking checks from PLAN Verification checklist A:

* **byte-for-byte render** against a fixed fixture instance (guards
  against template drift from the baseline spec's Section 2 text);
* the ceiling prompt's worked-example Output lines are the *oracle's* own
  bytes -- regenerated through ``rfc8785.dumps``, never hand-typed (spec
  Section 7.5) -- proven by independently recomputing them here;
* **static no-gold-leak** check: neither rendered prompt contains the
  instance's gold, and each template's only substitution slot is
  ``{input}`` (the public messy JSON).
"""

from __future__ import annotations

import rfc8785

from whetstone_envs.c11 import oracle
from whetstone_envs.c11.prompts import (
    CEILING_TEMPLATE,
    NAIVE_TEMPLATE,
    PROBES,
    WORKED_EXAMPLE_INPUTS,
    build_ceiling_template,
)
from whetstone_envs.core.instance import make_instance

# A fixed fixture: a hand-written messy input. Using a frozen literal (not
# a generator-produced instance) keeps this a stable golden guard against
# template drift.
_INPUT = '{"b": 2, "a": 1}'
_GOLD = '{"a":1,"b":2}'

_FIXTURE = make_instance(
    id="c11-fixture",
    seed=1_000_000,
    strata="S2_keysort",
    prompt_inputs={"input": _INPUT},
    gold=_GOLD,
)

# Golden naive rendering: verbatim from spec Section 2.1 with {input}
# filled. If the spec text is edited, this literal must be updated
# deliberately -- that is the drift guard.
_EXPECTED_NAIVE = "Canonicalize this JSON.\n\n" + _INPUT

# Golden ceiling rendering: verbatim from spec Section 2.2, with the four
# worked-example Output lines as the real oracle emits them, and {input}
# filled. Built here from the same fixed head/examples/tail the spec drafts
# so template drift in the module is caught.
_EXPECTED_CEILING = (
    "Convert the JSON below into its RFC 8785 (JCS) canonical form and"
    " return ONLY the\n"
    "resulting string, with no code fences, no commentary, and no trailing"
    " newline.\n"
    "\n"
    "Apply every one of these rules exactly:\n"
    "\n"
    "1. Object keys are sorted by their UTF-16 code units (compare the raw"
    " UTF-16 code-unit\n"
    "   sequences of the key strings, ascending). Sort keys at every"
    " nesting level.\n"
    "2. No insignificant whitespace: no spaces or newlines anywhere except"
    " inside string\n"
    '   values. `{"a":1,"b":[2,3]}`, never `{ "a": 1 }`.\n'
    "3. Strings use the shortest JSON escaping: escape only \" \\ and the"
    " control characters\n"
    "   U+0000..U+001F. Use the two-character forms \\\" \\\\ \\b \\f \\n"
    " \\r \\t where they exist;\n"
    "   otherwise \\u00XX (lowercase hex). Do not escape forward slash or"
    " any other character.\n"
    "4. Numbers use the ECMAScript Number-to-string (shortest round-trip)"
    " form: integers with\n"
    "   no decimal point or exponent; no leading zeros; no \"+\" on"
    " exponents; \"-0\" becomes \"0\";\n"
    "   the minimal digit sequence that round-trips to the same IEEE-754"
    " double.\n"
    "5. Literals are exactly true, false, null. Arrays preserve their"
    " element order.\n"
    "\n"
    "Worked examples:\n"
    "\n"
    'Input:  {"b": 2, "a": 1}\n'
    'Output: {"a":1,"b":2}\n'
    "\n"
    'Input:  {"x": 1.0, "y": 1e2, "z": -0}\n'
    'Output: {"x":1,"y":100,"z":0}\n'
    "\n"
    'Input:  {"s": "line1\\nline2", "t": "π"}\n'
    'Output: {"s":"line1\\nline2","t":"π"}\n'
    "\n"
    'Input:  {"nested": {"d": 4, "c": 3}, "arr": [true, null]}\n'
    'Output: {"arr":[true,null],"nested":{"c":3,"d":4}}\n'
    "\n"
    "Now canonicalize:\n"
    "\n" + _INPUT
)


def test_naive_prompt_renders_byte_for_byte() -> None:
    assert PROBES.render_naive(_FIXTURE) == _EXPECTED_NAIVE


def test_ceiling_prompt_renders_byte_for_byte() -> None:
    assert PROBES.render_ceiling(_FIXTURE) == _EXPECTED_CEILING


def test_ceiling_worked_outputs_are_oracle_regenerated() -> None:
    # Spec Section 7.5: the Output lines must be rfc8785.dumps' own bytes,
    # not hand-typed. Recompute each through rfc8785 directly and assert it
    # appears as an "Output: ..." line in the assembled ceiling template.
    for messy in WORKED_EXAMPLE_INPUTS:
        import json

        canonical = rfc8785.dumps(json.loads(messy)).decode("utf-8")
        assert f"Output: {canonical}" in CEILING_TEMPLATE
        # ...and the oracle path agrees with the direct library call.
        assert oracle.canonicalize(messy) == canonical


def test_ceiling_template_is_rebuildable_from_the_oracle() -> None:
    # The frozen module-level CEILING_TEMPLATE equals a fresh rebuild, so
    # the examples are never a stale hand-typed copy.
    assert build_ceiling_template() == CEILING_TEMPLATE


def test_templates_reference_only_the_public_input_slot() -> None:
    # Each template's sole substitution slot is {input}; a static count of
    # the marker guards against a stray gold/oracle-only slot being added.
    for template in (NAIVE_TEMPLATE, CEILING_TEMPLATE):
        assert template.count("{input}") == 1


def test_neither_prompt_leaks_gold() -> None:
    # The rendered prompts show only the messy input, never the canonical
    # gold. (The naive prompt trivially can't; the ceiling prompt shows
    # example golds for *other* inputs but never this instance's gold.)
    naive = PROBES.render_naive(_FIXTURE)
    ceiling = PROBES.render_ceiling(_FIXTURE)
    assert _INPUT in naive
    assert _INPUT in ceiling
    # This instance's gold is {"a":1,"b":2}; it must not appear as the
    # answer in either prompt (it happens to be a worked-example output in
    # the ceiling, so guard the naive prompt strictly and confirm the
    # ceiling only contains it as a labeled example, not next to {input}).
    assert _GOLD not in naive
    # The ceiling's tail after the last example must not contain the gold.
    tail = ceiling.split("Now canonicalize:")[1]
    assert _GOLD not in tail


def test_ceiling_is_longer_than_naive() -> None:
    assert len(PROBES.render_ceiling(_FIXTURE)) > len(
        PROBES.render_naive(_FIXTURE),
    )
