r"""The two c11 probe prompts for JSON canonicalization.

Probe (a), the naive prompt, is copied byte-for-byte from the spec's
Section 2.1 ``<pre><code>`` block. Probe (b), the ceiling prompt, states
the complete RFC 8785 rules and uses oracle-generated worked-example
outputs rather than hand-typed canonical strings.

So this module does exactly that: the ceiling template is *assembled at
import time* by running each worked-example input through the real oracle
(:func:`whetstone_envs.c11.oracle.canonicalize`, which delegates to
``rfc8785.dumps`` unmodified). The ``Output:`` lines are therefore never
hand-typed -- they are the oracle's own bytes, so they can never drift
from what the oracle would score. The single templated slot is
``{input}`` (the messy instance JSON), matching the spec's ``{input}``.

The byte-for-byte render test in ``tests/c11/test_prompts.py`` pins both
rendered prompts against a fixed fixture, and independently regenerates
the worked-example outputs through ``rfc8785.dumps`` to prove they match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.c11.oracle import canonicalize
from whetstone_envs.core.probes import ProbePair

if TYPE_CHECKING:
    from whetstone_envs.core.instance import Instance

# The only templated slot in either c11 prompt.
_INPUT_SLOT = "{input}"

# Probe (a) -- deliberately underspecified naive prompt (spec Section 2.1),
# copied byte-for-byte. Names the operation, states no JCS sub-rule.
NAIVE_TEMPLATE = """Canonicalize this JSON.

{input}"""

# The worked-example inputs from spec Section 2.2, plus one focused number
# example pinning ECMAScript's positive exponent sign. Their Output lines
# are regenerated below, never hand-typed (spec Section 7.5).
WORKED_EXAMPLE_INPUTS: tuple[str, ...] = (
    '{"b": 2, "a": 1}',
    '{"x": 1.0, "y": 1e2, "z": -0}',
    '{"s": "line1\\nline2", "t": "π"}',
    '{"nested": {"d": 4, "c": 3}, "arr": [true, null]}',
    '{"large": 1e30}',
)

# The fixed rule-and-example preamble of the ceiling prompt and the fixed
# tail after the oracle-generated examples.
_CEILING_HEAD = """Convert the JSON below into its RFC 8785 (JCS) canonical form and return ONLY the
resulting string, with no code fences, no commentary, and no trailing newline.

Apply every one of these rules exactly:

1. Object keys are sorted by their UTF-16 code units (compare the raw UTF-16 code-unit
   sequences of the key strings, ascending). Sort keys at every nesting level.
2. No insignificant whitespace: no spaces or newlines anywhere except inside string
   values. `{"a":1,"b":[2,3]}`, never `{ "a": 1 }`.
3. Strings use the shortest JSON escaping: escape only " \\ and the control characters
   U+0000..U+001F. Use the two-character forms \\" \\\\ \\b \\f \\n \\r \\t where they exist;
   otherwise \\u00XX (lowercase hex). Do not escape forward slash or any other character.
4. Numbers use the ECMAScript Number-to-string shortest round-trip form for finite
   IEEE-754 doubles. For nonzero magnitudes at least 1e-6 and below 1e21, use ordinary
   decimal notation; outside that range, use exponent notation. Exponents use lowercase
   "e", no leading zero, and "+" for a positive exponent (for example, 1e+30). Thus even
   an integer-valued number can use exponent notation at large magnitudes. "-0" becomes
   "0".
5. Literals are exactly true, false, null. Arrays preserve their element order.

Worked examples:
"""

_CEILING_TAIL = """
Now canonicalize:

{input}"""


def _worked_examples_block() -> str:
    """Render the worked-example block with oracle-regenerated outputs.

    Each ``Input:``/``Output:`` pair uses the verbatim spec input and the
    canonical form the *real oracle* produces for it, so the ceiling
    prompt's examples can never disagree with how the oracle scores.
    """
    lines: list[str] = []
    for src in WORKED_EXAMPLE_INPUTS:
        lines.append(f"Input:  {src}")
        lines.append(f"Output: {canonicalize(src)}")
        lines.append("")
    return "\n".join(lines)


def render_input_slot(template: str, instance: Instance) -> str:
    """Render ``template`` by substituting only the ``{input}`` slot.

    A plain replace (not ``str.format``) is used because both prompts
    contain literal ``{`` / ``}`` braces in their rule text and worked
    examples; a format call would misread those as fields. Substitution
    draws *only* from ``instance.prompt_inputs['input']`` -- the public
    messy JSON -- so a prompt can never interpolate gold/oracle-only
    state. A template missing the slot, or an instance missing the
    ``input`` field, raises loudly rather than rendering silently wrong.
    """
    if _INPUT_SLOT not in template:
        msg = "c11 template has no {input} slot to render"
        raise KeyError(msg)
    value = dict(instance.prompt_inputs)["input"]
    return template.replace(_INPUT_SLOT, value)


def build_ceiling_template() -> str:
    """Assemble the full ceiling template (head + oracle examples + tail).

    Built from the oracle at call time rather than stored as a literal, so
    a change in ``rfc8785`` output would flow into the prompt (and be
    caught by the render test) instead of silently disagreeing with the
    oracle.
    """
    return _CEILING_HEAD + "\n" + _worked_examples_block() + _CEILING_TAIL


# Probe (b) -- best-effort ceiling prompt (spec Section 2.2), with worked
# outputs regenerated through the oracle (spec Section 7.5).
CEILING_TEMPLATE = build_ceiling_template()


PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
    render=render_input_slot,
)
"""The naive/ceiling probe pair for c11, rendered via the shared core
:func:`~whetstone_envs.core.probes.render_with_prompt_inputs` (which
formats against ``prompt_inputs`` only, so a template can never
interpolate gold/oracle-only state)."""
