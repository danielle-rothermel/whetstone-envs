"""Naive and known-good prompts for C11 tasks."""

from whetstone_envs.c11.generation import INPUT_JSON_FIELD
from whetstone_envs.probes import ProbePair

NAIVE_TEMPLATE = """Canonicalize this JSON.

{input_json}"""

_WORKED_EXAMPLES = (
    ('{"b": 2, "a": 1}', '{"a":1,"b":2}'),
    ('{"x": 1.0, "y": 1e2, "z": -0}', '{"x":1,"y":100,"z":0}'),
    (
        '{"s": "line1\\nline2", "t": "π"}',
        '{"s":"line1\\nline2","t":"π"}',
    ),
    (
        '{"nested": {"d": 4, "c": 3}, "arr": [true, null]}',
        '{"arr":[true,null],"nested":{"c":3,"d":4}}',
    ),
    ('{"large": 1e30}', '{"large":1e+30}'),
)

_CEILING_INSTRUCTIONS = """Convert the JSON below into its RFC 8785 (JCS)
canonical form. Return only the resulting JSON text, with no code fence,
commentary, or trailing newline.

Apply every rule exactly:

1. Sort object keys by their UTF-16 code units at every nesting level.
2. Remove insignificant whitespace. Arrays preserve their element order.
3. Escape only quotation marks, backslashes, and U+0000 through U+001F.
   Use the short escapes where available and lowercase hexadecimal otherwise.
4. Render finite IEEE-754 numbers with ECMAScript's shortest round-trip form.
   Use lowercase `e`, include `+` for positive exponents, and render `-0` as
   `0`.
5. Render literals exactly as `true`, `false`, and `null`.

Worked examples:
"""


def _worked_examples_text() -> str:
    return "\n\n".join(
        f"Input:  {source}\nOutput: {expected}"
        for source, expected in _WORKED_EXAMPLES
    )


def _escape_literal_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


CEILING_TEMPLATE = (
    _escape_literal_braces(_CEILING_INSTRUCTIONS + _worked_examples_text())
    + f"\n\nNow canonicalize:\n\n{{{INPUT_JSON_FIELD}}}"
)

PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
)
