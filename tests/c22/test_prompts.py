"""Probe-prompt checks for c22.

Two blocking checks from PLAN Verification checklist A:

* **byte-for-byte render** against a fixed fixture instance (guards
  against template drift from the baseline spec's Section 2 text);
* **static no-gold-leak** check: neither rendered prompt contains any
  gold/oracle-only field (the constraint ids/kwargs), and the templates
  reference only the single public ``constraints_block`` field.
"""

from __future__ import annotations

import json

from whetstone_envs.c22.prompts import (
    CEILING_TEMPLATE,
    NAIVE_TEMPLATE,
    PROBES,
)
from whetstone_envs.core.instance import make_instance

# A fixed fixture: a hand-written base-task-and-constraints block. Using a
# frozen literal (not a generator-produced instance) keeps this test a
# stable golden guard against template drift.
_BLOCK = (
    "Name a color.\n"
    "\n"
    "1. Wrap your entire response with double quotation marks.\n"
    "2. In your entire response, refrain from the use of any commas.\n"
    "3. Finish your response with this exact phrase DONE. No other words"
    " should follow this phrase."
)

_FIXTURE = make_instance(
    id="c22-fixture",
    seed=1_000_000,
    strata="n3_easy",
    prompt_inputs={"constraints_block": _BLOCK},
    gold=json.dumps(
        {
            "instruction_id_list": [
                "startend:quotation",
                "punctuation:no_comma",
                "startend:end_checker",
            ],
            "kwargs_list": [{}, {}, {"end_phrase": "DONE"}],
        },
    ),
)

# Golden renderings, verbatim from the baseline spec Section 2 with the
# single templated slot filled by _BLOCK. If the spec text is edited, this
# literal must be updated deliberately -- that is the drift guard.
_EXPECTED_NAIVE = _BLOCK + "\n\nAnswer:"

_EXPECTED_CEILING = (
    "You must produce a response that satisfies EVERY constraint below."
    " The response\n"
    "is scored 1 only if all constraints pass a deterministic checker;"
    " missing even\n"
    "one constraint scores 0. There is no partial credit.\n"
    "\n"
    "Constraints (all must hold simultaneously):\n" + _BLOCK + "\n"
    "\n"
    "Follow these conventions of this task exactly:\n"
    "- Every constraint is stated explicitly above. Nothing is hidden;"
    " you do not need\n"
    "  to infer any unstated rule. Enumerate each constraint and satisfy"
    " it directly.\n"
    "- Output ONLY the answer text itself. Do not add a preamble,"
    " explanation,\n"
    "  restatement, label, quotation marks, or trailing commentary —"
    " extra text can\n"
    "  itself violate a length, word-count, casing, or forbidden-token"
    " constraint.\n"
    "- Do NOT use markdown, bold, bullets, or headers unless a constraint"
    " explicitly\n"
    "  requires them; stray formatting characters count against"
    " exact-match checks.\n"
    "- For any word-count or length constraint, count exactly and match"
    " the stated\n"
    '  number precisely (exact means exact, not "about").\n'
    "- For any forbidden-letter or forbidden-word constraint, scan your"
    " whole answer\n"
    "  and confirm the letter/word appears nowhere.\n"
    "- For any casing, start-token, or end-token constraint, verify the"
    " first/last\n"
    "  characters or tokens literally match what is required.\n"
    "- Before finalizing, silently check each constraint one more time;"
    " if any fails,\n"
    "  revise until all pass. Then output only the final answer.\n"
    "\n"
    "Answer:"
)


def test_naive_prompt_renders_byte_for_byte() -> None:
    assert PROBES.render_naive(_FIXTURE) == _EXPECTED_NAIVE


def test_ceiling_prompt_renders_byte_for_byte() -> None:
    assert PROBES.render_ceiling(_FIXTURE) == _EXPECTED_CEILING


def test_templates_reference_only_the_public_block_field() -> None:
    # Static structural guard: the only format field either template uses
    # is ``constraints_block`` -- so a template can never interpolate a
    # gold/oracle-only field even by accident.
    import string

    for template in (NAIVE_TEMPLATE, CEILING_TEMPLATE):
        fields = {
            name
            for _, name, _, _ in string.Formatter().parse(template)
            if name
        }
        assert fields == {"constraints_block"}


def test_neither_prompt_leaks_gold_only_fields() -> None:
    # Static no-gold-leak: the rendered prompts must not contain the raw
    # instruction ids or kwargs the oracle reads from gold.
    gold = json.loads(_FIXTURE.gold)
    naive = PROBES.render_naive(_FIXTURE)
    ceiling = PROBES.render_ceiling(_FIXTURE)
    for atom_id in gold["instruction_id_list"]:
        assert atom_id not in naive
        assert atom_id not in ceiling
    # The end-phrase kwarg value ("DONE") legitimately appears in the
    # human-readable constraint text, but the raw kwargs JSON must not.
    raw_kwargs = json.dumps(gold["kwargs_list"])
    assert raw_kwargs not in naive
    assert raw_kwargs not in ceiling


def test_ceiling_at_least_as_long_as_naive() -> None:
    # Sanity: the ceiling prompt strictly extends the naive one (it
    # embeds the same block plus the convention hints).
    assert len(PROBES.render_ceiling(_FIXTURE)) > len(
        PROBES.render_naive(_FIXTURE),
    )
