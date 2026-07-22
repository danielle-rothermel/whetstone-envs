"""Tests for probe pairing and shared prediction normalization."""

from __future__ import annotations

import pytest

from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.probes import (
    ProbePair,
    normalize,
    render_with_prompt_inputs,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  yes  ", "yes"),
        ("yes\n", "yes"),
        ("```\nyes\n```", "yes"),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("  ```\n  answer  \n```  ", "answer"),
        ("no fence here", "no fence here"),
        ("```only one backtick line", "```only one backtick line"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    raw = '```json\n{"a": 1}\n```'
    once = normalize(raw)
    assert normalize(once) == once


def test_normalize_preserves_internal_backticks() -> None:
    # A lone backtick inside the answer must not be eaten.
    assert normalize("a `b` c") == "a `b` c"


def test_render_uses_only_prompt_inputs() -> None:
    inst = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"question": "Q?", "hint": "H"},
        gold="SECRET",
    )
    pair = ProbePair(
        naive_template="Answer: {question}",
        ceiling_template="Think about {hint}, then answer {question}.",
        render=render_with_prompt_inputs,
    )
    assert pair.render_naive(inst) == "Answer: Q?"
    assert pair.render_ceiling(inst) == "Think about H, then answer Q?."


def test_render_cannot_leak_gold() -> None:
    inst = make_instance(
        id="t1", seed=1, strata="s", prompt_inputs={"q": "x"}, gold="LEAK"
    )
    pair = ProbePair(
        naive_template="{gold}",
        ceiling_template="{q}",
        render=render_with_prompt_inputs,
    )
    # gold is not a prompt input, so the template field cannot resolve.
    with pytest.raises(KeyError):
        pair.render_naive(inst)


def test_missing_field_raises_loudly() -> None:
    inst = make_instance(id="t1", seed=1, strata="s", prompt_inputs={"q": "x"})
    pair = ProbePair(
        naive_template="{does_not_exist}",
        ceiling_template="{q}",
        render=render_with_prompt_inputs,
    )
    with pytest.raises(KeyError):
        pair.render_naive(inst)
