"""Tests for probe-prompt pairing and rendering."""

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.probes import ProbePair, render_with_prompt_inputs


def test_render_uses_only_prompt_inputs() -> None:
    instance = make_instance(
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
    assert pair.render_naive(instance) == "Answer: Q?"
    assert pair.render_ceiling(instance) == "Think about H, then answer Q?."


def test_probe_pair_defaults_to_prompt_input_renderer() -> None:
    instance = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"question": "Q?", "hint": "H"},
    )
    pair = ProbePair(
        naive_template="Answer: {question}",
        ceiling_template="Use {hint} for {question}",
    )

    assert pair.render_naive(instance) == "Answer: Q?"
    assert pair.render_ceiling(instance) == "Use H for Q?"


def test_render_cannot_leak_gold() -> None:
    instance = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"q": "x"},
        gold="LEAK",
    )
    pair = ProbePair(
        naive_template="{gold}",
        ceiling_template="{q}",
        render=render_with_prompt_inputs,
    )
    with pytest.raises(KeyError):
        pair.render_naive(instance)


def test_missing_field_raises_loudly() -> None:
    instance = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"q": "x"},
    )
    pair = ProbePair(
        naive_template="{does_not_exist}",
        ceiling_template="{q}",
        render=render_with_prompt_inputs,
    )
    with pytest.raises(KeyError):
        pair.render_naive(instance)
