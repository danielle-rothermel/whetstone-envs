"""Tests for probe-prompt pairing and rendering."""

import pytest

from whetstone_envs.instances import Instance, make_instance
from whetstone_envs.probes import ProbePair, render_with_prompt_inputs


def test_default_renderer_is_restricted_to_prompt_inputs() -> None:
    instance = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"question": "Q?", "hint": "H"},
        gold="SECRET",
    )

    rendered = render_with_prompt_inputs("Answer: {question}", instance)
    assert rendered == "Answer: Q?"
    with pytest.raises(KeyError):
        render_with_prompt_inputs("{gold}", instance)


def test_probe_pair_routes_templates_through_default_renderer() -> None:
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


def test_custom_renderer_receives_templates_and_full_instance() -> None:
    instance = make_instance(
        id="t1",
        seed=1,
        strata="s",
        prompt_inputs={"q": "x"},
        gold="private answer",
    )
    calls: list[tuple[str, Instance]] = []

    def recording_renderer(template: str, received: Instance) -> str:
        calls.append((template, received))
        return f"{template}: {received.gold}"

    pair = ProbePair(
        naive_template="naive",
        ceiling_template="ceiling",
        render=recording_renderer,
    )

    assert pair.render_naive(instance) == "naive: private answer"
    assert pair.render_ceiling(instance) == "ceiling: private answer"
    assert calls == [("naive", instance), ("ceiling", instance)]
