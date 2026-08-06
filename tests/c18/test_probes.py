from __future__ import annotations

from whetstone_envs.c18.probes import CEILING_TEMPLATE, NAIVE_TEMPLATE, PROBES
from whetstone_envs.instances import Instance, make_instance

_QUESTION = "Sally is a brimpus. Every brimpus is sour."
_QUERY = "True or false: Sally is sour."


def _instance(gold: str = "True") -> Instance:
    return make_instance(
        id="c18-probe",
        seed=1_000_000_000,
        strata="D1",
        prompt_inputs={"question": _QUESTION, "query": _QUERY},
        gold=gold,
    )


def test_naive_probe_is_pinned() -> None:
    assert PROBES.render_naive(_instance()) == NAIVE_TEMPLATE.format(
        question=_QUESTION,
        query=_QUERY,
    )


def test_ceiling_probe_requests_a_terminal_verdict() -> None:
    rendered = PROBES.render_ceiling(_instance())
    assert rendered == CEILING_TEMPLATE.format(
        question=_QUESTION,
        query=_QUERY,
    )
    assert rendered.endswith("either\nTrue\nor\nFalse")


def test_probe_rendering_cannot_reach_gold() -> None:
    sentinel = "ZZ_SECRET_LABEL_ZZ"
    instance = _instance(gold=sentinel)
    assert sentinel not in PROBES.render_naive(instance)
    assert sentinel not in PROBES.render_ceiling(instance)
    assert "{gold}" not in NAIVE_TEMPLATE
    assert "{gold}" not in CEILING_TEMPLATE
