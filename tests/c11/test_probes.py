import pytest

from whetstone_envs.c11 import PROBES, canonicalize
from whetstone_envs.instances import make_instance
from whetstone_envs.probes import render_with_prompt_inputs

_INPUT_JSON = '{"b": 2, "a": 1}'
_GOLD_SENTINEL = "PRIVATE-GOLD-SENTINEL"
_INSTANCE = make_instance(
    id="c11-fixture",
    seed=1,
    strata="c11/key-order",
    prompt_inputs={"input_json": _INPUT_JSON},
    gold=_GOLD_SENTINEL,
)
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


def test_probes_use_the_shared_renderer() -> None:
    assert PROBES.render is render_with_prompt_inputs


def test_naive_probe_renders_exactly() -> None:
    assert PROBES.render_naive(_INSTANCE) == (
        "Canonicalize this JSON.\n\n" + _INPUT_JSON
    )


def test_ceiling_probe_renders_rules_examples_and_input() -> None:
    rendered = PROBES.render_ceiling(_INSTANCE)

    assert "RFC 8785" in rendered
    assert "include `+` for positive exponents" in rendered
    assert rendered.endswith("Now canonicalize:\n\n" + _INPUT_JSON)
    for source, expected in _WORKED_EXAMPLES:
        assert canonicalize(source) == expected
        assert f"Input:  {source}\nOutput: {expected}" in rendered


def test_probes_do_not_render_private_gold() -> None:
    assert _GOLD_SENTINEL not in PROBES.render_naive(_INSTANCE)
    assert _GOLD_SENTINEL not in PROBES.render_ceiling(_INSTANCE)


def test_missing_public_input_fails_loudly() -> None:
    instance = make_instance(
        id="missing-input",
        seed=1,
        strata="c11/key-order",
        prompt_inputs={"other": "value"},
        gold="private",
    )

    with pytest.raises(KeyError, match="input_json"):
        PROBES.render_naive(instance)
