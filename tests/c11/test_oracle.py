import json

import pytest
import rfc8785

from whetstone_envs.c11 import canonicalize

_EXAMPLES = (
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


@pytest.mark.parametrize(("source", "expected"), _EXAMPLES)
def test_canonicalize(source: str, expected: str) -> None:
    assert canonicalize(source) == expected


def test_oracle_delegates_to_rfc8785() -> None:
    source = '{"péché": "accent", "alpha": [1.0, -0, true]}'

    assert canonicalize(source) == rfc8785.dumps(json.loads(source)).decode()


def test_published_rfc8785_examples() -> None:
    french_example = {
        "peach": "This sorting order",
        "péché": "is wrong according to French",
        "pêche": "but canonicalization MUST",
        "sin": "ignore locale",
    }
    number_example = [
        333333333.33333329,
        1e30,
        4.5,
        2e-3,
        1e-27,
    ]

    assert canonicalize(json.dumps(french_example)) == (
        '{"peach":"This sorting order",'
        '"péché":"is wrong according to French",'
        '"pêche":"but canonicalization MUST",'
        '"sin":"ignore locale"}'
    )
    assert canonicalize(json.dumps(number_example)) == (
        "[333333333.3333333,1e+30,4.5,0.002,1e-27]"
    )


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(json.JSONDecodeError):
        canonicalize("{not-json")


def test_values_outside_the_rfc8785_domain_are_rejected() -> None:
    with pytest.raises(rfc8785.CanonicalizationError):
        canonicalize('{"too_large": 9007199254740992}')


def test_input_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="input_json must be a string"):
        canonicalize({"value": 1})  # ty: ignore[invalid-argument-type]
