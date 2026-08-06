from whetstone_envs.instances import make_instance, public_prompt_identity


def test_public_prompt_identity_uses_only_sorted_prompt_inputs() -> None:
    first = make_instance(
        id="first",
        seed=1,
        strata="alpha",
        prompt_inputs={"z": "last", "a": "first"},
        gold="first-gold",
    )
    second = make_instance(
        id="second",
        seed=2,
        strata="beta",
        prompt_inputs={"a": "first", "z": "last"},
        gold="second-gold",
    )

    expected = (("a", "first"), ("z", "last"))
    assert public_prompt_identity(first) == expected
    assert public_prompt_identity(second) == expected
