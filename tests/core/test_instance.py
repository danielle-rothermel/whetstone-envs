"""Tests for the frozen :class:`Instance` type."""

from __future__ import annotations

import dataclasses

import pytest

from whetstone_envs.core.instance import Instance, make_instance


def test_make_instance_normalizes_single_stratum() -> None:
    inst = make_instance(id="t1", seed=7, strata="easy", gold="A")
    assert inst.strata == ("easy",)
    assert inst.id == "t1"
    assert inst.seed == 7
    assert inst.gold == "A"


def test_make_instance_accepts_stratum_tuple() -> None:
    inst = make_instance(id="t1", seed=7, strata=("easy", "short"))
    assert inst.strata == ("easy", "short")


def test_instance_is_frozen() -> None:
    inst = make_instance(id="t1", seed=1, strata="s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.seed = 2  # ty: ignore[invalid-assignment]


def test_prompt_inputs_are_read_only_and_detached() -> None:
    source = {"a": "1"}
    inst = make_instance(id="t1", seed=1, strata="s", prompt_inputs=source)
    # Mutating the caller's dict must not change the frozen instance.
    source["a"] = "mutated"
    assert inst.prompt_inputs["a"] == "1"
    with pytest.raises(TypeError):
        inst.prompt_inputs["a"] = "x"  # ty: ignore[invalid-assignment]


def test_empty_id_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_instance(id="", seed=1, strata="s")


def test_empty_strata_rejected() -> None:
    with pytest.raises(ValueError, match="at least one stratum"):
        Instance(id="t1", seed=1, strata=())


def test_instances_are_hashable_and_value_equal() -> None:
    a = make_instance(id="t1", seed=1, strata="s", gold="g")
    b = make_instance(id="t1", seed=1, strata="s", gold="g")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_direct_construction_freezes_plain_mapping() -> None:
    inst = Instance(id="t1", seed=1, strata=("s",), prompt_inputs={"k": "v"})
    with pytest.raises(TypeError):
        inst.prompt_inputs["k"] = "x"  # ty: ignore[invalid-assignment]
