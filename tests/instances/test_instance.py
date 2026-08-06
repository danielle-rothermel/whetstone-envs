"""Tests for the frozen :class:`Instance` type."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from whetstone_envs.instances import Instance, make_instance
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_make_instance_normalizes_single_stratum() -> None:
    inst = make_instance(id="t1", seed=7, strata="easy", gold="A")
    assert inst.strata == ("easy",)
    assert inst.id == "t1"
    assert inst.seed == 7
    assert inst.gold == "A"


def test_make_instance_accepts_stratum_tuple() -> None:
    inst = make_instance(id="t1", seed=7, strata=("easy", "short"))
    assert inst.strata == ("easy", "short")


def test_direct_construction_rejects_bare_string_strata() -> None:
    with pytest.raises(TypeError, match=r"(?i)(?=.*strata)(?=.*tuple)"):
        Instance(
            id="t1",
            seed=7,
            strata=cast("tuple[str, ...]", "easy"),
        )


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


@pytest.mark.parametrize(
    "mapping_kind",
    ["dict", "proxy"],
    ids=["dict", "proxy"],
)
def test_direct_construction_detaches_prompt_inputs(
    mapping_kind: str,
) -> None:
    backing = {"k": "v"}
    prompt_inputs = (
        backing if mapping_kind == "dict" else MappingProxyType(backing)
    )
    inst = Instance(
        id="t1",
        seed=1,
        strata=("s",),
        prompt_inputs=prompt_inputs,
    )
    original_hash = hash(inst)
    members = {inst}

    backing["k"] = "mutated"

    assert inst.prompt_inputs["k"] == "v"
    assert hash(inst) == original_hash
    assert inst in members
    with pytest.raises(TypeError):
        inst.prompt_inputs["k"] = "x"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "prompt_inputs",
    [
        {1: "value"},
        {"key": 1},
    ],
    ids=["non-string-key", "non-string-value"],
)
def test_prompt_inputs_require_string_keys_and_values(
    prompt_inputs: dict[object, object],
) -> None:
    with pytest.raises(
        TypeError,
        match=r"(?i)(?=.*prompt_inputs)(?=.*string)",
    ):
        Instance(
            id="t1",
            seed=1,
            strata=("s",),
            prompt_inputs=cast("Mapping[str, str]", prompt_inputs),
        )


@pytest.mark.parametrize("stratum", ["", " ", "\t\n"])
def test_blank_stratum_rejected(stratum: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*strat)(?=.*(?:non-empty|blank))",
    ):
        make_instance(id="t1", seed=1, strata=stratum)


def test_repeated_strata_are_ordered_and_deduplicated_in_pool() -> None:
    inst = make_instance(
        id="t1",
        seed=1,
        strata=("easy", "hard", "easy", "hard"),
    )
    pool = TaskPool([inst])

    assert inst.strata == ("easy", "hard")
    assert pool.strata == ("easy", "hard")
    assert pool.stratum_counts() == {"easy": 1, "hard": 1}
    assert pool.in_stratum("easy") == (inst,)
    assert pool.in_stratum("hard") == (inst,)
