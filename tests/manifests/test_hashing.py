"""Tests for canonical task-pool content hashing."""

from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import content_hash
from whetstone_envs.pools import TaskPool


def test_content_hash_matches_pinned_vector() -> None:
    instance = make_instance(
        id="vector-1",
        seed=7,
        strata=("easy", "short"),
        prompt_inputs={"b": "2", "a": "1"},
        gold="yes",
    )
    assert content_hash(TaskPool([instance])) == (
        "765870a223a64fdd2d4cd0f00351b8d683a864e10b78b5906da05e3058988353"
    )


def test_content_hash_independent_of_prompt_input_order() -> None:
    first = make_instance(
        id="t",
        seed=1,
        strata="s",
        prompt_inputs={"a": "1", "b": "2"},
    )
    second = make_instance(
        id="t",
        seed=1,
        strata="s",
        prompt_inputs={"b": "2", "a": "1"},
    )
    assert content_hash(TaskPool([first])) == content_hash(TaskPool([second]))


def test_content_hash_changes_with_gold() -> None:
    first = TaskPool([make_instance(id="t", seed=1, strata="s", gold="A")])
    second = TaskPool([make_instance(id="t", seed=1, strata="s", gold="B")])
    assert content_hash(first) != content_hash(second)
