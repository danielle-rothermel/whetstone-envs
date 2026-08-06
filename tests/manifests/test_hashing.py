"""Tests for canonical task-pool content hashing."""

from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import content_hash
from whetstone_envs.pools import TaskPool


def test_content_hash_matches_pinned_vector() -> None:
    pool = TaskPool(
        [
            make_instance(
                id="vector-café",
                seed=7,
                strata=("easy", "短"),
                prompt_inputs={"z": "naïve", "a": "雪"},
                gold="sí",
            ),
            make_instance(
                id="vector-2",
                seed=11,
                strata=("hard", "long"),
                prompt_inputs={"emoji": "🧪", "b": "2"},
                gold="答案",
            ),
        ]
    )
    assert content_hash(pool) == (
        "3f1967dbec05232f926440c0b887c0b2acfa78ea213e645d5723c31117ca41a0"
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
