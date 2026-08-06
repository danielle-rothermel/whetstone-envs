from dr_serialize import Sha256Digest

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
        "49d37d086ef79b646e718820020fa29b9e7b5a5c6be0af8d34a3d07a85554eb5"
    )
    assert isinstance(content_hash(pool), Sha256Digest)


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
