from whetstone_envs.c23 import PROBES
from whetstone_envs.instances import make_instance


def test_prompts_include_the_public_task_and_exclude_gold() -> None:
    instance = make_instance(
        id="c23-test",
        seed=555_000_000,
        strata="S1",
        prompt_inputs={"demos_block": "ab -> ac", "query": "abab"},
        gold="private-gold",
    )

    naive = PROBES.render_naive(instance)
    ceiling = PROBES.render_ceiling(instance)
    for prompt in (naive, ceiling):
        assert instance.prompt_inputs["demos_block"] in prompt
        assert instance.prompt_inputs["query"] in prompt
        assert instance.gold not in prompt

    guidance = " ".join(ceiling.split()).casefold()
    assert "more than one rule may fit" in guidance
    assert "they agree on the query output" in guidance
