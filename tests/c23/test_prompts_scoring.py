from whetstone_envs.c23 import PROBES, score_gold
from whetstone_envs.instances import make_instance


def test_prompts_render_only_public_fields() -> None:
    instance = make_instance(
        id="c23-test",
        seed=555_000_000,
        strata="S1",
        prompt_inputs={"demos_block": "ab -> ac", "query": "abab"},
        gold="private-gold",
    )

    assert PROBES.render_naive(instance) == "ab -> ac\n\nabab -> "
    ceiling = PROBES.render_ceiling(instance)
    assert "ab -> ac" in ceiling
    assert "abab" in ceiling
    assert "private-gold" not in ceiling


def test_score_gold_extracts_last_case_insensitive_output_line() -> None:
    prediction = "Output: wrong\nreasoning\n  oUtPuT :  acac  \nignored"

    assert score_gold(prediction, "acac") == 1
    assert score_gold("acac", "acac") == 1
    assert score_gold("Output: wrong", "acac") == 0
