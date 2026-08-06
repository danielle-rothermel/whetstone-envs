from whetstone_envs.c23 import score_gold


def test_score_gold_uses_the_last_case_insensitive_output_line() -> None:
    prediction = "Output: wrong\nreasoning\n  oUtPuT :  acac  \nignored"

    assert score_gold(prediction, "acac") == 1
    assert score_gold("acac", "acac") == 1
    assert score_gold("Output: wrong", "acac") == 0
