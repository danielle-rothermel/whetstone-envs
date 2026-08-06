from __future__ import annotations

import re

from whetstone_envs.scoring import exact_match

_OUTPUT_LINE = re.compile(r"^\s*output\s*:\s*(?P<answer>.*)$", re.IGNORECASE)


def extract_last_output(prediction: str) -> str:
    answer: str | None = None
    for line in prediction.splitlines():
        if match := _OUTPUT_LINE.match(line):
            answer = match.group("answer")
    return prediction if answer is None else answer


def score_gold(prediction: str, gold: str) -> int:
    """Score the last case-insensitive Output line by shared exact match."""
    return exact_match(extract_last_output(prediction), gold)
