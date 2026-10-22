from __future__ import annotations

import re
from decimal import Decimal

from whetstone_envs.probes import normalize

GOLD_MARKER = "####"
STRICT_LINE = re.compile(r"#### (-?\d[\d,]*(?:\.\d+)?)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def canonical_number(text: str) -> str:
    value = Decimal(text.replace(",", ""))
    return format(value.normalize(), "f") if value else "0"


def gold_number(answer: str) -> str:
    return canonical_number(answer.rsplit(GOLD_MARKER, 1)[1].strip())


def strict_answer(response: str) -> str | None:
    lines = [line for line in normalize(response).splitlines() if line.strip()]
    if not lines:
        return None
    match = STRICT_LINE.fullmatch(lines[-1].strip())
    return None if match is None else canonical_number(match[1])


def forgiving_answer(response: str) -> str | None:
    plain = THINK_BLOCK.sub("", response).replace(",", "")
    numbers = NUMBER.findall(plain)
    return canonical_number(numbers[-1]) if numbers else None


def score_strict(response: str, gold: str) -> int:
    return int(strict_answer(response) == gold)


def score_forgiving(response: str, gold: str) -> int:
    return int(forgiving_answer(response) == gold)
