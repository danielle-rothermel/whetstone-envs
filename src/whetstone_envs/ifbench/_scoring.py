from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langdetect import DetectorFactory

from whetstone_envs.ifbench._vendor.ifbench import evaluation_lib

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

DetectorFactory.seed = 0


@dataclass(frozen=True, slots=True)
class Following:
    instruction_ids: tuple[str, ...]
    followed: tuple[bool, ...]

    @property
    def all_followed(self) -> bool:
        return all(self.followed)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, ok in zip(
                self.instruction_ids, self.followed, strict=True
            )
            if not ok
        )


def _check(response: str, gold: str, *, loose: bool) -> Following:
    spec: dict[str, Any] = json.loads(gold)
    example = evaluation_lib.InputExample(
        key=spec["key"],
        instruction_id_list=spec["instruction_id_list"],
        prompt=spec["prompt"],
        kwargs=spec["kwargs"],
    )
    test = (
        evaluation_lib.test_instruction_following_loose
        if loose
        else evaluation_lib.test_instruction_following_strict
    )
    output = test(example, {spec["prompt"]: response})
    return Following(
        instruction_ids=tuple(spec["instruction_id_list"]),
        followed=tuple(output.follow_instruction_list),
    )


def check_strict(response: str, gold: str) -> Following:
    return _check(response, gold, loose=False)


def check_loose(response: str, gold: str) -> Following:
    return _check(THINK_BLOCK.sub("", response), gold, loose=True)
