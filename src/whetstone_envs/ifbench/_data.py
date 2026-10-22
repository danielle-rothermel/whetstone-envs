from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from whetstone_envs.instances import Instance, InstanceGroup, InstancePool

SOURCE = "allenai/IFBench"
SOURCE_REVISION = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
INPUT_FILE = "IFBench_test.jsonl"
NAIVE_TEMPLATE = "{prompt}"


def load_rows() -> tuple[dict[str, Any], ...]:
    raw = files("whetstone_envs.ifbench").joinpath("data", INPUT_FILE)
    return tuple(
        json.loads(line) for line in raw.read_text("utf-8").splitlines()
    )


def gold_of(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": [
                {
                    name: value
                    for name, value in kwargs.items()
                    if value is not None
                }
                for kwargs in row["kwargs"]
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def generate_pool() -> InstancePool:
    instances = tuple(
        Instance(
            id=f"ifbench-{row['key']}",
            seed=0,
            categories=tuple(row["instruction_id_list"]),
            prompt_inputs={"prompt": row["prompt"]},
            gold=gold_of(row),
        )
        for row in load_rows()
    )
    return InstancePool(
        instance_group=InstanceGroup(name="ifbench", instances=instances)
    )
