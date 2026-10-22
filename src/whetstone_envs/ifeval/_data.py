from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from whetstone_envs.instances import Instance, InstanceGroup, InstancePool

DATASET = "google/IFEval"
DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
INPUT_FILE = "ifeval_input_data.jsonl"
NAIVE_TEMPLATE = "{prompt}"


def load_rows() -> tuple[dict[str, Any], ...]:
    raw = files("whetstone_envs.ifeval").joinpath("data", INPUT_FILE)
    return tuple(
        json.loads(line) for line in raw.read_text("utf-8").splitlines()
    )


def gold_of(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def generate_pool() -> InstancePool:
    instances = tuple(
        Instance(
            id=f"ifeval-{row['key']}",
            seed=0,
            categories=tuple(row["instruction_id_list"]),
            prompt_inputs={"prompt": row["prompt"]},
            gold=gold_of(row),
        )
        for row in load_rows()
    )
    return InstancePool(
        instance_group=InstanceGroup(name="ifeval", instances=instances)
    )
