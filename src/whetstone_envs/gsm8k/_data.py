from __future__ import annotations

import gzip
import json
from importlib.resources import files

from whetstone_envs.gsm8k._scoring import gold_number
from whetstone_envs.instances import Instance, InstanceGroup, InstancePool

DATASET = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TEST_FILE = "test.jsonl.gz"
CATEGORY = "gsm8k"


def load_test_rows() -> tuple[dict[str, str], ...]:
    raw = files("whetstone_envs.gsm8k").joinpath("data", TEST_FILE)
    text = gzip.decompress(raw.read_bytes()).decode("utf-8")
    return tuple(json.loads(line) for line in text.splitlines())


def generate_pool() -> InstancePool:
    instances = tuple(
        Instance(
            id=f"gsm8k-test-{index:04d}",
            seed=0,
            categories=(CATEGORY,),
            prompt_inputs={"question": row["question"]},
            gold=gold_number(row["answer"]),
        )
        for index, row in enumerate(load_test_rows())
    )
    return InstancePool(
        instance_group=InstanceGroup(name="gsm8k", instances=instances)
    )
