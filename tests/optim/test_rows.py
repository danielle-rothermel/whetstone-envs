from __future__ import annotations

from whetstone_envs.c19 import generate_pool
from whetstone_envs.optim import (
    task_row_from_instance,
    task_rows_from_instances,
)


def test_task_row_maps_instance_identity_and_gold() -> None:
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    instance = pool.instances[0]
    row = task_row_from_instance(instance)

    assert row.task_id == instance.id
    assert row.prompt_inputs == dict(instance.prompt_inputs)
    assert row.gold == instance.gold
    assert row.task_hash


def test_task_rows_preserve_pool_order() -> None:
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    rows = task_rows_from_instances(pool.instances[:3])

    assert [row.task_id for row in rows] == [
        instance.id for instance in pool.instances[:3]
    ]
