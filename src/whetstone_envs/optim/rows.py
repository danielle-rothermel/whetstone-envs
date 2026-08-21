from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from whetstone_envs.instances import Instance


@dataclass(frozen=True, slots=True)
class TaskRow:
    """One optimizer task row derived from a task instance."""

    task_id: str
    prompt_inputs: dict[str, str]
    gold: str

    @property
    def task_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "prompt_inputs": self.prompt_inputs,
            "gold": self.gold,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def task_row_from_instance(instance: Instance) -> TaskRow:
    """Map a task instance onto the row shape evaluation drivers require."""
    return TaskRow(
        task_id=instance.id,
        prompt_inputs=dict(instance.prompt_inputs),
        gold=instance.gold,
    )


def task_rows_from_instances(
    instances: Iterable[Instance],
) -> tuple[TaskRow, ...]:
    return tuple(task_row_from_instance(instance) for instance in instances)


__all__ = [
    "TaskRow",
    "task_row_from_instance",
    "task_rows_from_instances",
]
