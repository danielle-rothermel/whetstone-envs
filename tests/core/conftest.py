"""Synthetic fixtures shared by the core-harness tests.

None of these carry real task logic: they are hand-built instances and
pools whose gold labels and stratum membership are stated directly, so
the harness primitives can be exercised in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.core.instance import Instance


def _synthetic_instance(
    index: int,
    stratum: str | tuple[str, ...],
    *,
    gold: str = "yes",
) -> Instance:
    """Build a deterministic synthetic instance for tests."""
    stratum_label = stratum if isinstance(stratum, str) else "/".join(stratum)
    return make_instance(
        id=f"{stratum_label}-{index}",
        seed=1000 + index,
        strata=stratum,
        prompt_inputs={"question": f"q{index}", "hint": stratum_label},
        gold=gold,
    )


@pytest.fixture
def synthetic_instance() -> Callable[..., Instance]:
    """Return the synthetic-instance factory as a fixture."""
    return _synthetic_instance


@pytest.fixture
def two_stratum_pool() -> TaskPool:
    """A pool with two strata of three instances each (six total)."""
    instances = [_synthetic_instance(i, "easy") for i in range(3)] + [
        _synthetic_instance(i, "hard") for i in range(3, 6)
    ]
    return TaskPool(instances)
