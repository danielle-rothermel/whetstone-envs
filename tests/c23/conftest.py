from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c23 import generate_pool

if TYPE_CHECKING:
    from whetstone_envs.pools import TaskPool


@pytest.fixture(scope="session")
def c23_default_pool() -> TaskPool:
    return generate_pool()
