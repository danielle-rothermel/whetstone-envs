from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run opt-in integration tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="requires --run-integration outside CI",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def _synthetic_instance(
    index: int,
    stratum: str | tuple[str, ...],
    *,
    gold: str = "yes",
) -> Instance:
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
    return _synthetic_instance


@pytest.fixture
def two_stratum_pool() -> TaskPool:
    """A pool with two strata of three instances each (six total)."""
    instances = [_synthetic_instance(i, "easy") for i in range(3)] + [
        _synthetic_instance(i, "hard") for i in range(3, 6)
    ]
    return TaskPool(instances)
