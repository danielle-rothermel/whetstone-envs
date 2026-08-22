from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


#: The real-Codex opt-in variable, spelled here rather than imported.
#: This conftest loads for every suite, including installs without the
#: optional ``optim`` extra, so it must not import
#: ``whetstone_envs.optim.codex``. The two spellings are pinned equal by
#: ``tests/optim/test_codex.py``.
ALLOW_REAL_CODEX_ENV = "WHETSTONE_ENVS_ALLOW_REAL_CODEX"


@pytest.fixture(scope="session", autouse=True)
def _no_real_codex_opt_in() -> None:
    """No test may opt in to spawning the real, billed Codex CLI.

    ``run_optimizer`` refuses a Codex run that supplies neither a scripted
    seam nor the two-part opt-in, and one part is this environment
    variable. Clearing it for the whole session means the suite cannot
    reach the paid path even if the developer's shell exports it, and
    asserting it was unset first means a run that *would* have spent is
    reported rather than silently corrected.

    Session-scoped and autouse rather than per-test: the variable is
    process state, and a test that set it would be opting the rest of the
    session in.
    """
    present = os.environ.pop(ALLOW_REAL_CODEX_ENV, None)
    if present is not None:
        message = (
            f"{ALLOW_REAL_CODEX_ENV}={present!r} is set in this process. "
            "The suite must never be able to opt in to the real, billed "
            "Codex CLI; unset it before running the tests."
        )
        raise RuntimeError(message)


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
