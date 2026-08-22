"""A real fake-transport COPRO run, produced once per test session.

The audit reads persisted evidence, so its tests must run against evidence
whetstone actually wrote. A hand-built artifact would drift from the real
persisted shape and let an invariant pass against a format nobody produces.

The run is session-scoped because it costs a few seconds; it uses the fake
transport, so it makes no provider calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.run import RunSpec, run_optimizer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def copro_run_dir(tmp_path_factory) -> Path:
    """One completed fake-transport COPRO run directory."""
    output = tmp_path_factory.mktemp("audit-runs") / "copro"
    return run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            output_dir=output,
            run_id="c19-copro-audit-fixture",
        )
    )


@pytest.fixture
def mutable_run_dir(copro_run_dir, tmp_path) -> Path:
    """A per-test copy of the run, safe to mutate into a negative fixture."""
    from whetstone_envs.optim.audit._mutate import copy_run

    return copy_run(copro_run_dir, tmp_path / "mutated")
