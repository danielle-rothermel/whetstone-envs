"""Both CLIs must import cleanly in a fresh interpreter.

whetstone-ai <= 0.1.5 carries a provider <-> eval.drivers import cycle:
``whetstone.eval.schema`` reaches back into a partially initialized
``whetstone.experiment.binding`` for ``EvalConfigRef``. ``optim.cli`` works
around it by importing ``whetstone_envs.optim.run`` -- which walks whetstone's
modules in a cycle-resolving order -- before ``optim.gepa`` /
``optim.miprov2``.

That ordering is invisible at a glance and easy for a formatter, an
import-sorter, or a well-meaning edit to undo. These tests spend a subprocess
each to pin it: a reorder that reintroduces the cycle fails here rather than
at a user's first CLI invocation.

Each import runs in its own interpreter because the cycle only bites on a cold
module cache; once any test in this session has imported ``optim.run``,
``sys.modules`` hides the breakage.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Entry points a user reaches directly, via console script or ``python -m``.
CLI_MODULES = (
    "whetstone_envs.optim.cli",
    "whetstone_envs.reporting.cli",
)


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_imports_in_a_fresh_interpreter(module: str) -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, literal module
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"cold-importing {module} failed; whetstone-ai's provider/eval import "
        f"cycle is likely back. Check that whetstone_envs.optim.run is still "
        f"imported before optim.gepa / optim.miprov2.\n"
        f"stderr:\n{completed.stderr}"
    )
