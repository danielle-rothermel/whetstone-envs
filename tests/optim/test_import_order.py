"""Both CLIs must import cleanly in a fresh interpreter.

A CLI entry point is the first thing a user reaches, so an import-time failure
anywhere beneath it -- a dependency's import cycle, a missing optional module,
a side effect that needs a runtime it does not have -- surfaces as a broken
command rather than as a test failure. These tests spend a subprocess each to
catch that here instead.

Each import runs in its own interpreter because import-time breakage only bites
on a cold module cache: once any test in this session has imported the package,
``sys.modules`` hides it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# ``optim.cli`` reaches the optim extra; without it there is no CLI to guard.
pytest.importorskip("whetstone.experiment.env")

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
        f"cold-importing {module} failed, so the installed CLI is broken at "
        f"its entry point.\n"
        f"stderr:\n{completed.stderr}"
    )
