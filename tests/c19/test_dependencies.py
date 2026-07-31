"""Dependency contracts for c19's generated semantics."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_minigrid_is_exactly_pinned_in_pyproject() -> None:
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    minigrid_dependencies = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if dependency.startswith("minigrid")
    ]
    assert minigrid_dependencies == ["minigrid==3.1.0"]
