"""Dependency, publication, and persisted-pool contracts for C19."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import whetstone_envs
from whetstone_envs import c19
from whetstone_envs.c19 import (
    DEFAULT_SPLIT_SIZES,
    PROBES,
    Action,
    C19Fact,
    C19Scenario,
    C19Size,
    build_manifest,
    derive_fact,
    generate_pool,
)
from whetstone_envs.manifests import Manifest

_ROOT = Path(__file__).parents[2]
_SEMANTIC_DEPENDENCIES = {
    "gymnasium": "1.3.0",
    "minigrid": "3.1.0",
    "numpy": "2.5.1",
}


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def test_semantic_dependencies_are_exact_direct_runtime_pins() -> None:
    pyproject = _load_toml(_ROOT / "pyproject.toml")
    project = _mapping(pyproject["project"])
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)

    for name, version in _SEMANTIC_DEPENDENCIES.items():
        assert [
            dependency
            for dependency in dependencies
            if isinstance(dependency, str) and dependency.startswith(name)
        ] == [f"{name}=={version}"]


def test_lock_resolves_exact_semantic_dependency_versions() -> None:
    lock = _load_toml(_ROOT / "uv.lock")
    packages = lock["package"]
    assert isinstance(packages, list)
    resolved: dict[str, str] = {}
    for raw_package in packages:
        package = _mapping(raw_package)
        name = package.get("name")
        if name not in _SEMANTIC_DEPENDENCIES:
            continue
        version = package.get("version")
        assert isinstance(name, str)
        assert isinstance(version, str)
        resolved[name] = version

    assert resolved == _SEMANTIC_DEPENDENCIES


def test_c19_public_api_is_curated_and_root_api_stays_empty() -> None:
    assert whetstone_envs.__all__ == []
    assert c19.__all__ == [
        "DEFAULT_SPLIT_SIZES",
        "PROBES",
        "Action",
        "C19Fact",
        "C19Scenario",
        "C19Size",
        "build_manifest",
        "derive_fact",
        "generate_pool",
    ]
    assert (
        c19.DEFAULT_SPLIT_SIZES,
        c19.PROBES,
        c19.Action,
        c19.C19Fact,
        c19.C19Scenario,
        c19.C19Size,
        c19.build_manifest,
        c19.derive_fact,
        c19.generate_pool,
    ) == (
        DEFAULT_SPLIT_SIZES,
        PROBES,
        Action,
        C19Fact,
        C19Scenario,
        C19Size,
        build_manifest,
        derive_fact,
        generate_pool,
    )


def test_committed_manifest_matches_the_default_pool() -> None:
    manifest_path = Path(c19.__file__).with_name("manifest.json")
    pool = generate_pool()

    manifest = Manifest.read(manifest_path)
    assert manifest == build_manifest()
    assert manifest.matches_pool(pool)
