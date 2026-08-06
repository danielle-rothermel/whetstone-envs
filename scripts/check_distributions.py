#!/usr/bin/env python3
"""Validate built distributions before publication."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from email.message import Message

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
PACKAGE_DIR = PROJECT_ROOT / "src" / "whetstone_envs"
DISTRIBUTION_NAME = "whetstone-envs"


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _project_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        document = tomllib.load(file)
    project = document.get("project")
    if not isinstance(project, dict):
        _fail("pyproject.toml has no [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        _fail("pyproject.toml has no nonempty project version")
    return version


def _one_artifact(pattern: str, expected_name: str) -> Path:
    artifacts = sorted(
        path for path in DIST_DIR.glob(pattern) if path.is_file()
    )
    if len(artifacts) != 1:
        names = ", ".join(path.name for path in artifacts) or "none"
        _fail(f"expected one {pattern} artifact, found: {names}")
    artifact = artifacts[0]
    if artifact.name != expected_name:
        _fail(
            f"artifact filename {artifact.name!r}; expected {expected_name!r}"
        )
    return artifact


def _one_metadata_name(names: set[str], suffix: str, artifact: Path) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        _fail(
            f"expected one {suffix} record in {artifact.name}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _wheel_data(wheel: Path) -> tuple[Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        names = {
            member.filename
            for member in archive.infolist()
            if not member.is_dir()
        }
        metadata_name = _one_metadata_name(
            names,
            ".dist-info/METADATA",
            wheel,
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    return metadata, names


def _sdist_data(sdist: Path) -> tuple[Message, set[str], str]:
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = {
            member.name for member in archive.getmembers() if member.isfile()
        }
        metadata_name = _one_metadata_name(names, "/PKG-INFO", sdist)
        metadata_file = archive.extractfile(metadata_name)
        if metadata_file is None:
            _fail(f"could not read {metadata_name} from {sdist.name}")
        with metadata_file:
            metadata = BytesParser().parsebytes(metadata_file.read())
    root, separator, _ = metadata_name.partition("/")
    if not separator or not root:
        _fail(f"invalid sdist metadata path: {metadata_name}")
    return metadata, names, root


def _validate_metadata(
    metadata: Message,
    artifact: Path,
    expected_version: str,
) -> None:
    if metadata.get("Name") != DISTRIBUTION_NAME:
        _fail(
            f"{artifact.name} has distribution name {metadata.get('Name')!r}; "
            f"expected {DISTRIBUTION_NAME!r}"
        )
    if metadata.get("Version") != expected_version:
        _fail(
            f"{artifact.name} has version {metadata.get('Version')!r}; "
            f"expected {expected_version!r}"
        )


def _expected_source_files() -> set[str]:
    return {
        path.relative_to(PACKAGE_DIR.parent).as_posix()
        for path in PACKAGE_DIR.parent.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def _validate_package_files(
    artifact: Path,
    actual_names: set[str],
    expected_names: set[str],
) -> None:
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        details.append(f"missing:\n{formatted}")
    if unexpected:
        formatted = "\n".join(f"  - {name}" for name in unexpected)
        details.append(f"unexpected:\n{formatted}")
    _fail(f"{artifact.name} package file mismatch:\n" + "\n".join(details))


def _public_modules() -> list[str]:
    modules = ["whetstone_envs"]
    modules.extend(
        f"whetstone_envs.{path.name}"
        for path in sorted(PACKAGE_DIR.iterdir())
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    return modules


def _smoke_test_wheel(wheel: Path, expected_version: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        _fail("uv is required for the isolated wheel smoke test")
    environment = os.environ.copy()
    for variable in (
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        environment.pop(variable, None)
    environment["EXPECTED_VERSION"] = expected_version
    environment["SMOKE_MODULES"] = ",".join(_public_modules())
    code = """
import importlib
import os
from importlib.metadata import distribution
from pathlib import Path

expected = os.environ["EXPECTED_VERSION"]
installed = distribution("whetstone-envs")
actual = installed.version
if actual != expected:
    raise SystemExit(f"installed version {actual!r}; expected {expected!r}")
installation_root = Path(installed.locate_file("")).resolve()
for module_name in os.environ["SMOKE_MODULES"].split(","):
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise SystemExit(f"{module_name} has no import origin")
    origin = Path(module_file).resolve()
    if not origin.is_relative_to(installation_root):
        raise SystemExit(
            f"{module_name} imported from {origin}, "
            f"outside {installation_root}"
        )
print(f"installed wheel imports passed at version {actual}")
"""
    with tempfile.TemporaryDirectory(
        prefix="whetstone-envs-wheel-smoke-"
    ) as working_directory:
        subprocess.run(  # noqa: S603
            [
                uv,
                "run",
                "--isolated",
                "--with",
                str(wheel),
                "--",
                "python",
                "-c",
                code,
            ],
            check=True,
            cwd=working_directory,
            env=environment,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version")
    arguments = parser.parse_args()

    project_version = _project_version()
    expected_version = arguments.expected_version or project_version
    if expected_version != project_version:
        _fail(
            f"expected version {expected_version!r} does not match "
            f"pyproject.toml version {project_version!r}"
        )

    wheel = _one_artifact(
        "*.whl",
        f"whetstone_envs-{expected_version}-py3-none-any.whl",
    )
    sdist = _one_artifact(
        "*.tar.gz",
        f"whetstone_envs-{expected_version}.tar.gz",
    )
    wheel_metadata, wheel_names = _wheel_data(wheel)
    sdist_metadata, sdist_names, sdist_root = _sdist_data(sdist)
    _validate_metadata(wheel_metadata, wheel, expected_version)
    _validate_metadata(sdist_metadata, sdist, expected_version)

    expected_files = _expected_source_files()
    wheel_package_files = {
        name for name in wheel_names if ".dist-info/" not in name
    }
    _validate_package_files(wheel, wheel_package_files, expected_files)
    sdist_source_prefix = f"{sdist_root}/src/"
    sdist_package_files = {
        name.removeprefix(sdist_source_prefix)
        for name in sdist_names
        if name.startswith(sdist_source_prefix)
    }
    _validate_package_files(
        sdist,
        sdist_package_files,
        expected_files,
    )
    _smoke_test_wheel(wheel, expected_version)
    print(
        f"validated {wheel.name} and {sdist.name}: "
        f"{len(expected_files)} package files"
    )


if __name__ == "__main__":
    main()
