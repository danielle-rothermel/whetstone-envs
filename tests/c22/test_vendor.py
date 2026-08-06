from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

_VENDOR = "src/whetstone_envs/c22/_vendor/instruction_following_eval"
_UPSTREAM_SHA256 = {
    "evaluation_lib.py": (
        "01deab4c67bf7f30c3a48f59d7b0bb31ea165651a636af7f2a3af389a420edbb"
    ),
    "instructions.py": (
        "130f9c50e15ae44820c9ef5b4aa2aa948c4c0a17f4c44c2932b9271add22c6d7"
    ),
    "instructions_registry.py": (
        "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc"
    ),
    "instructions_util.py": (
        "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b"
    ),
}


def _apply(git: str, root: Path, *, reverse: bool) -> None:
    command = [git, "apply", "--unidiff-zero", "--whitespace=nowarn"]
    if reverse:
        command.append("--reverse")
    command.append("VENDORED_DIFF.patch")
    subprocess.run(command, cwd=root, check=True)  # noqa: S603


def test_vendor_patch_round_trip_and_upstream_hashes(tmp_path: Path) -> None:
    repository = tmp_path / "vendor"
    package = repository / "instruction_following_eval"
    source = Path(__file__).parents[2] / _VENDOR
    shutil.copytree(source, package)
    shutil.copy2(source / "VENDORED_DIFF.patch", repository)
    git = shutil.which("git")
    assert git is not None

    _apply(git, repository, reverse=True)
    assert {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in _UPSTREAM_SHA256
    } == _UPSTREAM_SHA256

    _apply(git, repository, reverse=False)
    for name in _UPSTREAM_SHA256:
        assert (package / name).read_bytes() == (source / name).read_bytes()
