from __future__ import annotations

from pathlib import Path

import pytest

from whetstone_envs.c19 import generation
from whetstone_envs.c19 import regenerate as regenerate_module
from whetstone_envs.c19.generation import (
    GENERATOR_VERSION,
    build_manifest,
    generate_pool,
)
from whetstone_envs.c19.regenerate import _main as regenerate_main
from whetstone_envs.c19.regenerate import regenerate
from whetstone_envs.manifests import Manifest

_CANONICAL_MANIFEST_PATH = Path(generation.__file__).with_name("manifest.json")


def test_committed_manifest_matches_the_default_pool() -> None:
    pool = generate_pool()
    manifest = Manifest.read(_CANONICAL_MANIFEST_PATH)

    assert manifest == build_manifest()
    assert manifest.matches_pool(pool)


def test_regeneration_is_canonical_and_repeatable(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = regenerate(
        first_path,
        n_per_stratum=2,
        seed_start=765_432,
    )
    second = regenerate(
        second_path,
        n_per_stratum=2,
        seed_start=765_432,
    )
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)

    assert (
        first
        == second
        == build_manifest(
            n_per_stratum=2,
            seed_start=765_432,
        )
    )
    assert first.generator_version == GENERATOR_VERSION
    assert first.seed_range == (765_432, 765_444)
    assert first.matches_pool(pool)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert Manifest.read(first_path) == first


def test_regenerate_rejects_custom_generation_at_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_build_manifest(*_args: object, **_kwargs: object) -> None:
        pytest.fail("target validation must precede manifest generation")

    monkeypatch.setattr(
        regenerate_module,
        "build_manifest",
        unexpected_build_manifest,
    )

    with pytest.raises(ValueError):
        regenerate(_CANONICAL_MANIFEST_PATH, n_per_stratum=2)


@pytest.mark.parametrize(
    "argv",
    [
        ["--n-per-stratum", "2"],
        [
            "--seed-start",
            "765432",
            "--manifest",
            str(_CANONICAL_MANIFEST_PATH),
        ],
    ],
)
def test_cli_rejects_custom_generation_at_canonical_manifest(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        regenerate_main(argv)

    assert raised.value.code == 2


def test_cli_writes_custom_generation_to_noncanonical_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "custom-manifest.json"

    assert (
        regenerate_main(
            [
                "--manifest",
                str(manifest_path),
                "--n-per-stratum",
                "2",
                "--seed-start",
                "765432",
            ],
        )
        == 0
    )
    assert Manifest.read(manifest_path) == build_manifest(
        n_per_stratum=2,
        seed_start=765_432,
    )
