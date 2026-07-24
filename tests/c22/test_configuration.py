"""Public Python and Typer configuration boundary checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from whetstone_envs.c22 import generate
from whetstone_envs.core.manifest import Manifest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_per_stratum": True},
        {"n_per_stratum": 0},
        {"n_per_stratum": "1"},
        {"constraint_counts": ()},
        {"constraint_counts": (True,)},
        {"constraint_counts": (0,)},
        {"constraint_counts": ("3",)},
        {"mixes": ()},
        {"mixes": ("unknown",)},
        {"seed_start": True},
        {"seed_start": generate.PUBLISHED_KEY_MAX},
    ],
    ids=[
        "bool-n",
        "zero-n",
        "string-n",
        "empty-count-axis",
        "bool-count",
        "zero-count",
        "string-count",
        "empty-mix-axis",
        "unknown-mix",
        "bool-seed",
        "published-seed",
    ],
)
def test_python_api_rejects_invalid_config_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    def unexpected_generation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("instance generation ran before configuration validation")

    monkeypatch.setattr(generate, "_make_instance", unexpected_generation)
    with pytest.raises(ValueError, match=r"."):
        generate.generate_pool(
            **overrides,  # ty: ignore[invalid-argument-type]
        )


def test_infeasible_depth_still_fails_loudly() -> None:
    with pytest.raises(ValueError, match="pool too small"):
        generate.generate_pool(
            n_per_stratum=1,
            constraint_counts=(100,),
            mixes=(generate.MIX_EASY,),
        )


def test_compatibility_retry_exhaustion_has_seed_and_attempt_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate, "MAX_COMPATIBILITY_ATTEMPTS", 3)
    monkeypatch.setattr(
        generate,
        "compatibility_error",
        lambda *_args: "synthetic conflict",
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"seed=1000000.*n_constraints=3.*mix='easy'.*"
            r"after 3 attempts.*synthetic conflict"
        ),
    ):
        generate.generate_pool(
            n_per_stratum=1,
            constraint_counts=(3,),
            mixes=(generate.MIX_EASY,),
        )


def test_typer_cli_generates_a_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    result = CliRunner().invoke(
        generate.app,
        [
            "--n-per-stratum",
            "1",
            "--constraint-counts",
            "3",
            "--mixes",
            "easy",
            "--seed-start",
            "4000",
            "--manifest",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = Manifest.read(output)
    assert manifest.seed_range == (4000, 4001)
    assert manifest.stratum_counts == {"n3_easy": 1}


def test_typer_hard_preset_rejects_ignored_axis_overrides() -> None:
    result = CliRunner().invoke(
        generate.app,
        [
            "--preset",
            "hard",
            "--constraint-counts",
            "0",
            "--mixes",
            "unknown",
            "--seed-start",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--n-per-stratum", "0"],
        ["--constraint-counts", "0"],
        ["--mixes", "unknown"],
        ["--seed-start", str(generate.PUBLISHED_KEY_MAX)],
        ["--preset", "unknown"],
    ],
)
def test_typer_cli_rejects_invalid_inputs(args: list[str]) -> None:
    result = CliRunner().invoke(generate.app, args)
    assert result.exit_code != 0
    assert "Invalid value" in result.output
