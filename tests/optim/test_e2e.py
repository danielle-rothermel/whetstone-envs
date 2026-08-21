from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.optim.contracts import OptimResult

from whetstone_envs.optim.cli import main
from whetstone_envs.optim.run import C19RunSpec, run_c19_optimizer


@pytest.mark.parametrize("optimizer", ["copro", "gepa"])
def test_fake_transport_completes(tmp_path, optimizer: str) -> None:
    output = tmp_path / f"{optimizer}-run"
    code = main(
        [
            "--family",
            "c19",
            "--optimizer",
            optimizer,
            "--transport",
            "fake",
            "--split-sizes",
            "2,2,0",
            "--run-id",
            f"c19-{optimizer}-e2e",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"
    assert result.proposals


def test_run_refuses_in_repo_output() -> None:
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=Path("artifacts") / "c19-run",
            )
        )


def test_run_refuses_in_repo_output_when_cwd_is_elsewhere(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=repo_root / "artifacts" / "c19-run",
            )
        )
