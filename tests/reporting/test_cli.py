from __future__ import annotations

from unittest.mock import patch

from whetstone_envs.optim.cli import main as optim_main
from whetstone_envs.reporting.cli import main as reporting_main
from whetstone_envs.reporting.publication import DurableRunError


def test_eval_cli_prints_created_run_directory_on_failure(
    tmp_path, capsys
) -> None:
    directory = tmp_path / "eval-failure"
    with patch(
        "whetstone_envs.reporting.execution.bind_openrouter_transport",
        side_effect=RuntimeError("transport failed"),
    ):
        status = reporting_main(
            [
                "run",
                "--transport",
                "openrouter",
                "--split-sizes",
                "1,1,0",
                "--output",
                str(directory),
                "--no-color",
            ]
        )

    captured = capsys.readouterr()
    assert status == 2
    assert "error: transport failed" in captured.out
    assert str(directory) in captured.out
    assert directory.is_dir()


def test_optimizer_cli_preserves_traceback_and_prints_run_directory(
    tmp_path, capsys
) -> None:
    directory = tmp_path / "optim-failure"
    directory.mkdir()
    error = DurableRunError(directory, RuntimeError("publication failed"))
    with patch(
        "whetstone_envs.optim.cli.run_c19_optimizer",
        side_effect=error,
    ):
        status = optim_main(
            [
                "--optimizer",
                "copro",
                "--output",
                str(directory),
            ]
        )

    captured = capsys.readouterr()
    assert status == 2
    assert "RuntimeError: publication failed" in captured.err
    assert str(directory) in captured.err


def test_html_cli_prints_absolute_expected_filename(
    fake_eval_output, capsys
) -> None:
    status = reporting_main(
        ["html", str(fake_eval_output.directory), "--no-color"]
    )

    captured = capsys.readouterr()
    expected = (fake_eval_output.directory / "eval-report.html").resolve()
    assert status == 0
    assert captured.out.strip() == str(expected)
    assert expected.is_file()


def test_trajectory_html_cli_prints_absolute_expected_filename(
    tmp_path, capsys
) -> None:
    expected = (tmp_path / "trajectory-report.html").resolve()
    expected.write_text("valid", encoding="utf-8")
    with patch(
        "whetstone_envs.reporting.html.publish_trajectory_html",
        return_value=expected,
    ):
        status = reporting_main(
            ["trajectory-html", str(tmp_path), "--no-color"]
        )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out.strip() == str(expected)
