from __future__ import annotations

import json
from pathlib import Path

import pytest
from dr_serialize import canonical_json_bytes
from pydantic import ValidationError

from whetstone_envs.reporting.publication import (
    EVAL_REPORT_NAME,
    load_eval_report,
    publish_eval_report,
    validate_output_root,
)


def test_eval_report_round_trips_canonical_file(fake_eval_output) -> None:
    loaded = load_eval_report(fake_eval_output.directory)
    assert loaded == fake_eval_output.report
    raw = (fake_eval_output.directory / EVAL_REPORT_NAME).read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw))


def test_invalid_publication_preserves_existing_file(
    fake_eval_output,
) -> None:
    path = fake_eval_output.directory / EVAL_REPORT_NAME
    before = path.read_bytes()
    invalid = fake_eval_output.report.model_copy(
        update={"schema_version": "unsupported"}
    )
    with pytest.raises(ValidationError):
        publish_eval_report(fake_eval_output.directory, invalid)
    assert path.read_bytes() == before


def test_output_guard_finds_repository_from_another_cwd(
    tmp_path, monkeypatch
) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="inside the repo"):
        validate_output_root(repository / "artifacts" / "run")


def test_output_guard_rejects_another_worktree_repository(tmp_path) -> None:
    repository = tmp_path / "other-worktree"
    repository.mkdir()
    (repository / ".git").write_text(
        "gitdir: /tmp/example.git/worktrees/other\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inside the repo"):
        validate_output_root(repository / "artifacts" / "run")
