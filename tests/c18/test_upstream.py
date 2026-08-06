from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from dr_serialize import DuplicateJsonKeyError, JsonByteLimitError

from whetstone_envs.c18 import upstream
from whetstone_envs.c18.config import DistractorMode
from whetstone_envs.c18.upstream import (
    RawInstance,
    UpstreamError,
    generate_raw,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_raw_returns_validated_rows() -> None:
    rows = generate_raw(hops=2, seed=1_000_100_001, num_trials=3)
    assert len(rows) == 3
    assert all(isinstance(row, RawInstance) for row in rows)
    assert all(row.hops == 2 for row in rows)
    assert all(row.answer in {"True", "False"} for row in rows)


def test_same_seed_is_deterministic() -> None:
    first = generate_raw(hops=3, seed=1_000_100_002, num_trials=3)
    second = generate_raw(hops=3, seed=1_000_100_002, num_trials=3)
    assert first == second


def test_deep_generation_supports_explicit_no_distractors() -> None:
    rows = generate_raw(
        hops=8,
        seed=2_000_100_008,
        num_trials=1,
        distractors=DistractorMode.NONE,
    )
    assert len(rows) == 1
    assert rows[0].hops == 8


def test_vendored_tree_is_not_written_to() -> None:
    before = {path.name for path in upstream._VENDOR_DIR.iterdir()}
    generate_raw(hops=1, seed=1_000_100_050, num_trials=1)
    after = {path.name for path in upstream._VENDOR_DIR.iterdir()}
    assert after == before


def test_parser_rejects_malformed_subprocess_json() -> None:
    with pytest.raises(UpstreamError, match="malformed"):
        upstream._parse_examples(
            {"example1": {"test_example": {"question": 1}}},
            hops=1,
        )


def test_output_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.json"
    output.write_bytes(b'{"answer":"False","answer":"True"}')

    with pytest.raises(UpstreamError, match="invalid strict JSON") as caught:
        upstream._read_output(output)

    assert isinstance(caught.value.__cause__, DuplicateJsonKeyError)


def test_output_reader_enforces_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "oversized.json"
    output.write_bytes(b"012345678")
    monkeypatch.setattr(upstream, "_MAX_OUTPUT_BYTES", 8)

    with pytest.raises(UpstreamError, match="invalid strict JSON") as caught:
        upstream._read_output(output)

    assert isinstance(caught.value.__cause__, JsonByteLimitError)


def test_generator_rejects_excessive_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "run_experiment.py"
    script.write_text(
        "import sys\nsys.stderr.write('x' * 100)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(upstream, "_MAX_DIAGNOSTIC_BYTES", 32)
    request = upstream._GenerationRequest(
        hops=1,
        seed=1_000_100_055,
        num_trials=1,
        distractors=DistractorMode.RELEVANT,
        timeout_s=5.0,
    )

    with pytest.raises(UpstreamError, match="diagnostic output limit"):
        upstream._run_generator(tmp_path, request=request)


@pytest.mark.parametrize("num_trials", [0, -1])
def test_generate_raw_rejects_nonpositive_counts(num_trials: int) -> None:
    with pytest.raises(ValueError, match="num_trials must be positive"):
        generate_raw(hops=1, seed=1_000_100_060, num_trials=num_trials)


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), 10**400])
def test_generate_raw_rejects_nonfinite_timeouts(
    timeout_s: float | int,
) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        generate_raw(
            hops=1,
            seed=1_000_100_061,
            num_trials=1,
            timeout_s=timeout_s,
        )
