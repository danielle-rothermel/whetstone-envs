from __future__ import annotations

import io
import subprocess
from typing import TYPE_CHECKING

import pytest
from dr_serialize import (
    CANONICAL_JSON_MAX_CONTAINER_DEPTH,
    DuplicateJsonKeyError,
    JsonByteLimitError,
    JsonDepthLimitError,
)

from whetstone_envs.c18 import upstream
from whetstone_envs.c18.config import DistractorMode
from whetstone_envs.c18.upstream import (
    RawInstance,
    UpstreamError,
    generate_raw,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_raw_ignores_hostile_pythonpath_and_returns_validated_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "numpy.py").write_text(
        "raise RuntimeError('ambient numpy was imported')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    rows = generate_raw(hops=2, seed=1_000_100_001, num_trials=3)
    assert len(rows) == 3
    assert all(isinstance(row, RawInstance) for row in rows)
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


def test_runtime_files_are_copied_byte_for_byte(tmp_path: Path) -> None:
    upstream._copy_runtime_files(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == set(
        upstream._VENDOR_FILES,
    )
    for name in upstream._VENDOR_FILES:
        assert (tmp_path / name).read_bytes() == (
            upstream._VENDOR_DIR / name
        ).read_bytes()


@pytest.mark.parametrize(
    "example",
    [
        {"question": 1},
        {
            "question": "",
            "query": "True or false: Sally is sour.",
            "answer": "False",
        },
    ],
)
def test_parser_rejects_malformed_subprocess_json(
    example: dict[str, object],
) -> None:
    with pytest.raises(UpstreamError):
        upstream._parse_examples({"example1": {"test_example": example}})


def test_output_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.json"
    output.write_bytes(b'{"answer":"False","answer":"True"}')

    with pytest.raises(UpstreamError) as caught:
        upstream._read_output(output)

    assert isinstance(caught.value.__cause__, DuplicateJsonKeyError)


def test_output_reader_enforces_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "oversized.json"
    output.write_bytes(b"012345678")
    monkeypatch.setattr(upstream, "_MAX_OUTPUT_BYTES", 8)

    with pytest.raises(UpstreamError) as caught:
        upstream._read_output(output)

    assert isinstance(caught.value.__cause__, JsonByteLimitError)


def test_output_reader_enforces_depth_limit(tmp_path: Path) -> None:
    output = tmp_path / "deep.json"
    depth = CANONICAL_JSON_MAX_CONTAINER_DEPTH + 1
    output.write_bytes(f"{'[' * depth}0{']' * depth}".encode())

    with pytest.raises(UpstreamError) as caught:
        upstream._read_output(output)

    assert isinstance(caught.value.__cause__, JsonDepthLimitError)


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

    with pytest.raises(UpstreamError):
        upstream._run_generator(tmp_path, request=request)


class _BlockedProcess:
    def __init__(self, failure: BaseException) -> None:
        self.stderr = io.BytesIO()
        self.failure = failure
        self.killed = False
        self.wait_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if timeout is not None:
            raise self.failure
        return -9

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        self.killed = True


def _request() -> upstream._GenerationRequest:
    return upstream._GenerationRequest(
        hops=1,
        seed=1_000_100_055,
        num_trials=1,
        distractors=DistractorMode.RELEVANT,
        timeout_s=5.0,
    )


def test_timeout_kills_and_reaps_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = subprocess.TimeoutExpired("generator", 5.0)
    process = _BlockedProcess(failure)
    monkeypatch.setattr(
        upstream.subprocess, "Popen", lambda *_a, **_kw: process
    )

    with pytest.raises(UpstreamError) as caught:
        upstream._run_generator(tmp_path, request=_request())

    assert caught.value.__cause__ is failure
    assert process.killed
    assert process.wait_calls == 2


def test_cancellation_kills_and_reaps_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _BlockedProcess(KeyboardInterrupt())
    monkeypatch.setattr(
        upstream.subprocess, "Popen", lambda *_a, **_kw: process
    )

    with pytest.raises(KeyboardInterrupt):
        upstream._run_generator(tmp_path, request=_request())

    assert process.killed
    assert process.wait_calls == 2


@pytest.mark.parametrize("num_trials", [0, -1])
def test_generate_raw_rejects_nonpositive_counts(num_trials: int) -> None:
    with pytest.raises(ValueError):
        generate_raw(hops=1, seed=1_000_100_060, num_trials=num_trials)


@pytest.mark.parametrize("seed", [-1, 1 << 32])
def test_generate_raw_rejects_out_of_range_seeds(seed: int) -> None:
    with pytest.raises(ValueError):
        generate_raw(hops=1, seed=seed, num_trials=1)


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), 10**400])
def test_generate_raw_rejects_nonfinite_timeouts(
    timeout_s: float | int,
) -> None:
    with pytest.raises(ValueError):
        generate_raw(
            hops=1,
            seed=1_000_100_061,
            num_trials=1,
            timeout_s=timeout_s,
        )
