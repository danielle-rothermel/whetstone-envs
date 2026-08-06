from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from whetstone_envs.c18.config import DistractorMode

_VENDOR_DIR = Path(__file__).parent / "_vendor" / "prontoqa"
_VENDOR_FILES = (
    "run_experiment.py",
    "theory.py",
    "syntax.py",
    "proof.py",
    "prompt.py",
    "fol.py",
    "bad_patterns.txt",
)
_FIXED_ONTOLOGY = "fictional"


class UpstreamError(RuntimeError):
    """The pinned PrOntoQA generator failed or violated its output contract."""


class _TestExample(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    question: str
    query: str
    answer: str


class _ExampleBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    test_example: _TestExample


@dataclass(frozen=True, slots=True)
class RawInstance:
    """The C18 fields projected from one validated upstream example."""

    question: str
    query: str
    answer: str
    hops: int


@dataclass(frozen=True, slots=True)
class _GenerationRequest:
    hops: int
    seed: int
    num_trials: int
    distractors: DistractorMode
    timeout_s: float


def _validate_generation_request(request: _GenerationRequest) -> None:
    for name, value in (
        ("hops", request.hops),
        ("seed", request.seed),
        ("num_trials", request.num_trials),
    ):
        if type(value) is not int:
            msg = f"C18 upstream {name} must be an int"
            raise TypeError(msg)
    if request.hops <= 0:
        msg = f"C18 upstream hops must be positive, got {request.hops}"
        raise ValueError(msg)
    if request.num_trials <= 0:
        msg = (
            "C18 upstream num_trials must be positive, "
            f"got {request.num_trials}"
        )
        raise ValueError(msg)
    if not isinstance(request.distractors, DistractorMode):
        msg = "C18 upstream distractors must be a DistractorMode"
        raise TypeError(msg)
    if isinstance(request.timeout_s, bool) or not isinstance(
        request.timeout_s,
        int | float,
    ):
        msg = "C18 upstream timeout_s must be a number"
        raise TypeError(msg)
    if request.timeout_s <= 0:
        msg = (
            f"C18 upstream timeout_s must be positive, got {request.timeout_s}"
        )
        raise ValueError(msg)


def _copy_runtime_files(work: Path) -> None:
    for name in _VENDOR_FILES:
        source = _VENDOR_DIR / name
        if not source.is_file():
            msg = f"C18 vendored runtime file is missing: {name}"
            raise UpstreamError(msg)
        shutil.copy2(source, work / name)


def _run_generator(
    work: Path,
    *,
    request: _GenerationRequest,
) -> Path:
    command = (
        sys.executable,
        "run_experiment.py",
        "--model-name",
        "json",
        "--model-size",
        "1",
        "--num-trials",
        str(request.num_trials),
        "--min-hops",
        str(request.hops),
        "--max-hops",
        str(request.hops),
        "--ontology",
        _FIXED_ONTOLOGY,
        "--distractors",
        request.distractors.value,
        "--test-distractors",
        request.distractors.value,
        "--seed",
        str(request.seed),
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and argv
            command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=request.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        msg = (
            f"C18 vendored generator timed out after {request.timeout_s}s "
            f"for D{request.hops} seed {request.seed}"
        )
        raise UpstreamError(msg) from error
    except OSError as error:
        msg = (
            "C18 vendored generator could not start for "
            f"D{request.hops} seed {request.seed}"
        )
        raise UpstreamError(msg) from error

    if completed.returncode != 0:
        stderr = completed.stderr[-500:]
        msg = (
            f"C18 vendored generator exited {completed.returncode} "
            f"for D{request.hops} seed {request.seed}: {stderr}"
        )
        raise UpstreamError(msg)

    outputs = tuple(work.glob("*.json"))
    if len(outputs) != 1:
        names = tuple(path.name for path in outputs)
        msg = (
            "C18 vendored generator must produce exactly one JSON file, "
            f"got {names!r} for D{request.hops} seed {request.seed}"
        )
        raise UpstreamError(msg)
    return outputs[0]


def _example_index(key: str) -> int:
    prefix = "example"
    if not key.startswith(prefix) or not key[len(prefix) :].isdigit():
        msg = f"C18 upstream output has invalid example key {key!r}"
        raise UpstreamError(msg)
    return int(key[len(prefix) :])


def _parse_examples(payload: object, *, hops: int) -> tuple[RawInstance, ...]:
    if not isinstance(payload, dict):
        msg = "C18 upstream output must be a JSON object"
        raise UpstreamError(msg)

    examples: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            msg = "C18 upstream output keys must be strings"
            raise UpstreamError(msg)
        examples[key] = value

    rows: list[RawInstance] = []
    for key in sorted(examples, key=_example_index):
        try:
            block = _ExampleBlock.model_validate(examples[key], strict=True)
        except ValidationError as error:
            msg = f"C18 upstream example {key!r} is malformed"
            raise UpstreamError(msg) from error
        example = block.test_example
        rows.append(
            RawInstance(
                question=example.question,
                query=example.query,
                answer=example.answer,
                hops=hops,
            )
        )
    if not rows:
        msg = "C18 upstream output contains no examples"
        raise UpstreamError(msg)
    return tuple(rows)


def generate_raw(
    *,
    hops: int,
    seed: int,
    num_trials: int,
    distractors: DistractorMode = DistractorMode.RELEVANT,
    timeout_s: float = 300.0,
) -> tuple[RawInstance, ...]:
    """Run the pinned fictional-ontology generator in an isolated directory."""
    request = _GenerationRequest(
        hops=hops,
        seed=seed,
        num_trials=num_trials,
        distractors=distractors,
        timeout_s=timeout_s,
    )
    _validate_generation_request(request)
    with tempfile.TemporaryDirectory(prefix="c18-prontoqa-") as temporary:
        work = Path(temporary)
        _copy_runtime_files(work)
        output = _run_generator(
            work,
            request=request,
        )
        try:
            payload: object = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError) as error:
            msg = f"C18 upstream output {output.name!r} is unreadable"
            raise UpstreamError(msg) from error
        return _parse_examples(payload, hops=hops)


__all__ = ["RawInstance", "UpstreamError", "generate_raw"]
