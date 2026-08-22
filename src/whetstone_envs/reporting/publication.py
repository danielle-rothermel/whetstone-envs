from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dr_store import CanonicalJsonFile
from pydantic import BaseModel

from whetstone_envs.reporting.schema import EvalReport, TrajectoryReport

if TYPE_CHECKING:
    from collections.abc import Iterator

EVAL_REPORT_NAME = "eval-report.json"
TRAJECTORY_REPORT_NAME = "trajectory-report.json"
# A canonical C19 manifest is small, but reports retain full prompts, outputs,
# and component traces for every candidate/repeat row.
MAX_REPORT_BYTES = 128 * 1024 * 1024


class DurableRunError(RuntimeError):
    """An operation failed after its durable run directory was created."""

    def __init__(self, directory: Path, cause: Exception) -> None:
        super().__init__(str(cause))
        self.directory = directory
        self.cause = cause


@contextmanager
def durable_run_boundary(directory: Path) -> Iterator[None]:
    try:
        yield
    except DurableRunError:
        raise
    except Exception as error:
        raise DurableRunError(directory, error) from error


def _git_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def repository_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        root = _git_root(start)
        if root is not None and root not in roots:
            roots.append(root)
    return tuple(roots)


def validate_output_root(path: Path) -> Path:
    """Resolve and reject a run directory inside any detected repository."""
    resolved = path.resolve()
    target_repository = _git_root(resolved)
    if target_repository is not None or any(
        resolved.is_relative_to(root) for root in repository_roots()
    ):
        raise ValueError("run artifacts must not be written inside the repo")
    return resolved


def prepare_output_root(path: Path) -> Path:
    resolved = validate_output_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _document(directory: Path, name: str) -> CanonicalJsonFile:
    return CanonicalJsonFile(directory, name, max_bytes=MAX_REPORT_BYTES)


def publish_eval_report(directory: Path, report: EvalReport) -> Path:
    validated = EvalReport.model_validate_json(report.model_dump_json())
    document = _document(directory, EVAL_REPORT_NAME)
    document.publish(validated.model_dump(mode="json"))
    return document.path


def publish_trajectory_report(
    directory: Path, report: TrajectoryReport
) -> Path:
    validated = TrajectoryReport.model_validate_json(report.model_dump_json())
    document = _document(directory, TRAJECTORY_REPORT_NAME)
    document.publish(validated.model_dump(mode="json"))
    return document.path


def _load[Report: BaseModel](
    directory_or_file: Path, name: str, model: type[Report]
) -> Report:
    path = directory_or_file.resolve()
    directory, filename = (
        (path.parent, path.name) if path.is_file() else (path, name)
    )
    raw = _document(directory, filename).read()
    return model.model_validate_json(json.dumps(raw))


def load_eval_report(directory_or_file: Path) -> EvalReport:
    return _load(directory_or_file, EVAL_REPORT_NAME, EvalReport)


def load_trajectory_report(directory_or_file: Path) -> TrajectoryReport:
    return _load(
        directory_or_file,
        TRAJECTORY_REPORT_NAME,
        TrajectoryReport,
    )


__all__ = [
    "EVAL_REPORT_NAME",
    "MAX_REPORT_BYTES",
    "TRAJECTORY_REPORT_NAME",
    "DurableRunError",
    "durable_run_boundary",
    "load_eval_report",
    "load_trajectory_report",
    "prepare_output_root",
    "publish_eval_report",
    "publish_trajectory_report",
    "repository_roots",
    "validate_output_root",
]
