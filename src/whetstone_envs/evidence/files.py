from __future__ import annotations

import json
from typing import TYPE_CHECKING

from dr_store import CanonicalJsonFile
from pydantic import BaseModel

from whetstone_envs.evidence.evaluation import EvalReport
from whetstone_envs.evidence.trajectory import TrajectoryReport

if TYPE_CHECKING:
    from pathlib import Path

EVAL_REPORT_NAME = "eval-report.json"
TRAJECTORY_REPORT_NAME = "trajectory-report.json"
MAX_REPORT_BYTES = 128 * 1024 * 1024


def _report_file(directory: Path, name: str) -> CanonicalJsonFile:
    return CanonicalJsonFile(directory, name, max_bytes=MAX_REPORT_BYTES)


def publish_eval_report(directory: Path, report: EvalReport) -> Path:
    # Revalidate nested dictionaries, which frozen models do not freeze.
    validated = EvalReport.model_validate_json(report.model_dump_json())
    document = _report_file(directory, EVAL_REPORT_NAME)
    document.publish(validated.model_dump(mode="json"))
    return document.path


def publish_trajectory_report(
    directory: Path, report: TrajectoryReport
) -> Path:
    # Revalidate nested dictionaries, which frozen models do not freeze.
    validated = TrajectoryReport.model_validate_json(report.model_dump_json())
    document = _report_file(directory, TRAJECTORY_REPORT_NAME)
    document.publish(validated.model_dump(mode="json"))
    return document.path


def _load_report[Report: BaseModel](
    directory_or_file: Path, name: str, model: type[Report]
) -> Report:
    path = directory_or_file.resolve()
    directory, filename = (
        (path.parent, path.name) if path.is_file() else (path, name)
    )
    raw = _report_file(directory, filename).read()
    return model.model_validate_json(json.dumps(raw))


def load_eval_report(directory_or_file: Path) -> EvalReport:
    return _load_report(directory_or_file, EVAL_REPORT_NAME, EvalReport)


def load_trajectory_report(directory_or_file: Path) -> TrajectoryReport:
    return _load_report(
        directory_or_file,
        TRAJECTORY_REPORT_NAME,
        TrajectoryReport,
    )
