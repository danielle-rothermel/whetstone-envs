from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from whetstone_envs.c19._info import C19_INFO
from whetstone_envs.reporting.derivations import compare_buckets
from whetstone_envs.reporting.publication import (
    load_eval_report,
    load_trajectory_report,
    validate_output_root,
)
from whetstone_envs.reporting.schema import (
    EvalReport,
    EvalSuccess,
    TrajectoryReport,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import JsonValue

EVAL_HTML_NAME = "eval-report.html"
TRAJECTORY_HTML_NAME = "trajectory-report.html"
_ASSET_PACKAGE = "whetstone_envs.reporting.assets"
_FACET_COUNT = 3


def _asset(name: str) -> str:
    return files(_ASSET_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _sha256_source(value: str) -> str:
    digest = hashlib.sha256(value.encode()).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


def _strict_embedded_json(value: Mapping[str, JsonValue]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    normalized = json.loads(raw)
    escaped = (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    if json.loads(escaped) != normalized:
        raise ValueError("embedded report JSON failed semantic round trip")
    return escaped


def _task_facets(strata: tuple[str, ...]) -> dict[str, str]:
    parts = strata[0].split("|")
    if len(parts) != _FACET_COUNT:
        raise ValueError("C19 report task requires scenario|size|fact stratum")
    return dict(zip(("scenario", "size", "fact"), parts, strict=True))


def _eval_view(report: EvalReport) -> dict[str, JsonValue]:
    task_facets = {
        task.task_id: _task_facets(task.strata) for task in report.tasks
    }
    summaries: dict[str, JsonValue] = {}
    matrices: dict[str, JsonValue] = {}
    for result in report.results:
        if isinstance(result, EvalSuccess):
            summaries[result.candidate_name] = {
                "kind": result.kind,
                "numerator": result.numerator,
                "denominator": result.denominator,
                "score": result.score,
                "accounting": result.accounting.model_dump(mode="json"),
                "classification": result.classification,
                "message": result.message,
            }
            matrices[result.candidate_name] = {
                item.stratum: item.model_dump(mode="json")
                for item in result.strata
            }
        else:
            summaries[result.candidate_name] = result.model_dump(mode="json")

    pairs: dict[str, JsonValue] = {}
    names = [candidate.name for candidate in report.candidates]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows = compare_buckets(report, left, right)
            key = f"{left_index}:{names.index(right)}"
            pairs[key] = {
                "left": left,
                "right": right,
                "rows": [
                    {
                        "bucket": row.bucket,
                        "task_id": row.task_id,
                        "seed_index": row.seed_index,
                    }
                    for row in rows
                ],
            }
    provider_errors = {
        name: sum(
            row.candidate_name == name and row.provider_error is not None
            for row in report.observations
        )
        for name in names
    }
    return {
        "task_facets": task_facets,
        "summaries": summaries,
        "matrices": matrices,
        "pairs": pairs,
        "provider_errors": provider_errors,
    }


def _comparison_view(
    parent: EvalReport, current: EvalReport
) -> dict[str, JsonValue]:
    parent_rows = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in parent.observations
    }
    current_rows = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in current.observations
    }
    planned = tuple(
        dict.fromkeys(
            (task.task_id, task.task_hash, seed_index)
            for report in (parent, current)
            for task in report.tasks
            for seed_index in range(report.run.repeats)
        )
    )
    changed_rows: list[JsonValue] = []
    counts = {
        "fail_to_pass": 0,
        "pass_to_fail": 0,
        "execution_mismatch": 0,
    }
    for coordinate in planned:
        before = parent_rows.get(coordinate)
        after = current_rows.get(coordinate)
        if (
            before is None
            or after is None
            or before.state.value != "scored"
            or after.state.value != "scored"
        ):
            bucket = "execution_mismatch"
        elif before.score == 0.0 and after.score == 1.0:
            bucket = "fail_to_pass"
        elif before.score == 1.0 and after.score == 0.0:
            bucket = "pass_to_fail"
        else:
            continue
        counts[bucket] += 1
        changed_rows.append(
            {
                "bucket": bucket,
                "task_id": coordinate[0],
                "seed_index": coordinate[2],
            }
        )

    parent_result = parent.results[0]
    current_result = current.results[0]
    overall: JsonValue = None
    strata: list[JsonValue] = []
    if isinstance(parent_result, EvalSuccess) and isinstance(
        current_result, EvalSuccess
    ):
        overall = {
            "before": parent_result.score,
            "after": current_result.score,
            "change": (
                current_result.score - parent_result.score
                if parent_result.score is not None
                and current_result.score is not None
                else None
            ),
            "before_numerator": parent_result.numerator,
            "after_numerator": current_result.numerator,
            "denominator": current_result.denominator,
        }
        current_strata = {item.stratum: item for item in current_result.strata}
        for before in parent_result.strata:
            after = current_strata.get(before.stratum)
            if after is None:
                continue
            strata.append(
                {
                    "stratum": before.stratum,
                    "before": before.score,
                    "after": after.score,
                    "change": (
                        after.score - before.score
                        if before.score is not None and after.score is not None
                        else None
                    ),
                    "before_numerator": before.numerator,
                    "after_numerator": after.numerator,
                    "denominator": after.denominator,
                }
            )
    return {
        "overall": overall,
        "strata": strata,
        "counts": counts,
        "changed_rows": changed_rows,
    }


def _trajectory_view(report: TrajectoryReport) -> dict[str, JsonValue]:
    eval_views: dict[str, JsonValue] = {}
    diagnoses: dict[str, JsonValue] = {}
    resolution_keys_by_candidate: dict[str, JsonValue] = {}
    latest_eval_by_ref: dict[str, EvalReport] = {}
    candidates_by_ref = {
        candidate.record_ref.content_hash: candidate
        for candidate in report.candidates
    }
    for row in report.resolutions:
        key = f"{row.step_index}:{row.resolution_index}"
        candidate_hash = row.candidate_ref.content_hash
        resolution_keys = resolution_keys_by_candidate.setdefault(
            candidate_hash, []
        )
        assert isinstance(resolution_keys, list)
        resolution_keys.append(key)
        if row.eval_report is not None:
            eval_views[key] = _eval_view(row.eval_report)
            candidate = candidates_by_ref[candidate_hash]
            parent = latest_eval_by_ref.get(candidate.base_ref.content_hash)
            if parent is not None:
                diagnosis = _comparison_view(parent, row.eval_report)
                counts = diagnosis["counts"]
                assert isinstance(counts, dict)
                expected = (
                    counts["fail_to_pass"],
                    counts["pass_to_fail"],
                    counts["execution_mismatch"],
                )
                recorded = (
                    row.gains,
                    row.regressions,
                    row.execution_mismatches,
                )
                if recorded != expected:
                    raise ValueError(
                        "trajectory comparison counts disagree with exact "
                        "then-current evaluation rows"
                    )
                diagnoses[key] = diagnosis
            latest_eval_by_ref[candidate_hash] = row.eval_report
    return {
        "eval_views": eval_views,
        "diagnoses": diagnoses,
        "candidate_by_ref": {
            candidate.record_ref.content_hash: index
            for index, candidate in enumerate(report.candidates)
        },
        "resolution_keys_by_candidate": resolution_keys_by_candidate,
    }


def _payload(
    report: EvalReport | TrajectoryReport,
    kind: Literal["eval", "trajectory"],
) -> dict[str, JsonValue]:
    return {
        "kind": kind,
        "report": report.model_dump(mode="json"),
        "info": asdict(C19_INFO),
        "view": (
            _eval_view(report)
            if isinstance(report, EvalReport)
            else _trajectory_view(report)
        ),
    }


def render_html_bytes(
    report: EvalReport | TrajectoryReport,
    *,
    kind: Literal["eval", "trajectory"],
) -> bytes:
    validated: EvalReport | TrajectoryReport
    if kind == "eval":
        validated = EvalReport.model_validate(report)
    else:
        validated = TrajectoryReport.model_validate(report)
    style = _asset("report.css")
    script = _asset("report.js")
    shell = _asset("shell.html")
    csp = "; ".join(
        (
            "default-src 'none'",
            f"script-src '{_sha256_source(script)}'",
            f"style-src '{_sha256_source(style)}'",
            "connect-src 'none'",
            "img-src 'none'",
            "font-src 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        )
    )
    trusted_replacements = {
        "@@CSP@@": csp,
        "@@STYLE@@": style,
        "@@SCRIPT@@": script,
    }
    markers = (*trusted_replacements, "@@REPORT@@")
    for marker in markers:
        if shell.count(marker) != 1:
            raise ValueError(f"HTML shell requires exactly one {marker}")
    if any(
        marker in value
        for value in trusted_replacements.values()
        for marker in markers
    ):
        raise ValueError("trusted HTML assets must not contain shell markers")
    for marker, value in trusted_replacements.items():
        shell = shell.replace(marker, value)
    shell = shell.replace(
        "@@REPORT@@", _strict_embedded_json(_payload(validated, kind))
    )
    return shell.encode("utf-8")


def _atomic_publish(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def publish_eval_html(directory_or_file: Path) -> Path:
    report = load_eval_report(directory_or_file)
    directory = (
        directory_or_file.resolve().parent
        if directory_or_file.resolve().is_file()
        else directory_or_file.resolve()
    )
    validate_output_root(directory)
    return _atomic_publish(
        directory / EVAL_HTML_NAME,
        render_html_bytes(report, kind="eval"),
    )


def publish_trajectory_html(directory_or_file: Path) -> Path:
    report = load_trajectory_report(directory_or_file)
    directory = (
        directory_or_file.resolve().parent
        if directory_or_file.resolve().is_file()
        else directory_or_file.resolve()
    )
    validate_output_root(directory)
    return _atomic_publish(
        directory / TRAJECTORY_HTML_NAME,
        render_html_bytes(report, kind="trajectory"),
    )


__all__ = [
    "EVAL_HTML_NAME",
    "TRAJECTORY_HTML_NAME",
    "publish_eval_html",
    "publish_trajectory_html",
    "render_html_bytes",
]
