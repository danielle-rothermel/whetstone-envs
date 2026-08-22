from __future__ import annotations

import base64
import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dr_serialize import canonical_json_bytes
from pydantic import ValidationError

from whetstone_envs.reporting.html import (
    EVAL_HTML_NAME,
    publish_eval_html,
    render_html_bytes,
)


def _embedded(html: bytes) -> dict[str, Any]:
    match = re.search(
        rb'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _csp_hash(asset: str) -> str:
    digest = hashlib.sha256(asset.encode()).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


def test_html_is_deterministic_strict_self_contained_and_round_trips(
    fake_eval_output,
) -> None:
    first = render_html_bytes(fake_eval_output.report, kind="eval")
    second = render_html_bytes(fake_eval_output.report, kind="eval")

    assert first == second
    embedded = _embedded(first)
    assert embedded["report"] == fake_eval_output.report.model_dump(
        mode="json"
    )
    assert b"http://" not in first
    assert b"https://" not in first
    assert b"<link" not in first
    assert b"<img" not in first
    assert b"fetch(" not in first
    assert b"XMLHttpRequest" not in first
    assert b"WebSocket" not in first
    assert b"innerHTML" not in first
    assert b"outerHTML" not in first
    assert b"insertAdjacentHTML" not in first
    assert b"document.write" not in first
    assert b"eval(" not in first
    assert b"new Function" not in first
    assert b"localStorage" not in first
    assert b"sessionStorage" not in first

    package = files("whetstone_envs.reporting.assets")
    style = package.joinpath("report.css").read_text(encoding="utf-8")
    script = package.joinpath("report.js").read_text(encoding="utf-8")
    assert _csp_hash(style).encode() in first
    assert _csp_hash(script).encode() in first
    assert b"default-src 'none'" in first
    assert b"connect-src 'none'" in first


def test_adversarial_report_text_remains_inert_and_exact(
    fake_eval_output,
) -> None:
    adversarial = (
        "  </script><img src=x onerror=alert(1)> & λ {grid}\n"
        + "x" * 4096
        + "\u2028\u2029  "
    )
    candidate = fake_eval_output.report.candidates[0].model_copy(
        update={
            "prompt_template": adversarial,
            "payload": {"prompt_template": adversarial},
        }
    )
    report = fake_eval_output.report.model_copy(
        update={
            "candidates": (
                candidate,
                *fake_eval_output.report.candidates[1:],
            )
        }
    )

    html = render_html_bytes(report, kind="eval")

    assert b"</script><img" not in html
    assert b"\\u003c/script>\\u003cimg" in html
    assert b"\\u0026" in html
    assert b"\\u2028\\u2029" in html
    embedded = _embedded(html)
    assert (
        embedded["report"]["candidates"][0]["prompt_template"] == adversarial
    )


@pytest.mark.parametrize(
    "marker", ["@@CSP@@", "@@STYLE@@", "@@REPORT@@", "@@SCRIPT@@"]
)
def test_report_text_round_trips_shell_marker_literals(
    fake_eval_output, marker
) -> None:
    text = f"before {marker} after"
    candidate = fake_eval_output.report.candidates[0].model_copy(
        update={
            "prompt_template": text,
            "payload": {"prompt_template": text},
        }
    )
    report = fake_eval_output.report.model_copy(
        update={
            "candidates": (
                candidate,
                *fake_eval_output.report.candidates[1:],
            )
        }
    )

    embedded = _embedded(render_html_bytes(report, kind="eval"))

    assert embedded["report"]["candidates"][0]["prompt_template"] == text


def test_failed_render_preserves_existing_html(fake_eval_output) -> None:
    path = publish_eval_html(fake_eval_output.directory)
    before = path.read_bytes()

    with (
        patch(
            "whetstone_envs.reporting.html._asset",
            side_effect=RuntimeError("asset failed"),
        ),
        pytest.raises(RuntimeError, match="asset failed"),
    ):
        publish_eval_html(fake_eval_output.directory)

    assert path.name == EVAL_HTML_NAME
    assert path.read_bytes() == before


def test_html_refuses_unsupported_report_schema(
    fake_eval_output, tmp_path
) -> None:
    payload = json.loads(
        (fake_eval_output.directory / "eval-report.json").read_bytes()
    )
    payload["schema_version"] = "unsupported"
    path = tmp_path / "unsupported.json"
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValidationError):
        publish_eval_html(path)


def test_html_publication_keeps_repository_guard(fake_eval_output) -> None:
    repository = Path(__file__).resolve().parents[2]
    with (
        patch(
            "whetstone_envs.reporting.html.load_eval_report",
            return_value=fake_eval_output.report,
        ),
        pytest.raises(ValueError, match="inside the repo"),
    ):
        publish_eval_html(repository / "eval-report.json")
