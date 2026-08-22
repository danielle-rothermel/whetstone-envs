from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Browser, Page, expect, sync_playwright

from whetstone_envs.optim.run import RunSpec, run_optimizer
from whetstone_envs.reporting.html import (
    publish_eval_html,
    publish_trajectory_html,
)
from whetstone_envs.reporting.publication import (
    load_eval_report,
    load_trajectory_report,
    publish_eval_report,
    publish_trajectory_report,
)
from whetstone_envs.reporting.schema import (
    EvalReport,
    EvalSuccess,
    ObservationState,
    ProviderErrorProjection,
    RowAccounting,
    StratumSummary,
    TrajectoryReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_SCREENSHOTS = Path(__file__).parent / "screenshots"


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


def _problem_report(source: Path, destination: Path) -> Path:
    report = load_eval_report(source)
    adversarial = "  </script><img src=x onerror=alert(1)> λ {grid}\n  "
    first = report.candidates[0].model_copy(
        update={
            "prompt_template": adversarial,
            "payload": {"prompt_template": adversarial},
        }
    )
    observations = []
    first_rows = [
        row for row in report.observations if row.candidate_name == first.name
    ]
    for index, row in enumerate(report.observations):
        if row.candidate_name != first.name:
            observations.append(row)
            continue
        observations.append(
            row.model_copy(
                update={
                    "score": None,
                    "state": (
                        ObservationState.FAILED
                        if index == 0
                        else ObservationState.MISSING
                    ),
                    "trace_state": ("failed" if index == 0 else "missing"),
                    "failure_code": "browser-fixture",
                    "provider_error": (
                        ProviderErrorProjection(
                            failure_class="timeout",
                            source="transport_failure",
                            recoverability="transient",
                        )
                        if index == 0
                        else None
                    ),
                }
            )
        )
    accounting = RowAccounting(
        planned=len(first_rows),
        present=0,
        missing=1,
        failed=1,
        invalid=0,
    )
    original = report.results[0]
    assert isinstance(original, EvalSuccess)
    strata = tuple(
        StratumSummary(
            stratum=item.stratum,
            numerator=0,
            denominator=1,
            accounting=RowAccounting(
                planned=1,
                present=0,
                missing=int(index == 1),
                failed=int(index == 0),
                invalid=0,
            ),
            score=None,
        )
        for index, item in enumerate(original.strata)
    )
    result = original.model_copy(
        update={
            "accounting": accounting,
            "evidence": original.evidence.model_copy(
                update={"row_accounting": accounting}
            ),
            "numerator": 0,
            "score": None,
            "strata": strata,
        }
    )
    changed = report.model_copy(
        update={
            "candidates": (first, *report.candidates[1:]),
            "observations": tuple(observations),
            "results": (result, *report.results[1:]),
        }
    )
    destination.mkdir()
    publish_eval_report(destination, changed)
    return publish_eval_html(destination)


def _guard_page(page: Page, errors: list[str]) -> None:
    page.on(
        "request",
        lambda request: (
            errors.append(f"network request: {request.url}")
            if not request.url.startswith("file:")
            else None
        ),
    )
    page.on("pageerror", lambda error: errors.append(f"page: {error}"))
    page.on(
        "console",
        lambda message: (
            errors.append(f"console {message.type}: {message.text}")
            if message.type == "error"
            else None
        ),
    )
    page.add_init_script(
        """
        window.__fallbackCopies = [];
        document.addEventListener('whetstone-copy-fallback', (event) => {
          window.__fallbackCopies.push(event.detail);
        });
        """
    )


def _dimensions(value: bytes) -> tuple[int, int]:
    assert value[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", value[16:24])


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def _rgb_pixels(value: bytes) -> tuple[int, int, bytes]:
    width, height = _dimensions(value)
    offset = 8
    compressed = bytearray()
    while offset < len(value):
        length = struct.unpack(">I", value[offset : offset + 4])[0]
        kind = value[offset + 4 : offset + 8]
        data = value[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            assert data[8:13] == bytes((8, 2, 0, 0, 0))
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
        offset += length + 12
    raw = zlib.decompress(compressed)
    stride = width * 3
    rows: list[bytearray] = []
    position = 0
    for _ in range(height):
        filter_kind = raw[position]
        source = raw[position + 1 : position + 1 + stride]
        position += stride + 1
        previous = rows[-1] if rows else bytearray(stride)
        decoded = bytearray(stride)
        for index, item in enumerate(source):
            left = decoded[index - 3] if index >= 3 else 0
            above = previous[index]
            upper_left = previous[index - 3] if index >= 3 else 0
            predictor = {
                0: 0,
                1: left,
                2: above,
                3: (left + above) // 2,
                4: _paeth(left, above, upper_left),
            }[filter_kind]
            decoded[index] = (item + predictor) & 0xFF
        rows.append(decoded)
    return width, height, b"".join(rows)


def _block_means(value: bytes, block: int = 16) -> tuple[tuple[int, ...], ...]:
    width, height, pixels = _rgb_pixels(value)
    means: list[tuple[int, ...]] = []
    for top in range(0, height, block):
        for left in range(0, width, block):
            totals = [0, 0, 0]
            count = 0
            for y in range(top, min(top + block, height)):
                for x in range(left, min(left + block, width)):
                    start = (y * width + x) * 3
                    for channel in range(3):
                        totals[channel] += pixels[start + channel]
                    count += 1
            means.append(tuple(total // count for total in totals))
    return tuple(means)


def _assert_screenshot_matches(current: bytes, expected: bytes) -> None:
    assert _dimensions(current) == _dimensions(expected)
    actual_blocks = _block_means(current)
    expected_blocks = _block_means(expected)
    assert len(actual_blocks) == len(expected_blocks)
    errors = tuple(
        sum(abs(a - b) for a, b in zip(actual, baseline, strict=True)) / 3
        for actual, baseline in zip(
            actual_blocks, expected_blocks, strict=True
        )
    )
    mean_error = sum(errors) / len(errors)
    materially_changed = sum(error > 48 for error in errors) / len(errors)
    assert mean_error < 20, f"screenshot mean block error {mean_error:.2f}"
    assert materially_changed < 0.25, (
        f"screenshot changed-block fraction {materially_changed:.3f}"
    )


def _screenshot(page: Page, name: str) -> None:
    current = page.screenshot(full_page=False)
    fixture = _SCREENSHOTS / name
    if os.environ.get("UPDATE_REPORT_SCREENSHOTS") == "1":
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(current)
    assert fixture.is_file()
    assert fixture.stat().st_size < 500_000
    _assert_screenshot_matches(current, fixture.read_bytes())


def _renamed_report(report: EvalReport, first_name: str) -> EvalReport:
    original = report.candidates[0].name
    payload = report.model_dump(mode="json")
    payload["candidates"][0]["name"] = first_name
    payload["results"][0]["candidate_name"] = first_name
    for row in payload["observations"]:
        if row["candidate_name"] == original:
            row["candidate_name"] = first_name
    return EvalReport.model_validate_json(json.dumps(payload))


def _first_row_pass(report: EvalReport) -> EvalReport:
    tasks = {task.task_id: task for task in report.tasks}
    observations = list(report.observations)
    first = observations[0]
    gold = tasks[first.task_id].gold
    observations[0] = first.model_copy(
        update={"output_text": gold, "normalized_output": gold, "score": 1.0}
    )
    result = report.results[0]
    assert isinstance(result, EvalSuccess)
    strata = []
    for summary in result.strata:
        rows = [
            row
            for row in observations
            if summary.stratum in tasks[row.task_id].strata
        ]
        numerator = sum(row.score == 1.0 for row in rows)
        strata.append(
            summary.model_copy(
                update={
                    "numerator": numerator,
                    "score": numerator / len(rows),
                }
            )
        )
    numerator = sum(row.score == 1.0 for row in observations)
    score = numerator / len(observations)
    updated_result = result.model_copy(
        update={
            "numerator": numerator,
            "score": score,
            "strata": tuple(strata),
            "evidence": result.evidence.model_copy(
                update={"aggregate_value": score}
            ),
        }
    )
    payload = report.model_copy(
        update={
            "observations": tuple(observations),
            "results": (updated_result,),
        }
    ).model_dump(mode="json")
    return EvalReport.model_validate_json(json.dumps(payload))


def _repeat_child_resolution(directory: Path) -> TrajectoryReport:
    report = load_trajectory_report(directory)
    child, base = report.resolutions
    assert child.candidate_ref != base.candidate_ref
    assert child.eval_report is not None
    assert base.eval_report is not None
    improved = _first_row_pass(child.eval_report)
    improved_result = improved.results[0]
    assert isinstance(improved_result, EvalSuccess)
    parent_rows = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in base.eval_report.observations
    }
    current_rows = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in improved.observations
    }
    coordinates = set(parent_rows) | set(current_rows)
    gains = regressions = mismatches = 0
    for coordinate in coordinates:
        parent = parent_rows.get(coordinate)
        current = current_rows.get(coordinate)
        if (
            parent is None
            or current is None
            or parent.state is not ObservationState.SCORED
            or current.state is not ObservationState.SCORED
        ):
            mismatches += 1
        elif parent.score == 0.0 and current.score == 1.0:
            gains += 1
        elif parent.score == 1.0 and current.score == 0.0:
            regressions += 1
    repeated = child.model_copy(
        update={
            "resolution_index": 2,
            "request_id": f"{child.request_id}:repeat",
            "reward": improved_result.score,
            "eval_report": improved,
            "gains": gains,
            "regressions": regressions,
            "execution_mismatches": mismatches,
        }
    )
    first_step = report.steps[0].model_copy(
        update={"resolution_indexes": (0, 1, 2)}
    )
    changed = report.model_copy(
        update={
            "steps": (first_step, *report.steps[1:]),
            "resolutions": (child, base, repeated),
        }
    )
    return TrajectoryReport.model_validate_json(changed.model_dump_json())


def test_eval_report_file_interactions_and_screenshots(
    browser: Browser, fake_eval_output, tmp_path
) -> None:
    path = _problem_report(fake_eval_output.directory, tmp_path / "eval")
    errors: list[str] = []
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    _guard_page(page, errors)
    page.goto(path.as_uri())

    assert page.get_by_text("PRIVATE ARTIFACT", exact=False).is_visible()
    assert page.get_by_text("Incomplete — no numeric score").first.is_visible()
    assert page.get_by_text("0.0%").first.is_visible()
    assert page.get_by_text("Failed").first.is_visible()
    assert page.get_by_text("Missing").first.is_visible()
    accounting = page.get_by_role(
        "heading", name="Outcome accounting · naive"
    ).locator("..")
    assert (
        accounting.get_by_text("Provider errors")
        .locator("..")
        .get_by_text("1", exact=True)
        .is_visible()
    )
    assert page.locator("img").count() == 0
    assert page.locator("tbody tr").count() == 4
    assert (
        page.locator(".diff-add")
        .get_by_text("</script><img", exact=False)
        .is_visible()
    )
    assert page.locator(".diff-del").count() > 0

    page.locator("button.cell").filter(has_text="navigation").first.click()
    assert "scenario=navigation" in page.url
    page.get_by_role("button", name="execution mismatch 2").click()
    assert "bucket=execution+mismatch" in page.url
    page.get_by_role("button", name="Open").first.click()
    assert page.get_by_text("Ordered component trace").first.is_visible()
    assert page.get_by_text("Full candidate template").first.is_visible()
    candidate_section = page.get_by_role(
        "heading", name="Full candidate template"
    ).first.locator("..")
    candidate_section.get_by_role("button", name="Copy", exact=True).click()
    expect(page.locator("html")).to_have_attribute(
        "data-copy-fallback", "complete"
    )
    assert page.evaluate("window.__fallbackCopies.length") == 1
    assert page.evaluate("window.__fallbackCopies[0]").startswith(
        "  </script>"
    )
    page.locator("#task-dialog").get_by_role(
        "button", name="Close", exact=True
    ).click()

    about = page.get_by_role("button", name="About C19")
    about.focus()
    page.keyboard.press("Enter")
    assert page.get_by_role("heading", name="About C19").is_visible()
    page.locator("#about-dialog").get_by_role(
        "button", name="Close", exact=True
    ).click()
    page.locator("aside .candidate").filter(has_text="naive").click()
    clean_accounting = page.get_by_role(
        "heading", name="Outcome accounting · ceiling"
    ).locator("..")
    assert (
        clean_accounting.get_by_text("Provider errors")
        .locator("..")
        .get_by_text("0", exact=True)
        .is_visible()
    )
    page.evaluate("window.scrollTo(0, 0)")
    _screenshot(page, "eval-wide.png")
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("window.scrollTo(0, 0)")
    _screenshot(page, "eval-narrow.png")

    page.goto(f"{path.as_uri()}#candidate=not-a-candidate")
    page.reload()
    assert page.get_by_text("Fragment notice", exact=False).is_visible()
    assert not errors
    page.close()


def test_trajectory_file_timeline_lineage_and_screenshots(
    browser: Browser, tmp_path
) -> None:
    directory = run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            output_dir=tmp_path / "trajectory",
            run_id="browser-trajectory",
        )
    )
    publish_trajectory_report(directory, _repeat_child_resolution(directory))
    path = publish_trajectory_html(directory)
    errors: list[str] = []
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    _guard_page(page, errors)
    page.goto(path.as_uri())

    assert page.get_by_text(
        "Ordered step and exact resolution timeline"
    ).is_visible()
    assert page.get_by_text("Exact candidate lineage").is_visible()
    assert page.get_by_text("does not imply causation").is_visible()
    assert page.locator(".timeline button").count() == 3
    page.locator('[data-resolution="0:0"]').click()
    assert "resolution=0%3A0" in page.url
    page.locator('[data-resolution="0:2"]').click()
    assert "resolution=0%3A2" in page.url
    assert page.get_by_text("Overall observed change").is_visible()
    assert page.get_by_text("Per-stratum observed changes").is_visible()
    assert page.get_by_text("Exact changed observation rows").is_visible()
    assert (
        page.get_by_role("button", name="fail-to-pass", exact=False).count()
        > 0
    )
    page.locator(".lineage button:not([disabled])").last.click()
    assert "candidate=" in page.url
    assert page.get_by_text("Full exact text").is_visible()
    page.evaluate("window.scrollTo(0, 0)")
    _screenshot(page, "trajectory-wide.png")
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("window.scrollTo(0, 0)")
    _screenshot(page, "trajectory-narrow.png")
    assert not errors
    page.close()


def test_comma_candidate_fragment_round_trips_exactly(
    browser: Browser, fake_eval_output, tmp_path
) -> None:
    first_name = "naive,with,commas"
    directory = tmp_path / "commas"
    directory.mkdir()
    report = _renamed_report(fake_eval_output.report, first_name)
    publish_eval_report(directory, report)
    path = publish_eval_html(directory)
    query = urlencode((("candidate", first_name), ("candidate", "ceiling")))
    page = browser.new_page()
    page.goto(f"{path.as_uri()}#{query}")

    assert (
        page.locator('[aria-pressed="true"]')
        .filter(has_text=first_name)
        .count()
        == 1
    )
    assert (
        page.locator('[aria-pressed="true"]')
        .filter(has_text="ceiling")
        .count()
        == 1
    )
    assert page.locator("#fragment-notice").is_hidden()
    page.reload()
    assert page.locator("#fragment-notice").is_hidden()
    page.close()


def test_screenshot_comparison_rejects_material_change(
    browser: Browser,
) -> None:
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.set_content(
        '<main style="position:fixed;inset:0;background:#f0f">'
        "materially different</main>"
    )
    with pytest.raises(AssertionError, match="screenshot"):
        _assert_screenshot_matches(
            page.screenshot(full_page=False),
            (_SCREENSHOTS / "eval-wide.png").read_bytes(),
        )
    page.close()
