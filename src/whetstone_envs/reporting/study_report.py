"""The Step 10 study report: a packet of ``report.md`` and ``report.html``.

The generator reads exactly two things: the study manifest, and the run
stores the manifest's pointers name. It never recomputes a statistic. Every
mean, interval, p-value, and call count the report prints is a field of
``study.json``, and the report renders it beside the
``(schema_name, content_hash)`` pointer the manifest cites for it.

**Why a value type rather than a formatting convention.** A report that
prints numbers and separately prints pointers can drift: a number gets added
and its pointer does not, and nothing catches it. So a number does not exist
here as a string. It exists as a :class:`Figure` -- a rendered value bound to
the pointer that backs it -- and both emitters render a ``Figure`` the same
way. :func:`figures_in` walks the built report and yields every one, which is
what makes "every number resolves to a manifest pointer" a mechanical test
over the report object rather than a regex over its output.

**What the manifest cannot supply.** Three of the assignment's content items
are not derivable from a v2 manifest, and the report says so in place rather
than inventing them:

* **Wall time.** ``RunSpendRecord`` carries calls, cached calls, tokens, and
  USD. It carries no duration, and neither does ``cost.json``. The cost
  column reports what the manifest holds and names wall time as unrecorded.
* **The trace-audit table.** ``RunRecord.audit_ref`` points at the
  ``audit.json`` the audit package writes. The report resolves that pointer
  through the store and renders whatever findings it holds; with no store, or
  with an audit package that has not landed, it prints the recorded
  ``audit_passed`` verdict and says the finding table was not resolved. It
  never fabricates an invariant list.
* **The internal trajectory and the proposed prompts.** Both live in the run
  artifacts rather than in the manifest. The generator reads them through
  :func:`~whetstone_envs.reporting.publication.load_trajectory_report` when
  the run's ``artifact_dir`` holds a projected trajectory report, and reports
  their absence otherwise.

The HTML follows the ``html-doc-polish`` kit and renders with no network
access: the stylesheet, the favicon, and every asset live inside the packet,
and the document loads no font, script, or highlighter from a CDN.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from whetstone_envs.optim.study.manifest import (
    PROVENANCE_AMENDED,
    STAGE_IDS,
    STUDY_MANIFEST_NAME,
    EvidencePointer,
    StageId,
    StudyManifest,
    TransportName,
)
from whetstone_envs.reporting.publication import (
    TRAJECTORY_REPORT_NAME,
    load_trajectory_report,
    validate_output_root,
)
from whetstone_envs.reporting.schema import TrajectoryReport

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from whetstone_envs.optim.study.manifest import (
        ArmRecord,
        EvidenceStore,
        HeldOutRecord,
        ProviderCallRecord,
        RunRecord,
        StageRecord,
    )

__all__ = [
    "ASSET_NAMES",
    "MISSING",
    "NO_PROVIDER_STAGE_DETAIL",
    "NULL_ARM_PREFIX",
    "REPORT_HTML_NAME",
    "REPORT_MARKDOWN_NAME",
    "STAGE_SPEND_COVERAGE_NOTE",
    "STUDY_MANIFEST_COPY",
    "UNLEDGERED_STAGE_DETAIL",
    "UNPRICED",
    "VALIDATION_CHECKLIST",
    "VERDICT_INCOMPLETE",
    "VERDICT_INVALID",
    "VERDICT_NOT_VALIDATED",
    "VERDICT_NO_IMPROVEMENT",
    "VERDICT_UNMEASURED",
    "VERDICT_VALIDATED",
    "Cell",
    "Figure",
    "RenderedText",
    "Row",
    "Section",
    "StudyReport",
    "Table",
    "build_study_report",
    "figures_in",
    "generate_study_report",
    "render_html",
    "render_markdown",
    "rendered_text_in",
    "study_leakage_failed",
]

# --------------------------------------------------------------------------
# Packet layout
# --------------------------------------------------------------------------

#: The packet's two renderings of one report. The Markdown is the source and
#: the HTML is the polished reading copy; both are emitted from the same
#: built :class:`StudyReport`, so they cannot disagree.
REPORT_MARKDOWN_NAME = "report.md"
REPORT_HTML_NAME = "report.html"

#: Presentation assets copied into the packet. They are packet-local because
#: the HTML must render with no network: nothing is fetched, so nothing can
#: fail to load.
ASSET_NAMES: tuple[str, ...] = ("doc.css", "favicon.svg")

_ASSET_PACKAGE = "whetstone_envs.reporting.assets.study"

#: The manifest is copied into the packet beside the report, because a report
#: whose evidence pointers name a document the reader does not have is a
#: report the reader cannot check.
STUDY_MANIFEST_COPY = STUDY_MANIFEST_NAME

#: How the report spells a fact the manifest does not carry. It is one
#: string so a reader can find every gap at once, and so the mechanical
#: pointer test can tell "absent" from "unbacked".
MISSING = "not recorded"

#: How an absent USD total renders. Never a zero: a role with unpriced calls
#: has a total nobody can compute, which is a different fact from free.
UNPRICED = "unpriced"

#: Arms whose ids start with this are controls rather than hypotheses. They
#: get a shorter section and no Holm correction, per the pre-registration.
NULL_ARM_PREFIX = "null-"

#: The percentile bootstrap cannot report a p-value below its own
#: resolution: with ``R`` resamples the smallest nonzero two-sided value is
#: ``2/R``, and an all-positive resample set is clamped there rather than to
#: zero. Holm then propagates that floor, so a corrected p-value at the floor
#: means "at or below the bootstrap's resolution", not an exact number.
PERCENTILE_P_FLOOR_CAVEAT = (
    "p-values are percentile-bootstrap values with a floor at the "
    "bootstrap's own resolution (2/resamples). A value at the floor means "
    "'at or below what {resamples:,} resamples can resolve', not an exact "
    "number, and Holm propagates that floor rather than dissolving it."
)

#: The manual checks Danielle must make before any of this is relied on.
#: Mandatory per review-patterns.md: these are experimental claims that
#: automated gates cannot settle.
VALIDATION_CHECKLIST: tuple[str, ...] = (
    (
        "Do the sampled optimized prompts read as genuine improvements, or as "
        "overfitting to the internal split's phrasing?"
    ),
    (
        "Is the Codex arm's pre-registered agent model a fair comparison "
        "against the arms whose proposer this study pinned, given that its "
        "own calls are never priced?"
    ),
    (
        "Does the null-A result look like selection-on-noise -- a positive "
        "delta produced by best-on-internal selection over perturbations "
        "alone?"
    ),
    (
        "Are the anchors behaving as intended on the task model, or is either "
        "the naive or the ceiling probe sitting at a floor or a ceiling?"
    ),
    (
        "Does the spend match the intent, and is the remaining key balance as "
        "expected?"
    ),
)


# --------------------------------------------------------------------------
# Figures: a number and the evidence that backs it
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Figure:
    """One rendered value and the manifest evidence that backs it.

    ``pointer`` is the manifest's own :class:`EvidencePointer` when the value
    is read from evidence the manifest cites, and ``None`` when the value is
    a field of the manifest itself. Both are backed -- a manifest field is
    backed by the manifest -- so ``source`` names which, and the mechanical
    test asserts that no figure is backed by neither.
    """

    value: str
    source: str
    pointer: EvidencePointer | None = None

    def evidence(self) -> str:
        """The provenance mark rendered beside the value.

        Both halves, when both exist: the manifest path says which field
        this number is, and the pointer says which stored record the
        manifest backs that field with. A reader checking the number needs
        the first to find it and the second to resolve it.
        """
        if self.pointer is None:
            return self.source
        return (
            f"{self.source} -> {self.pointer.schema_name}"
            f"@{self.pointer.content_hash[:12]}"
        )

    def backed(self) -> bool:
        """Whether this figure names where its value came from.

        A figure with neither a pointer nor a source names nothing, which is
        exactly the drift the value type exists to prevent.
        """
        return bool(self.pointer is not None or self.source.strip())


@dataclass(frozen=True, slots=True)
class Cell:
    """One table cell: either prose, or a figure with its evidence.

    Prose cells carry labels, ids, and verdicts -- things that are not
    numbers and have nothing to resolve. Figure cells carry every number the
    report prints.
    """

    text: str | None = None
    figure: Figure | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.figure is None):
            raise ValueError("a cell is either prose or a figure, not both")

    def rendered(self) -> str:
        """The cell's value, without its evidence mark.

        ``__post_init__`` guarantees exactly one of the two is set, but a
        checker cannot see that through the constructor, so the figure case
        is tested first and the prose case narrows to ``str``.
        """
        if self.figure is not None:
            return self.figure.value
        assert self.text is not None
        return self.text


@dataclass(frozen=True, slots=True)
class Row:
    """One table row."""

    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Table:
    """A titled table with a header row."""

    headers: tuple[str, ...]
    rows: tuple[Row, ...]
    caption: str | None = None

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row.cells) != len(self.headers):
                raise ValueError("every table row matches its header width")


@dataclass(frozen=True, slots=True)
class Section:
    """One report section: a question-first heading and its content.

    ``paragraphs`` are plain prose, ``tables`` are the evidence, and
    ``checklist`` is the one section that asks the reader to act. ``panels``
    holds child sections laid out side by side when the reader must
    cross-reference them.
    """

    heading: str
    tag: str | None = None
    paragraphs: tuple[str, ...] = ()
    tables: tuple[Table, ...] = ()
    checklist: tuple[str, ...] = ()
    code_blocks: tuple[tuple[str, str], ...] = ()
    panels: tuple[Section, ...] = ()
    level: int = 2


@dataclass(frozen=True, slots=True)
class StudyReport:
    """The whole built report, before either emitter renders it."""

    title: str
    dek: str
    byline: str
    sections: tuple[Section, ...]
    colophon: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())


def figures_in(report: StudyReport) -> Iterator[Figure]:
    """Every figure the report renders, in document order.

    This is the mechanical check's walk. A number added to the report
    without a figure is not reachable here -- but it is also not a number
    the emitters will render with evidence, because they render values
    through cells and a cell holding a number is a figure cell.
    """
    for section in report.sections:
        yield from _figures_in_section(section)


@dataclass(frozen=True, slots=True)
class RenderedText:
    """One rendered string that is *not* a figure, and where it came from.

    The figure walk covers every number the report backs with evidence.
    This covers everything else it renders -- paragraphs, prose cells,
    captions, headings, checklist items, code-block labels -- so a number
    that reached a reader without a pointer is findable mechanically rather
    than by reading the output.
    """

    kind: str
    location: str
    text: str


def rendered_text_in(report: StudyReport) -> Iterator[RenderedText]:
    """Every non-figure string the report renders, in document order.

    Deliberately exhaustive over the report model rather than over its
    Markdown: a number is unbacked because of how the report was *built*,
    and reading the emitted text back would also have to re-parse the
    provenance marks the emitters add.
    """
    yield RenderedText(kind="title", location="title", text=report.title)
    yield RenderedText(kind="dek", location="dek", text=report.dek)
    for index, note in enumerate(report.warnings):
        yield RenderedText(
            kind="warning", location=f"warnings[{index}]", text=note
        )
    for entry in report.colophon:
        yield RenderedText(kind="colophon", location="colophon", text=entry)
    for section in report.sections:
        yield from _text_in_section(section)


def _text_in_section(section: Section) -> Iterator[RenderedText]:
    where = section.tag or section.heading
    yield RenderedText(kind="heading", location=where, text=section.heading)
    for index, paragraph in enumerate(section.paragraphs):
        yield RenderedText(
            kind="paragraph",
            location=f"{where}.paragraphs[{index}]",
            text=paragraph,
        )
    for table_index, table in enumerate(section.tables):
        if table.caption:
            yield RenderedText(
                kind="caption",
                location=f"{where}.tables[{table_index}].caption",
                text=table.caption,
            )
        for header in table.headers:
            yield RenderedText(
                kind="header",
                location=f"{where}.tables[{table_index}].headers",
                text=header,
            )
        for row_index, row in enumerate(table.rows):
            for cell_index, cell in enumerate(row.cells):
                if cell.text is None:
                    continue
                yield RenderedText(
                    kind="cell",
                    location=(
                        f"{where}.tables[{table_index}]"
                        f".rows[{row_index}].cells[{cell_index}]"
                    ),
                    text=cell.text,
                )
    for index, item in enumerate(section.checklist):
        yield RenderedText(
            kind="checklist",
            location=f"{where}.checklist[{index}]",
            text=item,
        )
    for label, _body in section.code_blocks:
        yield RenderedText(
            kind="code_label", location=f"{where}.code_blocks", text=label
        )
    for panel in section.panels:
        yield from _text_in_section(panel)


def _figures_in_section(section: Section) -> Iterator[Figure]:
    for table in section.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.figure is not None:
                    yield cell.figure
    for panel in section.panels:
        yield from _figures_in_section(panel)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

_MANIFEST_SOURCE = f"{STUDY_MANIFEST_NAME}"


def _manifest_figure(value: str, field_path: str) -> Figure:
    """A figure whose value is a manifest field, named by its path."""
    return Figure(value=value, source=f"{_MANIFEST_SOURCE}:{field_path}")


def _pointer_figure(
    value: str, field_path: str, pointer: EvidencePointer
) -> Figure:
    """A figure whose value the manifest backs with a store pointer."""
    return Figure(
        value=value,
        source=f"{_MANIFEST_SOURCE}:{field_path}",
        pointer=pointer,
    )


def _proportion(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _interval(low: float, high: float) -> str:
    return f"[{low:+.4f}, {high:+.4f}]"


def _p_value(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 0.001:  # noqa: PLR2004
        return "<0.001"
    return f"{value:.3f}"


def _usd(value: float | None, unpriced: int, calls: int) -> str:
    """A role's USD total, or the honest statement of why there is none."""
    if value is not None:
        return f"${value:.4f}"
    return f"{UNPRICED} ({unpriced}/{calls})"


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

#: The three states an arm can be reported in. C1 gates C2 mechanically: an
#: arm whose fidelity audit failed is never a claim, whatever its interval
#: says, so the fidelity check comes first and the significance check cannot
#: overturn it.
VERDICT_VALIDATED = "validated"
VERDICT_NOT_VALIDATED = "not validated (fidelity)"
VERDICT_NO_IMPROVEMENT = "no detected improvement"
VERDICT_INCOMPLETE = "incomplete (not claimed)"
VERDICT_UNMEASURED = "not measured"
#: A leakage failure is not a property of one arm. It says the study's
#: measurement procedure did not hold, so every number it produced is
#: descriptive at best -- which is a stronger downgrade than a single arm's
#: failed fidelity audit, and a separate one.
VERDICT_INVALID = "invalid (leakage)"

_STATUS_BY_VERDICT = {
    VERDICT_VALIDATED: "ok",
    VERDICT_NOT_VALIDATED: "bad",
    VERDICT_NO_IMPROVEMENT: "warn",
    VERDICT_INCOMPLETE: "warn",
    VERDICT_UNMEASURED: "warn",
    VERDICT_INVALID: "bad",
}


def study_leakage_failed(manifest: StudyManifest) -> bool:
    """Whether this study's leakage rules did not establish a clean run.

    An **unrecorded** check counts as a failure, exactly as the CLI's
    ``leakage-check`` treats an unchecked rule: from the reader's side, a
    study whose leakage nobody verified and one whose leakage failed make
    the same claim, and the report must not present either as a result.
    """
    return manifest.leakage_check is None or not manifest.leakage_check.passed


def _arm_verdict(
    *,
    arm: ArmRecord,
    row: HeldOutRecord | None,
    backstop: float,
    leakage_failed: bool = False,
) -> str:
    """The arm's verdict: leakage first, then fidelity, then efficacy.

    The order is the gating order and no later check can overturn an
    earlier one. A failed or unrun leakage check invalidates the study's
    whole measurement procedure, so it outranks even a passing fidelity
    audit: an arm measured correctly against a contaminated split still
    reports a number nobody may claim. Fidelity comes next -- an arm with no
    runs, or with any run whose audit failed, is *not validated* -- and only
    then does the interval decide between an improvement and none.
    """
    if leakage_failed:
        return VERDICT_INVALID
    if not arm.runs or not all(run.audit_passed for run in arm.runs):
        return VERDICT_NOT_VALIDATED
    if row is None:
        return VERDICT_UNMEASURED
    if row.completeness < backstop:
        return VERDICT_INCOMPLETE
    if row.ci_low > 0.0:
        return VERDICT_VALIDATED
    return VERDICT_NO_IMPROVEMENT


def _held_out_by_name(
    manifest: StudyManifest,
) -> dict[str, HeldOutRecord]:
    return {row.candidate_name: row for row in manifest.held_out}


def _arm_held_out(
    manifest: StudyManifest, arm: ArmRecord
) -> HeldOutRecord | None:
    """The held-out row reported for ``arm``.

    Held-out rows are keyed by candidate name, and the study names an arm's
    representative candidate after the arm. A row that does not match is not
    guessed at: an arm with no matching row is reported unmeasured, which is
    what a missing held-out evaluation is.
    """
    return _held_out_by_name(manifest).get(arm.arm_id)


# --------------------------------------------------------------------------
# Evidence readers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunEvidence:
    """What the generator could read for one run, and what it could not.

    Every field is optional because every one of them is *outside* the
    manifest. A report that crashed when a run store was unavailable would
    be a report that cannot be regenerated from an archived study, so an
    unreadable artifact is a recorded absence rather than a failure.
    """

    run_id: str
    audit: JsonValue | None = None
    audit_detail: str = ""
    trajectory: TrajectoryReport | None = None
    trajectory_detail: str = ""


def _read_audit(
    run: RunRecord, store: EvidenceStore | None
) -> tuple[JsonValue | None, str]:
    if store is None:
        return None, "no evidence store was supplied"
    try:
        return store.get(run.audit_ref.as_object_reference()), ""
    except Exception as error:  # noqa: BLE001
        # Every store backend signals a missing or corrupt record with its
        # own exception type. The report's job is to say the pointer did not
        # resolve, not to propagate a backend's failure mode.
        return None, f"{type(error).__name__}: {error}"


def _read_trajectory(
    run: RunRecord,
) -> tuple[TrajectoryReport | None, str]:
    directory = Path(run.artifact_dir)
    path = directory / TRAJECTORY_REPORT_NAME
    if not path.is_file():
        return None, f"no {TRAJECTORY_REPORT_NAME} in {run.artifact_dir}"
    try:
        return load_trajectory_report(path), ""
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


def read_run_evidence(
    run: RunRecord, *, store: EvidenceStore | None
) -> RunEvidence:
    """Resolve one run's audit findings and internal trajectory."""
    audit, audit_detail = _read_audit(run, store)
    trajectory, trajectory_detail = _read_trajectory(run)
    return RunEvidence(
        run_id=run.run_id,
        audit=audit,
        audit_detail=audit_detail,
        trajectory=trajectory,
        trajectory_detail=trajectory_detail,
    )


#: Where an ``audit.json`` keeps its findings and its verdict. These are the
#: audit package's wire keys, read rather than owned here: this module is a
#: consumer of that format, and reading a key it does not own is why an
#: unrecognised document is reported as unreadable rather than mis-rendered.
AUDIT_FINDINGS_KEY = "findings"
AUDIT_INVARIANT_KEY = "invariant_id"
AUDIT_STATUS_KEY = "status"
AUDIT_DETAIL_KEY = "detail"
AUDIT_EVIDENCE_KEY = "evidence_refs"


def _audit_findings(audit: JsonValue | None) -> tuple[dict[str, str], ...]:
    """The finding rows an ``audit.json`` holds, or an empty tuple.

    A document whose shape this does not recognise yields nothing, and the
    caller renders "not resolved" -- the same outcome as a missing pointer,
    because from the reader's side an unreadable audit and an absent one
    make the same claim.
    """
    if not isinstance(audit, dict):
        return ()
    findings = audit.get(AUDIT_FINDINGS_KEY)
    if not isinstance(findings, list):
        return ()
    rows: list[dict[str, str]] = []
    for entry in findings:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "invariant": str(entry.get(AUDIT_INVARIANT_KEY, MISSING)),
                "status": str(entry.get(AUDIT_STATUS_KEY, MISSING)),
                "detail": str(entry.get(AUDIT_DETAIL_KEY, "")),
                "evidence": _audit_evidence(entry.get(AUDIT_EVIDENCE_KEY)),
            }
        )
    return tuple(rows)


def _audit_evidence(value: JsonValue) -> str:
    """The finding's evidence refs, rendered as one compact mark."""
    if not isinstance(value, list):
        return ""
    marks: list[str] = []
    for entry in value:
        if isinstance(entry, dict):
            schema = entry.get("schema_name") or entry.get("schema")
            digest = entry.get("content_hash")
            if isinstance(schema, str) and isinstance(digest, str):
                marks.append(f"{schema}@{digest[:12]}")
    return ", ".join(marks)


# --------------------------------------------------------------------------
# Trajectory and prompt samples
# --------------------------------------------------------------------------


def _best_so_far(
    trajectory: TrajectoryReport,
) -> tuple[tuple[int, float], ...]:
    """The best internal reward seen through each step, by step index.

    "Best so far" is a running maximum over the rewards the run's own
    resolutions recorded, which is what the optimizer was selecting on. It
    is a projection of already-persisted rewards, never a rescoring: no
    reward is computed here, only carried forward.
    """
    best: float | None = None
    points: list[tuple[int, float]] = []
    by_step: dict[int, float] = {}
    for row in trajectory.resolutions:
        if row.reward is None:
            continue
        current = by_step.get(row.step_index)
        if current is None or row.reward > current:
            by_step[row.step_index] = row.reward
    for step_index in sorted(by_step):
        reward = by_step[step_index]
        best = reward if best is None else max(best, reward)
        points.append((step_index, best))
    return tuple(points)


#: How many proposed prompts a per-optimizer section samples. The
#: assignment asks for first, best, and last; three is that, and a sample is
#: a sample rather than the run's whole proposal set.
PROMPT_SAMPLE_SIZE = 3


def _prompt_samples(
    trajectory: TrajectoryReport,
) -> tuple[tuple[str, str], ...]:
    """First, best, and last proposed prompt, labelled.

    "Best" is the candidate carrying the highest recorded internal reward,
    which is the same running maximum the trajectory chart plots. Duplicate
    selections collapse -- a two-candidate run whose best is also its last
    reports two samples, not three with a repeat.
    """
    candidates = trajectory.candidates
    if not candidates:
        return ()
    reward_by_ref: dict[str, float] = {}
    for row in trajectory.resolutions:
        if row.reward is None:
            continue
        key = row.candidate_ref.content_hash
        current = reward_by_ref.get(key)
        if current is None or row.reward > current:
            reward_by_ref[key] = row.reward
    best = max(
        candidates,
        key=lambda candidate: reward_by_ref.get(
            candidate.record_ref.content_hash, float("-inf")
        ),
    )
    picks = (
        ("first proposed", candidates[0]),
        ("best on internal", best),
        ("last proposed", candidates[-1]),
    )
    seen: set[str] = set()
    samples: list[tuple[str, str]] = []
    for label, candidate in picks:
        key = candidate.record_ref.content_hash
        if key in seen:
            continue
        seen.add(key)
        samples.append((label, candidate.mutation_text))
    return tuple(samples)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------


def _verdict_section(manifest: StudyManifest) -> Section:
    backstop = (
        manifest.design.completeness_backstop
        if manifest.design is not None
        else 1.0
    )
    leakage_failed = study_leakage_failed(manifest)
    rows: list[Row] = []
    for index, arm in enumerate(manifest.arms):
        row = _arm_held_out(manifest, arm)
        verdict = _arm_verdict(
            arm=arm,
            row=row,
            backstop=backstop,
            leakage_failed=leakage_failed,
        )
        path = f"held_out[{arm.arm_id}]"
        rows.append(
            Row(
                cells=(
                    Cell(text=arm.arm_id),
                    Cell(
                        text=(
                            "pass"
                            if arm.runs
                            and all(run.audit_passed for run in arm.runs)
                            else "fail"
                        ),
                        status=(
                            "ok"
                            if arm.runs
                            and all(run.audit_passed for run in arm.runs)
                            else "bad"
                        ),
                    ),
                    _delta_cell(row, path, index=index, arms=manifest.arms),
                    _ci_cell(row, path),
                    _p_cell(row, path, corrected=False),
                    _p_cell(row, path, corrected=True),
                    _spend_cell(arm, index=index),
                    Cell(
                        text=verdict,
                        status=_STATUS_BY_VERDICT[verdict],
                    ),
                )
            )
        )
    return Section(
        heading="Which optimizers improved held-out accuracy?",
        tag="verdict",
        paragraphs=_verdict_paragraphs(leakage_failed=leakage_failed),
        tables=(
            Table(
                headers=(
                    "arm",
                    "fidelity",
                    "delta vs naive",
                    "95% CI",
                    "p (uncorrected)",
                    "p (Holm)",
                    "spend",
                    "verdict",
                ),
                rows=tuple(rows),
                caption=(
                    "One row per arm. Nulls are controls, so their Holm "
                    "column is empty by design rather than by omission."
                ),
            ),
        ),
    )


def _verdict_paragraphs(*, leakage_failed: bool) -> tuple[str, ...]:
    """What the verdict column means, given whether leakage held.

    Stated conditionally rather than as fixed prose: a table whose every
    row reads *invalid (leakage)* beside a paragraph describing three
    efficacy states would misdescribe its own contents.
    """
    if leakage_failed:
        return (
            (
                "**Leakage gates everything.** This study's leakage rules "
                "did not establish a clean separation between the split its "
                "optimizers saw and the split it reports from, so every arm "
                "is reported **invalid (leakage)** and no number below is a "
                "claim -- whatever its interval or its fidelity audit says."
            ),
            (
                "The deltas and intervals are still printed, because "
                "withholding them would hide what was measured. They "
                "describe a run whose measurement procedure is not "
                "established, and nothing more."
            ),
        )
    return (
        (
            "Fidelity gates efficacy. An arm whose audit failed is "
            "reported *not validated* and its held-out number is "
            "descriptive only, never a claim, whatever its interval says."
        ),
        (
            "Three states only: **validated** (audit passed and the "
            "uncorrected 95% interval excludes zero), **not validated "
            "(fidelity)** (an audit failed), and **no detected "
            "improvement** (audit passed, interval includes zero)."
        ),
    )


def _delta_cell(
    row: HeldOutRecord | None,
    path: str,
    *,
    index: int,
    arms: Sequence[ArmRecord],
) -> Cell:
    del index, arms
    if row is None:
        return Cell(text=MISSING)
    return Cell(
        figure=_pointer_figure(
            _signed(row.delta_vs_naive),
            f"{path}.delta_vs_naive",
            row.per_task_scores_ref,
        )
    )


def _ci_cell(row: HeldOutRecord | None, path: str) -> Cell:
    if row is None:
        return Cell(text=MISSING)
    return Cell(
        figure=_pointer_figure(
            _interval(row.ci_low, row.ci_high),
            f"{path}.ci",
            row.per_task_scores_ref,
        )
    )


def _p_cell(row: HeldOutRecord | None, path: str, *, corrected: bool) -> Cell:
    if row is None:
        return Cell(text=MISSING)
    value = row.p_holm if corrected else row.p_bootstrap
    if corrected and value is None:
        return Cell(text="uncorrected control")
    return Cell(
        figure=_pointer_figure(
            _p_value(value),
            f"{path}.{'p_holm' if corrected else 'p_bootstrap'}",
            row.per_task_scores_ref,
        )
    )


def _spend_cell(arm: ArmRecord, *, index: int) -> Cell:
    """The arm's total task-model calls across its runs.

    A count, not a total in dollars: USD is per role and may be absent, and
    a verdict table that showed a dollar total would have to show an absent
    one as blank. Calls are always known, and the per-optimizer cost table
    carries the honest USD statement.
    """
    total = sum(entry.calls for run in arm.runs for entry in run.spend)
    return Cell(
        figure=_manifest_figure(
            f"{total:,} calls", f"arms[{index}].runs[*].spend[*].calls"
        )
    )


def _design_section(manifest: StudyManifest) -> Section:
    splits = manifest.splits
    design = manifest.design
    rows: list[Row] = [
        Row(
            cells=(
                Cell(text="internal split"),
                Cell(
                    figure=_manifest_figure(
                        f"{splits.internal.size}", "splits.internal.size"
                    )
                ),
                Cell(
                    text="what the optimizer sees; never official or held-out"
                ),
            )
        ),
        Row(
            cells=(
                Cell(text="official split"),
                Cell(
                    figure=_manifest_figure(
                        f"{splits.official.size}", "splits.official.size"
                    )
                ),
                Cell(text="selection only: one arg-max per arm"),
            )
        ),
        Row(
            cells=(
                Cell(text="held-out split"),
                Cell(
                    figure=_manifest_figure(
                        f"{splits.held_out.size}", "splits.held_out.size"
                    )
                ),
                Cell(text="one evaluation per reported candidate, ever"),
            )
        ),
    ]
    if design is not None:
        rows.extend(
            (
                Row(
                    cells=(
                        Cell(text="K_CAL (calibration repeats)"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.k_cal}", "design.k_cal"
                            )
                        ),
                        Cell(
                            text="a Stage-0 measurement input, not the "
                            "design's repeat count"
                        ),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="K_REPEAT (per-task repeats)"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.k_repeat}", "design.k_repeat"
                            )
                        ),
                        Cell(
                            text="the design's repeat count on every "
                            "reported evaluation"
                        ),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="measured MDE"),
                        Cell(
                            figure=_manifest_figure(
                                _proportion(design.mde_measured),
                                "design.mde_measured",
                            )
                        ),
                        Cell(text=design.mde_formula),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="tau^2 (between-task variance)"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.tau_sq:.6f}", "design.tau_sq"
                            )
                        ),
                        Cell(text="from Stage 0's variance decomposition"),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="sigma^2 (within-task variance)"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.sigma_sq:.6f}", "design.sigma_sq"
                            )
                        ),
                        Cell(text="estimated from the naive arm; see threats"),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="bootstrap resamples"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.resamples:,}", "design.resamples"
                            )
                        ),
                        Cell(
                            text=(
                                f"paired task-level percentile bootstrap at "
                                f"seed {design.bootstrap_seed}"
                            )
                        ),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="multiplicity correction"),
                        Cell(
                            figure=_manifest_figure(
                                f"{design.correction} (m={design.m})",
                                "design.correction",
                            )
                        ),
                        Cell(
                            text="over the real optimizers only; nulls are "
                            "controls"
                        ),
                    )
                ),
                Row(
                    cells=(
                        Cell(text="completeness backstop"),
                        Cell(
                            figure=_manifest_figure(
                                _proportion(design.completeness_backstop),
                                "design.completeness_backstop",
                            )
                        ),
                        Cell(text=design.completeness_rule),
                    )
                ),
            )
        )
    pre_registration = manifest.pre_registration
    if pre_registration is not None:
        rows.append(
            Row(
                cells=(
                    Cell(text="pre-registration hash"),
                    Cell(
                        figure=_manifest_figure(
                            pre_registration.design_hash[:12],
                            "pre_registration.design_hash",
                        )
                    ),
                    Cell(text="over the design fields fixed before any spend"),
                )
            )
        )
        # An amendment is the one thing about a pre-registration a reader
        # must not have to infer: a design that changed after Stage 0 is a
        # different design, and the hash it replaced is what makes that
        # checkable rather than asserted.
        amended_from = pre_registration.amended_from
        rows.append(
            Row(
                cells=(
                    Cell(text="pre-registration provenance"),
                    Cell(
                        figure=_manifest_figure(
                            pre_registration.provenance,
                            "pre_registration.provenance",
                        )
                    ),
                    Cell(
                        text=(
                            "amended after Stage 0; the design below is not "
                            "the one first registered"
                            if amended_from is not None
                            else "the design registered before any spend"
                        )
                    ),
                )
            )
        )
        if amended_from is not None:
            rows.append(
                Row(
                    cells=(
                        Cell(text="amended from"),
                        Cell(
                            figure=_manifest_figure(
                                amended_from[:12],
                                "pre_registration.amended_from",
                            )
                        ),
                        Cell(text="the pre-registration this one replaced"),
                    )
                )
            )
    models = manifest.models
    model_rows = (
        Row(
            cells=(
                Cell(text="task model"),
                Cell(text=models.task_model),
            )
        ),
        Row(
            cells=(
                Cell(text="proposer model"),
                Cell(text=models.proposer_model),
            )
        ),
        Row(
            cells=(
                Cell(text="temperature"),
                Cell(text=models.temperature),
            )
        ),
        Row(
            cells=(
                Cell(text="seed control"),
                Cell(text=models.seed_control),
            )
        ),
        Row(
            cells=(
                Cell(text="Codex agent model"),
                Cell(text=models.codex_agent_model),
            )
        ),
        # What each transport actually bound, which is a different fact
        # from which model the study meant to run: the route it resolved
        # and the request controls it did or did not set are what explain
        # the bill and what "the same experiment" means.
        *(
            Row(
                cells=(
                    Cell(text=f"provider call ({entry.transport})"),
                    Cell(
                        figure=_manifest_figure(
                            _provider_call_text(entry),
                            "models.provider_calls",
                        )
                    ),
                )
            )
            for entry in models.provider_calls
        ),
    )
    paragraphs = [
        (
            "Every number below is a field of the study manifest, and the "
            "report prints the manifest path it came from beside it."
        ),
    ]
    if design is not None:
        paragraphs.append(
            PERCENTILE_P_FLOOR_CAVEAT.format(resamples=design.resamples)
        )
    else:
        paragraphs.append(
            "No design is recorded, so this study has not run Stage 0 and "
            "no interval below is powered against a measured MDE."
        )
    return Section(
        heading="What design produced these numbers?",
        tag="design",
        paragraphs=tuple(paragraphs),
        tables=(
            Table(
                headers=("quantity", "value", "what it means"),
                rows=tuple(rows),
                caption="Splits, repeats, and the power arithmetic.",
            ),
            Table(
                headers=("model setting", "value"),
                rows=model_rows,
                caption=(
                    "Honest strings where the study does not control the "
                    "setting: an uncontrolled value is named, not omitted."
                ),
            ),
        ),
    )


def _leakage_section(manifest: StudyManifest) -> Section:
    check = manifest.leakage_check
    if check is None:
        return Section(
            heading="Did the pre-registered leakage rules hold?",
            tag="leakage",
            paragraphs=(
                (
                    "No leakage check is recorded. The study's rules L1-L6 "
                    "were not run over this manifest, so nothing here "
                    "establishes that the optimizer never saw official or "
                    "held-out data."
                ),
            ),
        )
    rows = tuple(
        Row(
            cells=(
                Cell(text=entry.check_id),
                Cell(
                    text="pass" if entry.passed else "FAIL",
                    status="ok" if entry.passed else "bad",
                ),
                Cell(text=entry.detail),
            )
        )
        for entry in check.checks
    )
    verdict = (
        "Every rule passed."
        if check.passed
        else "At least one rule failed; this study must not claim."
    )
    return Section(
        heading="Did the pre-registered leakage rules hold?",
        tag="leakage",
        paragraphs=(
            (
                f"L2 and L3 are prevented structurally rather than merely "
                f"detected: the selection is persisted before any held-out "
                f"call is issued. {verdict}"
            ),
        ),
        tables=(
            Table(
                headers=("rule", "verdict", "detail"),
                rows=rows,
                caption="The mechanical run of L1-L5, rolled up by L6.",
            ),
        ),
    )


def _stage_history_section(manifest: StudyManifest) -> Section:
    """What each stage recorded, and which gates it passed.

    The manifest does not carry a stage log, so the history is read from
    what each stage *writes*: Stage 0 writes the design, Wave 3 writes the
    GEPA sizing and the fan-out check, and Stages 1 and 2 write arms, runs,
    selections, and held-out rows. Naming the writer of each fact is what
    turns those presences into a history rather than a checklist.
    """
    design = manifest.design
    sizing = manifest.gepa_sizing
    fanout = manifest.fanout_check
    run_count = sum(len(arm.runs) for arm in manifest.arms)
    rows = (
        Row(
            cells=(
                Cell(text="Stage 0 -- anchor calibration"),
                _presence_cell(design is not None),
                Cell(
                    text=(
                        "records the design, the measured MDE, and the "
                        "variance decomposition"
                    )
                ),
            )
        ),
        *_transport_rows(manifest),
        Row(
            cells=(
                Cell(text="Stage-1 precondition -- GEPA sizing (F9)"),
                _presence_cell(sizing is not None),
                (
                    Cell(
                        text=(
                            "no step, wall-time, or store measurement recorded"
                        )
                    )
                    if sizing is None
                    else Cell(
                        figure=_manifest_figure(
                            _sizing_detail(manifest), "gepa_sizing"
                        )
                    )
                ),
            )
        ),
        Row(
            cells=(
                Cell(text="Stage-1 precondition -- fan-out check (F16)"),
                _presence_cell(fanout is not None),
                (
                    Cell(
                        text=("no minibatch fan-out measurement recorded"),
                    )
                    if fanout is None
                    else Cell(
                        figure=_manifest_figure(
                            f"{fanout.minibatch_intents} minibatch intents, "
                            f"{fanout.full_valset_intents} full-valset "
                            f"intents; "
                            f"{'passed' if fanout.passed else 'FAILED'}",
                            "fanout_check",
                        ),
                        status="ok" if fanout.passed else "bad",
                    )
                ),
            )
        ),
        Row(
            cells=(
                Cell(text="Stages 1-2 -- optimizer runs"),
                _presence_cell(run_count > 0),
                Cell(
                    figure=_manifest_figure(
                        f"{run_count} runs across {len(manifest.arms)} arms",
                        "arms[].runs",
                    )
                ),
            )
        ),
        Row(
            cells=(
                Cell(text="Selection -- arg-max on official"),
                _presence_cell(bool(manifest.selection)),
                Cell(
                    figure=_manifest_figure(
                        f"{_reported_selection_count(manifest)} of "
                        f"{len(manifest.arms)} arms selected",
                        "selection",
                    )
                ),
            )
        ),
        Row(
            cells=(
                Cell(text="Held-out -- one evaluation per candidate"),
                _presence_cell(bool(manifest.held_out)),
                Cell(
                    figure=_manifest_figure(
                        f"{len(manifest.held_out_claims)} claimed, "
                        f"{len(manifest.held_out)} reported",
                        "held_out_claims",
                    )
                ),
            )
        ),
    )
    return Section(
        heading="How far did the study get, and which gates did it pass?",
        tag="stages",
        paragraphs=(
            (
                "The manifest records no stage log, so this history is read "
                "from what each stage writes into it. A stage that did not "
                "write its record did not reach its gate."
            ),
            (
                "Each stage that ran also records the transport it ran on "
                "and what it spent. The transport is the difference between "
                "plumbing evidence and a study result: a stage on the fake "
                "transport answers from the experiment's own gold and "
                "measures nothing about a model. Every stage after Stage 0 "
                "is refused unless it runs on the transport the anchors "
                "were calibrated on, because every held-out delta is paired "
                "against those anchors."
            ),
            (STAGE_SPEND_COVERAGE_NOTE),
        ),
        tables=(
            Table(
                headers=("stage or gate", "recorded", "what it recorded"),
                rows=rows,
            ),
        ),
    )


def _transport_rows(manifest: StudyManifest) -> tuple[Row, ...]:
    """One row per stage that ran, naming its transport and its spend.

    Ordered by the stage sequence rather than by the manifest's storage
    order, so a study whose Stage 0 was re-run after Stage 1 still reads in
    the order the stages happen.
    """
    by_stage = {entry.stage: entry for entry in manifest.stages}
    rows: list[Row] = []
    for stage_id in STAGE_IDS:
        record = by_stage.get(stage_id)
        if record is None:
            continue
        rows.append(
            Row(
                cells=(
                    Cell(
                        text=(
                            f"{_stage_label(record.stage)} -- transport "
                            "and spend"
                        )
                    ),
                    _presence_cell(present=True),
                    Cell(
                        figure=_manifest_figure(
                            f"ran on {record.transport}; "
                            f"{_stage_spend_detail(record)}",
                            f"stages[{record.stage}]",
                        )
                    ),
                )
            )
        )
    return tuple(rows)


def _stage_label(stage: str) -> str:
    """The prose spelling of a stage id: ``stage0`` reads "Stage 0".

    The manifest stores the wire spelling and the report renders the one a
    reader reads. Keeping them distinct is also what keeps the stage name
    out of the rendered-number guard: ``stage0`` is a label whose digit
    means nothing, and a label indistinguishable from a measurement is
    precisely what the guard exists to catch.
    """
    return f"Stage {stage.removeprefix('stage')}"


#: What the report says the per-stage ledger covers.
#:
#: A stage spends by two routes and the row is the sum of both: its arms'
#: optimizer runs, each of which projected its own bill, and the reporting
#: pass -- official-selection scoring, the held-out evaluations, and the
#: anchors' re-measurement -- which reaches the provider through the
#: evaluation engine outside any run and is priced from its own persisted
#: rows. Saying so is what stops a reader from taking the run-side number
#: for the whole bill, which is what it used to be.
STAGE_SPEND_COVERAGE_NOTE = (
    "The per-stage spend below is the whole of what each stage bought: "
    "its arms' optimizer runs, plus the reporting pass -- "
    "official-selection scoring, the held-out evaluations, and the "
    "anchors' re-measurement -- folded onto the same row. Both routes are "
    "priced from the persisted output rows rather than accumulated while "
    "the stage ran."
)

#: What a paid stage that recorded no spend reports. Never the
#: fake-transport wording: this stage did reach a provider, and its bill is
#: unknown rather than absent.
UNLEDGERED_STAGE_DETAIL = (
    "UNLEDGERED -- ran on a paid transport and recorded no spend; this "
    "stage's bill is unknown, not zero"
)

#: What a fake-transport stage with no spend reports.
NO_PROVIDER_STAGE_DETAIL = "no provider reached (fake transport)"


def _stage_spend_detail(record: StageRecord) -> str:
    """What one stage spent, or why there is nothing to report.

    An empty ``spend`` tuple means one of two opposite things, and the
    transport is what tells them apart. A fake-transport stage reached no
    provider, so there is no bill. A **paid** stage that recorded nothing
    called a provider and lost track of what it bought -- reporting that as
    "reached no provider" would describe a fully billed stage as a free
    one, which is the reading this branch exists to prevent.
    """
    if not record.spend:
        if record.transport == TransportName.FAKE.value:
            return NO_PROVIDER_STAGE_DETAIL
        return UNLEDGERED_STAGE_DETAIL
    calls = sum(entry.calls for entry in record.spend)
    tokens = sum(
        entry.input_tokens + entry.output_tokens for entry in record.spend
    )
    unpriced = sum(entry.unpriced_calls for entry in record.spend)
    total = (
        None
        if any(entry.usd is None for entry in record.spend)
        else sum(entry.usd or 0.0 for entry in record.spend)
    )
    return (
        f"{calls:,} calls, {tokens:,} tokens, {_usd(total, unpriced, calls)}"
    )


def _reported_selection_count(manifest: StudyManifest) -> int:
    """How many arms the study's *reported* stage selected.

    Selections are recorded once per arm per stage, so counting every entry
    would report a study that ran a pilot and a full design as having
    selected twice as many arms as it has. The reported stage is the latest
    one that ran, which is the stage the held-out rows describe.
    """
    stages = {entry.stage for entry in manifest.selection}
    for candidate in (StageId.STAGE2.value, StageId.STAGE1.value):
        if candidate in stages:
            return sum(
                1 for entry in manifest.selection if entry.stage == candidate
            )
    return len(manifest.selection)


def _sizing_detail(manifest: StudyManifest) -> str:
    sizing = manifest.gepa_sizing
    if sizing is None:  # pragma: no cover - guarded by the caller
        return MISSING
    pinned = (
        f"max_metric_calls pinned to {sizing.max_metric_calls_pinned}"
        if sizing.max_metric_calls_pinned is not None
        else "no metric-call ceiling was forced"
    )
    return (
        f"{sizing.steps_per_run} steps, {sizing.wall_seconds:.1f}s wall, "
        f"{sizing.sqlite_bytes:,} store bytes; {pinned}"
    )


def _presence_cell(present: bool) -> Cell:  # noqa: FBT001
    return Cell(
        text="yes" if present else "no",
        status="ok" if present else "warn",
    )


def _optimizer_section(
    manifest: StudyManifest,
    arm: ArmRecord,
    *,
    index: int,
    evidence: Sequence[RunEvidence],
) -> Section:
    """One arm's full section, or a shorter one for a null control."""
    is_null = arm.arm_id.startswith(NULL_ARM_PREFIX)
    tables: list[Table] = [
        _held_out_table(manifest, arm),
        _cost_table(arm, index=index),
    ]
    paragraphs: list[str] = []
    if is_null:
        paragraphs.append(
            "A control, not a hypothesis: it is uncorrected, and its job is "
            "to say what this pipeline reports when nothing was optimized."
        )
    if not is_null:
        tables.append(_trajectory_table(arm, evidence))
    tables.append(_audit_table(arm, evidence))
    code_blocks = () if is_null else _prompt_blocks(evidence)
    if not is_null and not code_blocks:
        paragraphs.append(
            "No proposed prompts could be read: no run in this arm has a "
            f"projected {TRAJECTORY_REPORT_NAME} beside its artifacts."
        )
    return Section(
        heading=f"What did {arm.arm_id} do?",
        tag="null control" if is_null else "optimizer",
        level=3,
        paragraphs=tuple(paragraphs),
        tables=tuple(tables),
        code_blocks=code_blocks,
    )


#: The candidate name the naive anchor is recorded under, spelled here
#: rather than imported: this module reads persisted manifests and must not
#: pull in the optimizer stack to render one. Pinned equal to
#: ``optim.study.analysis.NAIVE_CANDIDATE_NAME`` by the reporting tests,
#: which is what keeps the two spellings from drifting apart.
NAIVE_CANDIDATE_NAME = "naive"


def _anchor_completeness_cell(
    manifest: StudyManifest, held: HeldOutRecord
) -> Cell:
    """The naive anchor's completeness, cited to the anchor's own vector.

    The figure points at the *anchor's* ``per_task_scores_ref`` rather
    than this row's: the number describes the anchor's evaluation, and
    citing the candidate's vector for it would attach the wrong evidence
    to a real measurement. Absent an anchor row -- or on the anchor's own
    row, which has nothing to compare against -- there is nothing to cite
    and the cell says so.
    """
    anchor = _held_out_by_name(manifest).get(NAIVE_CANDIDATE_NAME)
    if held.anchor_completeness is None or anchor is None:
        return Cell(text=MISSING)
    return Cell(
        figure=_pointer_figure(
            _proportion(held.anchor_completeness),
            f"held_out[{NAIVE_CANDIDATE_NAME}].completeness",
            anchor.per_task_scores_ref,
        )
    )


def _held_out_table(manifest: StudyManifest, arm: ArmRecord) -> Table:
    """The arm's held-out number against the anchors and the nulls.

    Every comparator is another held-out row of the same manifest, measured
    under the identical procedure, which is what makes the comparison
    paired rather than merely adjacent.
    """
    rows: list[Row] = []
    comparators = [arm.arm_id]
    comparators.extend(
        name for name in _held_out_by_name(manifest) if name != arm.arm_id
    )
    for name in comparators:
        held = _held_out_by_name(manifest).get(name)
        if held is None:
            continue
        path = f"held_out[{name}]"
        rows.append(
            Row(
                cells=(
                    Cell(
                        text=(
                            f"{name} (this arm)"
                            if name == arm.arm_id
                            else name
                        )
                    ),
                    Cell(
                        figure=_pointer_figure(
                            _proportion(held.mean),
                            f"{path}.mean",
                            held.eval_evidence_ref,
                        )
                    ),
                    _ci_cell(held, path),
                    _delta_cell(held, path, index=0, arms=()),
                    Cell(
                        figure=_pointer_figure(
                            _proportion(held.completeness),
                            f"{path}.completeness",
                            held.per_task_scores_ref,
                        )
                    ),
                    # The paired completeness above can be low because the
                    # anchor lost rows rather than this candidate, so the
                    # anchor's own side is shown beside it -- pointed at
                    # the anchor's own per-task vector, because it is a
                    # measurement like any other and needs the same
                    # provenance mark rather than being printed as prose.
                    _anchor_completeness_cell(manifest, held),
                )
            )
        )
    if not rows:
        rows.append(
            Row(
                cells=(
                    Cell(text=arm.arm_id),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                )
            )
        )
    return Table(
        headers=(
            "candidate",
            "held-out mean",
            "95% CI on delta",
            "delta vs naive",
            "completeness (paired)",
            "anchor completeness",
        ),
        rows=tuple(rows),
        caption=(
            "This arm beside every anchor and null the study measured on "
            "held-out under the identical procedure."
        ),
    )


def _cost_table(arm: ArmRecord, *, index: int) -> Table:
    """Per-run, per-role spend, with the honest unpriced statement.

    Wall time has no field in the manifest and none in ``cost.json``, so it
    is reported as unrecorded rather than estimated from anything else.
    """
    rows: list[Row] = []
    for run_index, run in enumerate(arm.runs):
        path = f"arms[{index}].runs[{run_index}]"
        for entry in run.spend:
            rows.append(  # noqa: PERF401
                Row(
                    cells=(
                        Cell(text=run.run_id),
                        Cell(text=entry.role),
                        Cell(
                            figure=_pointer_figure(
                                f"{entry.calls:,}",
                                f"{path}.spend[{entry.role}].calls",
                                run.cost_ref,
                            )
                        ),
                        Cell(
                            figure=_pointer_figure(
                                f"{entry.cached_calls:,}",
                                f"{path}.spend[{entry.role}].cached_calls",
                                run.cost_ref,
                            )
                        ),
                        Cell(
                            figure=_pointer_figure(
                                f"{entry.input_tokens:,} in / "
                                f"{entry.output_tokens:,} out",
                                f"{path}.spend[{entry.role}].tokens",
                                run.cost_ref,
                            )
                        ),
                        Cell(
                            figure=_pointer_figure(
                                _usd(
                                    entry.usd,
                                    entry.unpriced_calls,
                                    entry.calls,
                                ),
                                f"{path}.spend[{entry.role}].usd",
                                run.cost_ref,
                            )
                        ),
                        Cell(text=MISSING),
                    )
                )
            )
    if not rows:
        rows.append(
            Row(
                cells=(
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                )
            )
        )
    return Table(
        headers=(
            "run",
            "role",
            "calls",
            "cached",
            "tokens",
            "USD",
            "wall time",
        ),
        rows=tuple(rows),
        caption=(
            "An absent USD renders as 'unpriced (n/total)', never as zero: "
            "a role with unpriced calls has a total nobody can compute. "
            "Wall time is unrecorded because no manifest field or cost "
            "document carries a duration."
        ),
    )


def _trajectory_table(
    arm: ArmRecord, evidence: Sequence[RunEvidence]
) -> Table:
    """Best-so-far internal reward by step, per run.

    A running maximum over rewards the run already persisted. Nothing is
    rescored; a run whose trajectory could not be read is a row saying so.
    """
    by_run = {item.run_id: item for item in evidence}
    rows: list[Row] = []
    for run in arm.runs:
        item = by_run.get(run.run_id)
        if item is None or item.trajectory is None:
            detail = (
                item.trajectory_detail
                if item is not None
                else "no evidence was read for this run"
            )
            rows.append(
                Row(
                    cells=(
                        Cell(text=run.run_id),
                        Cell(text=MISSING),
                        Cell(text=detail),
                    )
                )
            )
            continue
        points = _best_so_far(item.trajectory)
        if not points:
            rows.append(
                Row(
                    cells=(
                        Cell(text=run.run_id),
                        Cell(text=MISSING),
                        Cell(
                            text=(
                                "the run's trajectory records no internal "
                                "reward"
                            )
                        ),
                    )
                )
            )
            continue
        rows.append(
            Row(
                cells=(
                    Cell(text=run.run_id),
                    Cell(
                        figure=_pointer_figure(
                            " -> ".join(
                                f"{step}:{value:.3f}" for step, value in points
                            ),
                            f"runs[{run.run_id}].trajectory.best_so_far",
                            run.result_ref,
                        )
                    ),
                    Cell(
                        figure=_pointer_figure(
                            f"{len(points)} steps; terminal "
                            f"{item.trajectory.terminal_status}",
                            f"runs[{run.run_id}].trajectory.steps",
                            run.result_ref,
                        )
                    ),
                )
            )
        )
    return Table(
        headers=("run", "best-so-far internal reward by step", "detail"),
        rows=tuple(rows),
        caption=(
            "A running maximum over rewards the run persisted, read from "
            f"its {TRAJECTORY_REPORT_NAME}. No score is recomputed here."
        ),
    )


def _audit_table(arm: ArmRecord, evidence: Sequence[RunEvidence]) -> Table:
    """The trace-audit findings behind each run's fidelity verdict."""
    by_run = {item.run_id: item for item in evidence}
    rows: list[Row] = []
    for run in arm.runs:
        item = by_run.get(run.run_id)
        findings = _audit_findings(item.audit if item is not None else None)
        if not findings:
            detail = (
                item.audit_detail
                if item is not None and item.audit_detail
                else "the audit document did not resolve to findings"
            )
            rows.append(
                Row(
                    cells=(
                        Cell(text=run.run_id),
                        Cell(text=MISSING),
                        Cell(
                            text=("pass" if run.audit_passed else "FAIL"),
                            status="ok" if run.audit_passed else "bad",
                        ),
                        Cell(text=detail),
                        Cell(
                            text=(
                                f"{run.audit_ref.schema_name}"
                                f"@{run.audit_ref.content_hash[:12]}"
                            )
                        ),
                    )
                )
            )
            continue
        for finding in findings:
            status = finding["status"].upper()
            rows.append(
                Row(
                    cells=(
                        Cell(text=run.run_id),
                        Cell(text=finding["invariant"]),
                        Cell(
                            text=status,
                            status=(
                                "bad"
                                if status == "FAIL"
                                else "ok"
                                if status == "PASS"
                                else "warn"
                            ),
                        ),
                        Cell(text=finding["detail"] or ""),
                        Cell(text=finding["evidence"] or MISSING),
                    )
                )
            )
    if not rows:
        rows.append(
            Row(
                cells=(
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                    Cell(text=MISSING),
                )
            )
        )
    return Table(
        headers=("run", "invariant", "status", "detail", "evidence"),
        rows=tuple(rows),
        caption=(
            "Read from each run's audit document through its manifest "
            "pointer. A run whose findings did not resolve shows the "
            "recorded verdict and says the table was not resolved."
        ),
    )


def _prompt_blocks(
    evidence: Sequence[RunEvidence],
) -> tuple[tuple[str, str], ...]:
    """First, best, and last proposed prompt from the arm's first readable
    run."""
    for item in evidence:
        if item.trajectory is None:
            continue
        samples = _prompt_samples(item.trajectory)
        if samples:
            return tuple(
                (f"{item.run_id} -- {label}", text)
                for label, text in samples[:PROMPT_SAMPLE_SIZE]
            )
    return ()


def _c18_section(manifest: StudyManifest) -> Section:
    record = manifest.c18
    if record is None:
        return Section(
            heading="Did the machinery carry to a second task family?",
            tag="generality",
            paragraphs=(
                (
                    "No second family was run. C3 -- generality -- is "
                    "unestablished by this study: nothing here shows the "
                    "runner reaches a second family without a domain change."
                ),
            ),
        )
    swap = record.adapter_swap
    modules = (
        ", ".join(swap.differing_modules) if swap.differing_modules else "none"
    )
    run_rows = tuple(
        Row(
            cells=(
                Cell(text=run.run_id),
                Cell(
                    text="pass" if run.audit_passed else "FAIL",
                    status="ok" if run.audit_passed else "bad",
                ),
                Cell(
                    figure=_pointer_figure(
                        f"{sum(entry.calls for entry in run.spend):,} calls",
                        f"c18.runs[{run.run_id}].spend[*].calls",
                        run.cost_ref,
                    )
                ),
                Cell(
                    figure=_pointer_figure(
                        _usd(
                            _total_usd(run),
                            sum(entry.unpriced_calls for entry in run.spend),
                            sum(entry.calls for entry in run.spend),
                        ),
                        f"c18.runs[{run.run_id}].spend[*].usd",
                        run.cost_ref,
                    )
                ),
            )
        )
        for run in record.runs
    )
    swap_rows = (
        Row(
            cells=(
                Cell(text="adapter-swap assertion"),
                Cell(
                    text="passed" if swap.passed else "FAILED",
                    status="ok" if swap.passed else "bad",
                ),
                Cell(text=f"differing modules: {modules}"),
            )
        ),
    )
    return Section(
        heading="Did the machinery carry to a second task family?",
        tag="generality",
        paragraphs=(
            (
                "C3 asks whether the runner reaches a second family through "
                "the identical entry point. A differing module outside the "
                "family adapter and the registry entry is the finding, not a "
                "detail: it would be a domain leak in the shared runner."
            ),
        ),
        tables=(
            Table(
                headers=("run", "fidelity", "calls", "USD"),
                rows=run_rows,
                caption="The second family's runs.",
            ),
            Table(
                headers=("assertion", "verdict", "detail"),
                rows=swap_rows,
            ),
        ),
    )


def _total_usd(run: RunRecord) -> float | None:
    """The run's USD total, or ``None`` when any role was unpriced.

    Absence is contagious on purpose: a total over roles where one role is
    unpriced would look authoritative while understating spend.
    """
    if any(entry.usd is None for entry in run.spend):
        return None
    return sum(entry.usd or 0.0 for entry in run.spend)


def _provider_call_text(entry: ProviderCallRecord) -> str:
    """One transport's effective provider call config, as one line.

    Every control appears, set or not: an omitted control would read as
    one the study chose, and "left to the provider's default" is the state
    that explains this study's per-call bill.
    """
    controls = ", ".join(
        f"{name} {value}"
        for name, value in (
            ("temperature", entry.temperature),
            ("top_p", entry.top_p),
            ("token limit", entry.token_limit),
            ("reasoning", entry.reasoning),
            ("seed", entry.seed),
        )
    )
    return (
        f"{entry.provider}/{entry.protocol} route {entry.model_route}; "
        f"{controls}; extensions {entry.extensions}"
    )


def _threats_section(manifest: StudyManifest) -> Section:
    design = manifest.design
    held_out_size = manifest.splits.held_out.size
    threats = (
        (
            "The Codex arm's agent model is pinned but unpriced",
            Cell(
                figure=_manifest_figure(
                    f"Pre-registered as {manifest.models.codex_agent_model}, "
                    "and the arm refuses to run an agent that disagrees. "
                    "Its own model calls still run off this study's key "
                    "entirely, so whetstone observes no usage evidence for "
                    "them and the manifest carries no cost role for them. "
                    "The arm's OpenRouter evaluation calls price normally; "
                    "the agent's do not appear at all.",
                    "models.codex_agent_model",
                )
            ),
            (
                "A comparison against arms whose proposer this study pinned. "
                "Danielle's judgement, not a number, decides whether that is "
                "fair."
            ),
        ),
        (
            (
                "Codex buys whole-split evaluations; MIPROv2 and GEPA buy "
                "minibatches"
            ),
            Cell(
                text=(
                    "The study capped Codex's admitted evaluate-calls per "
                    "run and audits that cap directly, which makes the arms "
                    "comparable in spend rather than in evaluation "
                    "granularity."
                )
            ),
            (
                "A residual incomparability the design chose not to engineer "
                "away: the arms are buying different things with their budget."
            ),
        ),
        (
            "Percentile intervals under-cover at small task counts",
            Cell(
                figure=_manifest_figure(
                    f"Intervals are paired task-level percentile bootstrap "
                    f"over the {held_out_size}-task held-out split, and the "
                    f"p-value floor is reported rather than hidden.",
                    "splits.held_out.size",
                )
            ),
            (
                "Percentile intervals are known to under-cover in small "
                "samples, so the nominal 95% is optimistic here."
            ),
        ),
        (
            (
                "Even the full held-out split does not reach a small true "
                "effect"
            ),
            (
                Cell(
                    text=(
                        "No MDE was measured, because Stage 0 is unrecorded."
                    )
                )
                if design is None
                else Cell(
                    figure=_manifest_figure(
                        f"The measured MDE is "
                        f"{_proportion(design.mde_measured)} at the design's "
                        f"repeat count.",
                        "design.mde_measured",
                    )
                )
            ),
            (
                "A true improvement of 3-5 points is below what this design "
                "can detect. A null result here is not evidence of no effect."
            ),
        ),
        (
            "Within-task variance is estimated from the naive arm only",
            Cell(
                text=(
                    "The variance decomposition estimates the within-task "
                    "component from the naive arm's base rate."
                )
            ),
            (
                "If naive and ceiling have very different base rates, the "
                "pooled estimate would differ and the MDE with it."
            ),
        ),
    )
    rows = tuple(
        Row(cells=(Cell(text=name), did, Cell(text=remains)))
        for name, did, remains in threats
    )
    return Section(
        heading="What could still be wrong with these conclusions?",
        tag="threats",
        tables=(
            Table(
                headers=("threat", "what the study did", "what remains"),
                rows=rows,
            ),
        ),
    )


def _checklist_section() -> Section:
    return Section(
        heading="What must Danielle check by hand before relying on this?",
        tag="validation",
        paragraphs=(
            (
                "These claims are experimental and need manual judgement. No "
                "automated gate in this study settles any of them."
            ),
        ),
        checklist=VALIDATION_CHECKLIST,
    )


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def build_study_report(
    manifest: StudyManifest, *, store: EvidenceStore | None = None
) -> StudyReport:
    """Assemble the report from the manifest and whatever evidence resolves.

    ``store`` is optional because a report must still be generable from an
    archived study whose run store has moved. Without it, the audit findings
    and anything else behind a pointer are reported as unresolved rather
    than as absent facts.
    """
    evidence_by_arm = {
        arm.arm_id: tuple(
            read_run_evidence(run, store=store) for run in arm.runs
        )
        for arm in manifest.arms
    }
    per_arm = tuple(
        _optimizer_section(
            manifest, arm, index=index, evidence=evidence_by_arm[arm.arm_id]
        )
        for index, arm in enumerate(manifest.arms)
    )
    arms_section = Section(
        heading="What did each arm do, and what did it cost?",
        tag="per arm",
        paragraphs=(
            (
                "Each arm's held-out number sits beside every anchor and null "
                "measured under the identical procedure, its own cost, its "
                "internal trajectory, and the audit findings behind its "
                "fidelity verdict."
            ),
        ),
        panels=per_arm,
    )
    sections = (
        _verdict_section(manifest),
        _design_section(manifest),
        _stage_history_section(manifest),
        _leakage_section(manifest),
        arms_section,
        _c18_section(manifest),
        _threats_section(manifest),
        _checklist_section(),
    )
    return StudyReport(
        title=_title(manifest),
        dek=(
            "What four optimizers did to held-out accuracy, and how far "
            "each result can be trusted."
        ),
        byline=f"whetstone-envs · {manifest.created_at[:10]}",
        sections=sections,
        colophon=_colophon(manifest),
        warnings=_warnings(manifest),
    )


def _title(manifest: StudyManifest) -> str:
    """A declarative h1 naming subject and consequence.

    The title states what the study established rather than what it is
    called, and it does not overstate in either direction. Three facts are
    distinguished, because collapsing them would be the report's most
    consequential lie: a study that measured nothing claims nothing; an arm
    whose fidelity audit failed is *unvalidated*, which is not the same as
    an arm that was measured and did not improve; and only an arm that
    passed both gates improved anything.
    """
    if not manifest.held_out:
        return (
            "This study has run no held-out evaluation, so it claims "
            "nothing yet"
        )
    if study_leakage_failed(manifest):
        # The strongest downgrade the title can state, and the reason it
        # comes before every per-arm reading: a leak means the numbers
        # below describe a procedure that did not hold, so no headline may
        # report an improvement, however wide the interval.
        if manifest.leakage_check is None:
            return (
                "This study's leakage rules were never run, so none of its "
                "held-out numbers may be claimed"
            )
        return (
            "A pre-registered leakage rule failed, so this study's held-out "
            "numbers are descriptive only and claim nothing"
        )
    backstop = (
        manifest.design.completeness_backstop
        if manifest.design is not None
        else 1.0
    )
    verdicts = {
        arm.arm_id: _arm_verdict(
            arm=arm, row=_arm_held_out(manifest, arm), backstop=backstop
        )
        for arm in manifest.arms
        if not arm.arm_id.startswith(NULL_ARM_PREFIX)
    }
    validated = tuple(
        arm_id
        for arm_id, verdict in verdicts.items()
        if verdict == VERDICT_VALIDATED
    )
    unvalidated = tuple(
        arm_id
        for arm_id, verdict in verdicts.items()
        if verdict == VERDICT_NOT_VALIDATED
    )
    fidelity = (
        f"; {', '.join(unvalidated)} failed fidelity and claims nothing"
        if unvalidated
        else ""
    )
    if not validated:
        return (
            "No optimizer in this study improved held-out accuracy "
            f"detectably{fidelity}"
        )
    remainder = len(verdicts) - len(validated) - len(unvalidated)
    rest = "; the rest did not detectably" if remainder else ""
    return f"{', '.join(validated)} improved held-out accuracy{rest}{fidelity}"


def _colophon(manifest: StudyManifest) -> tuple[str, str]:
    """Process metadata, per the kit: provenance at the bottom, not the top."""
    return (
        (
            f"Study {manifest.study_id}, created {manifest.created_at}, "
            f"schema {manifest.schema_}. Protocol "
            f"{manifest.protocol_doc_path} at sha256 "
            f"{manifest.protocol_doc_sha256[:12]}; assignment at sha256 "
            f"{manifest.assignment_doc_sha256[:12]}."
        ),
        (
            f"Every number in this report is a field of "
            f"{STUDY_MANIFEST_COPY} in this packet, or a value the manifest "
            "cites by (schema, content hash). Nothing was recomputed. The "
            "mark beside each number is its manifest path, and its store "
            "pointer where the manifest names one. Population "
            f"{manifest.population.family} at generator "
            f"{manifest.population.generator_version}, "
            f"n_per_stratum {manifest.population.n_per_stratum}, pool "
            f"{manifest.population.pool_manifest_content_hash[:12]}."
        ),
    )


def _warnings(manifest: StudyManifest) -> tuple[str, ...]:
    """Reader-relevant honesty, stated in context rather than buried."""
    notes: list[str] = []
    if manifest.design is None:
        notes.append(
            "No design is recorded: Stage 0 has not run, so no result here "
            "is powered against a measured detectable effect."
        )
    if not manifest.held_out:
        notes.append(
            "No held-out evaluation is recorded, so this report describes a "
            "design and its runs, not a result."
        )
    pre_registration = manifest.pre_registration
    if (
        pre_registration is not None
        and pre_registration.provenance == PROVENANCE_AMENDED
    ):
        notes.append(
            "The pre-registration was amended after Stage 0: this design "
            f"({pre_registration.design_hash[:12]}) replaced the one "
            f"registered at {(pre_registration.amended_from or '')[:12]}. "
            "Results below are pre-registered against the amended design, "
            "not the original."
        )
    if manifest.leakage_check is None:
        notes.append("The leakage rules were not run over this manifest.")
    elif not manifest.leakage_check.passed:
        notes.append(
            "A leakage rule failed. Nothing in this report may be claimed."
        )
    failed = tuple(
        arm.arm_id
        for arm in manifest.arms
        if arm.runs and not all(run.audit_passed for run in arm.runs)
    )
    if failed:
        notes.append(
            "Fidelity audits failed for "
            f"{', '.join(failed)}: those arms' numbers are descriptive "
            "only, never claims."
        )
    return tuple(notes)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_markdown(report: StudyReport) -> str:
    """The Markdown source of the packet.

    Markdown has no place to hang a provenance mark, so every figure renders
    its value followed by its evidence in braces. The rule is the same as
    the HTML's: a number never appears without what backs it.
    """
    lines = [
        f"# {report.title}",
        "",
        report.dek,
        "",
        f"*{report.byline}*",
        "",
    ]
    for note in report.warnings:
        lines.extend((f"> **Note.** {note}", ""))
    for section in report.sections:
        lines.extend(_markdown_section(section))
    if report.colophon:
        lines.extend(("---", ""))
        lines.extend(f"{entry}\n" for entry in report.colophon)
    return "\n".join(lines).rstrip() + "\n"


def _markdown_section(section: Section) -> Iterator[str]:
    heading = "#" * section.level
    tag = f" `{section.tag}`" if section.tag else ""
    yield f"{heading} {section.heading}{tag}"
    yield ""
    for paragraph in section.paragraphs:
        yield paragraph
        yield ""
    for table in section.tables:
        yield from _markdown_table(table)
    for label, body in section.code_blocks:
        yield f"**{label}**"
        yield ""
        yield "```text"
        yield body.rstrip("\n")
        yield "```"
        yield ""
    if section.checklist:
        for item in section.checklist:
            yield f"- [ ] {item}"
        yield ""
    for panel in section.panels:
        yield from _markdown_section(panel)


def _markdown_table(table: Table) -> Iterator[str]:
    yield "| " + " | ".join(table.headers) + " |"
    yield "| " + " | ".join("---" for _ in table.headers) + " |"
    for row in table.rows:
        yield (
            "| "
            + " | ".join(_markdown_cell(cell) for cell in row.cells)
            + " |"
        )
    yield ""
    if table.caption:
        yield f"*{table.caption}*"
        yield ""


def _markdown_cell(cell: Cell) -> str:
    text = _escape_pipes(cell.rendered())
    if cell.figure is None:
        return text
    return f"{text} {{{_escape_pipes(cell.figure.evidence())}}}"


def _escape_pipes(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def render_html(report: StudyReport) -> str:
    """The polished HTML reading copy, rendering with no network access.

    Every asset is packet-local: the stylesheet and favicon are files
    beside this document, and no font, script, or highlighter is fetched.
    Everything interpolated is escaped, so a prompt sample containing markup
    is text rather than structure.
    """
    body: list[str] = [
        f"<h1>{html.escape(report.title)}</h1>",
        f'<p class="dek">{html.escape(report.dek)}</p>',
        f'<div class="byline">{html.escape(report.byline)}</div>',
    ]
    body.extend(
        f'<div class="callout">{html.escape(note)}</div>'
        for note in report.warnings
    )
    for section in report.sections:
        body.extend(_html_section(section))
    if report.colophon:
        body.append('<div class="colophon">')
        body.extend(
            f"<p>{html.escape(entry)}</p>" for entry in report.colophon
        )
        body.append("</div>")
    return _HTML_DOCUMENT.format(
        title=html.escape(report.title),
        stylesheet=ASSET_NAMES[0],
        favicon=ASSET_NAMES[1],
        body="\n".join(body),
    )


#: The document shell. It carries no remote reference of any kind, which is
#: what "renders with no network" means concretely, and its only
#: substitution slots are the four named below -- a template with an
#: unresolved slot is a structural failure the tests assert against.
_HTML_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="{favicon}" type="image/svg+xml">
<link rel="stylesheet" href="{stylesheet}">
<title>whetstone-envs · {title}</title>
</head>
<body>
<main class="wide">
{body}
</main>
</body>
</html>
"""


def _html_section(section: Section) -> Iterator[str]:
    tag = (
        f'<span class="tag">{html.escape(section.tag)}</span>'
        if section.tag
        else ""
    )
    level = min(section.level, 3)
    yield (
        f'<div class="section-head"><h{level}>'
        f"{html.escape(section.heading)}</h{level}>{tag}</div>"
    )
    for paragraph in section.paragraphs:
        yield f"<p>{_inline(paragraph)}</p>"
    for table in section.tables:
        yield from _html_table(table)
    for label, code in section.code_blocks:
        yield '<div class="codeblock prompt">'
        yield (
            f'<div class="cb-head"><span>{html.escape(label)}</span>'
            "<span>proposed prompt</span></div>"
        )
        yield f"<pre><code>{html.escape(code)}</code></pre>"
        yield "</div>"
    if section.checklist:
        yield '<ul class="checklist">'
        for item in section.checklist:
            yield f"<li>{html.escape(item)}</li>"
        yield "</ul>"
    if section.panels:
        count = min(max(len(section.panels), 2), _MAX_PANELS)
        yield f'<div class="panels p{count}">'
        for panel in section.panels:
            yield '<section class="panel">'
            yield from _html_section(panel)
            yield "</section>"
        yield "</div>"


#: The kit defines panel grids for two, three, and four panels only. More
#: arms than that stack into the widest grid rather than inventing a class
#: the stylesheet has no rule for.
_MAX_PANELS = 4


def _html_table(table: Table) -> Iterator[str]:
    yield "<table>"
    heads = "".join(
        f"<th>{html.escape(header)}</th>" for header in table.headers
    )
    yield f"<tr>{heads}</tr>"
    for row in table.rows:
        cells = "".join(_html_cell(cell) for cell in row.cells)
        yield f"<tr>{cells}</tr>"
    yield "</table>"
    if table.caption:
        yield f'<div class="dcaption">{_inline(table.caption)}</div>'


def _html_cell(cell: Cell) -> str:
    if cell.figure is not None:
        return (
            '<td><span class="figure"><span class="value">'
            f"{html.escape(cell.figure.value)}</span></span>"
            f'<span class="evidence">'
            f"{html.escape(cell.figure.evidence())}</span></td>"
        )
    text = html.escape(cell.rendered())
    if cell.status is not None:
        return f'<td><span class="verdict {cell.status}">{text}</span></td>'
    return f"<td>{text}</td>"


def _inline(value: str) -> str:
    """Escape, then honour ``**bold**`` as the scan-anchor channel.

    Bold in the source prose marks the phrases carrying the meaning, and the
    kit's scan layer is ``<span class="key">`` in the anchor hue rather than
    bold. Escaping happens first, so the markers are the only markup that
    survives.
    """
    escaped = html.escape(value)
    parts = escaped.split("**")
    if len(parts) % 2 == 0:
        # An unbalanced marker is prose containing asterisks, not emphasis.
        return escaped
    return "".join(
        part if index % 2 == 0 else f'<span class="key">{part}</span>'
        for index, part in enumerate(parts)
    )


# --------------------------------------------------------------------------
# The packet
# --------------------------------------------------------------------------


def _copy_assets(out_dir: Path) -> None:
    for name in ASSET_NAMES:
        source = files(_ASSET_PACKAGE).joinpath(name)
        (out_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )


def generate_study_report(
    *,
    manifest: StudyManifest,
    out_dir: Path,
    store: EvidenceStore | None = None,
) -> Path:
    """Write the report packet for ``manifest`` and return its directory.

    This is the CLI's :class:`~whetstone_envs.optim.study.cli.ReportGenerator`
    and satisfies its contract exactly: it takes the manifest itself rather
    than a study directory, because the report is defined to read only the
    manifest and the evidence the manifest names.

    The packet holds the Markdown source, the polished HTML, the manifest
    that backs every number, and the presentation assets. It is a durable
    work document, so it is refused inside a repository, matching every
    other artifact this package writes.
    """
    resolved = validate_output_root(out_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    report = build_study_report(manifest, store=store)
    (resolved / REPORT_MARKDOWN_NAME).write_text(
        render_markdown(report), encoding="utf-8"
    )
    (resolved / REPORT_HTML_NAME).write_text(
        render_html(report), encoding="utf-8"
    )
    (resolved / STUDY_MANIFEST_COPY).write_text(
        json.dumps(
            manifest.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _copy_assets(resolved)
    return resolved
