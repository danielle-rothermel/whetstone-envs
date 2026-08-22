"""The Step 10 study report, generated from a manifest and checked for drift.

**Where the manifest under test comes from.** The study CLI's Stage-0 dry run
produces a manifest with real c19 splits over a real generated pool, and
that is what :func:`stage0_manifest` builds -- the same shape a paid Stage 0
would write, at toy sizes. There is **no fake-transport Stage-2 path**: the
stage harness needs an optimizer runner, an official scorer, and a held-out
evaluator, and wiring those on fakes is Wave 7's execution, not a unit test.
So the reported study is a **synthetic Stage-2 manifest**: the Stage-0
manifest's real population and real splits, with arms, runs, selections,
held-out claims, and held-out rows added through the manifest's own
constructors. Every one of those passes the same validation a paid stage's
write would, so what the report reads is a document the study could have
produced, not a hand-shaped dict.

The load-bearing assertion is mechanical rather than by eye: every number
the report renders is a :class:`Figure`, and every figure names the manifest
field it came from and the pointer the manifest cites for it. A number added
without its evidence fails the "names its evidence" test rather than
shipping unbacked.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.study.cli import (
    default_report_generator,
    main,
)
from whetstone_envs.optim.study.manifest import (
    SELECTION_RULE_ARGMAX_OFFICIAL,
    AdapterSwapRecord,
    ArmRecord,
    C18Record,
    DesignRecord,
    EvidencePointer,
    FanoutCheckRecord,
    GepaSizingRecord,
    HeldOutClaimRecord,
    HeldOutRecord,
    LeakageCheckEntry,
    LeakageCheckRecord,
    ModelsRecord,
    PopulationRecord,
    RunRecord,
    RunSpendRecord,
    SelectionRecord,
    SplitRecord,
    SplitsRecord,
    StudyManifest,
    write_study_manifest,
)
from whetstone_envs.reporting.study_report import (
    ASSET_NAMES,
    MISSING,
    REPORT_HTML_NAME,
    REPORT_MARKDOWN_NAME,
    STUDY_MANIFEST_COPY,
    UNPRICED,
    VALIDATION_CHECKLIST,
    VERDICT_NO_IMPROVEMENT,
    VERDICT_NOT_VALIDATED,
    VERDICT_VALIDATED,
    build_study_report,
    figures_in,
    generate_study_report,
    render_html,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------
# The manifest under test
# --------------------------------------------------------------------------

#: Toy sizes. The study's own are (88, 132, 220); these keep a real pool
#: generation inside a unit test's budget.
TOY_SPLIT_SIZES = (4, 4, 6)
TOY_N_PER_STRATUM = 1
TOY_POOL_SEED_START = 765_432


def _pointer(char: str) -> EvidencePointer:
    return EvidencePointer(
        schema_name="whetstone.eval_evidence", content_hash=char * 64
    )


def _stage0_splits() -> SplitsRecord:
    """The three splits of one deterministically generated c19 pool.

    Real hashes from the real experiment builder, not invented ones: a
    manifest of invented hashes validates and then describes tasks nothing
    could evaluate, which is exactly the failure a fixture must not hide.
    """
    from whetstone_envs.c19 import generate_pool
    from whetstone_envs.optim.experiment import prepare_c19_experiment
    from whetstone_envs.optim.rows import task_rows_from_instances

    pool = generate_pool(
        n_per_stratum=TOY_N_PER_STRATUM, seed_start=TOY_POOL_SEED_START
    )
    split = prepare_c19_experiment(
        pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=1
    ).split
    internal, official, held_out = TOY_SPLIT_SIZES

    def hashes(instances) -> tuple[str, ...]:
        return tuple(
            row.task_hash for row in task_rows_from_instances(instances)
        )

    return SplitsRecord(
        internal=SplitRecord(
            size=internal,
            task_hashes=hashes(split.internal_eval),
            eval_config_hash="toy-internal-config",
        ),
        official=SplitRecord(
            size=official,
            task_hashes=hashes(split.official),
            eval_config_hash="toy-official-config",
        ),
        held_out=SplitRecord(
            size=held_out,
            task_hashes=hashes(split.held_out),
            eval_config_hash="toy-held-out-config",
        ),
    )


def _spend(*, unpriced: bool) -> tuple[RunSpendRecord, ...]:
    """One run's per-role spend, with the proposer deliberately unpriced.

    The unpriced role is the point: an absent USD total is the report's most
    consequential number, so the fixture always carries one.
    """
    task_model = RunSpendRecord(
        role="task_model",
        calls=120,
        cached_calls=8,
        input_tokens=4_000,
        output_tokens=900,
        priced_calls=120,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.0421,
    )
    proposer = RunSpendRecord(
        role="proposer",
        calls=6,
        cached_calls=0,
        input_tokens=800,
        output_tokens=300,
        priced_calls=2 if unpriced else 6,
        unpriced_calls=4 if unpriced else 0,
        rows_missing_token_breakdown=0,
        usd=None if unpriced else 0.0033,
    )
    return (task_model, proposer)


def _run(
    run_id: str, *, seed: int, passed: bool, unpriced: bool = True
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        seed=seed,
        artifact_dir=f"/nonexistent/runs/{run_id}",
        result_ref=_pointer("a"),
        audit_ref=_pointer("b"),
        cost_ref=_pointer("c"),
        audit_passed=passed,
        spend=_spend(unpriced=unpriced),
    )


def _held_out_row(  # noqa: PLR0913
    name: str,
    *,
    mean: float,
    ci: tuple[float, float],
    delta: float,
    p_bootstrap: float,
    p_holm: float | None,
    completeness: float = 1.0,
) -> HeldOutRecord:
    return HeldOutRecord(
        candidate_name=name,
        eval_evidence_ref=_pointer("e"),
        per_task_scores_ref=_pointer("f"),
        mean=mean,
        ci_low=ci[0],
        ci_high=ci[1],
        delta_vs_naive=delta,
        p_bootstrap=p_bootstrap,
        p_holm=p_holm,
        completeness=completeness,
    )


def _claim(name: str, mean: float) -> HeldOutClaimRecord:
    return HeldOutClaimRecord(
        candidate_name=name,
        eval_config_hash="toy-held-out-config",
        repeats=3,
        mean=mean,
        completeness=1.0,
    )


@pytest.fixture(scope="module")
def stage0_manifest() -> StudyManifest:
    """A pre-Stage-0 manifest over a real c19 pool: no design, no runs."""
    return StudyManifest(
        study_id="step10-report-fixture",
        created_at="2026-08-22T12:00:00+00:00",
        protocol_doc_path="~/drotherm/data/.claude/protocol.md",
        protocol_doc_sha256="a" * 64,
        assignment_doc_sha256="b" * 64,
        population=PopulationRecord(
            family="c19",
            generator_version="whetstone-envs toy",
            n_per_stratum=TOY_N_PER_STRATUM,
            pool_seed_start=TOY_POOL_SEED_START,
            pool_manifest_content_hash="c" * 64,
            stratum_counts={"toy": sum(TOY_SPLIT_SIZES)},
        ),
        splits=_stage0_splits(),
        models=ModelsRecord(
            task_model="openai/gpt-5-nano",
            proposer_model="openai/gpt-5.4-nano",
            temperature="unset",
            provider="openrouter",
            seed_control="advertised",
            codex_agent_model="uncontrolled",
        ),
    )


@pytest.fixture(scope="module")
def reported_manifest(stage0_manifest: StudyManifest) -> StudyManifest:
    """A synthetic Stage-2 manifest: three arms, one of each verdict.

    ``copro`` passes its audits and its interval excludes zero (validated),
    ``gepa`` fails an audit (not validated, whatever its interval says), and
    ``null-random`` is a control whose interval spans zero and whose Holm
    column is empty by design.
    """
    arms = (
        ArmRecord(
            arm_id="copro",
            optimizer="copro",
            demo_mode=None,
            control_identity_hash="d" * 64,
            seed_note="provider-seed-control-only",
            runs=(
                _run("copro-1000", seed=1000, passed=True),
                _run("copro-1001", seed=1001, passed=True),
            ),
        ),
        ArmRecord(
            arm_id="gepa",
            optimizer="gepa",
            demo_mode=None,
            control_identity_hash="e" * 64,
            seed_note="control-seed-field",
            runs=(_run("gepa-3000", seed=3000, passed=False),),
        ),
        ArmRecord(
            arm_id="null-random",
            optimizer="null-random",
            demo_mode=None,
            control_identity_hash="f" * 64,
            seed_note="control-seed-field",
            runs=(_run("nullA-5000", seed=5000, passed=True, unpriced=False),),
        ),
    )
    return stage0_manifest.model_copy(
        update={
            "design": DesignRecord(
                k_cal=4,
                k_repeat=3,
                k_run_by_arm={"copro": 5, "gepa": 5, "null-random": 5},
                ci_level=0.95,
                resamples=10_000,
                bootstrap_seed=20_260_822,
                correction="holm-bonferroni",
                m=4,
                mde_formula=(
                    "MDE(T, K) = 2.8016 * sqrt((tau^2 + 2 sigma^2 / K) / T)"
                ),
                mde_measured=0.0812,
                tau_sq=0.0204,
                sigma_sq=0.1533,
                completeness_rule=(
                    "weight each task's delta by its achieved sample count"
                ),
                completeness_backstop=0.90,
            ),
            "gepa_sizing": GepaSizingRecord(
                steps_per_run=732,
                wall_seconds=4_210.5,
                sqlite_bytes=18_446_744,
                max_metric_calls_pinned=200,
            ),
            "fanout_check": FanoutCheckRecord(
                passed=True, minibatch_intents=40, full_valset_intents=6
            ),
            "arms": arms,
            "selection": tuple(
                SelectionRecord(
                    arm_id=arm.arm_id,
                    selected_run_id=arm.runs[0].run_id,
                    official_score=0.5,
                    rule=SELECTION_RULE_ARGMAX_OFFICIAL,
                )
                for arm in arms
            ),
            "held_out_claims": (
                _claim("copro", 0.6182),
                _claim("gepa", 0.4400),
                _claim("null-random", 0.4055),
                _claim("naive", 0.4000),
                _claim("ceiling", 0.9100),
            ),
            "held_out": (
                _held_out_row(
                    "copro",
                    mean=0.6182,
                    ci=(0.0930, 0.3410),
                    delta=0.2182,
                    p_bootstrap=0.0002,
                    p_holm=0.0008,
                ),
                _held_out_row(
                    "gepa",
                    mean=0.4400,
                    ci=(-0.0410, 0.1210),
                    delta=0.0400,
                    p_bootstrap=0.4120,
                    p_holm=0.8240,
                ),
                _held_out_row(
                    "null-random",
                    mean=0.4055,
                    ci=(-0.0320, 0.0430),
                    delta=0.0055,
                    p_bootstrap=0.7710,
                    p_holm=None,
                ),
                _held_out_row(
                    "naive",
                    mean=0.4000,
                    ci=(0.0, 0.0),
                    delta=0.0,
                    p_bootstrap=1.0,
                    p_holm=None,
                ),
                _held_out_row(
                    "ceiling",
                    mean=0.9100,
                    ci=(0.4180, 0.5820),
                    delta=0.5100,
                    p_bootstrap=0.0002,
                    p_holm=None,
                ),
            ),
            "leakage_check": LeakageCheckRecord(
                passed=True,
                checks=(
                    LeakageCheckEntry(
                        check_id="L2",
                        passed=True,
                        detail="one selection per arm",
                    ),
                    LeakageCheckEntry(
                        check_id="L3",
                        passed=True,
                        detail="one held-out evaluation per candidate",
                    ),
                    LeakageCheckEntry(
                        check_id="L5",
                        passed=True,
                        detail="split task hashes are pairwise disjoint",
                    ),
                ),
            ),
            "c18": C18Record(
                runs=(_run("c18-copro", seed=1000, passed=True),),
                adapter_swap=AdapterSwapRecord(
                    passed=True,
                    differing_modules=(
                        "whetstone_envs.optim.c18_experiment",
                        "whetstone_envs.optim.families",
                    ),
                ),
            ),
        }
    )


# --------------------------------------------------------------------------
# The mechanical evidence check
# --------------------------------------------------------------------------


def test_every_rendered_number_names_its_evidence(
    reported_manifest: StudyManifest,
) -> None:
    """The load-bearing assertion: no number renders unbacked.

    Every figure the report builds names the manifest field it came from,
    and a figure that names nothing fails here rather than reaching a
    reader as a number with no provenance.
    """
    report = build_study_report(reported_manifest)
    figures = tuple(figures_in(report))
    assert figures, "a reported study renders numbers"
    unbacked = [figure for figure in figures if not figure.backed()]
    assert unbacked == []


def test_every_figure_resolves_to_a_manifest_path(
    reported_manifest: StudyManifest,
) -> None:
    """Each figure's source is a real path into the manifest document.

    A source naming a field the manifest does not have would be provenance
    that looks checkable and is not, so the prefix is asserted rather than
    assumed.
    """
    report = build_study_report(reported_manifest)
    for figure in figures_in(report):
        assert figure.source.startswith("study.json:"), figure


def test_pointer_figures_cite_pointers_the_manifest_carries(
    reported_manifest: StudyManifest,
) -> None:
    """A figure's pointer is one the manifest itself cites.

    This is what stops the report from inventing a plausible-looking
    ``(schema, hash)`` pair: every pointer it prints must appear in the
    manifest's own pointer walk.
    """
    cited = set(reported_manifest.evidence_pointers())
    report = build_study_report(reported_manifest)
    printed = {
        figure.pointer
        for figure in figures_in(report)
        if figure.pointer is not None
    }
    assert printed
    assert printed <= cited


# --------------------------------------------------------------------------
# Both renderings
# --------------------------------------------------------------------------


def test_packet_holds_both_renderings_and_its_assets(
    reported_manifest: StudyManifest, tmp_path: Path, monkeypatch
) -> None:
    packet = tmp_path / "packet"
    monkeypatch.setattr(
        "whetstone_envs.reporting.study_report.validate_output_root",
        lambda path: path.resolve(),
    )
    result = generate_study_report(manifest=reported_manifest, out_dir=packet)
    assert result == packet.resolve()
    for name in (
        REPORT_MARKDOWN_NAME,
        REPORT_HTML_NAME,
        STUDY_MANIFEST_COPY,
        *ASSET_NAMES,
    ):
        assert (packet / name).is_file(), name


def test_the_packet_refuses_to_write_inside_the_repository(
    reported_manifest: StudyManifest,
) -> None:
    """A report packet is a durable work document, not a versioned one."""
    from pathlib import Path as RealPath

    inside = RealPath(__file__).parent / "would-be-packet"
    with pytest.raises(ValueError, match="must not be written inside"):
        generate_study_report(manifest=reported_manifest, out_dir=inside)


def test_html_has_no_unresolved_template_slots(
    reported_manifest: StudyManifest,
) -> None:
    """A leftover ``{slot}`` would be a rendering bug shipped as content."""
    document = render_html(build_study_report(reported_manifest))
    assert re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", document) is None


def test_html_validates_structurally(
    reported_manifest: StudyManifest,
) -> None:
    """The document is well-formed and carries the kit's shell."""
    document = render_html(build_study_report(reported_manifest))
    assert document.startswith("<!DOCTYPE html>")
    assert document.rstrip().endswith("</html>")
    for tag in ("html", "head", "body", "main", "table"):
        assert document.count(f"<{tag}") == document.count(f"</{tag}>"), tag
    assert '<main class="wide">' in document
    assert '<div class="colophon">' in document


def test_html_renders_with_no_network(
    reported_manifest: StudyManifest, tmp_path: Path, monkeypatch
) -> None:
    """No remote reference of any kind, in the document or in its assets.

    The report is read from a durable work directory that may have no
    network at all, so a CDN stylesheet or a webfont import would be a
    document that renders differently -- or not at all -- for its actual
    reader.

    What is asserted is *fetching*, not the string ``http``: the favicon's
    ``xmlns`` is an XML namespace identifier that no agent resolves, and
    banning the substring would fail on a correct asset.
    """
    packet = tmp_path / "packet"
    monkeypatch.setattr(
        "whetstone_envs.reporting.study_report.validate_output_root",
        lambda path: path.resolve(),
    )
    generate_study_report(manifest=reported_manifest, out_dir=packet)
    fetching = re.compile(
        r"""@import|url\(\s*['"]?https?:|(?:href|src)\s*=\s*['"](?:https?:)?//""",
        re.IGNORECASE,
    )
    for name in (REPORT_HTML_NAME, *ASSET_NAMES):
        text = (packet / name).read_text(encoding="utf-8")
        assert fetching.search(text) is None, name


def test_html_references_only_packet_local_assets(
    reported_manifest: StudyManifest,
) -> None:
    document = render_html(build_study_report(reported_manifest))
    for reference in re.findall(r'(?:href|src)="([^"]+)"', document):
        assert reference in ASSET_NAMES, reference


def test_markdown_and_html_report_the_same_numbers(
    reported_manifest: StudyManifest,
) -> None:
    """One built report, two emitters: they cannot disagree.

    Both renderings are checked to contain every figure's value, which is
    what makes the Markdown a source of the HTML rather than a second,
    drifting document.
    """
    report = build_study_report(reported_manifest)
    markdown = render_markdown(report)
    document = render_html(report)
    for figure in figures_in(report):
        assert figure.value in markdown, figure
        assert figure.value in document or _escaped(figure.value) in document


def _escaped(value: str) -> str:
    import html

    return html.escape(value)


# --------------------------------------------------------------------------
# Content the assignment names
# --------------------------------------------------------------------------


def test_unpriced_spend_renders_as_a_fraction_never_as_zero(
    reported_manifest: StudyManifest,
) -> None:
    """``unpriced (n/total)``: a total nobody can compute is not zero."""
    markdown = render_markdown(build_study_report(reported_manifest))
    assert f"{UNPRICED} (4/6)" in markdown
    assert "$0.0421" in markdown


def test_fidelity_gates_efficacy(
    reported_manifest: StudyManifest,
) -> None:
    """C1 gates C2: a failed audit is *not validated*, whatever the CI says.

    ``gepa``'s interval spans zero here anyway, so the sharper check is
    that a failed audit cannot produce ``validated`` even when the interval
    would: the verdict function is asked directly.
    """
    from whetstone_envs.reporting.study_report import _arm_verdict

    manifest = reported_manifest
    by_id = {arm.arm_id: arm for arm in manifest.arms}
    rows = {row.candidate_name: row for row in manifest.held_out}
    assert (
        _arm_verdict(arm=by_id["copro"], row=rows["copro"], backstop=0.9)
        == VERDICT_VALIDATED
    )
    # The failed-audit arm handed copro's significant interval still cannot
    # be validated: fidelity is checked first and cannot be outweighed.
    assert (
        _arm_verdict(arm=by_id["gepa"], row=rows["copro"], backstop=0.9)
        == VERDICT_NOT_VALIDATED
    )
    assert (
        _arm_verdict(
            arm=by_id["null-random"], row=rows["null-random"], backstop=0.9
        )
        == VERDICT_NO_IMPROVEMENT
    )


def test_the_title_separates_unvalidated_from_no_improvement(
    reported_manifest: StudyManifest,
) -> None:
    """The h1 must not read a fidelity failure as a measured null result.

    ``gepa`` failed its audit here. Saying it "did not improve" would claim
    a measurement the study is not entitled to, so the title says it failed
    fidelity and claims nothing.
    """
    title = build_study_report(reported_manifest).title
    assert "copro improved held-out accuracy" in title
    assert "gepa failed fidelity and claims nothing" in title
    assert "the rest did not detectably" not in title


def test_the_report_states_the_percentile_p_floor_caveat(
    reported_manifest: StudyManifest,
) -> None:
    markdown = render_markdown(build_study_report(reported_manifest))
    assert "percentile-bootstrap" in markdown
    assert "10,000 resamples" in markdown
    assert "Holm propagates that floor" in markdown


def test_the_report_carries_the_validation_checklist(
    reported_manifest: StudyManifest,
) -> None:
    """Mandatory: experimental claims need manual judgement."""
    report = build_study_report(reported_manifest)
    markdown = render_markdown(report)
    document = render_html(report)
    for item in VALIDATION_CHECKLIST:
        assert f"- [ ] {item}" in markdown
        assert _escaped(item) in document
    assert 'class="checklist"' in document


def test_the_report_shows_the_stage_and_gate_history(
    reported_manifest: StudyManifest,
) -> None:
    markdown = render_markdown(build_study_report(reported_manifest))
    assert "Stage 0 -- anchor calibration" in markdown
    assert "GEPA sizing (F9)" in markdown
    assert "fan-out check (F16)" in markdown
    assert "max_metric_calls pinned to 200" in markdown


def test_the_report_shows_the_second_family_and_its_swap(
    reported_manifest: StudyManifest,
) -> None:
    markdown = render_markdown(build_study_report(reported_manifest))
    assert "second task family" in markdown
    assert "whetstone_envs.optim.c18_experiment" in markdown


def test_the_report_names_the_threats_the_assignment_requires(
    reported_manifest: StudyManifest,
) -> None:
    markdown = render_markdown(build_study_report(reported_manifest))
    for phrase in (
        "agent model is uncontrolled",
        "whole-split evaluations",
        "under-cover at small task counts",
        "does not reach a small true",
    ):
        assert phrase in markdown, phrase


def test_wall_time_is_reported_unrecorded_not_invented(
    reported_manifest: StudyManifest,
) -> None:
    """The manifest carries no duration, so the report says so.

    Reporting a wall time the evidence does not contain would be the one
    thing this generator exists to prevent.
    """
    markdown = render_markdown(build_study_report(reported_manifest))
    assert "wall time" in markdown
    assert MISSING in markdown


def test_audit_findings_render_when_the_store_resolves_them(
    reported_manifest: StudyManifest,
) -> None:
    """A resolvable ``audit.json`` becomes the trace-audit table's rows.

    The audit package is Wave 2's, so this exercises the read path against
    a store double holding a document in that format rather than against
    the package itself.
    """

    class Store:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def get(self, reference):
            self.requested.append(reference.content_hash)
            return {
                "passed": True,
                "findings": [
                    {
                        "invariant_id": "COPRO_BREADTH_PER_DEPTH",
                        "status": "PASS",
                        "detail": "every step proposed the configured breadth",
                        "evidence_refs": [
                            {
                                "schema_name": "whetstone.optim_step_result",
                                "content_hash": "9" * 64,
                            }
                        ],
                    }
                ],
            }

    store = Store()
    markdown = render_markdown(
        build_study_report(reported_manifest, store=store)
    )
    assert store.requested
    assert "COPRO_BREADTH_PER_DEPTH" in markdown
    assert "whetstone.optim_step_result@999999999999" in markdown


def test_an_unresolvable_audit_reports_the_recorded_verdict(
    reported_manifest: StudyManifest,
) -> None:
    """A missing pointer is named, never rendered as a passing audit."""

    class Store:
        def get(self, reference):
            raise KeyError(reference.content_hash)

    markdown = render_markdown(
        build_study_report(reported_manifest, store=Store())
    )
    assert "KeyError" in markdown
    assert "did not resolve" in markdown or "not resolved" in markdown


# --------------------------------------------------------------------------
# The pre-result study
# --------------------------------------------------------------------------


def test_a_study_with_no_results_claims_nothing(
    stage0_manifest: StudyManifest,
) -> None:
    """The title does not overstate a study that has measured nothing."""
    report = build_study_report(stage0_manifest)
    assert "claims nothing yet" in report.title
    markdown = render_markdown(report)
    assert "No design is recorded" in markdown
    assert "No held-out evaluation is recorded" in markdown
    render_html(report)


# --------------------------------------------------------------------------
# The CLI wiring
# --------------------------------------------------------------------------


def test_the_study_cli_reports_by_default(
    reported_manifest: StudyManifest, tmp_path: Path, monkeypatch, capsys
) -> None:
    """``report`` needs no injected generator: the default is the real one.

    Both the manifest write and the packet write are pointed outside the
    repository check, because a test's ``tmp_path`` is legitimately outside
    a repository on a real run but not necessarily under this checkout's
    detection.
    """
    study_dir = tmp_path / "study"
    packet = tmp_path / "packet"
    for module in (
        "whetstone_envs.optim.study.manifest",
        "whetstone_envs.reporting.study_report",
    ):
        monkeypatch.setattr(
            f"{module}.validate_output_root", lambda path: path.resolve()
        )
    write_study_manifest(study_dir, reported_manifest)

    code = main(
        ["report", "--study-dir", str(study_dir), "--out", str(packet)]
    )
    assert code == 0
    assert str(packet.resolve()) in capsys.readouterr().out
    assert (packet / REPORT_HTML_NAME).is_file()
    assert (packet / REPORT_MARKDOWN_NAME).is_file()


def test_the_default_generator_matches_the_protocol(
    reported_manifest: StudyManifest, tmp_path: Path, monkeypatch
) -> None:
    """The CLI's default is callable through the injected contract exactly."""
    monkeypatch.setattr(
        "whetstone_envs.reporting.study_report.validate_output_root",
        lambda path: path.resolve(),
    )
    packet = default_report_generator(
        manifest=reported_manifest, out_dir=tmp_path / "packet"
    )
    assert (packet / REPORT_HTML_NAME).is_file()


# --------------------------------------------------------------------------
# The trajectory and prompt-sample read path, against a real run
# --------------------------------------------------------------------------


def test_a_real_run_supplies_its_trajectory_and_prompt_samples(
    reported_manifest: StudyManifest, tmp_path: Path
) -> None:
    """The synthetic manifest cannot exercise the artifact read path.

    Its ``artifact_dir`` values point nowhere on purpose, which proves the
    absence path and nothing else. So this drives one real fake-transport
    COPRO run, points an arm's run at the directory it produced, and asserts
    the report picks up the best-so-far trajectory and the proposed prompts
    that the run actually persisted -- the two content items that live in
    the artifacts rather than in the manifest.

    Zero provider calls: the transport is the c19 fake.
    """
    from whetstone_envs.optim.cli import main as run_cli
    from whetstone_envs.reporting.publication import TRAJECTORY_REPORT_NAME

    output = tmp_path / "copro-run"
    assert (
        run_cli(
            [
                "--family",
                "c19",
                "--optimizer",
                "copro",
                "--transport",
                "fake",
                "--split-sizes",
                "2,2,0",
                "--run-id",
                "c19-copro-report",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / TRAJECTORY_REPORT_NAME).is_file()

    arms = reported_manifest.arms
    first = arms[0]
    grounded = first.model_copy(
        update={
            "runs": (
                first.runs[0].model_copy(update={"artifact_dir": str(output)}),
            )
        }
    )
    manifest = reported_manifest.model_copy(
        update={"arms": (grounded, *arms[1:])}
    )

    # What the run actually persisted, so the assertions below are about
    # the report carrying it rather than about the run having produced
    # something.
    from whetstone_envs.reporting.publication import load_trajectory_report
    from whetstone_envs.reporting.study_report import (
        _best_so_far,
        _prompt_samples,
    )

    trajectory = load_trajectory_report(output)
    points = _best_so_far(trajectory)
    samples = _prompt_samples(trajectory)
    assert points, "the fake run recorded at least one internal reward"
    assert samples, "the fake run proposed at least one candidate"

    report = build_study_report(manifest)
    markdown = render_markdown(report)
    document = render_html(report)

    # The run's own trajectory, not the absence sentinel.
    assert f"no {TRAJECTORY_REPORT_NAME} in {output}" not in markdown
    for step, value in points:
        assert f"{step}:{value:.3f}" in markdown
    assert f"terminal {trajectory.terminal_status}" in markdown

    # The prompts the run proposed, rendered as code blocks in the HTML.
    assert "proposed prompt" in document
    for label, _text in samples:
        assert label in document
