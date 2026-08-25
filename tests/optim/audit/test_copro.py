"""COPRO's seven invariants, each against a real run and its own negative.

Every invariant here is tested twice: it PASSes on an unmutated
fake-transport COPRO run, and it FAILs on a fixture that violates exactly
it. The negative half is the one that matters -- an invariant that only ever
passes is indistinguishable from one that returns PASS unconditionally.

Each negative also asserts that *no other* invariant's status changed. A
mutation that broke several checks at once would flatter a sloppy one:
whichever invariant the test names would fail, and nobody would notice it
was failing for someone else's reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.optim.audit import copro_fixtures
from whetstone_envs.optim.audit._evidence import (
    COPRO_OPTIMIZER,
    load_run_evidence,
)
from whetstone_envs.optim.audit.copro import (
    COPRO_INVARIANTS,
    copro_best_so_far,
    copro_breadth_per_depth,
    copro_depth_steps,
    copro_distinct_bases,
    copro_internal_only,
    copro_no_search_evals,
    copro_terminal_provenance,
)
from whetstone_envs.optim.audit.registry import (
    SHARED_INVARIANTS,
    audit_run,
    invariants_for,
)
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId

if TYPE_CHECKING:
    from pathlib import Path

#: Each COPRO invariant paired with the builder that makes it, and only it,
#: fail. Parametrising over this is what makes "every invariant ships a
#: failing fixture" a property of the suite rather than a convention: adding
#: an invariant without a fixture leaves the registry check below failing.
NEGATIVES = (
    (
        InvariantId.COPRO_BREADTH_PER_DEPTH,
        copro_fixtures.over_configured_breadth,
    ),
    (
        InvariantId.COPRO_DEPTH_STEPS,
        copro_fixtures.short_of_configured_depth,
    ),
    (
        InvariantId.COPRO_INTERNAL_ONLY,
        copro_fixtures.evaluation_off_the_internal_split,
    ),
    (
        InvariantId.COPRO_BEST_SO_FAR,
        copro_fixtures.unselected_candidate_scored_higher,
    ),
    (
        InvariantId.COPRO_DISTINCT_BASES,
        copro_fixtures.two_proposals_sharing_one_base,
    ),
    (
        InvariantId.COPRO_NO_SEARCH_EVALS,
        copro_fixtures.evaluation_recorded_as_search,
    ),
    (
        InvariantId.COPRO_TERMINAL_PROVENANCE,
        copro_fixtures.seed_the_search_never_used,
    ),
)

#: The wire spelling of every COPRO invariant id, pinned. These are
#: persisted: an ``audit.json`` is cited by content hash from the study
#: manifest, so a rename here silently rewrites stored identity. Derived
#: spellings would not catch that; only a literal list does.
PINNED_IDS = {
    InvariantId.COPRO_BREADTH_PER_DEPTH: "copro_breadth_per_depth",
    InvariantId.COPRO_DEPTH_STEPS: "copro_depth_steps",
    InvariantId.COPRO_INTERNAL_ONLY: "copro_internal_only",
    InvariantId.COPRO_BEST_SO_FAR: "copro_best_so_far",
    InvariantId.COPRO_DISTINCT_BASES: "copro_distinct_bases",
    InvariantId.COPRO_NO_SEARCH_EVALS: "copro_no_search_evals",
    InvariantId.COPRO_TERMINAL_PROVENANCE: "copro_terminal_provenance",
}


def _statuses(run_dir: Path) -> dict[InvariantId, AuditStatus]:
    return {
        finding.invariant_id: finding.status
        for finding in audit_run(run_dir).findings
    }


def _finding(run_dir: Path, invariant: InvariantId):
    return next(
        finding
        for finding in audit_run(run_dir).findings
        if finding.invariant_id is invariant
    )


# --- Registration ----------------------------------------------------------


def test_every_copro_invariant_is_registered() -> None:
    registered = invariants_for(COPRO_OPTIMIZER)
    assert set(COPRO_INVARIANTS) <= set(registered)
    assert set(SHARED_INVARIANTS) <= set(registered)


def test_the_copro_audit_reports_the_seven_plus_the_shared_one() -> None:
    assert len(COPRO_INVARIANTS) == 7
    assert len(invariants_for(COPRO_OPTIMIZER)) == 7 + len(SHARED_INVARIANTS)


def test_copro_invariant_ids_are_pinned() -> None:
    for invariant, spelling in PINNED_IDS.items():
        assert invariant.value == spelling


def test_every_copro_invariant_ships_a_negative_fixture() -> None:
    """Section 3.2: an invariant with no failing fixture does not ship."""
    assert {invariant for invariant, _builder in NEGATIVES} == set(PINNED_IDS)


# --- The positive fixture --------------------------------------------------


def test_a_real_run_passes_every_copro_invariant(copro_run_dir) -> None:
    report = audit_run(copro_run_dir)
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def test_no_copro_invariant_reports_not_applicable(copro_run_dir) -> None:
    """Section 3.1: an always-inapplicable invariant is a defect.

    None of COPRO's seven is conditional on a dispatch mode or a demo
    regime, so a ``NOT_APPLICABLE`` here would mean an invariant that never
    judges anything.
    """
    for finding in audit_run(copro_run_dir).findings:
        assert finding.status is not AuditStatus.NOT_APPLICABLE


def test_every_finding_states_what_it_saw(copro_run_dir) -> None:
    """A detail that only says "check failed" is unusable when triaging."""
    for finding in audit_run(copro_run_dir).findings:
        assert finding.detail.strip()
        assert any(char.isdigit() for char in finding.detail), (
            f"{finding.invariant_id.value} names no observed quantity: "
            f"{finding.detail}"
        )


# --- The negatives ---------------------------------------------------------


@pytest.mark.parametrize(
    ("invariant", "builder"),
    NEGATIVES,
    ids=[invariant.value for invariant, _ in NEGATIVES],
)
def test_the_target_invariant_fails_on_its_negative_fixture(
    copro_run_dir, tmp_path, invariant, builder
) -> None:
    run_dir = builder(copro_run_dir, tmp_path / invariant.value)
    report = audit_run(run_dir)
    assert not report.passed
    failed = {
        finding.invariant_id
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    }
    assert failed == {invariant}


@pytest.mark.parametrize(
    ("invariant", "builder"),
    NEGATIVES,
    ids=[invariant.value for invariant, _ in NEGATIVES],
)
def test_no_other_invariants_status_changed(
    copro_run_dir, tmp_path, invariant, builder
) -> None:
    """A mutation that broke everything would flatter a sloppy invariant."""
    run_dir = builder(copro_run_dir, tmp_path / invariant.value)
    before = _statuses(copro_run_dir)
    after = _statuses(run_dir)
    assert set(before) == set(after)
    changed = {
        candidate
        for candidate, status in after.items()
        if before[candidate] is not status
    }
    assert changed == {invariant}


@pytest.mark.parametrize(
    ("invariant", "builder"),
    NEGATIVES,
    ids=[invariant.value for invariant, _ in NEGATIVES],
)
def test_a_negative_fixture_leaves_the_source_run_passing(
    copro_run_dir, tmp_path, invariant, builder
) -> None:
    """The builder must copy, never mutate the shared session fixture."""
    builder(copro_run_dir, tmp_path / invariant.value)
    assert audit_run(copro_run_dir).passed


# --- What each invariant actually reads ------------------------------------


def test_breadth_counts_occurrences_not_proposals(copro_run_dir) -> None:
    """The seed round proposes ``breadth - 1`` and measures ``breadth``.

    This is the trap the invariant exists to avoid: counting
    ``proposed_candidates`` would report an honest seed round as short by
    one, because the round's last occurrence is the re-measured initial
    candidate, which was not proposed.
    """
    evidence = load_run_evidence(copro_run_dir)
    seed_round = evidence.steps[0]
    assert len(seed_round.step.proposed_candidates) == 1
    assert len(seed_round.resolved_intents) == 2
    assert copro_breadth_per_depth(evidence).status is AuditStatus.PASS


def test_depth_steps_counts_the_finalizing_step(copro_run_dir) -> None:
    evidence = load_run_evidence(copro_run_dir)
    finding = copro_depth_steps(evidence)
    assert finding.status is AuditStatus.PASS
    assert len(evidence.steps) == 2
    assert not evidence.steps[-1].resolved_intents


def test_internal_only_reads_both_binding_and_recorded_role(
    copro_run_dir,
) -> None:
    """Either half alone is bypassable, so the finding cites both."""
    evidence = load_run_evidence(copro_run_dir)
    finding = copro_internal_only(evidence)
    assert finding.status is AuditStatus.PASS
    cited = {ref.schema_name for ref in finding.evidence_refs}
    assert "whetstone.eval_evidence" in cited
    assert "whetstone.copro_optimizer_config" in cited


def test_best_so_far_judges_the_finalizing_step(copro_run_dir) -> None:
    """A proposing step's ``accepted_candidates`` are entrants, not picks.

    The adapter sets them to the drafts it just minted, before any is
    measured, so judging them as selections would fail every honest run.
    """
    evidence = load_run_evidence(copro_run_dir)
    proposing = evidence.steps[0]
    assert proposing.step.accepted_candidates
    assert proposing.resolved_intents
    assert copro_best_so_far(evidence).status is AuditStatus.PASS


def test_best_so_far_cites_the_rewards_it_compared(copro_run_dir) -> None:
    finding = copro_best_so_far(load_run_evidence(copro_run_dir))
    assert finding.evidence_refs
    for ref in finding.evidence_refs:
        assert ref.schema_name == "whetstone.reward"


# --- Seed retention (whetstone-ai 0.1.16) ----------------------------------


def test_a_seed_retained_run_passes_every_copro_invariant(
    copro_seed_retained_run_dir,
) -> None:
    """The adoption's whole point: an honest retention is not a failure.

    Before 0.1.16 was adopted this run failed three invariants, and
    ``audit_passed=False`` demotes the arm to ``VERDICT_NOT_VALIDATED`` --
    so a run that truthfully reported "nothing beat the seed" would have
    unclaimed the arm.
    """
    report = audit_run(copro_seed_retained_run_dir)
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def test_a_seed_retained_early_terminal_passes_every_invariant(
    copro_seed_retained_early_run_dir,
) -> None:
    """The second emission point: retention short of the configured depth."""
    report = audit_run(copro_seed_retained_early_run_dir)
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def test_the_retained_fixtures_really_are_retentions(
    copro_seed_retained_run_dir, copro_seed_retained_early_run_dir
) -> None:
    """A fixture that lost its retention would pass for the wrong reason."""
    for run_dir in (
        copro_seed_retained_run_dir,
        copro_seed_retained_early_run_dir,
    ):
        evidence = load_run_evidence(run_dir)
        terminal = evidence.steps[-1]
        assert evidence.result.seed_retained
        assert terminal.step.seed_retained
        assert not terminal.step.accepted_candidates
        retained = terminal.step.retained_candidate_ref
        seed = evidence.result.run.record.initial_candidate_ref
        assert retained is not None
        assert seed is not None
        assert retained.record_ref == seed.record_ref


def test_the_early_terminal_fixture_stops_short_of_the_depth(
    copro_seed_retained_early_run_dir, copro_run_dir
) -> None:
    """Otherwise it would not exercise the ``COPRO_DEPTH_STEPS`` widening."""
    early = load_run_evidence(copro_seed_retained_early_run_dir)
    full = load_run_evidence(copro_run_dir)
    assert len(early.steps) < len(full.steps)


def test_best_so_far_checks_a_retention_rather_than_waving_it_through(
    copro_seed_retained_run_dir,
) -> None:
    """The exemption must restate the claim, not retire the invariant.

    A vacuous PASS would disable best-so-far precisely where it does its
    only interesting work -- deciding that nothing beat the starting point
    -- so the finding has to name the seed and the reward it compared.
    """
    finding = copro_best_so_far(load_run_evidence(copro_seed_retained_run_dir))
    assert finding.status is AuditStatus.PASS
    assert "retained its declared seed" in finding.detail
    assert finding.evidence_refs
    assert any(
        ref.schema_name == "whetstone.reward" for ref in finding.evidence_refs
    )


def test_best_so_far_fails_a_retention_that_discarded_a_better_candidate(
    copro_run_dir, tmp_path
) -> None:
    """The negative control for the widened exemption.

    The shape is a structurally perfect retention -- every clause the schema
    enforces still holds -- but the seed scores below a candidate this run
    measured. An exemption that trusted the flag would pass this.
    """
    run_dir = copro_fixtures.retention_keeping_a_candidate_that_lost(
        copro_run_dir, tmp_path / "losing-retention"
    )
    finding = copro_best_so_far(load_run_evidence(run_dir))
    assert finding.status is AuditStatus.FAIL
    assert "discarded a strictly better candidate" in finding.detail


def test_the_widened_exemptions_still_fail_a_genuine_shape_violation(
    copro_seed_retained_run_dir, tmp_path
) -> None:
    """Retention exempts the terminal shape, never the round cardinality.

    An overfilled round carries candidates nobody budgeted for, which no
    retention explains -- so widening the breadth exemption must not have
    made ``seed_retained`` a blanket amnesty for the whole run.
    """
    run_dir = copro_fixtures.over_configured_breadth(
        copro_seed_retained_run_dir, tmp_path / "retained-overfilled"
    )
    finding = _finding(run_dir, InvariantId.COPRO_BREADTH_PER_DEPTH)
    assert finding.status is AuditStatus.FAIL
    assert "more than the configured breadth" in finding.detail


def test_depth_steps_still_fails_a_short_run_that_declared_nothing(
    copro_run_dir, tmp_path
) -> None:
    """The un-widened half: short, with neither failure nor retention."""
    run_dir = copro_fixtures.short_of_configured_depth(
        copro_run_dir, tmp_path / "short-undeclared"
    )
    finding = _finding(run_dir, InvariantId.COPRO_DEPTH_STEPS)
    assert finding.status is AuditStatus.FAIL
    assert "neither a terminal failure nor retention" in finding.detail


def test_distinct_bases_says_so_when_it_had_no_pair_to_check(
    copro_run_dir,
) -> None:
    """A vacuous pass that reads like a checked one is worse than useless.

    At the smallest admissible breadth the seed round proposes one
    candidate, so there is no pair. The finding must say that rather than
    imply it compared something.
    """
    finding = copro_distinct_bases(load_run_evidence(copro_run_dir))
    assert finding.status is AuditStatus.PASS
    assert "no round could duplicate one" in finding.detail


def test_distinct_bases_really_compares_a_multi_draft_round(
    copro_multi_draft_run_dir,
) -> None:
    """The passing case must be a comparison, not an empty loop.

    A ``breadth`` of 3 over two iterations gives a seed round of two drafts
    and a history round of three, so this asserts the invariant actually
    reached multi-draft rounds and cited the proposals it compared.
    """
    evidence = load_run_evidence(copro_multi_draft_run_dir)
    multi_draft = [
        entry
        for entry in evidence.steps
        if len(entry.step.proposed_candidates) >= 2
    ]
    assert multi_draft, "the fixture produced no multi-draft round"

    finding = copro_distinct_bases(evidence)
    assert finding.status is AuditStatus.PASS
    assert "multi-draft rounds is a distinct candidate" in finding.detail
    # One ref per compared proposal, so the citation covers what it judged.
    assert len(finding.evidence_refs) == sum(
        len(entry.step.proposed_candidates) for entry in multi_draft
    )


def test_no_search_evals_reads_every_step(copro_run_dir) -> None:
    evidence = load_run_evidence(copro_run_dir)
    assert all(not entry.search_evidence for entry in evidence.steps)
    assert copro_no_search_evals(evidence).status is AuditStatus.PASS


def test_terminal_provenance_walks_back_to_the_declared_seed(
    copro_run_dir,
) -> None:
    evidence = load_run_evidence(copro_run_dir)
    finding = copro_terminal_provenance(evidence)
    assert finding.status is AuditStatus.PASS
    seed = evidence.result.run.record.initial_candidate_ref
    assert seed is not None
    assert any(
        ref.content_hash == seed.record_ref.content_hash
        for ref in finding.evidence_refs
    )


def test_the_audit_carries_no_task_family_vocabulary() -> None:
    """Section 3.5: an audit needing a family special case is a defect.

    Checked as a property of the module text rather than by inspection,
    because the family names are exactly what a later edit would reach for
    when an invariant is inconvenient to state generally.
    """
    from pathlib import Path

    import whetstone_envs.optim.audit.copro as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for family in ("c19", "c18", "c11"):
        assert family not in source.lower()


# --- Missing evidence fails loudly rather than crashing --------------------


def test_an_unreadable_control_fails_rather_than_raising(
    copro_run_dir, tmp_path
) -> None:
    """A crash would leave a defective run unjudged, not judged failing.

    Three of the seven need the control. Pointing the run at a ref nothing
    resolves is how a truncated or partially-copied artifact arrives in
    practice, and the audit must report on it rather than abort.
    """
    import json

    from whetstone_envs.optim.audit._evidence import RESULT_FILENAME
    from whetstone_envs.optim.audit._mutate import (
        copy_run,
        reseal_run_binding,
    )

    run_dir = copy_run(copro_run_dir, tmp_path / "unreadable-control")
    document = json.loads(
        (run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    )
    document["run"]["record"]["optimizer_config"]["record_ref"][
        "content_hash"
    ] = "0" * 64
    reseal_run_binding(document)
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    report = audit_run(run_dir)
    assert not report.passed
    failed = {
        finding.invariant_id
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    }
    assert failed == {
        InvariantId.COPRO_BREADTH_PER_DEPTH,
        InvariantId.COPRO_DEPTH_STEPS,
        InvariantId.COPRO_INTERNAL_ONLY,
    }
    for invariant in failed:
        assert "does not resolve" in _finding(run_dir, invariant).detail
