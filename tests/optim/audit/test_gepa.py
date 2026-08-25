"""GEPA's eight invariants, each with the fixture that makes it FAIL.

Section 3.2 of the Step 10 assignment makes a failing fixture a shipping
requirement: an invariant with no negative fixture is not yet an invariant.
Every fixture here starts as a *real* fake-transport GEPA run and has
exactly one evidence field rewritten through ``_mutate.py``, so it stays in
the format whetstone actually persists.

Four positive runs cover the four shapes GEPA's evidence takes, because an
invariant that only ever sees one shape is untested for the others:

``gepa_run_dir``
    the ordinary case -- a reflection is accepted, so the pool holds a
    mutated candidate with a recorded parent.
``gepa_seed_retained_run_dir``
    the reflection returns the seed template, so the search finds nothing
    better and the run terminalizes with ``seed_retained``.
``gepa_skipped_run_dir``
    the reflection returns a template missing required placeholders, so the
    format contract rejects it twice and the run records both attempts, the
    second ``exhausted``.
``gepa_multistep_run_dir``
    a larger ceiling, so the run takes four steps rather than two.

Each negative test asserts both that its target invariant FAILs *and* that
no other invariant's status changed -- a mutation that broke everything
would flatter a sloppy invariant.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.core.identity import TypedRef, compute_identity_hash
from whetstone.optim.contracts import TerminalFailure
from whetstone.optim.gepa.contracts import GepaCandidateComponent
from whetstone.optim.gepa.harness_adapter import (
    GEPA_SKIPPED_MUTATIONS_KEY,
    GEPA_TERMINAL_ARTIFACT_KEY,
)
from whetstone.optim.gepa.result_artifact import (
    GEPA_DETAILED_RESULT_RECORD_SCHEMA,
    GEPA_RUN_RESULT_ARTIFACT_SCHEMA,
)
from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY

from whetstone_envs.optim.audit._evidence import (
    GEPA_OPTIMIZER,
    RESULT_FILENAME,
    GepaTerminalEvidence,
    RunEvidence,
    load_run_evidence,
)
from whetstone_envs.optim.audit._mutate import (
    MutationError,
    copy_run,
    mutate_json_field,
    put_record,
    reseal_all,
    reseal_run_binding,
)
from whetstone_envs.optim.audit.gepa import (
    EVALUATE_EFFECT,
    GEPA_INVARIANTS,
    GEPA_MAX_METRIC_CALLS_HYPERPARAMETER,
    PROPOSE_EFFECT,
    SEMANTIC_CANDIDATE_SCHEMA,
    SEMANTIC_CANDIDATE_SCHEMA_VERSION,
    SKIPPED_MUTATION_EXHAUSTED_FIELD,
    SKIPPED_MUTATION_KEY_NAME,
    _as_skip_record,
    gepa_metric_call_budget,
    gepa_repeats_as_recorded,
    gepa_terminal_artifact_present,
    gepa_train_val_disjoint,
)
from whetstone_envs.optim.audit.registry import audit_run, invariants_for
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId

if TYPE_CHECKING:
    from pathlib import Path

#: The state-snapshot and history-snapshot record schemas the runner writes.
#: Re-putting a mutated snapshot must reuse them or the reader will not
#: recognise the record it dereferences.
STATE_SNAPSHOT_SCHEMA = "whetstone.optim_state_snapshot"
HISTORY_SNAPSHOT_SCHEMA = "whetstone.optim_history_snapshot"


# --- helpers ---------------------------------------------------------------


def _read(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))


def _write(run_dir: Path, document: dict[str, Any]) -> Path:
    """Re-seal every integrity ref and write the mutated result back.

    ``_mutate.reseal_all`` covers the request refs, candidate wrappers, and
    snapshot rethreading a GEPA mutation invalidates, so this is only the
    write.
    """
    reseal_all(document)
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    return run_dir


def _mutate_field(
    source: Path,
    destination: Path,
    path: tuple[str | int, ...],
    rewrite,
) -> Path:
    """Copy, violate exactly one named field, and re-seal."""
    run_dir = copy_run(source, destination)
    document = _read(run_dir)
    mutate_json_field(document, path, rewrite)
    return _write(run_dir, document)


def _gepa_terminal(evidence: RunEvidence) -> GepaTerminalEvidence:
    """The run's terminal artifact, asserted present.

    Tests that read the artifact are asserting against a healthy fixture, so
    absence is a broken fixture rather than a case to branch on.
    """
    terminal = evidence.gepa_terminal
    assert terminal is not None
    return terminal


def _statuses(run_dir: Path) -> dict[InvariantId, AuditStatus]:
    return {
        finding.invariant_id: finding.status
        for finding in audit_run(run_dir).findings
    }


def _detail(run_dir: Path, invariant_id: InvariantId) -> str:
    finding = next(
        item
        for item in audit_run(run_dir).findings
        if item.invariant_id is invariant_id
    )
    assert finding.status is AuditStatus.FAIL, finding.detail
    return finding.detail


def assert_only_this_failed(
    healthy: Path,
    mutated: Path,
    invariant_id: InvariantId,
) -> None:
    """The mutation moved exactly one invariant, and moved it to FAIL."""
    before = _statuses(healthy)
    after = _statuses(mutated)
    assert set(before) == set(after)
    changed = {
        invariant
        for invariant, status in after.items()
        if before[invariant] is not status
    }
    assert changed == {invariant_id}, {
        invariant: (before[invariant], after[invariant])
        for invariant in changed | {invariant_id}
    }
    assert after[invariant_id] is AuditStatus.FAIL


def _terminal_step_index(document: dict[str, Any]) -> int:
    return len(document["step_results"]) - 1


def _rewrite_state(
    run_dir: Path,
    document: dict[str, Any],
    step_index: int,
    rewrite,
) -> None:
    """Re-put one step's state snapshot with ``rewrite`` applied.

    The state lives in the store, not in ``result.json``, so a state
    mutation puts a new record and repoints the step's ``state_ref``.
    """
    record = document["step_results"][step_index]["record"]
    assert record["state_ref"]["schema_name"] == STATE_SNAPSHOT_SCHEMA
    current = _load_stored(run_dir, record["state_ref"])
    updated = rewrite(json.loads(json.dumps(current)))
    record["state_ref"] = put_record(run_dir, STATE_SNAPSHOT_SCHEMA, updated)


def _rewrite_history(
    run_dir: Path,
    document: dict[str, Any],
    step_index: int,
    rewrite,
) -> None:
    record = document["step_results"][step_index]["record"]
    assert record["history_ref"]["schema_name"] == HISTORY_SNAPSHOT_SCHEMA
    current = _load_stored(run_dir, record["history_ref"])
    updated = rewrite(json.loads(json.dumps(current)))
    record["history_ref"] = put_record(
        run_dir, HISTORY_SNAPSHOT_SCHEMA, updated
    )


def _load_stored(run_dir: Path, ref: dict[str, str]) -> Any:
    """Dereference a serialized ``TypedRef`` against the run's own store.

    Fixture tooling reads the store directly on purpose: a negative fixture
    is built by rewriting a stored record, which is not something the audit
    itself is ever allowed to do.
    """
    with open_sqlite(str(run_dir / "runtime.sqlite")) as store:
        return store.get(TypedRef.model_validate(ref).reference)


def _rewrite_detailed_result(
    run_dir: Path,
    document: dict[str, Any],
    rewrite,
) -> None:
    """Re-put the terminal artifact with a mutated ``GepaDetailedResult``.

    The artifact addresses the detailed result by content hash, so changing
    the result means re-putting both records and repointing the terminal
    step's history at the new artifact.
    """
    index = _terminal_step_index(document)
    record = document["step_results"][index]["record"]
    history = json.loads(
        json.dumps(_load_stored(run_dir, record["history_ref"]))
    )
    artifact_ref = history[GEPA_TERMINAL_ARTIFACT_KEY]
    artifact = json.loads(json.dumps(_load_stored(run_dir, artifact_ref)))
    detailed = json.loads(
        json.dumps(_load_stored(run_dir, artifact["detailed_result_ref"]))
    )
    artifact["detailed_result_ref"] = put_record(
        run_dir, GEPA_DETAILED_RESULT_RECORD_SCHEMA, rewrite(detailed)
    )
    history[GEPA_TERMINAL_ARTIFACT_KEY] = put_record(
        run_dir, GEPA_RUN_RESULT_ARTIFACT_SCHEMA, artifact
    )
    record["history_ref"] = put_record(
        run_dir, HISTORY_SNAPSHOT_SCHEMA, history
    )


def _rewrite_artifact(
    run_dir: Path, document: dict[str, Any], rewrite
) -> None:
    index = _terminal_step_index(document)
    record = document["step_results"][index]["record"]
    history = json.loads(
        json.dumps(_load_stored(run_dir, record["history_ref"]))
    )
    artifact = json.loads(
        json.dumps(_load_stored(run_dir, history[GEPA_TERMINAL_ARTIFACT_KEY]))
    )
    history[GEPA_TERMINAL_ARTIFACT_KEY] = put_record(
        run_dir, GEPA_RUN_RESULT_ARTIFACT_SCHEMA, rewrite(artifact)
    )
    record["history_ref"] = put_record(
        run_dir, HISTORY_SNAPSHOT_SCHEMA, history
    )


# --- the invariant set -----------------------------------------------------


def test_gepa_registers_exactly_ten_invariants() -> None:
    """Ten: the protocol's eight less F5, plus two later additions.

    F5 deleted GEPA_REFLECTION_MINIBATCH -- nothing persisted witnesses a
    reflection's minibatch size, so it ships not at all rather than as a
    permanent NOT_APPLICABLE, and ``GEPA_TERMINAL_ARTIFACT_PRESENT``
    replaces it. ``GEPA_TRAIN_VAL_DISJOINT`` is the ninth, added with the
    required train/val split, and ``GEPA_REPEATS_AS_RECORDED`` the tenth,
    added when whetstone-ai 0.1.11 let GEPA search at more than one repeat
    and began recording the count it resolved to.
    """
    assert len(GEPA_INVARIANTS) == 10
    assert {invariant.__name__ for invariant in GEPA_INVARIANTS} == {
        "gepa_terminal_artifact_present",
        "gepa_pareto_front",
        "gepa_mutation_traces_to_reflection",
        "gepa_metric_call_budget",
        "gepa_skipped_mutations_recorded",
        "gepa_step_evidence_present",
        "gepa_no_forged_terminal",
        "gepa_platform_resume_identity",
        "gepa_train_val_disjoint",
        "gepa_repeats_as_recorded",
    }


def test_no_reflection_minibatch_invariant_exists() -> None:
    """GEPA ships no reflection-minibatch invariant.

    Scoped to GEPA's own ``GEPA_`` namespace: MIPROv2 ships a genuine
    ``MIPRO_MINIBATCH_SIZING``, so an unscoped scan of ``InvariantId`` would
    fail on another optimizer's shipped invariant rather than on the one
    this guards.
    """
    offenders = [
        member.name
        for member in InvariantId
        if member.name.startswith("GEPA_") and "MINIBATCH" in member.name
    ]
    assert not offenders, (
        "GEPA_REFLECTION_MINIBATCH has no failing fixture and must not ship"
    )


def test_every_gepa_invariant_is_registered(gepa_run_dir) -> None:
    registered = invariants_for(GEPA_OPTIMIZER)
    assert set(GEPA_INVARIANTS) <= set(registered)
    report = audit_run(gepa_run_dir)
    assert report.optimizer == GEPA_OPTIMIZER
    assert len(report.findings) == len(registered)


@pytest.fixture
def overlapping_split_run(gepa_run_dir, tmp_path) -> Path:
    """A control whose valset repeats its trainset.

    This is DSPy's own trainset = valset default, which is exactly what
    the required-partition contract refuses at build time -- so the only
    way to witness the audit catching it is to rewrite the persisted
    control after the fact.
    """
    from whetstone.core.identity import TypedRef
    from whetstone.optim.gepa.control import (
        GEPA_CONTROL_SCHEMA,
        GepaControl,
    )

    run_dir = copy_run(gepa_run_dir, tmp_path / "overlapping-split")
    document = _read(run_dir)
    ref = TypedRef.model_validate(
        document["run"]["record"]["optimizer_config"]["record_ref"]
    )
    with open_sqlite(str(run_dir / "runtime.sqlite")) as store:
        control = GepaControl.model_validate(store.get(ref.reference))
    variant = control.model_copy(
        update={
            "valset_task_hashes": control.trainset_task_hashes,
            "source_valset_task_hashes": control.trainset_task_hashes,
        }
    )
    if variant == control:
        raise MutationError("the control was already overlapping")
    document["run"]["record"]["optimizer_config"] = {
        "record_ref": put_record(
            run_dir, GEPA_CONTROL_SCHEMA, variant.model_dump(mode="json")
        ),
        "record_hash": variant.identity_hash(),
    }
    reseal_run_binding(document)
    return _write(run_dir, document)


def test_an_overlapping_train_val_split_fails(
    overlapping_split_run,
) -> None:
    """Fails-before evidence for the disjointness invariant."""
    statuses = _statuses(overlapping_split_run)
    assert statuses[InvariantId.GEPA_TRAIN_VAL_DISJOINT] is AuditStatus.FAIL
    detail = _detail(
        overlapping_split_run, InvariantId.GEPA_TRAIN_VAL_DISJOINT
    )
    assert "share" in detail


def test_a_faithful_run_passes_the_train_val_invariant(gepa_run_dir) -> None:
    statuses = _statuses(gepa_run_dir)
    assert statuses[InvariantId.GEPA_TRAIN_VAL_DISJOINT] is AuditStatus.PASS


def _evidence_with_intent_tasks(evidence, task_hashes):
    """``evidence`` whose first step resolves one intent over ``task_hashes``.

    Substituted on loaded evidence rather than on disk, for the same reason
    ``test_miprov2.py`` substitutes a control there: in-process GEPA
    dispatches no evaluation intents at all, so a run that resolved one
    over a foreign task cannot be produced as a fixture. A thin view over
    the step swaps exactly the field the predicate reads.
    """
    from dataclasses import replace as replace_dataclass
    from types import SimpleNamespace

    resolution = SimpleNamespace(
        optim_eval_request=SimpleNamespace(task_hashes=tuple(task_hashes))
    )

    class _Entry:
        def __init__(self, entry, intents):
            self._entry = entry
            self._intents = intents

        def __getattr__(self, name):
            return getattr(self._entry, name)

        @property
        def resolved_intents(self):
            return self._intents

    steps = (
        _Entry(evidence.steps[0], (resolution,)),
        *evidence.steps[1:],
    )
    return replace_dataclass(evidence, steps=steps)


def test_an_intent_outside_the_declared_partition_fails(gepa_run_dir) -> None:
    """The second half of the invariant: the run must stay inside the split.

    The control's own two sets being disjoint is not enough. GEPA reflects
    on the trainset and scores its frontier on the valset, so an evaluation
    that reached a task in *neither* scored something the declared
    partition never admitted -- the fan-out failure seen from the split's
    side, and the reason the invariant checks the run as well as the
    control.

    Fails-before evidence for that branch specifically: it had no negative
    test, because no fixture can reach it.
    """
    evidence = load_run_evidence(gepa_run_dir)
    finding = gepa_train_val_disjoint(
        _evidence_with_intent_tasks(evidence, ("f" * 64,))
    )
    assert finding.status is AuditStatus.FAIL, finding.detail
    assert "outside the declared" in finding.detail
    assert finding.evidence_refs


def test_an_intent_inside_the_declared_partition_passes(gepa_run_dir) -> None:
    """The same substitution with an admitted task must not fail.

    Without this, the test above would pass on a predicate that failed
    every resolved intent regardless of which tasks it named.
    """
    from whetstone.optim.gepa.control import GepaControl

    evidence = load_run_evidence(gepa_run_dir)
    control = GepaControl.model_validate(evidence.control_record)
    finding = gepa_train_val_disjoint(
        _evidence_with_intent_tasks(evidence, control.trainset_task_hashes[:1])
    )
    assert finding.status is AuditStatus.PASS, finding.detail


def test_gepa_invariant_wire_values_are_pinned() -> None:
    """``audit.json`` is cited by content hash, so these are stored identity.

    Pinned literally rather than derived from the member names: deriving
    them is exactly the drift that would silently rewrite what a study
    manifest already cites.
    """
    assert {
        member.name: member.value
        for member in InvariantId
        if member.name.startswith("GEPA_")
    } == {
        "GEPA_TERMINAL_ARTIFACT_PRESENT": "gepa_terminal_artifact_present",
        "GEPA_PARETO_FRONT": "gepa_pareto_front",
        "GEPA_MUTATION_TRACES_TO_REFLECTION": (
            "gepa_mutation_traces_to_reflection"
        ),
        "GEPA_METRIC_CALL_BUDGET": "gepa_metric_call_budget",
        "GEPA_SKIPPED_MUTATIONS_RECORDED": "gepa_skipped_mutations_recorded",
        "GEPA_STEP_EVIDENCE_PRESENT": "gepa_step_evidence_present",
        "GEPA_NO_FORGED_TERMINAL": "gepa_no_forged_terminal",
        "GEPA_PLATFORM_RESUME_IDENTITY": "gepa_platform_resume_identity",
        "GEPA_TRAIN_VAL_DISJOINT": "gepa_train_val_disjoint",
        "GEPA_REPEATS_AS_RECORDED": "gepa_repeats_as_recorded",
    }


def test_repeats_as_recorded_fails_on_an_overstated_repeat_count(
    gepa_run_dir, tmp_path
) -> None:
    """A detailed result claiming more repeats than the search bought.

    ``validation_num_seeds`` is what an envs manifest is diffed against, and
    nothing else re-derives it: a GEPA metric call is one candidate-task
    evaluation at any repeat count, so repeats move provider rows without
    moving ``total_metric_calls``. A run that searched at one repeat under a
    design registering three is therefore indistinguishable in the budget
    the ceiling is stated in, and only this comparison catches it.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "overstated-repeats")
    document = _read(run_dir)
    recorded = _gepa_terminal(load_run_evidence(gepa_run_dir))
    before = recorded.detailed_result.validation_num_seeds

    _rewrite_detailed_result(
        run_dir,
        document,
        lambda detailed: {**detailed, "validation_num_seeds": before + 1},
    )
    mutated = _write(run_dir, document)

    assert f"not the recorded {before + 1}" in _detail(
        mutated, InvariantId.GEPA_REPEATS_AS_RECORDED
    )
    assert_only_this_failed(
        gepa_run_dir, mutated, InvariantId.GEPA_REPEATS_AS_RECORDED
    )


def test_repeats_as_recorded_passes_on_the_run_as_issued(gepa_run_dir) -> None:
    """The same predicate on the unmutated run must PASS.

    Without this the test above would pass on a predicate that failed every
    run regardless of what its detailed result recorded.
    """
    finding = gepa_repeats_as_recorded(load_run_evidence(gepa_run_dir))
    assert finding.status is AuditStatus.PASS, finding.detail
    assert finding.evidence_refs


def test_persisted_read_path_literals_are_pinned() -> None:
    """These are whetstone's stored spellings, not names we may rename."""
    assert SEMANTIC_CANDIDATE_SCHEMA == "whetstone.gepa.semantic_candidate"
    assert SEMANTIC_CANDIDATE_SCHEMA_VERSION == 1
    assert GEPA_MAX_METRIC_CALLS_HYPERPARAMETER == "max_metric_calls"
    assert EVALUATE_EFFECT == "evaluate"
    assert PROPOSE_EFFECT == "propose"
    #: The audit's own spelling of the skip key must equal the reader's
    #: imported constant, or a drift would make every skip look absent.
    assert SKIPPED_MUTATION_KEY_NAME == GEPA_SKIPPED_MUTATIONS_KEY
    assert GEPA_TERMINAL_ARTIFACT_KEY == "terminal_artifact_ref"
    assert GEPA_STATE_KEY == "gepa_checkpoint"


# --- the four healthy runs -------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    [
        "gepa_run_dir",
        "gepa_seed_retained_run_dir",
        "gepa_skipped_run_dir",
        "gepa_multistep_run_dir",
    ],
)
def test_every_healthy_run_shape_passes(fixture_name, request) -> None:
    """An audit that fires on a correct run is worse than no audit."""
    run_dir = request.getfixturevalue(fixture_name)
    report = audit_run(run_dir)
    assert report.passed, [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


def test_the_ordinary_run_accepted_a_traced_mutation(gepa_run_dir) -> None:
    """The positive fixture is not vacuous: it really mutated a candidate."""
    evidence = load_run_evidence(gepa_run_dir)
    detailed = _gepa_terminal(evidence).detailed_result
    assert len(detailed.candidates) > 1
    assert any(
        parent is not None
        for parents in detailed.parents
        for parent in parents
    )
    assert evidence.result.seed_retained is False


def test_the_skipped_run_really_recorded_rejections(
    gepa_skipped_run_dir,
) -> None:
    """The skip fixture exercises the recorded path, not an empty one."""
    evidence = load_run_evidence(gepa_skipped_run_dir)
    skips = [
        record
        for entry in evidence.steps
        for record in entry.gepa_skipped_mutations()
    ]
    assert skips
    narrowed = [_as_skip_record(record) for record in skips]
    assert any(
        record is not None and record[SKIPPED_MUTATION_EXHAUSTED_FIELD] is True
        for record in narrowed
    )


def test_the_multistep_run_took_more_than_two_steps(
    gepa_multistep_run_dir,
) -> None:
    assert len(load_run_evidence(gepa_multistep_run_dir).steps) > 2


# --- 1 · GEPA_TERMINAL_ARTIFACT_PRESENT ------------------------------------


@pytest.fixture
def no_terminal_artifact_run(gepa_run_dir, tmp_path) -> Path:
    """A run whose terminal step's history dropped the artifact key.

    Without it the search result is unreachable, so nothing downstream can
    be judged -- which is exactly why this invariant is the precondition.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "no-terminal-artifact")
    document = _read(run_dir)
    _rewrite_history(
        run_dir,
        document,
        _terminal_step_index(document),
        lambda history: {
            key: value
            for key, value in history.items()
            if key != GEPA_TERMINAL_ARTIFACT_KEY
        },
    )
    return _write(run_dir, document)


def test_a_missing_terminal_artifact_fails(no_terminal_artifact_run) -> None:
    statuses = _statuses(no_terminal_artifact_run)
    assert (
        statuses[InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT]
        is AuditStatus.FAIL
    )
    assert "no step's history carries a GEPA terminal artifact" in _detail(
        no_terminal_artifact_run, InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT
    )


def test_a_missing_artifact_fails_every_dependent_rather_than_crashing(
    no_terminal_artifact_run,
) -> None:
    """FAIL, never crash, and never a flattering NOT_APPLICABLE.

    Losing the artifact loses the whole search record, so every GEPA
    invariant that reads *through* the artifact reports the defect. This
    is the one mutation permitted to move more than one invariant,
    because the precondition is what those others read through: reporting
    them PASS on absent evidence would be the fabrication this package
    exists to prevent.

    ``gepa_train_val_disjoint`` is exempt: it reads the persisted control,
    not the terminal artifact, so a lost artifact genuinely does not stop
    it from judging the declared partition. Asserting it FAILs here would
    be asserting a dependency it does not have.
    """
    statuses = _statuses(no_terminal_artifact_run)
    artifact_dependent = tuple(
        invariant
        for invariant in GEPA_INVARIANTS
        if invariant.__name__ != "gepa_train_val_disjoint"
    )
    assert len(artifact_dependent) == len(GEPA_INVARIANTS) - 1
    for invariant in artifact_dependent:
        member = InvariantId(invariant.__name__)
        assert statuses[member] is AuditStatus.FAIL, member
    assert statuses[InvariantId.GEPA_TRAIN_VAL_DISJOINT] is AuditStatus.PASS, (
        "the split invariant reads the control, not the artifact"
    )
    assert (
        statuses[InvariantId.REPORTED_NUMBERS_RESOLVE] is AuditStatus.PASS
    ), "the shared invariant reads none of the GEPA artifact"


@pytest.fixture
def foreign_terminal_artifact_run(gepa_run_dir, tmp_path) -> Path:
    """A terminal artifact whose detailed result names another control.

    A neighbouring run's artifact in a shared store would otherwise let
    every downstream invariant pass against another run's search.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "foreign-terminal-artifact")
    document = _read(run_dir)
    _rewrite_detailed_result(
        run_dir,
        document,
        lambda detailed: {**detailed, "control_identity_hash": "b" * 64},
    )
    return _write(run_dir, document)


def test_an_artifact_from_another_control_fails(
    gepa_run_dir, foreign_terminal_artifact_run
) -> None:
    assert "does not equal the artifact's" in _detail(
        foreign_terminal_artifact_run,
        InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT,
    )
    assert_only_this_failed(
        gepa_run_dir,
        foreign_terminal_artifact_run,
        InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT,
    )


# --- 2 · GEPA_PARETO_FRONT -------------------------------------------------


@pytest.fixture
def dominated_front_run(gepa_run_dir, tmp_path) -> Path:
    """A front naming a candidate that is dominated on that instance.

    This is the infidelity the invariant exists for: the pool is presented
    as a Pareto front, but the recorded front is not the per-instance argmax
    over the scores it claims to be derived from.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "dominated-front")
    document = _read(run_dir)

    def rewrite(detailed: dict[str, Any]) -> dict[str, Any]:
        front = dict(detailed["per_val_instance_best_candidates"])
        instance = min(front)
        # Name candidate 0 -- the seed, scored strictly below the winner on
        # this instance -- as the instance's best.
        front[instance] = [0]
        return {**detailed, "per_val_instance_best_candidates": front}

    _rewrite_detailed_result(run_dir, document, rewrite)
    return _write(run_dir, document)


def test_a_front_that_is_not_the_argmax_fails(
    gepa_run_dir, dominated_front_run
) -> None:
    assert "argmax" in _detail(
        dominated_front_run, InvariantId.GEPA_PARETO_FRONT
    )
    assert_only_this_failed(
        gepa_run_dir, dominated_front_run, InvariantId.GEPA_PARETO_FRONT
    )


@pytest.fixture
def selection_off_the_front_run(gepa_run_dir, tmp_path) -> Path:
    """A run whose selected candidate is dominated on every instance."""
    run_dir = copy_run(gepa_run_dir, tmp_path / "selection-off-front")
    document = _read(run_dir)
    _rewrite_detailed_result(
        run_dir,
        document,
        lambda detailed: {**detailed, "best_idx": 0},
    )
    return _write(run_dir, document)


def test_selecting_a_dominated_candidate_fails(
    selection_off_the_front_run,
) -> None:
    """Selection must come from the front, not from the discovery history.

    Repointing ``best_idx`` also changes which candidate the terminal step
    claims to have returned, so ``GEPA_NO_FORGED_TERMINAL`` fails too --
    correctly: both statements are false about this artifact.
    """
    statuses = _statuses(selection_off_the_front_run)
    assert statuses[InvariantId.GEPA_PARETO_FRONT] is AuditStatus.FAIL
    assert "dominated on every instance" in _detail(
        selection_off_the_front_run, InvariantId.GEPA_PARETO_FRONT
    )


# --- 3 · GEPA_MUTATION_TRACES_TO_REFLECTION --------------------------------


@pytest.fixture
def untraced_mutation_run(gepa_run_dir, tmp_path) -> Path:
    """A mutated candidate whose recorded parent no reflection touched.

    Rewriting the seed's text changes its semantic identity, so the propose
    effect recorded against the original seed no longer matches the parent
    the pool now names -- the mutation has no reflection to trace to.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "untraced-mutation")
    document = _read(run_dir)

    def rewrite(detailed: dict[str, Any]) -> dict[str, Any]:
        candidates = [dict(candidate) for candidate in detailed["candidates"]]
        component = next(iter(candidates[0]))
        candidates[0][component] = (
            candidates[0][component] + "\n(unrecorded edit)"
        )
        return {**detailed, "candidates": candidates}

    _rewrite_detailed_result(run_dir, document, rewrite)
    return _write(run_dir, document)


def test_a_mutation_with_no_reflection_fails(untraced_mutation_run) -> None:
    assert "no propose effect reflected over it" in _detail(
        untraced_mutation_run, InvariantId.GEPA_MUTATION_TRACES_TO_REFLECTION
    )


@pytest.fixture
def unknown_parent_run(gepa_run_dir, tmp_path) -> Path:
    """A candidate descending from a parent index that does not exist."""
    run_dir = copy_run(gepa_run_dir, tmp_path / "unknown-parent")
    document = _read(run_dir)

    def rewrite(detailed: dict[str, Any]) -> dict[str, Any]:
        parents = [list(row) for row in detailed["parents"]]
        parents[1] = [99]
        return {**detailed, "parents": parents}

    _rewrite_detailed_result(run_dir, document, rewrite)
    return _write(run_dir, document)


def test_a_candidate_naming_an_unknown_parent_fails(
    gepa_run_dir, unknown_parent_run
) -> None:
    assert "not one of the" in _detail(
        unknown_parent_run, InvariantId.GEPA_MUTATION_TRACES_TO_REFLECTION
    )
    assert_only_this_failed(
        gepa_run_dir,
        unknown_parent_run,
        InvariantId.GEPA_MUTATION_TRACES_TO_REFLECTION,
    )


def test_the_semantic_hash_matches_what_the_recorder_stamped(
    gepa_run_dir,
) -> None:
    """The audit recomputes whetstone's own candidate identity, not its own.

    If this drifts, the mutation-provenance invariant would silently report
    every accepted mutation as untraced.
    """
    evidence = load_run_evidence(gepa_run_dir)
    detailed = _gepa_terminal(evidence).detailed_result
    stamped = {
        entry.semantic_candidate_identity_hash
        for entry in _gepa_terminal(evidence).effect_transcript.entries
    }
    recomputed = {
        compute_identity_hash(
            schema=SEMANTIC_CANDIDATE_SCHEMA,
            schema_version=SEMANTIC_CANDIDATE_SCHEMA_VERSION,
            payload=[
                GepaCandidateComponent(name=name, text=text).model_dump(
                    mode="json"
                )
                for name, text in candidate.items()
            ],
        )
        for candidate in detailed.candidates
    }
    assert stamped <= recomputed


# --- 4 · GEPA_METRIC_CALL_BUDGET -------------------------------------------


@pytest.fixture
def continued_past_ceiling_run(gepa_run_dir, tmp_path) -> Path:
    """A run whose ceiling is below what its terminal step consumed.

    Lowering the advertised ceiling to 1 makes the first step -- which
    consumed 2 -- a step that continued at or past the ceiling instead of
    terminalizing, which is the overspend this invariant exists to catch.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "continued-past-ceiling")
    return _mutate_field(
        run_dir,
        tmp_path / "continued-past-ceiling-mutated",
        (
            "step_results",
            0,
            "record",
            "request",
            "record",
            "hyperparameters",
            GEPA_MAX_METRIC_CALLS_HYPERPARAMETER,
        ),
        lambda _value: 1,
    )


def test_continuing_past_the_ceiling_fails(continued_past_ceiling_run) -> None:
    detail = _detail(
        continued_past_ceiling_run, InvariantId.GEPA_METRIC_CALL_BUDGET
    )
    assert "ceiling" in detail


@pytest.fixture
def counter_moved_backwards_run(gepa_run_dir, tmp_path) -> Path:
    """A later step reporting fewer metric calls than an earlier one.

    A resumed run whose counter regressed would re-spend a prefix it
    already paid for.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "counter-backwards")
    document = _read(run_dir)
    _rewrite_state(
        run_dir,
        document,
        0,
        lambda state: {
            **state,
            GEPA_STATE_KEY: {
                **state[GEPA_STATE_KEY],
                "metric_calls_consumed": 99,
            },
        },
    )
    return _write(run_dir, document)


def test_a_counter_that_moved_backwards_fails(
    counter_moved_backwards_run,
) -> None:
    detail = _detail(
        counter_moved_backwards_run, InvariantId.GEPA_METRIC_CALL_BUDGET
    )
    assert "below step" in detail or "continued at" in detail


@pytest.fixture
def unaccounted_total_run(gepa_run_dir, tmp_path) -> Path:
    """A detailed result reporting a total no checkpoint accounts for."""
    run_dir = copy_run(gepa_run_dir, tmp_path / "unaccounted-total")
    document = _read(run_dir)
    _rewrite_detailed_result(
        run_dir,
        document,
        lambda detailed: {
            **detailed,
            "total_metric_calls": detailed["total_metric_calls"] + 40,
        },
    )
    return _write(run_dir, document)


def test_an_unaccounted_metric_call_total_fails(
    gepa_run_dir, unaccounted_total_run
) -> None:
    assert "accounts for" in _detail(
        unaccounted_total_run, InvariantId.GEPA_METRIC_CALL_BUDGET
    )
    assert_only_this_failed(
        gepa_run_dir,
        unaccounted_total_run,
        InvariantId.GEPA_METRIC_CALL_BUDGET,
    )


def test_the_ceiling_is_not_checked_against_the_outcome(gepa_run_dir) -> None:
    """The measured overshoot must not read as a violation.

    ``run_one_gepa_iteration`` requests ``min(ceiling, consumed + 1)`` and
    upstream finishes the iteration it started, so a healthy run's
    ``total_metric_calls`` exceeds the ceiling whenever the ceiling is not a
    multiple of a full valset pass. This pins that the smoke fixture really
    does overshoot, so the restatement is load-bearing rather than
    theoretical.
    """
    evidence = load_run_evidence(gepa_run_dir)
    ceiling = dict(evidence.steps[0].step.request.record.hyperparameters)[
        GEPA_MAX_METRIC_CALLS_HYPERPARAMETER
    ]
    assert isinstance(ceiling, int)
    total = _gepa_terminal(evidence).detailed_result.total_metric_calls
    assert total is not None
    assert total > ceiling
    assert audit_run(gepa_run_dir).passed


# --- 5 · GEPA_SKIPPED_MUTATIONS_RECORDED -----------------------------------


@pytest.fixture
def skip_key_dropped_run(gepa_skipped_run_dir, tmp_path) -> Path:
    """A step that persisted GEPA state but no skipped-mutation key.

    An absent key is indistinguishable from "nothing was rejected", so the
    key must be present on every step even when empty.
    """
    run_dir = copy_run(gepa_skipped_run_dir, tmp_path / "skip-key-dropped")
    document = _read(run_dir)
    _rewrite_state(
        run_dir,
        document,
        0,
        lambda state: {
            key: value
            for key, value in state.items()
            if key != GEPA_SKIPPED_MUTATIONS_KEY
        },
    )
    return _write(run_dir, document)


def test_a_step_without_the_skip_key_fails(
    gepa_skipped_run_dir, skip_key_dropped_run
) -> None:
    assert "without a" in _detail(
        skip_key_dropped_run, InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED
    )
    assert_only_this_failed(
        gepa_skipped_run_dir,
        skip_key_dropped_run,
        InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED,
    )


@pytest.fixture
def skip_lost_from_step_state_run(gepa_skipped_run_dir, tmp_path) -> Path:
    """A rejection the transcript reports that no step's state carries.

    This is the caveat the ``GepaSkippedMutation`` docstring states: the
    run-level record is the union over every step's state, so a skip present
    only in the terminal transcript means a step lost its durable copy.
    """
    run_dir = copy_run(gepa_skipped_run_dir, tmp_path / "skip-lost-from-state")
    document = _read(run_dir)
    _rewrite_state(
        run_dir,
        document,
        _terminal_step_index(document),
        lambda state: {**state, GEPA_SKIPPED_MUTATIONS_KEY: []},
    )
    return _write(run_dir, document)


def test_a_skip_missing_from_every_steps_state_fails(
    gepa_skipped_run_dir, skip_lost_from_step_state_run
) -> None:
    assert "no step's state carries" in _detail(
        skip_lost_from_step_state_run,
        InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED,
    )
    assert_only_this_failed(
        gepa_skipped_run_dir,
        skip_lost_from_step_state_run,
        InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED,
    )


@pytest.fixture
def illegible_skip_run(gepa_skipped_run_dir, tmp_path) -> Path:
    """A rejection recorded without the fields that make it legible."""
    run_dir = copy_run(gepa_skipped_run_dir, tmp_path / "illegible-skip")
    document = _read(run_dir)

    def rewrite(state: dict[str, Any]) -> dict[str, Any]:
        skips = [dict(record) for record in state[GEPA_SKIPPED_MUTATIONS_KEY]]
        skips[0].pop("exhausted")
        return {**state, GEPA_SKIPPED_MUTATIONS_KEY: skips}

    _rewrite_state(run_dir, document, _terminal_step_index(document), rewrite)
    return _write(run_dir, document)


def test_a_skip_missing_the_exhausted_flag_fails(illegible_skip_run) -> None:
    """Only ``exhausted=True`` means a mutation was dropped.

    Without the flag the run's dropped-mutation count is unknowable, so the
    rejection record is not doing its job.
    """
    assert "omits exhausted" in _detail(
        illegible_skip_run, InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED
    )


# --- 6 · GEPA_STEP_EVIDENCE_PRESENT ----------------------------------------


@pytest.fixture
def paying_step_without_evidence_run(gepa_run_dir, tmp_path) -> Path:
    """A step whose counter advanced but which shows nothing for it."""
    run_dir = copy_run(gepa_run_dir, tmp_path / "paying-step-no-evidence")
    document = _read(run_dir)
    document["step_results"][0]["record"]["search_evidence"] = []
    return _write(run_dir, document)


def test_a_paying_step_with_no_evidence_fails(
    paying_step_without_evidence_run,
) -> None:
    assert "carries neither a resolved intent nor search evidence" in _detail(
        paying_step_without_evidence_run,
        InvariantId.GEPA_STEP_EVIDENCE_PRESENT,
    )


def test_a_pure_replay_step_is_exempt_not_failed(
    gepa_run_dir, tmp_path
) -> None:
    """The F9 restatement: a step that bought nothing owes nothing.

    Freezing step 0's counter at step 1's value makes step 1 a pure-replay
    step. Stripping its evidence must then be exempt rather than a failure,
    which is the whole point of keying the predicate to the counter instead
    of to ``budget_delta`` -- ``GepaStepCheckpoint.budget_delta`` reports
    ``metric_calls: 1`` for every non-degenerate step and would exempt
    nothing.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "pure-replay")
    document = _read(run_dir)
    terminal = _terminal_step_index(document)
    consumed = _load_stored(
        run_dir, document["step_results"][terminal]["record"]["state_ref"]
    )[GEPA_STATE_KEY]["metric_calls_consumed"]
    _rewrite_state(
        run_dir,
        document,
        0,
        lambda state: {
            **state,
            GEPA_STATE_KEY: {
                **state[GEPA_STATE_KEY],
                "metric_calls_consumed": consumed,
            },
        },
    )
    document["step_results"][terminal]["record"]["search_evidence"] = []
    _write(run_dir, document)

    statuses = _statuses(run_dir)
    assert statuses[InvariantId.GEPA_STEP_EVIDENCE_PRESENT] is AuditStatus.PASS
    finding = next(
        item
        for item in audit_run(run_dir).findings
        if item.invariant_id is InvariantId.GEPA_STEP_EVIDENCE_PRESENT
    )
    assert "pure-replay" in finding.detail


# --- 7 · GEPA_NO_FORGED_TERMINAL -------------------------------------------


@pytest.fixture
def forged_terminal_run(gepa_run_dir, tmp_path) -> Path:
    """A terminal candidate whose text is in no candidate the run searched.

    This is the forgery the invariant names: a prompt returned as the run's
    result that its own recorded search never produced.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "forged-terminal")
    document = _read(run_dir)
    # GEPA returns the same candidate as proposed, accepted, and the run's
    # proposal. ``OptimStepResult`` requires the accepted multiset to be
    # contained in the proposed one, and ``OptimResult`` requires its
    # proposals to derive from the accepted candidates, so the injection has
    # to land on all three or the artifact never reaches an invariant.
    injected = "\n(injected after the search)"
    for field in ("proposed_candidates", "accepted_candidates"):
        mutate_json_field(
            document,
            (
                "step_results",
                1,
                "record",
                field,
                0,
                "record",
                "payload",
                "prompt_template",
            ),
            lambda text: text + injected,
        )
    mutate_json_field(
        document,
        ("proposals", 0, "candidate", "record", "payload", "prompt_template"),
        lambda text: text + injected,
    )
    return _write(run_dir, document)


def test_a_terminal_candidate_the_search_never_produced_fails(
    gepa_run_dir, forged_terminal_run
) -> None:
    assert "is not the text of selected candidate" in _detail(
        forged_terminal_run, InvariantId.GEPA_NO_FORGED_TERMINAL
    )
    assert_only_this_failed(
        gepa_run_dir, forged_terminal_run, InvariantId.GEPA_NO_FORGED_TERMINAL
    )


def test_a_dishonest_seed_retention_is_structurally_unrepresentable(
    gepa_seed_retained_run_dir, tmp_path
) -> None:
    """This clause has no negative fixture, and cannot have one.

    ``OptimStepResult._validate`` (``optim/contracts.py:1166-1193``) already
    refuses a ``seed_retained`` step whose ``retained_candidate_ref`` is not
    the run's exact ``initial_candidate_ref``. Rewriting the retained
    candidate to anything else does not produce a run that FAILs the audit
    -- it produces an artifact whetstone refuses to validate at all.

    Recording that here rather than shipping a fixture that quietly tests
    something else: the invariant keeps the clause as defence in depth
    against an upstream weakening, and its live-draft clauses carry the
    fixtures.
    """
    with pytest.raises(MutationError, match="retain the exact run"):
        _mutate_field(
            gepa_seed_retained_run_dir,
            tmp_path / "dishonest-retention",
            (
                "step_results",
                1,
                "record",
                "retained_candidate_ref",
                "record",
                "payload",
                "prompt_template",
            ),
            lambda text: f"{text}\n(not the seed after all)",
        )


@pytest.fixture
def parentless_terminal_run(gepa_run_dir, tmp_path) -> Path:
    """A selected candidate the search records no parent for.

    A candidate in the pool with no recorded parent was never produced by a
    reflection in this run, so returning it as the terminal result is the
    forgery this clause names.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "parentless-terminal")
    document = _read(run_dir)

    def rewrite(detailed: dict[str, Any]) -> dict[str, Any]:
        parents = [list(row) for row in detailed["parents"]]
        parents[detailed["best_idx"]] = [None]
        return {**detailed, "parents": parents}

    _rewrite_detailed_result(run_dir, document, rewrite)
    return _write(run_dir, document)


def test_a_terminal_candidate_with_no_recorded_parent_fails(
    parentless_terminal_run,
) -> None:
    assert "records no parent" in _detail(
        parentless_terminal_run, InvariantId.GEPA_NO_FORGED_TERMINAL
    )


def test_a_terminal_candidate_matching_a_probe_is_not_itself_a_failure(
    gepa_run_dir,
) -> None:
    """Provenance, not text: the fake transport's draft *is* the ceiling probe.

    A textual check against ``PROBES.ceiling_template`` would fail every
    healthy CI run while still missing a forgery whose text differed. This
    pins that the fixture really does return the ceiling probe and still
    passes.
    """
    from whetstone_envs.c19 import PROBES

    evidence = load_run_evidence(gepa_run_dir)
    accepted = evidence.steps[-1].step.accepted_candidates[0].record
    assert accepted.payload["prompt_template"] == PROBES.ceiling_template
    assert audit_run(gepa_run_dir).passed


# --- 8 · GEPA_PLATFORM_RESUME_IDENTITY -------------------------------------


@pytest.fixture
def deferred_run_that_failed(gepa_run_dir, tmp_path) -> Path:
    """A deferral episode that did not reach a clean terminal result.

    ``StepStatus`` has no deferred member; a GEPA deferral returns
    ``CONTINUE`` carrying the persisted intent, and in-process GEPA emits no
    intents at all, so a step carrying a ``resolved_intents`` entry *is* a
    deferral episode. This fixture promotes one of the run's own recorded
    search-evidence entries into that intent shape -- the same candidate,
    the same eval evidence, the same reward -- and then records a terminal
    failure, which is what a resume that could not replay its paid prefix
    looks like.
    """
    run_dir = copy_run(gepa_run_dir, tmp_path / "deferred-failed")
    document = _read(run_dir)
    step = document["step_results"][0]["record"]
    search = step["search_evidence"][0]
    # The resolution must cite one of the step's own request candidates
    # (``OptimStepResult._validate``), which for GEPA step 0 is the seed.
    candidate = step["request"]["record"]["candidates"][0]
    step["resolved_intents"] = [
        {
            "schema_version": -1,
            "optim_eval_request": {
                "optim_run_id": search["optim_run_id"],
                "optim_step_index": search["optim_step_index"],
                "eval_request": {
                    "request_id": search["eval_request_id"],
                    "candidate": candidate,
                },
                "expected_reward_policy_hash": _reward_policy_hash(document),
                "task_hashes": None,
            },
            "outcome": "completed",
            "detail": {
                "classification": "measured",
                "message": "replayed from the effect cache after deferral",
            },
            "eval_result_ref": search["eval_result_ref"],
            "reward_evidence_refs": search["reward_evidence_refs"],
            "resolved_eval_config": _eval_config(run_dir, document),
            "reward_ref": search["reward_ref"],
        }
    ]
    failure = {
        "code": "gepa_effect_conflict",
        "message": "GEPA replay changed the semantic effect",
    }
    # ``OptimResult`` requires its failure to match the final Step Result's,
    # so a resume that could not replay fails at both levels, as it would in
    # a real run.
    document["terminal_failure"] = failure
    final = document["step_results"][-1]["record"]
    final["terminal_failure"] = failure
    final["status"] = "failed"
    final["accepted_candidates"] = []
    final["proposed_candidates"] = []
    document["proposals"] = []
    return _write(run_dir, document)


def _reward_policy_hash(document: dict[str, Any]) -> str:
    return dict(
        document["step_results"][0]["record"]["request"]["record"][
            "hyperparameters"
        ]
    )["reward_policy_hash"]


def _eval_config(run_dir: Path, document: dict[str, Any]) -> dict[str, Any]:
    """The run's own internal Eval Config ref, taken from its GEPA control."""
    config = document["run"]["record"]["optimizer_config"]["record_ref"]
    control = _load_stored(run_dir, config)
    return control["metric"]


def test_a_deferral_episode_is_recognised_and_judged(
    deferred_run_that_failed,
) -> None:
    """NOT_APPLICABLE must be conditional on dispatch, not permanent.

    An invariant that is always NOT_APPLICABLE for its own optimizer is a
    defect -- that is what killed ``GEPA_REFLECTION_MINIBATCH``. This
    fixture reaches the checks, so the status is earned rather than assumed.
    """
    statuses = _statuses(deferred_run_that_failed)
    assert (
        statuses[InvariantId.GEPA_PLATFORM_RESUME_IDENTITY] is AuditStatus.FAIL
    )
    assert "terminal failure" in _detail(
        deferred_run_that_failed, InvariantId.GEPA_PLATFORM_RESUME_IDENTITY
    )


def test_an_in_process_run_reports_not_applicable_with_a_reason(
    gepa_run_dir,
) -> None:
    finding = next(
        item
        for item in audit_run(gepa_run_dir).findings
        if item.invariant_id is InvariantId.GEPA_PLATFORM_RESUME_IDENTITY
    )
    assert finding.status is AuditStatus.NOT_APPLICABLE
    assert "dispatched in process" in finding.detail
    assert finding.evidence_refs


# --- evidence citation -----------------------------------------------------


def test_every_gepa_finding_cites_the_records_it_read(gepa_run_dir) -> None:
    """A finding with no evidence ref cannot be checked by a reader."""
    evidence = load_run_evidence(gepa_run_dir)
    for invariant in GEPA_INVARIANTS:
        finding = invariant(evidence)
        assert finding.evidence_refs, finding.invariant_id
        for ref in finding.evidence_refs:
            assert len(ref.content_hash) == 64


def test_the_precondition_cites_the_whole_read_chain(gepa_run_dir) -> None:
    finding = gepa_terminal_artifact_present(load_run_evidence(gepa_run_dir))
    schemas = {ref.schema_name for ref in finding.evidence_refs}
    assert schemas == {
        GEPA_RUN_RESULT_ARTIFACT_SCHEMA,
        GEPA_DETAILED_RESULT_RECORD_SCHEMA,
        "whetstone.gepa.effect_transcript",
    }


# --- 4b · the declared-terminal-failure exemption ---------------------------


def _with_declared_failure(evidence: RunEvidence, message: str) -> RunEvidence:
    """The same evidence, with a terminal failure declared on the result.

    Substituted onto the loaded evidence rather than baked into a fixture
    on disk: ``OptimResult`` requires a failed run to claim no proposals
    at all, so a result.json that both searched and failed cannot be
    built by mutating a healthy run. What this invariant reads is the
    declaration, and this puts exactly that in place.
    """
    failed = evidence.result.model_copy(
        update={
            "terminal_failure": TerminalFailure(
                code="provider_unavailable",
                message=message,
                details={},
            )
        }
    )
    return dataclasses.replace(evidence, result=failed)


def _raised_ceiling_evidence(gepa_run_dir, tmp_path, ceiling: int):
    """Evidence whose advertised ceiling sits above what it consumed."""
    run_dir = copy_run(gepa_run_dir, tmp_path / f"ceiling-{ceiling}")
    document = _read(run_dir)
    for step in document["step_results"]:
        hyper = step["record"]["request"]["record"]["hyperparameters"]
        hyper[GEPA_MAX_METRIC_CALLS_HYPERPARAMETER] = ceiling
    _write(run_dir, document)
    return load_run_evidence(run_dir)


def test_terminalizing_below_the_ceiling_fails_when_nothing_is_declared(
    gepa_run_dir, tmp_path
) -> None:
    """The check the exemption sits inside still has to bite.

    A ceiling far above what the run consumed makes its terminal step a
    below-ceiling terminalization. Undeclared, that is exactly the silent
    truncation this invariant exists to catch.
    """
    evidence = _raised_ceiling_evidence(gepa_run_dir, tmp_path, 10_000)
    assert evidence.result.terminal_failure is None

    finding = gepa_metric_call_budget(evidence)
    assert finding.status is AuditStatus.FAIL
    assert "below the ceiling" in finding.detail


def test_a_declared_terminal_failure_exempts_the_short_stop(
    gepa_run_dir, tmp_path
) -> None:
    """**Fails-before: flagged, with no exemption GEPA could offer.**

    COPRO already exempts a short search that declares a terminal failure
    (``copro_search_depth``); GEPA did not. So a run that stopped because
    the provider stopped answering was reported twice -- once honestly as
    its terminal failure, and again here as a budget violation, which
    downgraded the arm to ``VERDICT_NOT_VALIDATED``. That is the audit
    blaming the harness for the infrastructure's behaviour.

    Same evidence as the test above, plus the declaration.
    """
    evidence = _with_declared_failure(
        _raised_ceiling_evidence(gepa_run_dir, tmp_path, 10_000),
        "the task model stopped answering mid-search",
    )

    finding = gepa_metric_call_budget(evidence)
    assert finding.status is AuditStatus.PASS, finding.detail


def test_the_exemption_does_not_excuse_continuing_past_the_ceiling(
    gepa_run_dir, tmp_path
) -> None:
    """A declared failure covers stopping early, not overspending.

    The past-the-ceiling check stays a thing the harness controls whatever
    upstream did, so the exemption is scoped to the below-ceiling stop
    alone. A ceiling of 1 makes the run's own steps overshoot it.
    """
    evidence = _with_declared_failure(
        _raised_ceiling_evidence(gepa_run_dir, tmp_path, 1),
        "declared, but the run still overspent",
    )

    finding = gepa_metric_call_budget(evidence)
    assert finding.status is AuditStatus.FAIL
    assert "at or past the ceiling" in finding.detail
