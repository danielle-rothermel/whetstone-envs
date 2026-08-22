"""The six Codex fidelity invariants, each with a failing fixture.

Section 3.2 makes a negative fixture a shipping requirement: an invariant
with no evidence that makes it FAIL is not yet an invariant. Every
invariant below therefore has a mutation that violates exactly it, and
every negative asserts *both* that its target FAILs and that no other
invariant's status changed -- a mutation that broke everything would let
a sloppy invariant look sound.

**Why these fixtures are committed rather than built here.** The other
optimizers' audits build their evidence by running a real fake-transport
run in ``conftest.py``. A Codex run cannot be built that way: the
Codex-direct adapter, the one-tool MCP surface, and
``ToolAdmissionAuthority.admitted_entries`` are all 0.1.7 surface, and
the installed whetstone-ai is 0.1.6. So the runs were produced once
against a checkout carrying that surface (see
``fixtures/generate.py``) and committed, and these tests skip until an
install can read them.

**The version skew is real and worth naming.** Neither tip alone can
produce these fixtures. ``08-22-codex`` carries the Codex surface but
predates ``OptimStepResult.proposer_usage``, which the 0.1.6 tip added to
the step record's identity payload; the 0.1.6 tip has ``proposer_usage``
but no Codex. The committed fixtures were generated against the merge of
the two, which is what 0.1.7 will be -- the two branches' source merges
cleanly. ``requires_codex_surface`` skips on anything else, so this suite
turns itself on when 0.1.7 lands rather than needing to be edited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from whetstone_envs.optim.audit._evidence import (
    RESULT_FILENAME,
    load_run_evidence,
)
from whetstone_envs.optim.audit._mutate import (
    copy_run,
    delete_ledger_entry,
    mutate_ledger_entry,
    mutate_run,
)
from whetstone_envs.optim.audit.codex import (
    CODEX_ACCEPTED_CALL_COUNT_KEY,
    CODEX_CAPACITY_CAP,
    CODEX_INVARIANTS,
    CODEX_OUTPUT_ARTIFACT_REF_KEY,
    TASK_MODEL_COST_ROLE,
)
from whetstone_envs.optim.audit.registry import audit_run
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId

FIXTURES = Path(__file__).parent / "fixtures"
COMPLETED_RUN = FIXTURES / "codex-completed"
FAILED_RUN = FIXTURES / "codex-failed"

#: Every ``InvariantId`` this module owns.
CODEX_INVARIANT_IDS = frozenset(
    {
        InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS,
        InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
        InvariantId.CODEX_CAPACITY_RESPECTED,
        InvariantId.CODEX_LEASE_BINDS_ARTIFACT,
        InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
        InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
    }
)


def _codex_surface_reason() -> str | None:
    """Why this install cannot read the committed Codex fixtures, if so."""
    try:
        from whetstone.optim.tools.facade import ToolAdmissionAuthority
    except ImportError as error:  # pragma: no cover
        return str(error)
    if not hasattr(ToolAdmissionAuthority, "admitted_entries"):
        return "ToolAdmissionAuthority has no admitted_entries (pre-0.1.7)"
    try:
        from whetstone.optim.codex.adapter import CodexOutputArtifact
    except ImportError as error:  # pragma: no cover
        return str(error)
    if "selected_call_id" not in CodexOutputArtifact.model_fields:
        return "CodexOutputArtifact predates Codex-direct (pre-0.1.7)"
    return None


requires_codex_surface = pytest.mark.skipif(
    _codex_surface_reason() is not None,
    reason=f"Codex-direct surface unavailable: {_codex_surface_reason()}",
)


# --------------------------------------------------------------------------
# Registration and shape. These hold on every install, so they do not skip:
# a Codex run must never audit vacuously, whatever whetstone-ai is pinned.
# --------------------------------------------------------------------------


def test_the_six_invariants_are_registered() -> None:
    from whetstone_envs.optim.audit._evidence import CODEX_OPTIMIZER
    from whetstone_envs.optim.audit.registry import invariants_for

    registered = invariants_for(CODEX_OPTIMIZER)
    assert set(CODEX_INVARIANTS) <= set(registered)
    assert len(CODEX_INVARIANTS) == 6


def test_registration_does_not_import_the_codex_module() -> None:
    """Registering must not require whetstone-ai to carry Codex at all.

    An eager import would make ``optim/audit`` unimportable on any
    install without the Codex-direct surface, taking the other three
    optimizers' audits down with it.
    """
    import subprocess
    import sys

    # whetstone-ai itself imports ``whetstone.optim.codex.adapter`` eagerly
    # from ``whetstone.optim.copro.control``, so the mere presence of that
    # module in ``sys.modules`` says nothing about *this* package. What is
    # ours to guarantee is that registering adds no Codex import beyond the
    # one whetstone-ai already performs for COPRO, which is what registering
    # on top of an already-imported COPRO control measures.
    probe = (
        "import sys;"
        "import whetstone.optim.copro.control;"
        "baseline = {m for m in sys.modules if 'codex' in m};"
        "import whetstone_envs.optim.audit.registry as r;"
        "assert r.INVARIANTS_BY_OPTIMIZER;"
        "added = {m for m in sys.modules if 'codex' in m} - baseline;"
        "print(sorted(m for m in added if m.startswith('whetstone.')))"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "[]", completed.stdout


def test_the_cost_role_is_named_from_the_owning_enum() -> None:
    """OQ1: there is no ``codex_agent`` role, so this is the only one."""
    from whetstone.optim.cost import CostRole

    assert TASK_MODEL_COST_ROLE == "task_model"
    assert TASK_MODEL_COST_ROLE in {role.value for role in CostRole}
    assert "codex_agent" not in {role.value for role in CostRole}


def test_the_capacity_cap_is_the_pre_registered_value() -> None:
    """D2 fixes the per-run admission cap at 8 evaluate-calls."""
    assert CODEX_CAPACITY_CAP == 8


def test_the_artifact_state_key_matches_the_evidence_readers() -> None:
    """One spelling, in the reader and in the invariant that uses it."""
    from whetstone_envs.optim.audit._evidence import STATE_RECORD_REF_KEYS

    assert CODEX_OUTPUT_ARTIFACT_REF_KEY == "codex_output_artifact_ref"
    assert CODEX_OUTPUT_ARTIFACT_REF_KEY in STATE_RECORD_REF_KEYS


def test_a_codex_run_without_the_surface_fails_rather_than_skipping() -> None:
    """An un-auditable Codex arm must never read as validated.

    ``NOT_APPLICABLE`` would be the wrong verdict: the invariant genuinely
    applies to a Codex run, and reporting it inapplicable on an old
    install would let the arm pass its fidelity gate having checked
    nothing.
    """
    if _codex_surface_reason() is None:
        pytest.skip("this install carries the Codex surface")
    report = audit_run(COMPLETED_RUN)
    statuses = {
        finding.invariant_id: finding.status
        for finding in report.findings
        if finding.invariant_id in CODEX_INVARIANT_IDS
    }
    unjudgeable = {
        invariant_id
        for invariant_id, status in statuses.items()
        if status is AuditStatus.NOT_APPLICABLE
    }
    assert unjudgeable <= {InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED}
    assert not report.passed


# --------------------------------------------------------------------------
# The positives.
# --------------------------------------------------------------------------


@requires_codex_surface
def test_a_real_completed_codex_run_passes_every_invariant() -> None:
    report = audit_run(COMPLETED_RUN)
    assert report.optimizer == "codex"
    assert report.passed, [
        (finding.invariant_id, finding.detail)
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]


@requires_codex_surface
def test_the_completed_run_judges_five_invariants_and_defers_one() -> None:
    """Only the failure-evidence invariant is conditional here."""
    statuses = _statuses(audit_run(COMPLETED_RUN))
    assert statuses[InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED] is (
        AuditStatus.NOT_APPLICABLE
    )
    judged = [
        invariant_id
        for invariant_id in CODEX_INVARIANT_IDS
        if statuses[invariant_id] is AuditStatus.PASS
    ]
    assert len(judged) == 5


@requires_codex_surface
def test_invariants_cite_the_ledger_entries_they_read() -> None:
    """A finding a human cannot trace back is not evidence."""
    report = audit_run(COMPLETED_RUN)
    for finding in report.findings:
        if finding.invariant_id in {
            InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS,
            InvariantId.CODEX_CAPACITY_RESPECTED,
            InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
        }:
            assert finding.evidence_refs, finding.invariant_id
            for ref in finding.evidence_refs:
                assert ref.schema_name == "whetstone.tool_call"
                assert len(ref.content_hash) == 64


@requires_codex_surface
def test_a_failed_run_still_accounts_for_what_it_paid_for() -> None:
    """The sixth invariant's positive: the spend record survives failure.

    This run's artifact named a call that was never issued, so the Step
    failed under ``codex_selection_unevaluated`` -- and it still recorded
    the one evaluation it genuinely paid for, which is what a wall stop
    or an interrupted evaluation must equally preserve.
    """
    statuses = _statuses(audit_run(FAILED_RUN))
    assert statuses[InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED] is (
        AuditStatus.PASS
    )
    assert statuses[InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED] is (
        AuditStatus.FAIL
    )
    assert statuses[InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS] is (
        AuditStatus.PASS
    )


# --------------------------------------------------------------------------
# The negatives. One per invariant, each asserting nothing else moved.
# --------------------------------------------------------------------------


def _statuses(report) -> dict[InvariantId, AuditStatus]:
    return {
        finding.invariant_id: finding.status for finding in report.findings
    }


def _assert_only(
    mutated: Path,
    target: InvariantId,
    *,
    baseline: Path = COMPLETED_RUN,
) -> None:
    """The mutation FAILed ``target`` and moved nothing else.

    Without the second half, a mutation that corrupted the artifact
    wholesale would satisfy the first half while proving nothing about
    the invariant it claims to test.
    """
    before = _statuses(audit_run(baseline))
    after = _statuses(audit_run(mutated))
    assert after[target] is AuditStatus.FAIL, (
        f"{target} did not fail against its own negative fixture"
    )
    moved = {
        invariant_id
        for invariant_id, status in after.items()
        if status is not before[invariant_id]
    }
    assert moved == {target}, f"collateral damage: {sorted(moved)}"


@requires_codex_surface
def test_an_admitted_call_absent_from_tool_evidence_fails_totality(
    tmp_path,
) -> None:
    """Ledger totality: paid work must stay reachable from the Step.

    Dropping a call from ``tool_evidence`` leaves it on the ledger with
    nothing citing it -- exactly the shortfall the adapter fails under
    ``codex_unreported_evaluation`` at run time.
    """
    # Drop the *unselected* call. Dropping the selected one would also
    # break CODEX_FINAL_CANDIDATE_EVALUATED, and a mutation that moves two
    # invariants proves neither.
    mutated = mutate_run(
        COMPLETED_RUN,
        tmp_path / "unreported",
        ("step_results", 0, "record", "tool_evidence"),
        lambda evidence: evidence[1:],
    )
    _assert_only(mutated, InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS)


@requires_codex_surface
def test_a_selection_with_no_ledger_entry_fails(tmp_path) -> None:
    """The returned candidate must resolve to a real admitted call.

    The artifact's ``selected_call_id`` is repointed rather than the
    ledger row deleted: deleting the row would *also* break ledger
    totality, and a mutation that moves two invariants proves neither.
    """
    mutated = copy_run(COMPLETED_RUN, tmp_path / "unselected")
    _rewrite_artifact(mutated, selected_call_id="never-issued")
    _assert_only(mutated, InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED)


@requires_codex_surface
def test_a_reported_call_outside_the_ledger_scope_fails(tmp_path) -> None:
    """Totality's reverse direction, which scoping makes non-obvious.

    ``admitted_entries`` selects on the Tool Config's identity hash, so
    tampering with a durable entry's Tool Config does not make it read as
    wrong -- it makes it vanish from the scope. Only the reported-side
    check catches that.
    """
    mutated = copy_run(COMPLETED_RUN, tmp_path / "outofscope")
    delete_ledger_entry(mutated, "c1")
    _assert_only(mutated, InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS)


@requires_codex_surface
def test_a_debit_ordinal_past_the_cap_fails(tmp_path) -> None:
    """Ordinals are dense and one-based, so one past the cap is overspend."""
    mutated = copy_run(COMPLETED_RUN, tmp_path / "overcap")
    mutate_ledger_entry(
        mutated,
        "c2",
        lambda entry: {
            **entry,
            "capacity_debit_ordinal": CODEX_CAPACITY_CAP + 1,
        },
    )
    _assert_only(mutated, InvariantId.CODEX_CAPACITY_RESPECTED)


@requires_codex_surface
def test_a_malformed_lease_token_hash_fails(tmp_path) -> None:
    """An empty hash is what the adapter's own mismatch path rejects."""
    mutated = copy_run(COMPLETED_RUN, tmp_path / "unbound")
    _rewrite_artifact(mutated, lease_token_hash="")
    _assert_only(mutated, InvariantId.CODEX_LEASE_BINDS_ARTIFACT)


@requires_codex_surface
def test_a_widened_tool_definition_is_caught(tmp_path) -> None:
    """A tampered Tool Config cannot hide by leaving the ledger scope.

    Widening a durable entry's declared ``input_fields`` is what an
    extension of the tool surface looks like on the wire. Because the
    entry's Tool Config identity keys the capacity scope, the widened
    entry stops matching ``admitted_entries`` altogether -- so the
    tampering surfaces as a reported call with no entry in the run's own
    scope rather than as a wider surface. Either way it does not pass,
    which is the property that matters; asserting the specific verdict
    pins which invariant owns it.
    """
    # The unselected call, so the selection invariant stays untouched.
    mutated = copy_run(COMPLETED_RUN, tmp_path / "widened")
    mutate_ledger_entry(mutated, "c1", _widened_tool_definition)
    _assert_only(mutated, InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS)


@requires_codex_surface
def test_a_second_granted_tool_fails_the_minimal_surface() -> None:
    """A run may hold a capability it never exercised.

    The ledger cannot show this: a run that called one tool while being
    granted two looks minimal from its entries alone. So the grant is
    checked on ``OptimRun.tool_configs``, and the negative is built
    against the invariant directly -- ``OptimRun``'s content hash is the
    RUN-scoped capacity subject every Tool Call binds to, so granting a
    second config through ``result.json`` would invalidate every Tool
    Evidence entry before the audit could judge the grant.
    """
    from dataclasses import replace

    from whetstone_envs.optim.audit.codex import codex_tool_surface_minimal

    evidence = load_run_evidence(COMPLETED_RUN)
    run = evidence.result.run
    granted = run.record.tool_configs
    doubled = run.model_copy(
        update={
            "record": run.record.model_copy(
                update={"tool_configs": (*granted, *granted)}
            )
        }
    )
    two_tools = replace(
        evidence,
        result=evidence.result.model_copy(update={"run": doubled}),
    )

    assert codex_tool_surface_minimal(evidence).status is AuditStatus.PASS
    finding = codex_tool_surface_minimal(two_tools)
    assert finding.status is AuditStatus.FAIL
    assert "granted 2 Tool Configs" in finding.detail


@requires_codex_surface
def test_a_failed_run_understating_its_spend_fails(tmp_path) -> None:
    """The sixth invariant's negative: a recorded spend below the ledger.

    Nothing upstream reconciles ``harness_store_accepted_call_count``
    against the admission entries it was read from, so an artifact whose
    recorded spend is lower than what it actually bought loads fine and
    would be trusted by the study's cost reconciliation.
    """
    mutated = copy_run(FAILED_RUN, tmp_path / "understated")
    _rewrite_state(mutated, {CODEX_ACCEPTED_CALL_COUNT_KEY: 0})
    _assert_only(
        mutated,
        InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
        baseline=FAILED_RUN,
    )


def _as_object(record: object) -> dict[str, Any]:
    """A stored record as a mutable mapping.

    ``ObjectStore.get`` is typed as returning ``object`` -- it decodes
    whatever schema it was handed -- so the narrowing happens once here
    rather than at each rewrite site.
    """
    assert isinstance(record, dict)
    return dict(record)


def _widened_tool_definition(entry: dict) -> dict:
    """Add an input field to the entry's Tool Definition, resealing refs.

    ``ToolDefinitionRef``, ``ToolConfigRef``, and ``ToolCallRef`` are each
    self-verifying, and a completed entry's ``EffectTerminal`` must belong
    to the exact Tool request -- whose hash covers the Tool Config. So
    one added field invalidates four nested identities. Resealing all of
    them keeps the fixture a *decodable* entry violating only the
    semantic invariant; one rejected at decode would prove nothing.

    The call's own ``args`` are widened to match, because ``ToolCall``
    requires args to equal the definition's ``input_fields`` exactly --
    which is why the invariant pins the *definition* against
    ``CODEX_EVAL_INPUT_FIELDS`` rather than only comparing args to it.
    """
    from whetstone.optim.tools.admission import tool_effect_request
    from whetstone.optim.tools.contracts import (
        ToolCall,
        ToolConfig,
        ToolDefinition,
        tool_call_reference,
        tool_config_reference,
        tool_definition_reference,
    )

    mutated = json.loads(json.dumps(entry))
    definition = mutated["tool_config"]["record"]["definition"]["record"]
    definition["input_fields"] = [*definition["input_fields"], "task_ids"]
    sealed_definition = tool_definition_reference(
        ToolDefinition.model_validate(definition)
    ).model_dump(mode="json")

    config = mutated["tool_config"]["record"]
    config["definition"] = sealed_definition
    sealed_config = tool_config_reference(
        ToolConfig.model_validate(config)
    ).model_dump(mode="json")
    mutated["tool_config"] = sealed_config

    call = mutated["tool_call"]["record"]
    call["tool_config"] = sealed_config
    call["args"] = {**call["args"], "task_ids": []}
    exact_call = ToolCall.model_validate(call)
    mutated["tool_call"] = tool_call_reference(exact_call).model_dump(
        mode="json"
    )
    terminal = mutated.get("effect_terminal")
    if terminal is not None:
        terminal["request"] = tool_effect_request(exact_call).model_dump(
            mode="json"
        )
    return mutated


def _rewrite_state(run_dir: Path, updates: dict) -> None:
    """Re-put the terminal step's state snapshot with ``updates`` applied."""
    from dr_store.sync import open_sqlite

    evidence = load_run_evidence(run_dir)
    step = evidence.steps[-1]
    assert step.step.state_ref is not None
    with open_sqlite(str(run_dir / "runtime.sqlite")) as store:
        snapshot = _as_object(store.get(step.step.state_ref.reference))
        snapshot.update(updates)
        snapshot_ref, _status = store.put(
            step.step.state_ref.schema_name, snapshot
        )
    document = json.loads(
        (run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    )
    document["step_results"][0]["record"]["state_ref"] = {
        "schema_name": snapshot_ref.schema,
        "content_hash": snapshot_ref.content_hash,
    }
    from whetstone_envs.optim.audit._mutate import reseal_step_chain

    reseal_step_chain(document)
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )


def _rewrite_artifact(run_dir: Path, **updates) -> None:
    """Re-put the Codex output artifact with ``updates`` applied.

    The artifact is content-addressed, so the step's state snapshot must
    point at the rewritten record. Both are rewritten together, which
    keeps the fixture a run the audit can still load.
    """
    from dr_store.sync import open_sqlite
    from whetstone.core.identity import TypedRef

    evidence = load_run_evidence(run_dir)
    step = evidence.steps[-1]
    assert step.state is not None
    assert step.step.state_ref is not None
    artifact_ref = TypedRef.model_validate(
        step.state[CODEX_OUTPUT_ARTIFACT_REF_KEY]
    )
    with open_sqlite(str(run_dir / "runtime.sqlite")) as store:
        artifact = _as_object(store.get(artifact_ref.reference))
        artifact.update(updates)
        reference, _status = store.put(artifact_ref.schema_name, artifact)
        snapshot = _as_object(store.get(step.step.state_ref.reference))
        snapshot[CODEX_OUTPUT_ARTIFACT_REF_KEY] = {
            "schema_name": reference.schema,
            "content_hash": reference.content_hash,
        }
        snapshot_ref, _status = store.put(
            step.step.state_ref.schema_name, snapshot
        )
    document = json.loads(
        (run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    )
    document["step_results"][0]["record"]["state_ref"] = {
        "schema_name": snapshot_ref.schema,
        "content_hash": snapshot_ref.content_hash,
    }
    from whetstone_envs.optim.audit._mutate import reseal_step_chain

    reseal_step_chain(document)
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
