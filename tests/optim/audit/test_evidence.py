"""The reader resolves a real run's evidence through public whetstone APIs.

These run against an actual fake-transport COPRO run rather than a
hand-built artifact, so a change in what whetstone persists surfaces here
instead of silently invalidating every invariant built on top.
"""

from __future__ import annotations

import pytest

from whetstone_envs.optim.audit._evidence import (
    COPRO_OPTIMIZER,
    RESULT_FILENAME,
    RUNTIME_STORE_FILENAME,
    AuditEvidenceError,
    load_run_evidence,
)


@pytest.fixture(scope="session")
def evidence(copro_run_dir):
    return load_run_evidence(copro_run_dir)


def test_optimizer_is_read_from_the_result_not_the_caller(evidence) -> None:
    assert evidence.optimizer == COPRO_OPTIMIZER


def test_run_id_comes_from_the_result(evidence) -> None:
    assert evidence.run_id == "c19-copro-audit-fixture"


def test_every_step_result_is_unwrapped(evidence) -> None:
    assert evidence.steps
    assert [entry.index for entry in evidence.steps] == list(
        range(len(evidence.result.step_results))
    )
    for entry in evidence.steps:
        assert entry.step.step_index == entry.index


def test_intents_resolve_to_eval_evidence(evidence) -> None:
    """The reader dereferences each cited eval result out of the store."""
    cited = [
        resolution.eval_result_ref
        for entry in evidence.steps
        for resolution in entry.resolved_intents
        if resolution.eval_result_ref is not None
    ]
    assert cited, "the fixture run recorded no eval results"
    for ref in cited:
        found = evidence.eval_evidence(ref)
        assert found is not None
        assert found.task_hashes
        assert found.eval_role.value == "internal"


def test_all_eval_evidence_matches_the_resolved_set(evidence) -> None:
    collected = dict(evidence.all_eval_evidence())
    assert collected
    for ref, record in collected.items():
        assert ref.schema_name == "whetstone.eval_evidence"
        assert evidence.eval_evidence(ref) is record


def test_copro_run_carries_no_gepa_terminal_artifact(evidence) -> None:
    """Absence is reported, not raised; an invariant decides what it means."""
    assert evidence.gepa_terminal is None


def test_copro_state_carries_no_gepa_or_miprov2_slice(evidence) -> None:
    for entry in evidence.steps:
        assert entry.gepa_checkpoint() is None
        assert entry.miprov2_state() is None
        assert entry.gepa_skipped_mutations() == ()


def test_missing_result_is_an_evidence_error(tmp_path) -> None:
    (tmp_path / RUNTIME_STORE_FILENAME).write_bytes(b"")
    with pytest.raises(AuditEvidenceError, match=r"result\.json is missing"):
        load_run_evidence(tmp_path)


def test_missing_store_is_an_evidence_error(tmp_path) -> None:
    (tmp_path / RESULT_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(
        AuditEvidenceError, match=r"runtime\.sqlite is missing"
    ):
        load_run_evidence(tmp_path)


def test_unparseable_result_is_an_evidence_error(tmp_path) -> None:
    (tmp_path / RESULT_FILENAME).write_text('{"run": 1}', encoding="utf-8")
    (tmp_path / RUNTIME_STORE_FILENAME).write_bytes(b"")
    with pytest.raises(AuditEvidenceError, match="not a valid OptimResult"):
        load_run_evidence(tmp_path)
