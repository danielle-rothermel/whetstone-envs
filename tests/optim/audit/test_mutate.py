"""The negative-fixture mechanism, and the worked example negative.

Section 3.2 of the Step 10 assignment makes a failing fixture a shipping
requirement: an invariant with no fixture that makes it FAIL is not yet an
invariant. These tests pin the helper the four per-optimizer waves use to
build those fixtures, plus one worked end-to-end negative for the shared
invariant so each wave has a pattern to copy.
"""

from __future__ import annotations

import json

import pytest

from whetstone_envs.optim.audit._evidence import (
    RESULT_FILENAME,
    load_run_evidence,
)
from whetstone_envs.optim.audit._mutate import (
    MutationError,
    copy_run,
    mutate_json_field,
    mutate_run,
    put_record,
)
from whetstone_envs.optim.audit.registry import audit_run
from whetstone_envs.optim.audit.schema import AuditStatus, InvariantId

#: The first completed intent's eval-result ref inside ``result.json``.
#: Rewriting it is the mutation that violates REPORTED_NUMBERS_RESOLVE.
EVAL_RESULT_REF_PATH = (
    "step_results",
    0,
    "record",
    "resolved_intents",
    0,
    "eval_result_ref",
)


def test_mutate_json_field_rewrites_a_nested_value() -> None:
    document = {"a": [{"b": 1}]}
    mutate_json_field(document, ("a", 0, "b"), lambda value: value + 1)
    assert document == {"a": [{"b": 2}]}


def test_mutate_json_field_refuses_a_no_op() -> None:
    """A silent no-op makes a negative test pass for the wrong reason."""
    document = {"a": 1}
    with pytest.raises(MutationError, match="identical value"):
        mutate_json_field(document, ("a",), lambda value: value)


def test_mutate_json_field_refuses_an_absent_path() -> None:
    with pytest.raises(MutationError, match="is absent"):
        mutate_json_field({"a": 1}, ("b",), lambda _value: 2)


def test_mutate_json_field_refuses_an_absent_index() -> None:
    with pytest.raises(MutationError, match="is absent"):
        mutate_json_field({"a": []}, ("a", 3), lambda _value: 2)


def test_copy_run_leaves_the_source_untouched(copro_run_dir, tmp_path) -> None:
    before = (copro_run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    copied = copy_run(copro_run_dir, tmp_path / "copy")
    (copied / RESULT_FILENAME).write_text("{}", encoding="utf-8")
    after = (copro_run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    assert after == before


def test_put_record_round_trips_through_the_runs_store(
    mutable_run_dir,
) -> None:
    ref = put_record(
        mutable_run_dir, "whetstone_envs.audit_test_record", {"value": 7}
    )
    assert set(ref) == {"schema_name", "content_hash"}
    assert ref["schema_name"] == "whetstone_envs.audit_test_record"
    assert len(ref["content_hash"]) == 64


def test_put_record_is_content_addressed(mutable_run_dir) -> None:
    """Two different records cannot collide onto one ref."""
    first = put_record(mutable_run_dir, "s", {"value": 1})
    second = put_record(mutable_run_dir, "s", {"value": 2})
    again = put_record(mutable_run_dir, "s", {"value": 1})
    assert first["content_hash"] != second["content_hash"]
    assert first == again


def test_mutate_run_produces_a_readable_but_violated_artifact(
    copro_run_dir, tmp_path
) -> None:
    """The mutation stays in the persisted format; the reader still parses."""
    mutated = mutate_run(
        copro_run_dir,
        tmp_path / "negative",
        EVAL_RESULT_REF_PATH,
        lambda ref: {**ref, "content_hash": "0" * 64},
    )
    evidence = load_run_evidence(mutated)
    assert evidence.optimizer == "copro"
    assert evidence.steps


def test_mutate_run_leaves_the_source_run_passing(
    copro_run_dir, tmp_path
) -> None:
    mutate_run(
        copro_run_dir,
        tmp_path / "negative",
        EVAL_RESULT_REF_PATH,
        lambda ref: {**ref, "content_hash": "0" * 64},
    )
    assert audit_run(copro_run_dir).passed


# --- The worked negative fixture ------------------------------------------


@pytest.fixture
def dangling_eval_ref_run(copro_run_dir, tmp_path):
    """A run whose first reported number cites a record nobody stored.

    This is the violation ``REPORTED_NUMBERS_RESOLVE`` exists to catch: the
    optimizer reports a score whose backing evidence is not in the run's own
    store, so the number cannot be audited.
    """
    return mutate_run(
        copro_run_dir,
        tmp_path / "dangling-eval-ref",
        EVAL_RESULT_REF_PATH,
        lambda ref: {**ref, "content_hash": "0" * 64},
    )


def test_the_target_invariant_fails_on_the_negative_fixture(
    dangling_eval_ref_run,
) -> None:
    report = audit_run(dangling_eval_ref_run)
    assert not report.passed
    failed = {
        finding.invariant_id
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    }
    assert failed == {InvariantId.REPORTED_NUMBERS_RESOLVE}


def test_the_failing_finding_names_what_it_saw(dangling_eval_ref_run) -> None:
    report = audit_run(dangling_eval_ref_run)
    finding = next(
        item
        for item in report.findings
        if item.invariant_id is InvariantId.REPORTED_NUMBERS_RESOLVE
    )
    assert finding.status is AuditStatus.FAIL
    assert "do not resolve" in finding.detail
    assert "step 0 intent 0" in finding.detail


def test_no_other_invariants_status_changed(
    copro_run_dir, dangling_eval_ref_run
) -> None:
    """A mutation that broke everything would flatter a sloppy invariant."""
    before = {
        finding.invariant_id: finding.status
        for finding in audit_run(copro_run_dir).findings
    }
    after = {
        finding.invariant_id: finding.status
        for finding in audit_run(dangling_eval_ref_run).findings
    }
    assert set(before) == set(after)
    changed = {
        invariant
        for invariant, status in after.items()
        if before[invariant] is not status
    }
    assert changed == {InvariantId.REPORTED_NUMBERS_RESOLVE}


def test_the_mutation_touched_one_semantic_field(
    copro_run_dir, dangling_eval_ref_run
) -> None:
    """Only the target field, plus the integrity refs resealing recomputes.

    ``OptimResult`` self-verifies its step wrapper refs and chains each
    request to the prior step, so a mutated artifact must be re-sealed or it
    fails schema validation before any invariant can judge it. Those
    recomputed hashes are bookkeeping; the only *semantic* change is the
    targeted field.
    """
    original = json.loads(
        (copro_run_dir / RESULT_FILENAME).read_text(encoding="utf-8")
    )
    mutated = json.loads(
        (dangling_eval_ref_run / RESULT_FILENAME).read_text(encoding="utf-8")
    )

    def differing(left, right, path=()):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left.keys() | right.keys():
                yield from differing(
                    left.get(key), right.get(key), (*path, key)
                )
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                yield path
            else:
                for index, (a, b) in enumerate(zip(left, right, strict=True)):
                    yield from differing(a, b, (*path, index))
        elif left != right:
            yield path

    changed = list(differing(original, mutated))
    target = (*EVAL_RESULT_REF_PATH, "content_hash")
    assert target in changed

    #: Paths resealing is allowed to rewrite: a step wrapper's own ref, and
    #: the request refs that chain one step to the prior one.
    def _is_integrity_ref(path: tuple[object, ...]) -> bool:
        return path[-1] == "content_hash" and (
            path[-2:] == ("record_ref", "content_hash")
            or "prior_step_result_ref" in path
        )

    unexpected = [
        path
        for path in changed
        if path != target and not _is_integrity_ref(path)
    ]
    assert unexpected == []
