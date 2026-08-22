"""Build a negative fixture by violating one field of a real run artifact.

Every invariant must ship a fixture that makes exactly that invariant FAIL.
Hand-writing such a fixture is how audits rot: a hand-built artifact drifts
from what whetstone actually persists, and the invariant then passes against
reality while its fixture keeps testing a shape nobody produces.

So a negative fixture starts as a *real* fake-transport run and has exactly
one evidence field rewritten. That guarantees the fixture stays in the
persisted format, and it makes the mutation itself legible: the test names
the field it violated.

The mechanism relies on content addressing, and ``OptimResult`` is
self-verifying in two ways that a naive rewrite trips over:

1. Each ``OptimStepResultRef`` carries its record inline *and* a
   ``record_ref`` that must address that exact record, so mutating a field
   invalidates the wrapper hash ("Step Result record_ref must address the
   exact result").
2. Step requests form a chain: step *i*'s request must cite step *i-1*'s
   exact ``record_ref``, state ref, and history ref ("each later Step
   Request must cite the prior exact result").
3. The ``OptimRun`` record is embedded once at the top and again inside
   every step request, each copy verifying its own ``record_ref`` and
   ``config_hash``.

So :func:`mutate_run` re-seals after mutating -- it recomputes each step's
wrapper ref from its mutated record and re-threads the chain forward. That
keeps the negative fixture a *valid* ``OptimResult`` that violates only the
semantic invariant under test, which is the point: an artifact that failed
schema validation would prove nothing about the invariant.
:func:`reseal_run_binding` does the same for a mutation of the run record
itself, which is how an invariant over the run's *configuration* -- its
optimizer control, its seed candidate -- gets a negative fixture.

This module is test tooling, not part of the audited path.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from typing import TYPE_CHECKING, Any

from dr_store.sync import open_sqlite
from whetstone.core.identity import typed_ref_for_record
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.contracts import (
    OPTIM_RUN_SCHEMA,
    OptimRun,
    OptimStepRequest,
    OptimStepResult,
    optimization_run_reference,
    step_request_reference,
    step_result_reference,
)

from whetstone_envs.optim.audit._evidence import (
    RESULT_FILENAME,
    RUNTIME_STORE_FILENAME,
    AuditEvidenceError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class MutationError(RuntimeError):
    """The requested mutation did not apply.

    Raised when the named path is absent or the rewrite left the value
    unchanged. A silent no-op is the dangerous case: the negative test would
    then assert against an unmutated artifact and pass for the wrong reason.
    """


def _resolve(document: Any, path: tuple[str | int, ...]) -> Any:
    cursor = document
    for step in path:
        try:
            cursor = cursor[step]
        except (KeyError, IndexError, TypeError) as error:
            raise MutationError(
                f"path {'.'.join(str(part) for part in path)} is absent "
                f"from the artifact"
            ) from error
    return cursor


def _assign(document: Any, path: tuple[str | int, ...], value: Any) -> None:
    parent = _resolve(document, path[:-1])
    try:
        parent[path[-1]] = value
    except (KeyError, IndexError, TypeError) as error:
        raise MutationError(
            f"path {'.'.join(str(part) for part in path)} is not assignable"
        ) from error


def mutate_json_field(
    document: dict[str, Any],
    path: tuple[str | int, ...],
    rewrite: Callable[[Any], Any],
) -> dict[str, Any]:
    """Rewrite one field of ``document`` in place, refusing a no-op.

    ``path`` addresses the field by successive keys and list indices, e.g.
    ``("step_results", 0, "record", "resolved_intents", 0,
    "eval_result_ref")``. ``rewrite`` receives the current value and returns
    the replacement.
    """
    current = _resolve(document, path)
    replacement = rewrite(current)
    if replacement == current:
        raise MutationError(
            f"rewriting {'.'.join(str(part) for part in path)} produced an "
            f"identical value; the fixture would not be a negative"
        )
    _assign(document, path, replacement)
    return document


def put_record(run_dir: Path, schema: str, record: Any) -> dict[str, str]:
    """Store ``record`` in the run's own sqlite store, returning its ref.

    The returned mapping is a serialized ``TypedRef``, ready to splice into
    ``result.json`` wherever the original ref appeared.
    """
    store_path = run_dir / RUNTIME_STORE_FILENAME
    if not store_path.is_file():
        raise AuditEvidenceError(f"{store_path} is missing")
    with open_sqlite(str(store_path)) as store:
        reference, _status = store.put(schema, record)
    return {
        "schema_name": reference.schema,
        "content_hash": reference.content_hash,
    }


#: The relational table the Tool admission ledger persists its entries
#: in. Rewriting a row is how a Codex negative fixture violates a ledger
#: invariant: the entry's own model validators reject most edits made
#: through ``result.json``, so the violation has to be made where the
#: authority will read it back. This is whetstone's persisted table name,
#: pinned by a golden test.
TOOL_ADMISSION_ENTRY_TABLE = "whetstone_tool_admission_entry"


def mutate_ledger_entry(
    run_dir: Path,
    call_id: str,
    rewrite: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Rewrite one durable admission entry, refusing a no-op.

    The audit reads the ledger through ``admitted_entries``, never by
    raw SQL. Building a negative fixture is the opposite direction --
    it must *write* a row the reader will then decode -- and there is no
    public writer for an already-terminal entry, so the fixture builder
    goes through the table directly. That asymmetry is deliberate: test
    tooling may reach for the storage, the audited path may not.
    """
    store_path = run_dir / RUNTIME_STORE_FILENAME
    if not store_path.is_file():
        raise AuditEvidenceError(f"{store_path} is missing")
    connection = sqlite3.connect(store_path)
    try:
        rows = connection.execute(
            f"SELECT store_namespace_key, entry_json "  # noqa: S608
            f"FROM {TOOL_ADMISSION_ENTRY_TABLE} WHERE call_id = ?",
            (call_id,),
        ).fetchall()
        if not rows:
            raise MutationError(
                f"no admission entry for call {call_id!r} to mutate"
            )
        for namespace, raw in rows:
            current = json.loads(raw)
            replacement = rewrite(dict(current))
            if replacement == current:
                raise MutationError(
                    f"rewriting admission entry {call_id!r} produced an "
                    f"identical value; the fixture would not be a negative"
                )
            connection.execute(
                f"UPDATE {TOOL_ADMISSION_ENTRY_TABLE} "  # noqa: S608
                f"SET entry_json = ? "
                f"WHERE store_namespace_key = ? AND call_id = ?",
                (
                    json.dumps(replacement, sort_keys=True),
                    namespace,
                    call_id,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def delete_ledger_entry(run_dir: Path, call_id: str) -> None:
    """Drop one durable admission entry, refusing a no-op.

    An entry present in ``tool_evidence`` but absent from the ledger is
    the shape no honest run produces, which is exactly why it is the
    negative for the reverse direction of ledger totality.
    """
    store_path = run_dir / RUNTIME_STORE_FILENAME
    if not store_path.is_file():
        raise AuditEvidenceError(f"{store_path} is missing")
    connection = sqlite3.connect(store_path)
    try:
        cursor = connection.execute(
            f"DELETE FROM {TOOL_ADMISSION_ENTRY_TABLE} "  # noqa: S608
            f"WHERE call_id = ?",
            (call_id,),
        )
        if cursor.rowcount < 1:
            raise MutationError(
                f"no admission entry for call {call_id!r} to delete"
            )
        connection.commit()
    finally:
        connection.close()


def copy_run(source: Path, destination: Path) -> Path:
    """Copy a run directory so a mutation never touches the original.

    A fixture generator that mutated its source in place would corrupt the
    positive fixture the same suite asserts against.
    """
    if not (source / RESULT_FILENAME).is_file():
        raise AuditEvidenceError(f"{source / RESULT_FILENAME} is missing")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def reseal_step_chain(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute each step's wrapper ref and re-thread the request chain.

    ``OptimResult`` verifies that every step's ``record_ref`` addresses its
    own record, that each later request cites the prior step's exact
    refs, and that the run wrapper addresses its own record. After a
    mutation those are stale, so this reseals the run, threads it into
    every request, then walks the steps in order, recomputing each
    wrapper ref from the mutated record and writing the new ref into the
    next step's ``prior_step_result_ref``.

    The mutated field itself is untouched -- only the integrity refs that
    would otherwise reject the artifact before any invariant could judge it.
    """
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    run_ref = _resealed_run(document)
    prior_ref: dict[str, Any] | None = None
    for index, wrapper in enumerate(steps):
        if not isinstance(wrapper, dict):
            raise MutationError(f"step {index} is not a JSON object")
        record = wrapper.get("record")
        if not isinstance(record, dict):
            raise MutationError(f"step {index} carries no record")
        request = record.get("request")
        if not isinstance(request, dict) or not isinstance(
            request.get("record"), dict
        ):
            raise MutationError(f"step {index} carries no request record")
        if run_ref is not None:
            # Every request cites the run, so a mutated run record makes
            # each one stale too.
            request["record"]["run"] = run_ref
        if prior_ref is not None:
            request["record"]["prior_step_result_ref"] = prior_ref
        if run_ref is not None or prior_ref is not None:
            # The request is itself a self-verifying wrapper.
            request["record_ref"] = _request_ref(request["record"])
        try:
            recomputed = step_result_reference(
                OptimStepResult.model_validate(record)
            )
        except ValueError as error:
            raise MutationError(
                f"mutated step {index} is not a valid OptimStepResult: {error}"
            ) from error
        wrapper["record_ref"] = recomputed.record_ref.model_dump(mode="json")
        prior_ref = wrapper["record_ref"]
    return document


def _resealed_run(document: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute the run wrapper's ref, when the run record was mutated.

    ``OptimResult`` verifies that ``run.record_ref`` addresses its own
    record, and every step request cites the run wrapper, so mutating a
    field of the run leaves both stale. Returns None when the ref was
    already exact, so an untouched run is left byte-identical.
    """
    run = document.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("record"), dict):
        raise MutationError("result.json carries no run record")
    try:
        recomputed = optimization_run_reference(
            OptimRun.model_validate(run["record"])
        )
    except ValueError as error:
        raise MutationError(
            f"mutated run is not a valid OptimRun: {error}"
        ) from error
    resealed = recomputed.model_dump(mode="json")
    if resealed == run:
        return None
    run.clear()
    run.update(resealed)
    return resealed


def _request_ref(request_record: dict[str, Any]) -> dict[str, Any]:
    reference = step_request_reference(
        OptimStepRequest.model_validate(request_record)
    )
    return reference.record_ref.model_dump(mode="json")


def reseal_run_binding(document: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the run wrapper from its record and re-embed it everywhere.

    Some invariants are only violable by changing what the run says it was
    *configured* to do -- its optimizer control, its initial candidate, its
    mutation field. Those live on the ``OptimRun`` record, which is embedded
    once at the top of ``result.json`` and again inside every step request,
    each copy self-verifying its own ``record_ref`` and ``config_hash``.

    So a control-level mutation needs three things resealed, in order: the
    run wrapper's own refs, the identical wrapper inside every step request,
    and then the step chain. :func:`reseal_step_chain` does the last, but it
    reseals step 0's request only when a prior step forced it to -- which a
    run-level mutation does not. This does all of it.

    Mutate ``document["run"]["record"]`` first, then call this.
    """
    wrapper = document.get("run")
    if not isinstance(wrapper, dict) or not isinstance(
        wrapper.get("record"), dict
    ):
        raise MutationError("result.json carries no run record")
    record = wrapper["record"]
    try:
        run = OptimRun.model_validate(record)
    except ValueError as error:
        raise MutationError(
            f"mutated run is not a valid OptimRun: {error}"
        ) from error
    resealed = {
        "record": record,
        "record_ref": typed_ref_for_record(
            OPTIM_RUN_SCHEMA, run.record_content()
        ).model_dump(mode="json"),
        "config_hash": run.identity_hash(),
    }
    document["run"] = resealed
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    for index, step_wrapper in enumerate(steps):
        request = step_wrapper.get("record", {}).get("request")
        if not isinstance(request, dict) or not isinstance(
            request.get("record"), dict
        ):
            raise MutationError(f"step {index} carries no request record")
        request["record"]["run"] = resealed
        request["record_ref"] = _request_ref(request["record"])
    return reseal_step_chain(document)


def reseal_request_refs(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute every step's own ``request.record_ref``.

    :func:`reseal_step_chain` recomputes a request ref only for steps with a
    predecessor, because its own worked mutation targets a step *result*
    field. A mutation may instead target the request itself -- GEPA's
    metric-call ceiling lives in ``request.record.hyperparameters`` -- and
    step 0 has no predecessor, so its stale ref would fail validation before
    any invariant could judge the artifact.
    """
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    for index, wrapper in enumerate(steps):
        request = wrapper.get("record", {}).get("request")
        if not isinstance(request, dict) or not isinstance(
            request.get("record"), dict
        ):
            raise MutationError(f"step {index} carries no request record")
        request["record_ref"] = _request_ref(request["record"])
    return document


def reseal_candidate_refs(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute each step's candidate wrapper refs.

    ``CandidateRef`` self-verifies that ``record_ref`` addresses its own
    record, so a mutation to a candidate payload leaves the wrapper stale
    and the artifact fails validation before an invariant can judge it.
    Recomputing is integrity bookkeeping; the semantic violation is the
    mutated payload.
    """
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    proposals = document.get("proposals") or []
    for index, wrapper in enumerate(steps):
        record = wrapper.get("record")
        if not isinstance(record, dict):
            raise MutationError(f"step {index} carries no record")
        candidates = [
            *record.get("proposed_candidates", []),
            *record.get("accepted_candidates", []),
            *(
                proposal["candidate"]
                for proposal in proposals
                if isinstance(proposal, dict) and "candidate" in proposal
            ),
        ]
        retained = record.get("retained_candidate_ref")
        if retained is not None:
            candidates.append(retained)
        for candidate_wrapper in candidates:
            try:
                reference = candidate_reference(
                    Candidate.model_validate(candidate_wrapper["record"])
                )
            except ValueError as error:
                raise MutationError(
                    f"mutated step {index} carries an invalid candidate: "
                    f"{error}"
                ) from error
            candidate_wrapper["record_ref"] = reference.record_ref.model_dump(
                mode="json"
            )
            candidate_wrapper["identity_hash"] = reference.identity_hash
    return document


def rethread_snapshot_refs(document: dict[str, Any]) -> dict[str, Any]:
    """Re-point each request's ``prior_state_ref`` / ``prior_history_ref``.

    ``OptimStepRequest`` cites the prior step's *exact* state and history
    refs, so re-putting a mutated snapshot invalidates the next step's
    request before any invariant can judge the artifact.
    :func:`reseal_step_chain` rethreads ``prior_step_result_ref``, which is
    all a ``result.json``-only mutation needs; a state or history mutation
    additionally needs these two. Both are integrity bookkeeping -- the
    semantic violation is the mutated snapshot itself.
    """
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    for index in range(1, len(steps)):
        prior = steps[index - 1].get("record")
        request = steps[index].get("record", {}).get("request")
        if not isinstance(prior, dict) or not isinstance(request, dict):
            raise MutationError(f"step {index} carries no request record")
        request["record"]["prior_state_ref"] = prior["state_ref"]
        request["record"]["prior_history_ref"] = prior["history_ref"]
    return document


def reseal_all(document: dict[str, Any]) -> dict[str, Any]:
    """Re-seal every integrity ref a single-field mutation can invalidate.

    The four passes run in dependency order: snapshot refs feed the request
    records, candidate wrappers and request refs are each self-verifying,
    and the step chain is recomputed last from the now-consistent records.
    """
    rethread_snapshot_refs(document)
    reseal_candidate_refs(document)
    reseal_request_refs(document)
    return reseal_step_chain(document)


def mutate_run(
    source: Path,
    destination: Path,
    path: tuple[str | int, ...],
    rewrite: Callable[[Any], Any],
) -> Path:
    """Copy ``source`` to ``destination`` and violate one evidence field.

    Returns the mutated run directory, ready to hand to ``audit_run``. The
    result stays a schema-valid ``OptimResult`` -- only the semantic
    invariant under test is violated.

    The caller asserts that its target invariant FAILs against it and that no
    other invariant's status changed -- a mutation that broke everything
    would make a sloppy invariant look sound.
    """
    run_dir = copy_run(source, destination)
    result_path = run_dir / RESULT_FILENAME
    document = json.loads(result_path.read_text(encoding="utf-8"))
    mutate_json_field(document, path, rewrite)
    reseal_all(document)
    result_path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    return run_dir


__all__ = [
    "TOOL_ADMISSION_ENTRY_TABLE",
    "MutationError",
    "copy_run",
    "delete_ledger_entry",
    "mutate_json_field",
    "mutate_ledger_entry",
    "mutate_run",
    "put_record",
    "reseal_all",
    "reseal_candidate_refs",
    "reseal_request_refs",
    "reseal_run_binding",
    "reseal_step_chain",
    "rethread_snapshot_refs",
]
