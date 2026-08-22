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
from typing import TYPE_CHECKING, Any

from dr_store.sync import open_sqlite
from whetstone.core.identity import typed_ref_for_record
from whetstone.optim.contracts import (
    OPTIM_RUN_SCHEMA,
    OptimRun,
    OptimStepRequest,
    OptimStepResult,
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
    own record and that each later request cites the prior step's exact
    refs. After a mutation both are stale, so this walks the steps in order,
    recomputes the wrapper ref from the mutated record, and writes the new
    ref into the next step's ``prior_step_result_ref``.

    The mutated field itself is untouched -- only the integrity refs that
    would otherwise reject the artifact before any invariant could judge it.
    """
    steps = document.get("step_results")
    if not isinstance(steps, list):
        raise MutationError("result.json carries no step_results list")
    prior_ref: dict[str, Any] | None = None
    for index, wrapper in enumerate(steps):
        if not isinstance(wrapper, dict):
            raise MutationError(f"step {index} is not a JSON object")
        record = wrapper.get("record")
        if not isinstance(record, dict):
            raise MutationError(f"step {index} carries no record")
        if prior_ref is not None:
            request = record.get("request")
            if not isinstance(request, dict) or not isinstance(
                request.get("record"), dict
            ):
                raise MutationError(f"step {index} carries no request record")
            request["record"]["prior_step_result_ref"] = prior_ref
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
    reseal_step_chain(document)
    result_path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    return run_dir


__all__ = [
    "MutationError",
    "copy_run",
    "mutate_json_field",
    "mutate_run",
    "put_record",
    "reseal_run_binding",
    "reseal_step_chain",
]
