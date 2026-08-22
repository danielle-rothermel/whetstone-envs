"""``cost.json``: one run's spend, projected beside its ``result.json``.

Every run writes this file immediately after ``result.json``. It is a
**projection**, not a second source of truth: ``OptimResult.cost`` in
``result.json`` remains the authority, and every number here is copied from
it without rederivation. What the projection buys is a stable, pinned-key
document the study manifest can cite by content hash and a reader can open
without parsing a whole optimization result.

The per-role records are the manifest's own
:class:`~whetstone_envs.optim.study.manifest.RunSpendRecord`, so the study's
accounting surface and the run's own artifact carry the same fields under
the same wire keys. The honesty split travels with them: an absent ``usd``
plus a nonzero ``unpriced_calls`` is the study's most consequential number,
and reporting one without the other would be a total that looks
authoritative while understating spend.

A run whose result carries no cost report writes no ``cost.json``. Writing
an all-zero document would claim the run was free rather than unmeasured.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from dr_store import CanonicalJsonFile
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)
from whetstone.optim.cost import CostRole, RoleCost, RunCostReport

from whetstone_envs.optim.study.manifest import RunSpendRecord

if TYPE_CHECKING:
    from pathlib import Path

    from dr_store import ObjectReference
    from dr_store.sync import BlockingObjectStore
    from whetstone.optim.contracts import OptimResult

__all__ = [
    "MAX_RUN_COST_BYTES",
    "RUN_COST_NAME",
    "RUN_COST_SCHEMA",
    "RUN_COST_SCHEMA_NAME",
    "RUN_COST_SCHEMA_VERSION",
    "RunCostDocument",
    "project_run_cost",
    "read_run_cost",
    "write_run_cost",
]

#: The artifact's filename inside a run directory.
RUN_COST_NAME = "cost.json"

#: Persisted-format contract, pinned by ``tests/optim/test_run_cost.py``.
#: Never derive these from model attribute names.
RUN_COST_SCHEMA_NAME = "whetstone_envs.run_cost"
RUN_COST_SCHEMA_VERSION = 1
RUN_COST_SCHEMA = f"{RUN_COST_SCHEMA_NAME}/v{RUN_COST_SCHEMA_VERSION}"

#: A cost document is one record per provider role, so it stays tiny.
MAX_RUN_COST_BYTES = 256 * 1024


class RunCostDocument(BaseModel):
    """One run's per-role spend, in the manifest's own record shape.

    ``cost_report_schema_version`` is the upstream ``RunCostReport``'s
    version, carried through so a reader can tell which upstream format the
    numbers were projected from without re-opening ``result.json``.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: StrictInt = RUN_COST_SCHEMA_VERSION
    run_id: StrictStr
    cost_report_schema_version: StrictInt
    spend: tuple[RunSpendRecord, ...]

    @model_validator(mode="after")
    def _validate_document(self) -> RunCostDocument:
        if self.schema_version != RUN_COST_SCHEMA_VERSION:
            raise ValueError(
                f"expected schema version {RUN_COST_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if not self.run_id.strip():
            raise ValueError("a cost document names its run")
        roles = [entry.role for entry in self.spend]
        if len(set(roles)) != len(roles):
            raise ValueError("each provider role is reported once")
        if not self.spend:
            raise ValueError(
                "a cost document reports at least one provider role"
            )
        return self


def _spend_record(role: CostRole, cost: RoleCost) -> RunSpendRecord:
    return RunSpendRecord(
        role=role.value,
        calls=cost.calls,
        cached_calls=cost.cached_calls,
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        priced_calls=cost.priced_calls,
        unpriced_calls=cost.unpriced_calls,
        rows_missing_token_breakdown=cost.rows_missing_token_breakdown,
        usd=cost.usd,
    )


def project_run_cost(
    result: OptimResult, *, run_id: str
) -> RunCostDocument | None:
    """Project ``result.cost`` into a cost document, or ``None``.

    ``OptimResult.cost`` travels as a serialized ``RunCostReport``; an empty
    object means the run recorded no cost report at all, which is a
    different fact from a run that spent nothing.
    """
    payload = result.cost.to_json()
    if not payload:
        return None
    report = RunCostReport.model_validate(payload)
    return RunCostDocument(
        run_id=run_id,
        cost_report_schema_version=report.schema_version,
        spend=(
            _spend_record(CostRole.TASK_MODEL, report.task_model),
            _spend_record(CostRole.PROPOSER, report.proposer),
        ),
    )


def _cost_document(directory: Path) -> CanonicalJsonFile:
    return CanonicalJsonFile(
        directory, RUN_COST_NAME, max_bytes=MAX_RUN_COST_BYTES
    )


def write_run_cost(
    directory: Path,
    document: RunCostDocument,
    *,
    store: BlockingObjectStore | None = None,
) -> tuple[Path, ObjectReference | None]:
    """Write ``cost.json`` and, given a store, content-address it there.

    Validation round-trips through the persisted JSON rather than trusting
    the in-memory object, matching how every other artifact in this package
    is published; the same validated payload is what lands in both places,
    so the file and the stored record are the same bytes.

    The store copy is what the study manifest's ``cost_ref`` cites and what
    ``manifest check`` resolves. Without it a manifest could name a cost
    document nothing verifies, which is the one thing the pointer check
    exists to prevent.
    """
    validated = RunCostDocument.model_validate_json(document.model_dump_json())
    payload = validated.model_dump(mode="json")
    file = _cost_document(directory)
    file.publish(payload)
    reference = None
    if store is not None:
        reference, _ = store.put(RUN_COST_SCHEMA_NAME, payload)
    return file.path, reference


def read_run_cost(directory_or_file: Path) -> RunCostDocument:
    """Load and validate the cost document at ``directory_or_file``."""
    path = directory_or_file.resolve()
    directory, filename = (
        (path.parent, path.name) if path.is_file() else (path, RUN_COST_NAME)
    )
    raw = CanonicalJsonFile(
        directory, filename, max_bytes=MAX_RUN_COST_BYTES
    ).read()
    # Via JSON, not the parsed object: strict mode reads a JSON array as a
    # tuple but refuses a Python list, so validating the document the way it
    # was written is what makes the round trip exact.
    return RunCostDocument.model_validate_json(json.dumps(raw))
