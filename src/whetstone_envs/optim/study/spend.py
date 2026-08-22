"""What a stage's own evaluations cost, projected from persisted evidence.

An optimizer run reports its spend through ``OptimResult.cost``, which
whetstone aggregates from the evidence each Step cites. Stage 0 has no
optimizer and no ``OptimResult``: its anchors reach the provider straight
through the evaluation engine, so the same rows exist and nothing collects
them. That is the gap this module closes, and it closes it the same way --
by re-deriving every number from the persisted output rows rather than from
a counter held while the stage ran.

**Every number is read back, never accumulated.** The stage hands over the
:class:`~whetstone.eval.schema.EvalEvidence` its evaluations produced, this
module dereferences each one's ``outputs_ref``, and the per-row usage
fields are aggregated through whetstone's own
:func:`~whetstone.optim.cost.aggregate_role_cost`. That is what makes a
stage total and a run total the same kind of fact: the honesty split
(cached, priced, unpriced, missing token breakdown) and the rule that an
absent ``usd`` is "not knowable" rather than "zero" both come from the
shared aggregator instead of being restated here.

Evidence is de-duplicated by reference, for the reason a run's aggregation
de-duplicates: one evaluation cited twice was paid for once. Anchors are
distinct evaluations on distinct splits, so today no key repeats -- but a
stage that ever re-cited one would double its own bill, and that is not a
failure worth leaving to chance.

**Only the task model spends here.** A stage's own evaluations are task
evaluations; there is no proposer in Stage 0. A proposer role is therefore
omitted from the projection rather than reported as an all-zero row, which
would claim the stage measured a proposer that spent nothing.

This module prices whatever evidence it is given and asks no questions
about where the evidence came from. The harness decides which stages are
worth pricing -- a fake-transport stage produces real rows that would total
to a bill nobody owes -- and that judgement lives at the call site rather
than here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from whetstone.eval.schema import EvalOutputRow, EvalOutputsRecord
from whetstone.execution.call_support import evidences_provider_response
from whetstone.optim.cost import (
    CostRole,
    RoleCost,
    UsageObservation,
    aggregate_role_cost,
)

from whetstone_envs.optim.study.manifest import RunSpendRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dr_store import ObjectStore
    from whetstone.eval.schema import EvalEvidence

__all__ = [
    "row_observation",
    "run_spend_records",
    "stage_spend_records",
]


def row_observation(row: EvalOutputRow) -> UsageObservation | None:
    """Classify one output row as billable, cached, or no call at all.

    The classification mirrors whetstone's own row rule, because a stage's
    rows and a run's rows are the same rows and must be counted the same
    way:

    * a row the prompt cache replayed contributes only a cached call --
      its tokens and price were charged when the original call was made;
    * a row with no usage telemetry that went missing, or that failed
      before the provider answered, evidences no provider call at all and
      is dropped -- but a row the provider *did* answer and the classifier
      then rejected was billed like any other, so it counts;
    * anything else is a billable call, priced only when the provider
      reported a price, and flagged as missing its token breakdown unless
      both directions are present.

    The distinction between "any token count" and "a token breakdown"
    matters twice over: one direction is enough to prove the call happened,
    but only both make the token totals complete, and collapsing the two
    would either drop a call or hide an understated token total.
    """
    has_any_token_count = (
        row.prompt_tokens is not None or row.completion_tokens is not None
    )
    has_token_breakdown = (
        row.prompt_tokens is not None and row.completion_tokens is not None
    )
    has_telemetry = has_any_token_count or row.provider_cost is not None
    if row.cache_hit:
        return UsageObservation(cached=True)
    if not has_telemetry and row.missing:
        return None
    if (
        not has_telemetry
        and row.failed
        and not evidences_provider_response(
            None
            if row.provider_error is None
            else dict(row.provider_error.to_json())
        )
    ):
        return None
    return UsageObservation(
        input_tokens=row.prompt_tokens or 0,
        output_tokens=row.completion_tokens or 0,
        usd=row.provider_cost,
        missing_token_breakdown=not has_token_breakdown,
    )


def _observations(
    store: ObjectStore, evidence: Iterable[EvalEvidence]
) -> tuple[UsageObservation, ...]:
    """Every billable observation behind a stage's distinct evaluations."""
    seen: set[tuple[str, str]] = set()
    observations: list[UsageObservation] = []
    for record in evidence:
        outputs_ref = record.outputs_ref
        key = (outputs_ref.schema_name, outputs_ref.content_hash)
        if key in seen:
            continue
        seen.add(key)
        outputs = EvalOutputsRecord.model_validate_json(
            json.dumps(store.get(outputs_ref.reference))
        )
        for row in outputs.outputs:
            observation = row_observation(row)
            if observation is not None:
                observations.append(observation)
    return tuple(observations)


def _spend_record(cost: RoleCost) -> RunSpendRecord:
    return RunSpendRecord(
        role=CostRole.TASK_MODEL.value,
        calls=cost.calls,
        cached_calls=cost.cached_calls,
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        priced_calls=cost.priced_calls,
        unpriced_calls=cost.unpriced_calls,
        rows_missing_token_breakdown=cost.rows_missing_token_breakdown,
        usd=cost.usd,
    )


def stage_spend_records(
    *,
    store: ObjectStore,
    evidence: Iterable[EvalEvidence],
) -> tuple[RunSpendRecord, ...]:
    """One stage's task-model spend, or nothing when it evidenced none.

    An empty tuple is the honest answer for a stage whose evaluations left
    no rows evidencing a provider call. Reporting a zero-call task-model
    role instead would say the stage measured its spend and found it free,
    which is a different and untrue claim.

    Whether a stage should be priced at all is the caller's decision, not
    this function's: a fake-transport stage produces real rows that this
    module would happily total, and the harness declines to ask. Keeping
    that judgement at the call site leaves this function a pure projection
    of whatever evidence it is handed.
    """
    observations = _observations(store, evidence)
    if not observations:
        return ()
    return (_spend_record(aggregate_role_cost(observations)),)


def run_spend_records(
    runs: Iterable[RunSpendRecord],
) -> tuple[RunSpendRecord, ...]:
    """Every run's per-role spend, folded into one per-role total.

    An arm stage does not evaluate through the engine the way Stage 0 does:
    it spends through optimizer runs, and each run already re-derived its
    own per-role bill from its own evidence. The stage total is therefore
    a **sum of those records**, not a second measurement of the same calls
    -- re-reading the rows here would risk counting a call twice and would
    make the stage row and the run rows able to disagree.

    The honesty rules survive the fold because they are re-applied to the
    total rather than carried from the parts:

    * ``usd`` is ``None`` for a role as soon as *any* contributing run left
      it ``None``. A sum over the priced runs alone would look
      authoritative while understating the role's bill, which is the same
      rule ``RunSpendRecord`` enforces within one run.
    * The counters add, so ``priced_calls + unpriced_calls == calls``
      continues to hold and the model's own validator re-checks it.

    Roles are returned in first-seen order, so a stage whose arms ran in a
    fixed order renders deterministically.
    """
    totals: dict[str, list[int]] = {}
    usd_by_role: dict[str, float | None] = {}
    order: list[str] = []
    for entry in runs:
        if entry.role not in totals:
            order.append(entry.role)
            totals[entry.role] = [0, 0, 0, 0, 0, 0, 0]
            usd_by_role[entry.role] = 0.0
        running = totals[entry.role]
        for index, value in enumerate(
            (
                entry.calls,
                entry.cached_calls,
                entry.input_tokens,
                entry.output_tokens,
                entry.priced_calls,
                entry.unpriced_calls,
                entry.rows_missing_token_breakdown,
            )
        ):
            running[index] += value
        known = usd_by_role[entry.role]
        usd_by_role[entry.role] = (
            None if known is None or entry.usd is None else known + entry.usd
        )
    return tuple(
        RunSpendRecord(
            role=role,
            calls=totals[role][0],
            cached_calls=totals[role][1],
            input_tokens=totals[role][2],
            output_tokens=totals[role][3],
            priced_calls=totals[role][4],
            unpriced_calls=totals[role][5],
            rows_missing_token_breakdown=totals[role][6],
            usd=usd_by_role[role],
        )
        for role in order
    )
