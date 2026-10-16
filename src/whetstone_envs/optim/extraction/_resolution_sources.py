from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue
from whetstone.core.identity import TypedRef
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.reward import RewardRef
from whetstone.optim.codex.mcp_bridge import CODEX_EVAL_INPUT_FIELDS

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.optim.contracts import (
        IntentResolution,
        OptimResult,
        OptimStepResult,
    )


@dataclass(frozen=True, slots=True)
class _TrajectoryResolutionSource:
    candidate: CandidateRef
    request_id: str
    outcome: str
    eval_result_ref: TypedRef | None
    reward_ref: RewardRef | None
    classification: str | None
    message: str | None
    terminal_failure: dict[str, JsonValue] | None


def _intent_resolution_source(
    resolution: IntentResolution,
) -> _TrajectoryResolutionSource:
    return _TrajectoryResolutionSource(
        candidate=candidate_reference(
            resolution.optim_eval_request.eval_request.candidate
        ),
        request_id=resolution.optim_eval_request.eval_request.request_id,
        outcome=resolution.outcome.value,
        eval_result_ref=resolution.eval_result_ref,
        reward_ref=resolution.reward_ref,
        classification=resolution.detail.classification.value,
        message=resolution.detail.message,
        terminal_failure=(
            None
            if resolution.terminal_failure is None
            else cast(
                "dict[str, JsonValue]",
                resolution.terminal_failure.model_dump(mode="json"),
            )
        ),
    )


def _evidence_reward_ref(
    *, store: ObjectStore, ref: TypedRef | None
) -> RewardRef | None:
    if ref is None or ref.schema_name != EVAL_EVIDENCE_SCHEMA:
        return None
    raw = store.get(ref.reference)
    reward = raw.get("reward_ref") if isinstance(raw, dict) else None
    if reward is None:
        return None
    return RewardRef.model_validate(reward)


def _tool_evidence_sources(
    step: OptimStepResult, *, store: ObjectStore
) -> tuple[tuple[int, _TrajectoryResolutionSource], ...]:
    sources: list[tuple[int, _TrajectoryResolutionSource]] = []
    for evidence in step.tool_evidence:
        entry = evidence.store_entry
        ordinal = entry.capacity_debit_ordinal
        # Refusals before admission consume no capacity and have no evaluation.
        if ordinal is None:
            continue
        record = evidence.result.record
        call = entry.tool_call.record
        base_ref_arg, _route_arg, template_arg = CODEX_EVAL_INPUT_FIELDS
        candidate = candidate_reference(
            Candidate(
                candidate_id=str(call.call_id),
                base_ref=TypedRef.model_validate(call.args[base_ref_arg]),
                payload={
                    str(call.tool_config.record.candidate_template_field): (
                        call.args[template_arg]
                    )
                },
            )
        )
        eval_result_ref = (
            record.evaluation_evidence_refs[0]
            if record.evaluation_evidence_refs
            else None
        )
        if record.terminal_failure is None:
            outcome = "completed"
        # Admission can debit capacity before validation rejects a call.
        elif eval_result_ref is None:
            outcome = "rejected"
        else:
            outcome = "failed"
        sources.append(
            (
                ordinal,
                _TrajectoryResolutionSource(
                    candidate=candidate,
                    request_id=f"tool:{call.call_id}",
                    outcome=outcome,
                    eval_result_ref=eval_result_ref,
                    # Tool and evaluation rewards have different provenance.
                    reward_ref=_evidence_reward_ref(
                        store=store, ref=eval_result_ref
                    ),
                    classification=None,
                    message=(
                        None
                        if record.terminal_failure is None
                        else record.terminal_failure.message
                    ),
                    terminal_failure=(
                        None
                        if record.terminal_failure is None
                        or outcome == "rejected"
                        else cast(
                            "dict[str, JsonValue]",
                            record.terminal_failure.model_dump(mode="json"),
                        )
                    ),
                ),
            )
        )
    return tuple(sources)


def _gepa_transcript_sources(  # noqa: PLR0912
    *, store: ObjectStore, optimizer_result: OptimResult
) -> tuple[tuple[int, int, _TrajectoryResolutionSource], ...]:
    from whetstone.optim.gepa.contracts import (  # noqa: PLC0415
        GEPA_EVALUATION_REQUEST_RECORD_SCHEMA,
        GEPA_EVALUATION_RESULT_RECORD_SCHEMA,
        GepaEffectTranscript,
        GepaEvaluationEffectRequest,
        GepaEvaluationEffectResult,
    )
    from whetstone.optim.gepa.harness_adapter import (  # noqa: PLC0415
        GEPA_TERMINAL_ARTIFACT_KEY,
    )
    from whetstone.optim.gepa.result_artifact import (  # noqa: PLC0415
        GEPA_RUN_RESULT_ARTIFACT_SCHEMA,
        GepaRunResultArtifact,
    )

    if not optimizer_result.step_results:
        return ()
    history_ref = optimizer_result.step_results[-1].record.history_ref
    if history_ref is None:
        return ()
    history = store.get(history_ref.reference)
    if not isinstance(history, dict):
        raise TypeError("optimizer history snapshot must be a JSON object")
    raw_artifact_ref = history.get(GEPA_TERMINAL_ARTIFACT_KEY)
    if raw_artifact_ref is None:
        return ()
    artifact_ref = TypedRef.model_validate(raw_artifact_ref)
    if artifact_ref.schema_name != GEPA_RUN_RESULT_ARTIFACT_SCHEMA:
        raise ValueError("GEPA terminal artifact ref has the wrong schema")
    artifact = GepaRunResultArtifact.model_validate_json(
        json.dumps(store.get(artifact_ref.reference))
    )
    transcript = GepaEffectTranscript.model_validate_json(
        json.dumps(store.get(artifact.effect_transcript_ref.reference))
    )
    if artifact.context != transcript.context:
        raise ValueError("GEPA artifact and transcript contexts disagree")
    if transcript.context.run_id != optimizer_result.run_id:
        raise ValueError("GEPA transcript belongs to another optimizer run")
    sources: list[tuple[int, int, _TrajectoryResolutionSource]] = []
    for entry in transcript.entries:
        if entry.effect_kind != "evaluate":
            continue
        if (
            entry.request_ref.schema_name
            != GEPA_EVALUATION_REQUEST_RECORD_SCHEMA
            or entry.result_ref.schema_name
            != GEPA_EVALUATION_RESULT_RECORD_SCHEMA
        ):
            raise ValueError("GEPA evaluation effect refs have wrong schemas")
        request = GepaEvaluationEffectRequest.model_validate_json(
            json.dumps(store.get(entry.request_ref.reference))
        )
        effect_result = GepaEvaluationEffectResult.model_validate_json(
            json.dumps(store.get(entry.result_ref.reference))
        )
        if (
            request.slot.context != transcript.context
            or request.slot.invocation_ordinal != entry.invocation_ordinal
            or effect_result.request_hash != request.identity_hash()
            or tuple(row.data for row in effect_result.rows) != request.data
        ):
            raise ValueError("GEPA transcript evaluation binding disagrees")
        if effect_result.resolution is None:
            raise ValueError(
                "GEPA transcript evaluation has no harness resolution"
            )
        resolution = effect_result.resolution
        if (
            str(resolution.optim_eval_request.optim_run_id)
            != transcript.context.run_id
        ):
            raise ValueError(
                "GEPA evaluation belongs to another optimizer run"
            )
        step_index = int(resolution.optim_eval_request.optim_step_index)
        if step_index < 0 or step_index >= len(optimizer_result.step_results):
            raise ValueError("GEPA evaluation cites an unknown optimizer step")
        sources.append(
            (
                step_index,
                entry.invocation_ordinal,
                _intent_resolution_source(resolution),
            )
        )
    return tuple(sources)


def _step_resolution_sources(
    step: OptimStepResult,
    *,
    store: ObjectStore,
    gepa_sources: tuple[tuple[int, int, _TrajectoryResolutionSource], ...],
) -> tuple[tuple[int, _TrajectoryResolutionSource], ...]:
    sources = (
        tuple(
            (index, _intent_resolution_source(resolution))
            for index, resolution in enumerate(step.resolved_intents)
        )
        + tuple(
            (ordinal, source)
            for source_step, ordinal, source in gepa_sources
            if source_step == step.step_index
        )
        + _tool_evidence_sources(step, store=store)
    )
    if len({index for index, _source in sources}) != len(sources):
        raise ValueError("trajectory resolution indexes overlap")
    return sources
