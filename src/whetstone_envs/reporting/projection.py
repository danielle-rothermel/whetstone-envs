from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, cast, get_args

from pydantic import JsonValue
from whetstone.core.identity import TypedRef
from whetstone.eval import (
    EvalEvidence,
    EvalEvidenceWithRef,
    EvalFailureEvidence,
    EvalOutputRow,
    EvalOutputsRecord,
    EvalTraces,
)
from whetstone.eval import (
    EvalRejected as WhetstoneEvalRejected,
)
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.reward import RewardRef
from whetstone.optim.codex.mcp_bridge import CODEX_EVAL_INPUT_FIELDS
from whetstone.optim.contracts import (
    IntentResolution,
    OptimResult,
    optimization_result_reference,
)
from whetstone.optim.cost import RoleCost, RunCostReport

from whetstone_envs.optim.study.spend import stage_spend_records
from whetstone_envs.probes import normalize
from whetstone_envs.reporting.schema import (
    EVAL_REPORT_SCHEMA,
    SPEND_SCHEMA_VERSION,
    TRAJECTORY_REPORT_SCHEMA,
    CandidateRecord,
    CandidateSource,
    EvalFailed,
    EvalRejected,
    EvalReport,
    EvalRoleName,
    EvalRun,
    EvalSpend,
    EvalSuccess,
    EvaluationResult,
    FamilyName,
    Observation,
    ObservationState,
    ProviderErrorProjection,
    ReportedEvidence,
    ReportRef,
    RoleSpend,
    RoleSpendName,
    RowAccounting,
    RunSpend,
    StratumSummary,
    TaskRecord,
    TrajectoryCandidate,
    TrajectoryReport,
    TrajectoryResolution,
    TrajectoryStep,
    two_stage_task_mean,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval import EvalResult
    from whetstone.experiment.sampling import EvalSplit
    from whetstone.optim.contracts import OptimStepResult

    from whetstone_envs.instances import Instance
    from whetstone_envs.optim.experiment import PreparedExperiment


def _ref(value: TypedRef | None) -> ReportRef | None:
    if value is None:
        return None
    return ReportRef(
        schema_name=value.schema_name,
        content_hash=str(value.content_hash),
    )


def _required_ref(value: TypedRef) -> ReportRef:
    result = _ref(value)
    assert result is not None
    return result


#: The package prefix every task family's namespace carries. A prepared
#: experiment names itself ``whetstone_envs.<family>``, so the report's
#: family field is read from that namespace rather than passed in and
#: possibly disagreeing with the evidence it labels.
_FAMILY_NAMESPACE_PREFIX = "whetstone_envs."


def _family_name(prepared: PreparedExperiment) -> FamilyName:
    """Name the task family that authored this evidence.

    ``FamilyName`` is a persisted-format literal, so an unrecognised
    namespace is refused here rather than written into a report the schema
    would reject on read-back.
    """
    env_name = prepared.experiment.env_name
    family = env_name.removeprefix(_FAMILY_NAMESPACE_PREFIX)
    if family not in get_args(FamilyName):
        raise ValueError(
            f"experiment namespace {env_name!r} names no reportable family"
        )
    return cast("FamilyName", family)


def _instances_for_role(
    prepared: PreparedExperiment, role: EvalRoleName
) -> tuple[Instance, ...]:
    if role == "internal":
        return prepared.split.internal_eval
    if role == "official":
        return prepared.split.official
    if role == "held_out":
        return prepared.split.held_out
    raise ValueError(f"unsupported evaluation role {role!r}")


def _eval_split_for_role(
    prepared: PreparedExperiment, role: EvalRoleName
) -> EvalSplit:
    """Resolve the prepared eval split for one reporting role.

    Held-out is optional upstream, so an absent held-out split is refused by
    name rather than silently reported against another role's tasks.
    """
    configs = prepared.experiment.eval_configs
    if role == "internal":
        return configs.internal
    if role == "official":
        return configs.official
    if role == "held_out":
        if configs.held_out is None:
            raise ValueError(
                "this experiment has no held_out split; request a "
                "positive held-out split size to evaluate the held_out role"
            )
        return configs.held_out
    raise ValueError(f"unsupported evaluation role {role!r}")


def _candidate_record(
    *, name: str, source: CandidateSource, candidate: CandidateRef
) -> CandidateRecord:
    template = candidate.record.payload.get("prompt_template")
    if type(template) is not str:
        raise ValueError(
            f"candidate {candidate.record.candidate_id!r} at "
            f"{candidate.record_ref.content_hash} has no strict string "
            "prompt_template"
        )
    return CandidateRecord(
        name=name,
        candidate_id=candidate.record.candidate_id,
        source=source,
        record_ref=_required_ref(candidate.record_ref),
        identity_hash=str(candidate.identity_hash),
        payload=candidate.record.payload.to_json(),
        prompt_template=template,
    )


def _row_state(row: EvalOutputRow) -> ObservationState:
    if row.score is not None:
        return ObservationState.SCORED
    if row.failed:
        return ObservationState.FAILED
    if row.missing:
        return ObservationState.MISSING
    if row.invalid:
        return ObservationState.INVALID
    raise ValueError("output row has no reportable state")


def _provider_error_projection(
    value: object | None,
) -> ProviderErrorProjection | None:
    if value is None:
        return None
    to_json = getattr(value, "to_json", None)
    if not callable(to_json):
        raise TypeError("provider error is not a JSON object")
    payload = to_json()
    if not isinstance(payload, dict):
        raise TypeError("provider error is not a JSON object")
    failure_class = payload.get("failure_class")
    transport_failure = payload.get("transport_failure")
    rejected_response = payload.get("rejected_response")
    if isinstance(transport_failure, dict) == isinstance(
        rejected_response, dict
    ):
        raise ValueError(
            "provider error requires exactly one typed failure source"
        )
    if isinstance(transport_failure, dict):
        return ProviderErrorProjection.model_validate(
            {
                "failure_class": failure_class,
                "source": "transport_failure",
                "recoverability": transport_failure.get("recoverability"),
                "status_code": transport_failure.get("status_code"),
                "timeout_containment": transport_failure.get("containment"),
            }
        )
    return ProviderErrorProjection.model_validate(
        {
            "failure_class": failure_class,
            "source": "rejected_response",
        }
    )


def _accounting(observations: tuple[Observation, ...]) -> RowAccounting:
    counts = Counter(row.state for row in observations)
    return RowAccounting(
        planned=len(observations),
        present=counts[ObservationState.SCORED],
        missing=counts[ObservationState.MISSING],
        failed=counts[ObservationState.FAILED],
        invalid=counts[ObservationState.INVALID],
    )


def _summaries(
    observations: tuple[Observation, ...], tasks: tuple[TaskRecord, ...]
) -> tuple[StratumSummary, ...]:
    task_by_id = {task.task_id: task for task in tasks}
    labels = tuple(
        dict.fromkeys(label for task in tasks for label in task.strata)
    )
    summaries: list[StratumSummary] = []
    for label in labels:
        rows = tuple(
            row
            for row in observations
            if label in task_by_id[row.task_id].strata
        )
        accounting = _accounting(rows)
        # Strata partition tasks, not rows, so a stratum's score gets the
        # same two-stage treatment as the overall score: each task's
        # repeats reduce to one value, then those are meaned across the
        # stratum's tasks. ``numerator``/``denominator`` stay row-level.
        numerator = sum(row.score == 1.0 for row in rows)
        summaries.append(
            StratumSummary(
                stratum=label,
                numerator=numerator,
                denominator=len(rows),
                accounting=accounting,
                score=two_stage_task_mean(rows),
            )
        )
    return tuple(summaries)


def _success_projection(
    *,
    store: ObjectStore,
    name: str,
    result: EvalEvidenceWithRef,
    instances: tuple[Instance, ...],
) -> tuple[EvalSuccess, tuple[TaskRecord, ...], tuple[Observation, ...]]:
    evidence = result.evidence
    if not isinstance(evidence, EvalEvidence):
        raise TypeError("success projection requires EvalEvidence")
    outputs = EvalOutputsRecord.model_validate_json(
        json.dumps(store.get(evidence.outputs_ref.reference))
    )
    traces = EvalTraces.model_validate_json(
        json.dumps(store.get(evidence.traces_ref.reference))
    )
    if outputs.traces_ref != evidence.traces_ref:
        raise ValueError("outputs do not cite the evidence trace record")
    if (
        outputs.candidate != evidence.candidate
        or traces.candidate != evidence.candidate
    ):
        raise ValueError(
            "evidence, outputs, and traces must cite one candidate"
        )
    if (
        outputs.task_hashes != evidence.task_hashes
        or traces.task_hashes != evidence.task_hashes
    ):
        raise ValueError("evidence task plans disagree")
    if len(instances) != len(outputs.task_hashes):
        raise ValueError("source split does not match evidence task count")

    tasks: list[TaskRecord] = []
    for index, (instance, task_hash) in enumerate(
        zip(instances, outputs.task_hashes, strict=True)
    ):
        output = outputs.outputs[index * outputs.num_seeds]
        if output.task_id != instance.id or output.task_hash != task_hash:
            raise ValueError(
                "evidence task does not exactly join source instance"
            )
        tasks.append(
            TaskRecord(
                task_id=instance.id,
                task_hash=task_hash,
                seed=instance.seed,
                strata=instance.strata,
                prompt_inputs=dict(instance.prompt_inputs),
                gold=instance.gold,
            )
        )

    observations: list[Observation] = []
    for output, trace_row in zip(outputs.outputs, traces.rows, strict=True):
        if output.task_trial_key() != trace_row.task_trial_key():
            raise ValueError("output and trace coordinates disagree")
        state = _row_state(output)
        observations.append(
            Observation(
                candidate_name=name,
                task_id=output.task_id,
                task_hash=output.task_hash,
                task_index=output.task_index,
                seed_index=output.seed_index,
                rendered_prompt=output.rendered_prompt,
                output_text=output.output_text,
                normalized_output=(
                    normalize(output.output_text)
                    if output.output_text is not None
                    else None
                ),
                score=output.score,
                state=state,
                trace_state=trace_row.trace.row_state.value,
                failure_code=output.failure_code,
                finish_reason=output.finish_reason,
                provider_error=_provider_error_projection(
                    output.provider_error
                ),
                max_budget=output.max_budget,
                over_budget=output.over_budget,
                submission_result=(
                    None
                    if output.submission_result is None
                    else output.submission_result.model_dump(mode="json")
                ),
                component_trace=tuple(
                    step.model_dump(mode="json")
                    for step in trace_row.trace.trace_steps
                ),
            )
        )
    projected = tuple(observations)
    accounting = _accounting(projected)
    reported = RowAccounting.model_validate(
        evidence.row_accounting.model_dump(mode="json")
    )
    if accounting != reported:
        raise ValueError("recomputed row accounting disagrees with evidence")
    numerator = sum(row.score == 1.0 for row in projected)
    # Recomputed from the projected rows, deliberately *not* read off
    # ``evidence.per_task_values``: this comparison is the independent
    # check that the persisted aggregate matches the rows it claims to
    # summarize, and sourcing the per-task values from the same evidence
    # would make it compare evidence against itself. Independence belongs
    # in the inputs, not the arithmetic -- the arithmetic must match
    # whetstone-ai's ``unweighted_task_mean`` bit for bit.
    score = two_stage_task_mean(projected)
    if evidence.aggregate_value != score:
        raise ValueError("recomputed aggregate disagrees with evidence")
    if (
        evidence.reward_ref is not None
        and evidence.reward_ref.record.value != score
    ):
        raise ValueError("recomputed score disagrees with evidence reward")
    success = EvalSuccess(
        kind="success",
        candidate_name=name,
        classification="measured",
        message="evaluation completed",
        evidence=ReportedEvidence(
            evidence_ref=_required_ref(result.evidence_ref),
            outputs_ref=_required_ref(evidence.outputs_ref),
            traces_ref=_required_ref(evidence.traces_ref),
            aggregate_ref=_required_ref(evidence.aggregate_ref),
            reward_ref=(
                None
                if evidence.reward_ref is None
                else _required_ref(evidence.reward_ref.record_ref)
            ),
            aggregate_name=evidence.aggregate_name,
            aggregate_value=evidence.aggregate_value,
            aggregate_status=evidence.aggregate_status,
            row_accounting=reported,
        ),
        accounting=accounting,
        numerator=numerator,
        denominator=len(projected),
        score=score,
        strata=_summaries(projected, tuple(tasks)),
    )
    return success, tuple(tasks), projected


def project_eval_report(  # noqa: PLR0913
    *,
    store: ObjectStore,
    prepared: PreparedExperiment,
    run_id: str,
    transport: str,
    model: str,
    role: EvalRoleName,
    split_sizes: tuple[int, int, int],
    candidates: tuple[tuple[str, CandidateSource, CandidateRef], ...],
    results: tuple[EvalResult, ...],
    task_hashes: tuple[str, ...] | None = None,
) -> EvalReport:
    if len(candidates) != len(results):
        raise ValueError("candidate/result counts disagree")
    all_instances = _instances_for_role(prepared, role)
    candidate_records = tuple(
        _candidate_record(name=name, source=source, candidate=candidate)
        for name, source, candidate in candidates
    )
    projected_results: list[EvaluationResult] = []
    report_observations: list[Observation] = []
    graph_hash = prepared.experiment.rollout_graph.graph_hash
    eval_split = _eval_split_for_role(prepared, role)
    all_tasks = tuple(
        zip(
            all_instances,
            eval_split.tasks,
            eval_split.task_set.task_hashes,
            strict=True,
        )
    )
    if task_hashes is None:
        selected_tasks = all_tasks
    else:
        by_hash = {
            task_hash: (instance, row, task_hash)
            for instance, row, task_hash in all_tasks
        }
        if len(by_hash) != len(all_tasks):
            raise ValueError("prepared evaluation task hashes must be unique")
        try:
            selected_tasks = tuple(
                by_hash[task_hash] for task_hash in task_hashes
            )
        except KeyError as error:
            raise ValueError(
                "evaluation evidence cites a task outside the prepared split"
            ) from error
        if len(set(task_hashes)) != len(task_hashes):
            raise ValueError("evaluation evidence task hashes must be unique")
    instances = tuple(instance for instance, _row, _hash in selected_tasks)
    report_task_records: list[TaskRecord] = []
    for instance, row, task_hash in selected_tasks:
        if row.task_id != instance.id:
            raise ValueError("prepared task does not match source instance")
        report_task_records.append(
            TaskRecord(
                task_id=instance.id,
                task_hash=task_hash,
                seed=instance.seed,
                strata=instance.strata,
                prompt_inputs=dict(instance.prompt_inputs),
                gold=instance.gold,
            )
        )
    report_tasks = tuple(report_task_records)
    for (name, _source, _candidate), result in zip(
        candidates, results, strict=True
    ):
        if isinstance(result, WhetstoneEvalRejected):
            projected_results.append(
                EvalRejected(
                    kind="rejected",
                    candidate_name=name,
                    classification=result.detail.classification.value,
                    message=result.detail.message,
                )
            )
            continue
        if isinstance(result.evidence, EvalFailureEvidence):
            projected_results.append(
                EvalFailed(
                    kind="failed",
                    candidate_name=name,
                    classification="execution",
                    message=result.evidence.message,
                    evidence_ref=_required_ref(result.evidence_ref),
                    exception_type=result.evidence.exception_type,
                )
            )
            continue
        success, tasks, observations = _success_projection(
            store=store,
            name=name,
            result=result,
            instances=instances,
        )
        if report_tasks != tasks:
            raise ValueError(
                "candidate evaluations do not share one exact task plan"
            )
        projected_results.append(success)
        report_observations.extend(observations)
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
        run=EvalRun(
            run_id=run_id,
            family=_family_name(prepared),
            transport=transport,
            model=model,
            role=role,
            split_sizes=split_sizes,
            repeats=eval_split.seed_plan.num_seeds,
            dataset_revision=eval_split.task_set.dataset_revision,
            graph_hash=graph_hash,
            eval_config_hash=eval_split.eval_config.config_hash,
            package_version=version("whetstone-envs"),
        ),
        candidates=candidate_records,
        tasks=report_tasks,
        observations=tuple(report_observations),
        results=tuple(projected_results),
        # Projected here rather than by the caller, so an evaluation cannot
        # be published without its bill having been looked for.
        spend=project_eval_spend(store=store, results=results),
    )


def _dispositions_append(
    dispositions: dict[TypedRef, list[str]], ref: TypedRef, value: str
) -> None:
    values = dispositions.setdefault(ref, [])
    if value not in values:
        values.append(value)


def _comparison(
    parent: EvalReport | None, current: EvalReport | None
) -> tuple[int | None, int | None, int | None]:
    if parent is None or current is None:
        return None, None, None
    prior = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in parent.observations
    }
    present = {
        (row.task_id, row.task_hash, row.seed_index): row
        for row in current.observations
    }
    planned = tuple(
        dict.fromkeys(
            (
                task.task_id,
                task.task_hash,
                seed_index,
            )
            for report in (parent, current)
            for task in report.tasks
            for seed_index in range(report.run.repeats)
        )
    )
    gains = regressions = mismatches = 0
    for coordinate in planned:
        other = prior.get(coordinate)
        row = present.get(coordinate)
        if (
            row is None
            or other is None
            or row.state is not ObservationState.SCORED
            or other.state is not ObservationState.SCORED
        ):
            mismatches += 1
        elif other.score == 0.0 and row.score == 1.0:
            gains += 1
        elif other.score == 1.0 and row.score == 0.0:
            regressions += 1
    return gains, regressions, mismatches


class _TrajectoryEvaluationHistory:
    def __init__(self) -> None:
        self._latest: dict[TypedRef, EvalReport] = {}

    def compare_then_remember(
        self,
        *,
        candidate_ref: TypedRef,
        base_ref: TypedRef,
        report: EvalReport | None,
    ) -> tuple[int | None, int | None, int | None]:
        comparison = _comparison(self._latest.get(base_ref), report)
        if report is not None:
            self._latest[candidate_ref] = report
        return comparison


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
    """The Reward the ``EvalEvidence`` at ``ref`` cites, if any."""
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
    """Trajectory rows for the evaluations a TOOL_USING step paid for.

    Codex is the only such optimizer. It resolves no intent and mints no
    search evidence by design -- its paid evaluations are cited from
    ``tool_evidence`` instead -- so a projection reading only the intent
    path renders a Codex run as having evaluated nothing, and the report
    shows a terminal candidate with no measurement behind it.

    Each row is attributed to the candidate the Tool Call actually built:
    the ``base_ref`` the call named, mutated by the ``template`` it
    submitted. That is the same reconstruction the adapter performs when
    it rebuilds the accepted candidate, so the report attributes an
    evaluation to what was measured rather than to the step's seed.

    A refused call contributes no row: it bought nothing. A call that
    terminalized with a failure does contribute one, carrying that
    failure, because the run paid for it.

    **A paid call can terminalize without ever being evaluated.** The
    evaluator admits a call, then validates it, so a real agent that
    submits a template the family's render contract does not accept --
    an unavailable field, say -- gets ``tool_evaluation_rejected`` *after*
    admission: capacity is debited, and no ``EvalEvidence`` is ever
    minted. That is ``rejected``, not ``failed``: the two differ in the
    report exactly as they do on the intent path, where ``failed`` means
    an evaluation ran and ended badly while ``rejected`` means none
    happened. Projecting it as ``failed`` produced a row claiming a
    failure evaluation it had no ref for, and the schema refused it --
    which took down publication for the whole run, turning one wasted
    call into a lost run. The fake CLI is handed valid arguments, so it
    can never reach this state.

    The reward cited is the ``EvalEvidence``'s own, not the Tool
    Result's. The two are different records: the Tool Result's Reward
    cites the evidence and carries the call's ``provenance_ordinal``,
    while the report's embedded evaluation is projected from the
    evidence, so citing the Tool Result's would make every row disagree
    with the evidence rendered beneath it.
    """
    sources: list[tuple[int, _TrajectoryResolutionSource]] = []
    for evidence in step.tool_evidence:
        entry = evidence.store_entry
        ordinal = entry.capacity_debit_ordinal
        if ordinal is None:
            # A refusal debits no capacity and evaluated nothing.
            continue
        record = evidence.result.record
        call = entry.tool_call.record
        # The wire keys are fixed by the Tool's input schema, whose one
        # owner is whetstone-ai's ``CODEX_EVAL_INPUT_FIELDS`` -- the same
        # constant ``optim/audit/codex.py`` checks the ledger against.
        # Re-spelling them here would let this projection and the audit
        # disagree about the surface a foreign agent was handed. The
        # unpack pins the schema's arity as well as its names, so a
        # widened or narrowed tool surface fails here rather than
        # silently reading the wrong key.
        #
        # ``model_route`` is deliberately unread: it is the route the call
        # asserted, which ``EngineToolEvaluator`` already validated
        # against the engine, not part of the candidate being rebuilt.
        #
        # The wire key is not the payload field it lands in -- that is the
        # run's mutation field, which the Tool Config names. They are
        # different names for a reason and must not be conflated:
        # ``EngineToolEvaluator`` reads the first and writes the second,
        # and this reconstruction mirrors it.
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
        elif eval_result_ref is None:
            # Paid for, never evaluated: rejected after admission.
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
                    reward_ref=_evidence_reward_ref(
                        store=store, ref=eval_result_ref
                    ),
                    classification=None,
                    # A rejected row carries no durable eval result, so
                    # every *evidence* field the schema forbids on it
                    # stays absent -- the refs, the reward, the hydrated
                    # report, and the structured failure below.
                    #
                    # The message is not evidence and the schema permits
                    # it on a rejected row, so it is preserved: it is the
                    # only place the reason a post-admission call was
                    # rejected survives in the projected trajectory.
                    # Dropping it left the row saying a call was rejected
                    # with no readable account of why.
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


def _role_spend(role: RoleSpendName, report: RoleCost) -> RoleSpend:
    return RoleSpend(
        role=role,
        calls=report.calls,
        cached_calls=report.cached_calls,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        priced_calls=report.priced_calls,
        unpriced_calls=report.unpriced_calls,
        rows_missing_token_breakdown=report.rows_missing_token_breakdown,
        usd=report.usd,
    )


def project_run_spend(result: OptimResult) -> RunSpend | None:
    """Project the run's cost report, when the result carries one.

    ``OptimResult.cost`` travels as a serialized ``RunCostReport``; an empty
    object means the run recorded no cost report at all.
    """
    payload = result.cost.to_json()
    if not payload:
        return None
    cost = RunCostReport.model_validate(payload)
    if cost.schema_version != SPEND_SCHEMA_VERSION:
        raise ValueError(
            "unsupported whetstone-ai cost report schema version "
            f"{cost.schema_version}; this package projects only "
            f"{SPEND_SCHEMA_VERSION}"
        )
    return RunSpend(
        schema_version=SPEND_SCHEMA_VERSION,
        task_model=_role_spend("task_model", cost.task_model),
        proposer=_role_spend("proposer", cost.proposer),
    )


def project_eval_spend(
    *, store: ObjectStore, results: tuple[EvalResult, ...]
) -> EvalSpend | None:
    """What a standalone evaluation's provider calls cost, or ``None``.

    An optimizer run reports its bill through ``OptimResult.cost``, which
    whetstone aggregates from the evidence each Step cites. A standalone
    evaluation has no ``OptimResult`` and no Step: it reaches the provider
    straight through the engine, so the same rows exist and nothing
    collects them. This closes that gap the way the study's Stage 0 route
    does -- by re-deriving every number from the persisted output rows
    through whetstone's own aggregator, so an evaluation's bill and a run's
    bill are the same kind of fact under the same honesty rules.

    Only the task model appears: an evaluation runs no proposer, and an
    all-zero proposer row would claim it measured one and found it free.

    ``None`` when the rows evidence no provider call at all -- which is the
    fake transport's honest answer, and a paid evaluation's honest answer
    when its rows carried no usage telemetry. An all-zero block would say
    the evaluation was measured and cost nothing.
    """
    evidence = tuple(
        result.evidence
        for result in results
        if not isinstance(result, WhetstoneEvalRejected)
        and isinstance(result.evidence, EvalEvidence)
    )
    records = stage_spend_records(store=store, evidence=evidence)
    if not records:
        return None
    (record,) = records
    return EvalSpend(
        schema_version=SPEND_SCHEMA_VERSION,
        task_model=RoleSpend(
            role="task_model",
            calls=record.calls,
            cached_calls=record.cached_calls,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            priced_calls=record.priced_calls,
            unpriced_calls=record.unpriced_calls,
            rows_missing_token_breakdown=(record.rows_missing_token_breakdown),
            usd=record.usd,
        ),
    )


def project_trajectory_report(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    store: ObjectStore,
    prepared: PreparedExperiment,
    result_ref: TypedRef,
    result: OptimResult,
    transport: str,
    model: str,
    split_sizes: tuple[int, int, int],
) -> TrajectoryReport:
    if optimization_result_reference(result) != result_ref:
        raise ValueError(
            "terminal OptimResult reference does not address result"
        )
    ordered: list[CandidateRef] = []
    dispositions: dict[TypedRef, list[str]] = {}
    first_step: dict[TypedRef, int] = {}

    def discover(candidate: CandidateRef, step: int, disposition: str) -> None:
        ref = candidate.record_ref
        if ref not in first_step:
            first_step[ref] = step
            ordered.append(candidate)
        _dispositions_append(dispositions, ref, disposition)

    trajectory_steps: list[TrajectoryStep] = []
    trajectory_resolutions: list[TrajectoryResolution] = []
    evaluation_history = _TrajectoryEvaluationHistory()
    gepa_sources = _gepa_transcript_sources(
        store=store, optimizer_result=result
    )
    for step_ref in result.step_results:
        step = step_ref.record
        request_candidates = tuple(
            candidate_reference(candidate)
            for candidate in step.request.record.candidates
        )
        for candidate in request_candidates:
            discover(candidate, step.step_index, "requested")
        for candidate in step.proposed_candidates:
            discover(candidate, step.step_index, "proposed")
        for candidate in step.accepted_candidates:
            discover(candidate, step.step_index, "accepted")
        accepted_refs = {
            candidate.record_ref for candidate in step.accepted_candidates
        }
        for candidate in step.proposed_candidates:
            if candidate.record_ref not in accepted_refs:
                _dispositions_append(
                    dispositions, candidate.record_ref, "rejected"
                )
        resolution_indexes: list[int] = []
        resolution_sources = (
            tuple(
                (index, _intent_resolution_source(resolution))
                for index, resolution in enumerate(step.resolved_intents)
            )
            + tuple(
                (invocation_ordinal, source)
                for source_step, invocation_ordinal, source in gepa_sources
                if source_step == step.step_index
            )
            + _tool_evidence_sources(step, store=store)
        )
        if len({index for index, _source in resolution_sources}) != len(
            resolution_sources
        ):
            raise ValueError("trajectory resolution indexes overlap")
        for resolution_index, resolution in resolution_sources:
            candidate = resolution.candidate
            discover(candidate, step.step_index, "evaluated")
            if resolution.outcome in {"rejected", "failed"}:
                _dispositions_append(
                    dispositions,
                    candidate.record_ref,
                    resolution.outcome,
                )
            resolution_indexes.append(resolution_index)
            embedded: EvalReport | None = None
            raw_result = None
            selected_task_hashes: tuple[str, ...] | None = None
            if resolution.eval_result_ref is not None:
                raw = store.get(resolution.eval_result_ref.reference)
                if resolution.outcome == "completed":
                    evidence = EvalEvidence.model_validate_json(
                        json.dumps(raw)
                    )
                    raw_result = EvalEvidenceWithRef(
                        evidence, resolution.eval_result_ref
                    )
                    selected_task_hashes = evidence.task_hashes
                else:
                    failure = EvalFailureEvidence.model_validate_json(
                        json.dumps(raw)
                    )
                    raw_result = EvalEvidenceWithRef(
                        failure, resolution.eval_result_ref
                    )
            if raw_result is not None:
                embedded = project_eval_report(
                    store=store,
                    prepared=prepared,
                    run_id=f"{result.run_id}:{step.step_index}:{resolution_index}",
                    transport=transport,
                    model=model,
                    role="internal",
                    split_sizes=split_sizes,
                    candidates=(
                        (
                            candidate.record.candidate_id,
                            "optimized",
                            candidate,
                        ),
                    ),
                    results=(raw_result,),
                    task_hashes=selected_task_hashes,
                )
            embedded_result = None if embedded is None else embedded.results[0]
            classification = resolution.classification
            message = resolution.message
            if classification is None:
                classification = (
                    resolution.outcome
                    if embedded_result is None
                    else embedded_result.classification
                )
            if message is None:
                message = (
                    f"search evaluation {resolution.outcome}"
                    if embedded_result is None
                    else embedded_result.message
                )
            if (
                embedded is not None
                and resolution.reward_ref is not None
                and isinstance(embedded.results[0], EvalSuccess)
                and embedded.results[0].score
                != resolution.reward_ref.record.value
            ):
                raise ValueError(
                    "recomputed evaluation score disagrees with "
                    "resolution reward"
                )
            gains, regressions, mismatches = (
                evaluation_history.compare_then_remember(
                    candidate_ref=candidate.record_ref,
                    base_ref=candidate.record.base_ref,
                    report=embedded,
                )
            )
            trajectory_resolutions.append(
                TrajectoryResolution(
                    step_index=step.step_index,
                    resolution_index=resolution_index,
                    request_id=resolution.request_id,
                    candidate_ref=_required_ref(candidate.record_ref),
                    outcome=resolution.outcome,
                    classification=classification,
                    message=message,
                    eval_result_ref=_ref(resolution.eval_result_ref),
                    reward_ref=(
                        None
                        if resolution.reward_ref is None
                        else _required_ref(resolution.reward_ref.record_ref)
                    ),
                    reward=(
                        None
                        if resolution.reward_ref is None
                        else resolution.reward_ref.record.value
                    ),
                    terminal_failure=(
                        None
                        if resolution.terminal_failure is None
                        else resolution.terminal_failure
                    ),
                    eval_report=embedded,
                    gains=gains,
                    regressions=regressions,
                    execution_mismatches=mismatches,
                )
            )
        trajectory_steps.append(
            TrajectoryStep(
                step_index=step.step_index,
                status=step.status.value,
                request_candidates=tuple(
                    _required_ref(item.record_ref)
                    for item in request_candidates
                ),
                proposed_candidates=tuple(
                    _required_ref(item.record_ref)
                    for item in step.proposed_candidates
                ),
                accepted_candidates=tuple(
                    _required_ref(item.record_ref)
                    for item in step.accepted_candidates
                ),
                resolution_indexes=tuple(resolution_indexes),
                budget_delta_consumed=step.budget_delta.consumed.to_json(),
                budget_cumulative_consumed=step.budget.consumed.to_json(),
                budget_remaining=step.budget.remaining.to_json(),
                terminal_failure=(
                    None
                    if step.terminal_failure is None
                    else step.terminal_failure.model_dump(mode="json")
                ),
            )
        )
    for proposal in result.proposals:
        discover(proposal.candidate, len(result.step_results) - 1, "terminal")

    exact_by_ref = {candidate.record_ref: candidate for candidate in ordered}
    candidates: list[TrajectoryCandidate] = []
    for candidate in ordered:
        mutation = candidate.record.payload.get(
            result.run.record.mutation_field
        )
        if type(mutation) is not str:
            raise ValueError(
                f"candidate {candidate.record.candidate_id!r} at "
                f"{candidate.record_ref.content_hash} has malformed mutation "
                f"field {result.run.record.mutation_field!r}"
            )
        base = exact_by_ref.get(candidate.record.base_ref)
        candidates.append(
            TrajectoryCandidate(
                first_step=first_step[candidate.record_ref],
                candidate_id=candidate.record.candidate_id,
                record_ref=_required_ref(candidate.record_ref),
                identity_hash=str(candidate.identity_hash),
                base_ref=_required_ref(candidate.record.base_ref),
                base_candidate_ref=(
                    None if base is None else _required_ref(base.record_ref)
                ),
                payload=candidate.record.payload.to_json(),
                mutation_text=mutation,
                dispositions=tuple(dispositions[candidate.record_ref]),
            )
        )
    return TrajectoryReport(
        schema_version=TRAJECTORY_REPORT_SCHEMA,
        result_ref=_required_ref(result_ref),
        run_id=result.run_id,
        mutation_field=result.run.record.mutation_field,
        terminal_status=result.status.value,
        candidates=tuple(candidates),
        steps=tuple(trajectory_steps),
        resolutions=tuple(trajectory_resolutions),
        terminal_candidate_refs=tuple(
            _required_ref(proposal.candidate.record_ref)
            for proposal in result.proposals
        ),
        spend=project_run_spend(result),
    )


__all__ = [
    "project_eval_report",
    "project_run_spend",
    "project_trajectory_report",
]
