from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

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
from whetstone.experiment.candidate import (
    CandidateRef,
)

from whetstone_envs.evidence import (
    EVAL_REPORT_SCHEMA,
    SPLIT_ROLE_BY_REPORT_ROLE,
    CandidateRecord,
    CandidateSource,
    EvalFailed,
    EvalRejected,
    EvalReport,
    EvalRoleName,
    EvalRun,
    EvalSuccess,
    EvaluationResult,
    Observation,
    ObservationState,
    ProviderErrorProjection,
    ReportedEvidence,
    RowAccounting,
    TaskRecord,
    account_observations,
    summarize_categories,
    two_stage_task_mean,
)
from whetstone_envs.instances import Instance
from whetstone_envs.optim.extraction._references import _required_ref
from whetstone_envs.probes import normalize

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval import EvalResult
    from whetstone.experiment.sampling import EvalSplit

    from whetstone_envs.optim.experiment import PreparedExperiment


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    name: str
    source: CandidateSource
    candidate: CandidateRef


def _eval_split_for_role(
    prepared: PreparedExperiment, role: EvalRoleName
) -> EvalSplit:
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
    *,
    name: str,
    source: CandidateSource,
    candidate: CandidateRef,
    mutation_field: str,
) -> CandidateRecord:
    template = candidate.record.payload.get(mutation_field)
    if type(template) is not str:
        raise ValueError(
            f"candidate {candidate.record.candidate_id!r} at "
            f"{candidate.record_ref.content_hash} has no strict string "
            f"payload field {mutation_field!r}"
        )
    return CandidateRecord(
        name=name,
        candidate_id=candidate.record.candidate_id,
        source=source,
        record_ref=_required_ref(candidate.record_ref),
        identity_hash=str(candidate.identity_hash),
        payload=candidate.record.payload.to_json(),
        mutation_field=mutation_field,
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


def _load_success_records(
    store: ObjectStore,
    evidence: EvalEvidence,
) -> tuple[EvalOutputsRecord, EvalTraces]:
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

    return outputs, traces


def _project_evidence_tasks(
    outputs: EvalOutputsRecord,
    instances: tuple[Instance, ...],
) -> tuple[TaskRecord, ...]:
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
                categories=instance.categories,
                prompt_inputs=dict(instance.prompt_inputs),
                gold=instance.gold,
            )
        )

    return tuple(tasks)


def _project_observations(
    name: str,
    outputs: EvalOutputsRecord,
    traces: EvalTraces,
) -> tuple[Observation, ...]:
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
    return tuple(observations)


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
    outputs, traces = _load_success_records(store, evidence)
    tasks = _project_evidence_tasks(outputs, instances)
    projected = _project_observations(name, outputs, traces)
    accounting = account_observations(projected)
    reported = RowAccounting.model_validate(
        evidence.row_accounting.model_dump(mode="json")
    )
    if accounting != reported:
        raise ValueError("recomputed row accounting disagrees with evidence")
    numerator = sum(row.score == 1.0 for row in projected)
    # Recompute from rows independently of the persisted aggregate.
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
        categories=summarize_categories(tuple(tasks), projected),
    )
    return success, tuple(tasks), projected


@dataclass(frozen=True, slots=True)
class _TaskPlan:
    instances: tuple[Instance, ...]
    records: tuple[TaskRecord, ...]


def _select_task_plan(
    prepared: PreparedExperiment,
    role: EvalRoleName,
    task_hashes: tuple[str, ...] | None,
) -> _TaskPlan:
    eval_split = _eval_split_for_role(prepared, role)
    all_instances = prepared.split.splits[
        SPLIT_ROLE_BY_REPORT_ROLE[role]
    ].instances
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
                categories=instance.categories,
                prompt_inputs=dict(instance.prompt_inputs),
                gold=instance.gold,
            )
        )
    return _TaskPlan(instances=instances, records=tuple(report_task_records))


def project_eval_report(  # noqa: PLR0913
    *,
    store: ObjectStore,
    prepared: PreparedExperiment,
    run_id: str,
    transport: str,
    model: str,
    role: EvalRoleName,
    split_sizes: tuple[int, int, int],
    mutation_field: str,
    candidates: tuple[EvaluationCandidate, ...],
    results: tuple[EvalResult, ...],
    task_hashes: tuple[str, ...] | None = None,
) -> EvalReport:
    if len(candidates) != len(results):
        raise ValueError("candidate/result counts disagree")
    eval_split = _eval_split_for_role(prepared, role)
    plan = _select_task_plan(prepared, role, task_hashes)
    candidate_records = tuple(
        _candidate_record(
            name=item.name,
            source=item.source,
            candidate=item.candidate,
            mutation_field=mutation_field,
        )
        for item in candidates
    )
    projected_results: list[EvaluationResult] = []
    report_observations: list[Observation] = []
    for candidate, result in zip(candidates, results, strict=True):
        name = candidate.name
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
            instances=plan.instances,
        )
        if plan.records != tasks:
            raise ValueError(
                "candidate evaluations do not share one exact task plan"
            )
        projected_results.append(success)
        report_observations.extend(observations)
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
        run=EvalRun(
            run_id=run_id,
            family=prepared.experiment.env_name,
            transport=transport,
            model=model,
            role=role,
            split_sizes=split_sizes,
            repeats=eval_split.seed_plan.num_seeds,
            dataset_revision=eval_split.task_set.dataset_revision,
            graph_hash=prepared.experiment.rollout_graph.graph_hash,
            eval_config_hash=eval_split.eval_config.config_hash,
            package_version=version("whetstone-envs"),
        ),
        candidates=candidate_records,
        tasks=plan.records,
        observations=tuple(report_observations),
        results=tuple(projected_results),
    )


__all__ = ["EvaluationCandidate", "project_eval_report"]
