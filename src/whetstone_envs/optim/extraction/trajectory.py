from __future__ import annotations

import json
from typing import TYPE_CHECKING

from whetstone.eval import (
    EvalEvidence,
    EvalEvidenceWithRef,
    EvalFailureEvidence,
)
from whetstone.experiment.candidate import (
    CandidateRef,
    candidate_reference,
)
from whetstone.optim.contracts import (
    OptimResult,
    optimization_result_reference,
)

from whetstone_envs.evidence import (
    TRAJECTORY_REPORT_SCHEMA,
    ComparisonOutcome,
    EvalReport,
    EvalSuccess,
    TrajectoryCandidate,
    TrajectoryReport,
    TrajectoryResolution,
    TrajectoryStep,
    classify_pair,
)
from whetstone_envs.optim.extraction._references import _ref, _required_ref
from whetstone_envs.optim.extraction._resolution_sources import (
    _gepa_transcript_sources,
    _step_resolution_sources,
    _TrajectoryResolutionSource,
)
from whetstone_envs.optim.extraction.evaluation import (
    EvaluationCandidate,
    project_eval_report,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.core.identity import TypedRef
    from whetstone.optim.contracts import OptimStepResult

    from whetstone_envs.evaluation.experiment import PreparedSplitExperiment


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
        outcome = classify_pair(prior.get(coordinate), present.get(coordinate))
        if outcome is ComparisonOutcome.EXECUTION_MISMATCH:
            mismatches += 1
        elif outcome is ComparisonOutcome.RIGHT_ONLY:
            gains += 1
        elif outcome is ComparisonOutcome.LEFT_ONLY:
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


class _TrajectoryCandidates:
    def __init__(self) -> None:
        self._ordered: list[CandidateRef] = []
        self._dispositions: dict[TypedRef, list[str]] = {}
        self._first_step: dict[TypedRef, int] = {}

    def discover(
        self, candidate: CandidateRef, step: int, disposition: str
    ) -> None:
        ref = candidate.record_ref
        if ref not in self._first_step:
            self._first_step[ref] = step
            self._ordered.append(candidate)
        _dispositions_append(self._dispositions, ref, disposition)

    def records(self, mutation_field: str) -> tuple[TrajectoryCandidate, ...]:
        exact_by_ref = {
            candidate.record_ref: candidate for candidate in self._ordered
        }
        candidates: list[TrajectoryCandidate] = []
        for candidate in self._ordered:
            mutation = candidate.record.payload.get(mutation_field)
            if type(mutation) is not str:
                raise ValueError(
                    f"candidate {candidate.record.candidate_id!r} at "
                    f"{candidate.record_ref.content_hash} has malformed "
                    "mutation "
                    f"field {mutation_field!r}"
                )
            base = exact_by_ref.get(candidate.record.base_ref)
            candidates.append(
                TrajectoryCandidate(
                    first_step=self._first_step[candidate.record_ref],
                    candidate_id=candidate.record.candidate_id,
                    record_ref=_required_ref(candidate.record_ref),
                    identity_hash=str(candidate.identity_hash),
                    base_ref=_required_ref(candidate.record.base_ref),
                    base_candidate_ref=(
                        None
                        if base is None
                        else _required_ref(base.record_ref)
                    ),
                    payload=candidate.record.payload.to_json(),
                    mutation_text=mutation,
                    dispositions=tuple(
                        self._dispositions[candidate.record_ref]
                    ),
                )
            )
        return tuple(candidates)


def _hydrate_evaluation_report(  # noqa: PLR0913
    *,
    store: ObjectStore,
    prepared: PreparedSplitExperiment,
    run_id: str,
    transport: str,
    resolution: _TrajectoryResolutionSource,
    mutation_field: str,
) -> EvalReport | None:
    ref = resolution.eval_result_ref
    if ref is None:
        return None
    raw = store.get(ref.reference)
    evidence: EvalEvidence | EvalFailureEvidence
    selected_task_hashes: tuple[str, ...] | None = None
    if resolution.outcome == "completed":
        evidence = EvalEvidence.model_validate_json(json.dumps(raw))
        selected_task_hashes = evidence.task_hashes
    else:
        evidence = EvalFailureEvidence.model_validate_json(json.dumps(raw))
    candidate = resolution.candidate
    return project_eval_report(
        store=store,
        prepared=prepared,
        run_id=run_id,
        transport=transport,
        role="internal",
        mutation_field=mutation_field,
        candidates=(
            EvaluationCandidate(
                name=candidate.record.candidate_id,
                source="optimized",
                candidate=candidate,
            ),
        ),
        results=(EvalEvidenceWithRef(evidence, ref),),
        task_hashes=selected_task_hashes,
    )


def _project_resolution(
    *,
    step_index: int,
    resolution_index: int,
    resolution: _TrajectoryResolutionSource,
    embedded: EvalReport | None,
    evaluation_history: _TrajectoryEvaluationHistory,
) -> TrajectoryResolution:
    candidate = resolution.candidate
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
        and embedded.results[0].score != resolution.reward_ref.record.value
    ):
        raise ValueError(
            "recomputed evaluation score disagrees with resolution reward"
        )
    gains, regressions, mismatches = evaluation_history.compare_then_remember(
        candidate_ref=candidate.record_ref,
        base_ref=candidate.record.base_ref,
        report=embedded,
    )
    return TrajectoryResolution(
        step_index=step_index,
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


def _project_step(
    step: OptimStepResult,
    request_candidates: tuple[CandidateRef, ...],
    resolution_indexes: tuple[int, ...],
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step.step_index,
        status=step.status.value,
        request_candidates=tuple(
            _required_ref(item.record_ref) for item in request_candidates
        ),
        proposed_candidates=tuple(
            _required_ref(item.record_ref) for item in step.proposed_candidates
        ),
        accepted_candidates=tuple(
            _required_ref(item.record_ref) for item in step.accepted_candidates
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


def project_trajectory_report(
    *,
    store: ObjectStore,
    prepared: PreparedSplitExperiment,
    result_ref: TypedRef,
    result: OptimResult,
    transport: str,
) -> TrajectoryReport:
    if optimization_result_reference(result) != result_ref:
        raise ValueError(
            "terminal OptimResult reference does not address result"
        )
    candidates = _TrajectoryCandidates()
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
            candidates.discover(candidate, step.step_index, "requested")
        for candidate in step.proposed_candidates:
            candidates.discover(candidate, step.step_index, "proposed")
        for candidate in step.accepted_candidates:
            candidates.discover(candidate, step.step_index, "accepted")
        accepted_refs = {
            candidate.record_ref for candidate in step.accepted_candidates
        }
        for candidate in step.proposed_candidates:
            if candidate.record_ref not in accepted_refs:
                candidates.discover(candidate, step.step_index, "rejected")
        resolution_indexes: list[int] = []
        resolution_sources = _step_resolution_sources(
            step,
            store=store,
            gepa_sources=gepa_sources,
        )
        for resolution_index, resolution in resolution_sources:
            candidate = resolution.candidate
            candidates.discover(candidate, step.step_index, "evaluated")
            if resolution.outcome in {"rejected", "failed"}:
                candidates.discover(
                    candidate, step.step_index, resolution.outcome
                )
            resolution_indexes.append(resolution_index)
            embedded = _hydrate_evaluation_report(
                store=store,
                prepared=prepared,
                run_id=f"{result.run_id}:{step.step_index}:{resolution_index}",
                transport=transport,
                resolution=resolution,
                mutation_field=result.run.record.mutation_field,
            )
            trajectory_resolutions.append(
                _project_resolution(
                    step_index=step.step_index,
                    resolution_index=resolution_index,
                    resolution=resolution,
                    embedded=embedded,
                    evaluation_history=evaluation_history,
                )
            )
        trajectory_steps.append(
            _project_step(step, request_candidates, tuple(resolution_indexes))
        )
    for proposal in result.proposals:
        candidates.discover(
            proposal.candidate, len(result.step_results) - 1, "terminal"
        )

    return TrajectoryReport(
        schema_version=TRAJECTORY_REPORT_SCHEMA,
        result_ref=_required_ref(result_ref),
        run_id=result.run_id,
        mutation_field=result.run.record.mutation_field,
        terminal_status=result.status.value,
        candidates=candidates.records(result.run.record.mutation_field),
        steps=tuple(trajectory_steps),
        resolutions=tuple(trajectory_resolutions),
        terminal_candidate_refs=tuple(
            _required_ref(proposal.candidate.record_ref)
            for proposal in result.proposals
        ),
    )


__all__ = ["project_trajectory_report"]
