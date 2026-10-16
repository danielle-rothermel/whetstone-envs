from __future__ import annotations

from typing import Literal

from pydantic import (
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone_envs.evidence._models import ReportRef, _StrictModel
from whetstone_envs.evidence.evaluation import (
    EvalFailed,
    EvalRejected,
    EvalReport,
    EvalSuccess,
)

TRAJECTORY_REPORT_SCHEMA = "whetstone_envs.trajectory_report/v3"


class TrajectoryCandidate(_StrictModel):
    first_step: StrictInt
    candidate_id: StrictStr
    record_ref: ReportRef
    identity_hash: StrictStr
    base_ref: ReportRef
    base_candidate_ref: ReportRef | None
    payload: dict[StrictStr, JsonValue]
    mutation_text: StrictStr
    dispositions: tuple[StrictStr, ...]


class TrajectoryResolution(_StrictModel):
    step_index: StrictInt
    resolution_index: StrictInt
    request_id: StrictStr
    candidate_ref: ReportRef
    outcome: StrictStr
    classification: StrictStr
    message: StrictStr
    eval_result_ref: ReportRef | None
    reward_ref: ReportRef | None
    reward: StrictFloat | None
    terminal_failure: dict[StrictStr, JsonValue] | None
    eval_report: EvalReport | None
    gains: StrictInt | None = None
    regressions: StrictInt | None = None
    execution_mismatches: StrictInt | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> TrajectoryResolution:
        if self.outcome == "rejected":
            if any(
                value is not None
                for value in (
                    self.eval_result_ref,
                    self.reward_ref,
                    self.reward,
                    self.terminal_failure,
                    self.eval_report,
                )
            ):
                raise ValueError(
                    "rejected resolution has no durable eval result"
                )
        elif self.outcome == "completed":
            if self.eval_result_ref is None or self.eval_report is None:
                raise ValueError(
                    "completed resolution requires hydrated evaluation"
                )
            if self.terminal_failure is not None:
                raise ValueError(
                    "completed resolution has no terminal failure"
                )
            if (self.reward_ref is None) != (self.reward is None):
                raise ValueError(
                    "reward value and reference must appear together"
                )
        elif self.outcome == "failed":
            if (
                self.eval_result_ref is None
                or self.eval_report is None
                or self.terminal_failure is None
                or self.reward_ref is not None
                or self.reward is not None
            ):
                raise ValueError(
                    "failed resolution requires failure evaluation only"
                )
        else:
            raise ValueError("unsupported trajectory resolution outcome")
        return self


class TrajectoryStep(_StrictModel):
    step_index: StrictInt
    status: StrictStr
    request_candidates: tuple[ReportRef, ...]
    proposed_candidates: tuple[ReportRef, ...]
    accepted_candidates: tuple[ReportRef, ...]
    resolution_indexes: tuple[StrictInt, ...]
    budget_delta_consumed: dict[StrictStr, JsonValue]
    budget_cumulative_consumed: dict[StrictStr, JsonValue]
    budget_remaining: dict[StrictStr, JsonValue]
    terminal_failure: dict[StrictStr, JsonValue] | None

    @model_validator(mode="after")
    def _validate_status(self) -> TrajectoryStep:
        if (self.status == "failed") != (self.terminal_failure is not None):
            raise ValueError(
                "failed step requires exactly one terminal failure"
            )
        return self


class TrajectoryReport(_StrictModel):
    schema_version: Literal["whetstone_envs.trajectory_report/v3"]
    result_ref: ReportRef
    run_id: StrictStr
    mutation_field: StrictStr
    terminal_status: StrictStr
    candidates: tuple[TrajectoryCandidate, ...]
    steps: tuple[TrajectoryStep, ...]
    resolutions: tuple[TrajectoryResolution, ...]
    terminal_candidate_refs: tuple[ReportRef, ...]

    @model_validator(mode="after")
    def _validate_order(self) -> TrajectoryReport:
        self._validate_candidates()
        self._validate_step_order()
        self._validate_resolution_order()
        self._validate_candidate_refs()
        self._validate_embedded_evaluations()
        return self

    def _validate_candidates(self) -> None:
        refs = tuple(candidate.record_ref for candidate in self.candidates)
        if len(set(refs)) != len(refs):
            raise ValueError(
                "trajectory candidates must be unique by exact ref"
            )
        if any(
            candidate.payload.get(self.mutation_field)
            != candidate.mutation_text
            for candidate in self.candidates
        ):
            raise ValueError(
                "trajectory candidate mutation text must match its payload"
            )
        if self.steps and any(
            candidate.first_step < 0 or candidate.first_step >= len(self.steps)
            for candidate in self.candidates
        ):
            raise ValueError("candidate first step is outside the trajectory")
        by_ref = {
            candidate.record_ref: candidate for candidate in self.candidates
        }
        for candidate in self.candidates:
            if candidate.base_candidate_ref is None:
                continue
            base = by_ref.get(candidate.base_candidate_ref)
            if base is None or base.record_ref != candidate.base_ref:
                raise ValueError(
                    "resolved candidate base must equal its exact base ref"
                )

    def _validate_step_order(self) -> None:
        if tuple(step.step_index for step in self.steps) != tuple(
            range(len(self.steps))
        ):
            raise ValueError("trajectory step indexes must be contiguous")

    def _validate_resolution_order(self) -> None:
        actual = tuple(
            (row.step_index, row.resolution_index) for row in self.resolutions
        )
        expected = tuple(
            (step.step_index, index)
            for step in self.steps
            for index in step.resolution_indexes
        )
        if actual != expected:
            raise ValueError(
                "trajectory resolutions must follow exact step order"
            )

    def _validate_candidate_refs(self) -> None:
        known_refs = {candidate.record_ref for candidate in self.candidates}
        cited_refs = tuple(
            ref
            for step in self.steps
            for ref in (
                *step.request_candidates,
                *step.proposed_candidates,
                *step.accepted_candidates,
            )
        ) + tuple(row.candidate_ref for row in self.resolutions)
        if any(ref not in known_refs for ref in cited_refs):
            raise ValueError("trajectory records cite an unknown candidate")
        if any(ref not in known_refs for ref in self.terminal_candidate_refs):
            raise ValueError("terminal proposals cite an unknown candidate")

    def _validate_embedded_evaluations(self) -> None:
        for resolution in self.resolutions:
            _validate_resolution_evaluation(resolution)


def _validate_resolution_evaluation(resolution: TrajectoryResolution) -> None:
    embedded = resolution.eval_report
    if embedded is None:
        return
    if (
        len(embedded.candidates) != 1
        or embedded.candidates[0].record_ref != resolution.candidate_ref
    ):
        raise ValueError(
            "embedded evaluation must cite the exact resolution candidate"
        )
    result = embedded.results[0]
    if resolution.outcome == "completed":
        _validate_completed_evaluation(resolution, result)
    elif resolution.outcome == "failed" and (
        not isinstance(result, EvalFailed)
        or result.evidence_ref != resolution.eval_result_ref
    ):
        raise ValueError(
            "failed resolution ref must match embedded failure evidence"
        )


def _validate_completed_evaluation(
    resolution: TrajectoryResolution,
    result: EvalSuccess | EvalFailed | EvalRejected,
) -> None:
    if not isinstance(result, EvalSuccess):
        raise ValueError(  # noqa: TRY004 - Pydantic wraps ValueError.
            "completed resolution requires successful evaluation evidence"
        )
    if (
        result.evidence.evidence_ref != resolution.eval_result_ref
        or result.evidence.reward_ref != resolution.reward_ref
    ):
        raise ValueError(
            "completed resolution refs must match embedded evidence"
        )
    if resolution.reward_ref is not None and result.score != resolution.reward:
        raise ValueError(
            "completed resolution reward must match embedded score"
        )
