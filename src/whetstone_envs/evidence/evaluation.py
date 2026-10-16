from __future__ import annotations

from collections import Counter
from enum import UNIQUE, StrEnum, verify
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone_envs.evidence._models import ReportRef, _StrictModel
from whetstone_envs.probes import normalize

EVAL_REPORT_SCHEMA = "whetstone_envs.eval_report/v5"
CandidateSource = Literal["naive", "ceiling", "custom", "optimized"]
EvalRoleName = Literal["internal", "official", "held_out"]

SPLIT_ROLE_BY_REPORT_ROLE: dict[EvalRoleName, str] = {
    "internal": "internal_eval",
    "official": "official",
    "held_out": "held_out",
}


class EvalRun(_StrictModel):
    run_id: StrictStr
    family: StrictStr
    transport: StrictStr
    model: StrictStr
    role: EvalRoleName
    split_sizes: tuple[StrictInt, StrictInt, StrictInt]
    repeats: StrictInt
    dataset_revision: StrictStr
    graph_hash: StrictStr
    eval_config_hash: StrictStr
    package_version: StrictStr

    @model_validator(mode="after")
    def _validate_run(self) -> EvalRun:
        values = (
            self.run_id,
            self.family,
            self.transport,
            self.model,
            self.dataset_revision,
            self.graph_hash,
            self.eval_config_hash,
            self.package_version,
        )
        if any(not value.strip() for value in values):
            raise ValueError("evaluation run identifiers must be nonblank")
        if self.repeats < 1:
            raise ValueError("evaluation repeats must be positive")
        if any(size < 0 for size in self.split_sizes):
            raise ValueError("evaluation split sizes must be non-negative")
        return self


class CandidateRecord(_StrictModel):
    name: StrictStr
    candidate_id: StrictStr
    source: CandidateSource
    record_ref: ReportRef
    identity_hash: StrictStr
    payload: dict[StrictStr, JsonValue]
    mutation_field: StrictStr
    prompt_template: StrictStr

    @model_validator(mode="after")
    def _validate_candidate(self) -> CandidateRecord:
        if not self.name.strip() or not self.candidate_id.strip():
            raise ValueError("candidate name and ID must be nonblank")
        if not self.prompt_template:
            raise ValueError("candidate prompt template must be nonempty")
        if not self.mutation_field.strip():
            raise ValueError("candidate mutation field must be nonblank")
        if self.payload.get(self.mutation_field) != self.prompt_template:
            raise ValueError(
                "candidate payload must contain the exact template"
            )
        return self


class TaskRecord(_StrictModel):
    task_id: StrictStr
    task_hash: StrictStr
    seed: StrictInt
    categories: tuple[StrictStr, ...]
    prompt_inputs: dict[StrictStr, StrictStr]
    gold: StrictStr

    @model_validator(mode="after")
    def _validate_task(self) -> TaskRecord:
        if not self.task_id.strip() or not self.task_hash.strip():
            raise ValueError("task ID and hash must be nonblank")
        if not self.categories or any(
            not label.strip() for label in self.categories
        ):
            raise ValueError("tasks require nonblank categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("task categories must be unique and ordered")
        if not self.prompt_inputs:
            raise ValueError("tasks require at least one prompt input")
        if any(not name.strip() for name in self.prompt_inputs):
            raise ValueError("task prompt input names must be nonblank")
        return self


@verify(UNIQUE)
class ObservationState(StrEnum):
    SCORED = "scored"
    FAILED = "failed"
    MISSING = "missing"
    INVALID = "invalid"


ProviderFailureClass = Literal[
    "transport-error",
    "rate-limit",
    "timeout",
    "provider-rejection",
    "blank-provider-generation",
    "malformed-response",
]
ProviderFailureSource = Literal["transport_failure", "rejected_response"]
ProviderRecoverability = Literal[
    "permanent",
    "transient",
    "rate_limited",
    "resource_exhaustion",
    "unknown",
]


class ProviderErrorProjection(_StrictModel):
    failure_class: ProviderFailureClass
    source: ProviderFailureSource
    recoverability: ProviderRecoverability | None = None
    status_code: StrictInt | None = None
    timeout_containment: Literal["contained", "uncontained"] | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> ProviderErrorProjection:
        transport_values = (
            self.recoverability,
            self.status_code,
            self.timeout_containment,
        )
        if self.source == "transport_failure":
            if self.recoverability is None:
                raise ValueError(
                    "transport provider errors require recoverability"
                )
        elif any(value is not None for value in transport_values):
            raise ValueError(
                "rejected responses have no projected transport metadata"
            )
        if self.status_code is not None and self.status_code < 0:
            raise ValueError("provider status code must be non-negative")
        return self


class Observation(_StrictModel):
    candidate_name: StrictStr
    task_id: StrictStr
    task_hash: StrictStr
    task_index: StrictInt
    seed_index: StrictInt
    rendered_prompt: StrictStr
    output_text: StrictStr | None
    normalized_output: StrictStr | None
    score: StrictFloat | None
    state: ObservationState
    trace_state: Literal["success", "failed", "missing", "invalid"]
    failure_code: StrictStr
    finish_reason: StrictStr | None
    provider_error: ProviderErrorProjection | None
    max_budget: StrictInt | None
    over_budget: StrictBool | None
    submission_result: dict[StrictStr, JsonValue] | None
    component_trace: tuple[dict[StrictStr, JsonValue], ...]

    @model_validator(mode="after")
    def _validate_state(self) -> Observation:
        if (self.state is ObservationState.SCORED) != (self.score is not None):
            raise ValueError("only scored observations carry a score")
        if self.score is not None and self.score not in {0.0, 1.0}:
            raise ValueError(
                "scored observations require an exact binary score"
            )
        if (
            self.output_text is not None
            and self.normalized_output != normalize(self.output_text)
        ):
            raise ValueError(
                "normalized output must match shared prediction normalization"
            )
        if self.output_text is None and self.normalized_output is not None:
            raise ValueError("absent output text has no normalized output")
        expected_trace = {
            ObservationState.SCORED: "success",
            ObservationState.FAILED: "failed",
            ObservationState.MISSING: "missing",
            ObservationState.INVALID: "invalid",
        }[self.state]
        if self.trace_state != expected_trace:
            raise ValueError("trace state disagrees with report row state")
        return self


def two_stage_task_mean(
    rows: tuple[Observation, ...],
) -> float | None:
    # Preserve two-stage arithmetic and row order to match runtime floats.
    # Average scored repeats; omit tasks with no scores.
    if not rows:
        return None
    by_task: dict[int, list[float]] = {}
    for row in rows:
        scores = by_task.setdefault(row.task_index, [])
        if row.score is not None:
            scores.append(row.score)
    task_means = [
        sum(scores) / len(scores) for scores in by_task.values() if scores
    ]
    if not task_means:
        return None
    return sum(task_means) / len(task_means)


class RowAccounting(_StrictModel):
    planned: StrictInt
    present: StrictInt
    missing: StrictInt
    failed: StrictInt
    invalid: StrictInt

    @model_validator(mode="after")
    def _validate_total(self) -> RowAccounting:
        values = (
            self.planned,
            self.present,
            self.missing,
            self.failed,
            self.invalid,
        )
        if any(value < 0 for value in values):
            raise ValueError("row accounting values must be non-negative")
        if (
            self.present + self.missing + self.failed + self.invalid
            != self.planned
        ):
            raise ValueError("row accounting must exhaust the planned matrix")
        return self


class CategorySummary(_StrictModel):
    category: StrictStr
    numerator: StrictInt
    denominator: StrictInt
    accounting: RowAccounting
    score: StrictFloat | None

    @model_validator(mode="after")
    def _validate_summary(self) -> CategorySummary:
        if self.denominator != self.accounting.planned:
            raise ValueError("category denominator must equal planned rows")
        if self.numerator < 0 or self.numerator > self.accounting.present:
            raise ValueError("category numerator is outside present rows")
        scored_anything = self.accounting.present > 0
        if scored_anything != (self.score is not None):
            raise ValueError(
                "category score must be present iff some row was scored"
            )
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("category score must be a unit-interval mean")
        return self


class ReportedEvidence(_StrictModel):
    evidence_ref: ReportRef | None
    outputs_ref: ReportRef | None
    traces_ref: ReportRef | None
    aggregate_ref: ReportRef | None
    reward_ref: ReportRef | None
    aggregate_name: StrictStr | None
    aggregate_value: StrictFloat | None
    aggregate_status: StrictStr | None
    row_accounting: RowAccounting | None


class EvalSuccess(_StrictModel):
    kind: Literal["success"]
    candidate_name: StrictStr
    classification: StrictStr
    message: StrictStr
    evidence: ReportedEvidence
    accounting: RowAccounting
    numerator: StrictInt
    denominator: StrictInt
    score: StrictFloat | None
    categories: tuple[CategorySummary, ...]


class EvalFailed(_StrictModel):
    kind: Literal["failed"]
    candidate_name: StrictStr
    classification: StrictStr
    message: StrictStr
    evidence_ref: ReportRef
    exception_type: StrictStr


class EvalRejected(_StrictModel):
    kind: Literal["rejected"]
    candidate_name: StrictStr
    classification: StrictStr
    message: StrictStr


EvaluationResult = Annotated[
    EvalSuccess | EvalFailed | EvalRejected,
    Field(discriminator="kind"),
]


def account_observations(rows: tuple[Observation, ...]) -> RowAccounting:
    counts = Counter(row.state for row in rows)
    return RowAccounting(
        planned=len(rows),
        present=counts[ObservationState.SCORED],
        missing=counts[ObservationState.MISSING],
        failed=counts[ObservationState.FAILED],
        invalid=counts[ObservationState.INVALID],
    )


def summarize_categories(
    tasks: tuple[TaskRecord, ...], rows: tuple[Observation, ...]
) -> tuple[CategorySummary, ...]:
    task_by_id = {task.task_id: task for task in tasks}
    labels = dict.fromkeys(
        label for task in tasks for label in task.categories
    )
    return tuple(
        _summarize_category(
            label,
            tuple(
                row
                for row in rows
                if label in task_by_id[row.task_id].categories
            ),
        )
        for label in labels
    )


def _summarize_category(
    category: str, rows: tuple[Observation, ...]
) -> CategorySummary:
    return CategorySummary(
        category=category,
        numerator=sum(row.score == 1.0 for row in rows),
        denominator=len(rows),
        accounting=account_observations(rows),
        score=two_stage_task_mean(rows),
    )


def _validate_result_accounting(
    result: EvalSuccess, rows: tuple[Observation, ...]
) -> None:
    accounting = account_observations(rows)
    if (
        result.accounting != accounting
        or result.evidence.row_accounting != accounting
        or result.numerator != sum(row.score == 1.0 for row in rows)
        or result.denominator != len(rows)
        or result.score != two_stage_task_mean(rows)
    ):
        raise ValueError(
            "evaluation result disagrees with observation accounting"
        )


class EvalReport(_StrictModel):
    schema_version: Literal["whetstone_envs.eval_report/v5"]
    run: EvalRun
    candidates: tuple[CandidateRecord, ...]
    tasks: tuple[TaskRecord, ...]
    observations: tuple[Observation, ...]
    results: tuple[EvaluationResult, ...]

    @model_validator(mode="after")
    def _validate_collections(self) -> EvalReport:
        self._validate_candidate_and_task_identities()
        self._validate_result_order()
        self._validate_observation_matrix()
        self._validate_observation_tasks()
        self._validate_result_summaries()
        return self

    def _validate_candidate_and_task_identities(self) -> None:
        names = tuple(candidate.name for candidate in self.candidates)
        if len(set(names)) != len(names) or not names:
            raise ValueError("candidate names must be nonempty and unique")
        refs = tuple(candidate.record_ref for candidate in self.candidates)
        if len(set(refs)) != len(refs):
            raise ValueError("candidate record refs must be unique")
        task_ids = tuple(task.task_id for task in self.tasks)
        task_hashes = tuple(task.task_hash for task in self.tasks)
        if len(set(task_ids)) != len(task_ids) or len(set(task_hashes)) != len(
            task_hashes
        ):
            raise ValueError("task IDs and hashes must each be unique")

    def _validate_result_order(self) -> None:
        names = tuple(candidate.name for candidate in self.candidates)
        if tuple(result.candidate_name for result in self.results) != names:
            raise ValueError("results must follow exact candidate order")

    def _validate_observation_matrix(self) -> None:
        successful = {
            result.candidate_name
            for result in self.results
            if isinstance(result, EvalSuccess)
        }
        expected = tuple(
            (candidate.name, task.task_id, task_index, seed_index)
            for candidate in self.candidates
            if candidate.name in successful
            for task_index, task in enumerate(self.tasks)
            for seed_index in range(self.run.repeats)
        )
        actual = tuple(
            (row.candidate_name, row.task_id, row.task_index, row.seed_index)
            for row in self.observations
        )
        if actual != expected:
            raise ValueError(
                "observations must cover exact candidate-major matrix order"
            )

    def _validate_observation_tasks(self) -> None:
        task_by_id = {task.task_id: task for task in self.tasks}
        for row in self.observations:
            task = task_by_id.get(row.task_id)
            if task is None or task.task_hash != row.task_hash:
                raise ValueError(
                    "observation references an unknown exact task"
                )
            if (
                row.state is ObservationState.SCORED
                and row.output_text is None
            ):
                raise ValueError("scored observations require output text")

    def _validate_result_summaries(self) -> None:
        for result in self.results:
            if not isinstance(result, EvalSuccess):
                continue
            rows = tuple(
                row
                for row in self.observations
                if row.candidate_name == result.candidate_name
            )
            _validate_result_accounting(result, rows)
            if result.categories != summarize_categories(self.tasks, rows):
                raise ValueError(
                    "evaluation categories disagree with "
                    "observation accounting"
                )
