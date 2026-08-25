from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone_envs.probes import normalize
from whetstone_envs.scoring.families import family_score

# Persisted literals are owned here and pinned directly by golden tests.

#: v2 adds ``spend``: what the evaluation's own provider calls cost.
#: v1 published a report for a run that had reached a provider and said
#: nothing about the bill, so a held-out evaluation -- the role every
#: efficacy claim is finally made against -- spent money no artifact
#: recorded.
EVAL_REPORT_SCHEMA = "whetstone_envs.eval_report/v2"
TRAJECTORY_REPORT_SCHEMA = "whetstone_envs.trajectory_report/v1"
#: The only whetstone-ai cost-report schema version this package projects.
#: An embedded report at any other version carries semantics this schema has
#: not been checked against, so it is rejected rather than reinterpreted.
SPEND_SCHEMA_VERSION = 1
CandidateSource = Literal["naive", "ceiling", "custom", "optimized"]
EvalRoleName = Literal["internal", "official", "held_out"]
#: Every task family a persisted report may name. Persisted-format values:
#: a report's family string is pinned here, not derived from a module name,
#: so a family rename is a deliberate schema change with a golden test.
FamilyName = Literal["c19", "c18"]
RoleSpendName = Literal["task_model", "proposer"]

#: The reported role name mapped to the upstream experiment split role. The
#: keys are this package's persisted ``EvalRun.role`` literals; the values are
#: whetstone's ``SPLIT_ROLES`` spellings, which differ for ``internal``.
SPLIT_ROLE_BY_REPORT_ROLE: dict[EvalRoleName, str] = {
    "internal": "internal_eval",
    "official": "official",
    "held_out": "held_out",
}


def _family_score(*, family: str, output_text: str, gold: str) -> float:
    """What ``family``'s own scorer yields for one observation.

    The report's scores are not the schema's to invent -- they are the
    run's -- so validating them means re-deriving them the way the run
    did. :mod:`whetstone_envs.scoring.families` is the single owner of
    "how a generation becomes a score", and both the eval runners and this
    check route through it rather than restating any family's rule.

    That registry holds only the pure per-family rule and imports no part
    of the optimizer stack, so validating a scored report works on a base
    install. Routing this check through the optimizer's family registry
    instead pulled in whetstone-ai, which the ``optim`` extra installs only
    on Python 3.13+, and made reading any scored report fail where it was
    merely being read.

    Hard-coding normalized exact match, which is what this check used to
    do, is a c19 rule wearing a family-agnostic name: c18 scores the
    *terminal verdict* it extracts from a reasoned reply
    (:func:`whetstone_envs.c18.score_gold`), so a correct c18 answer
    ending in ``True`` scored 1.0 while the schema recomputed 0.0 and
    refused the whole report. That failure had nothing to do with the
    row and everything to do with the check, and it took down
    publication for the entire run.
    """
    return family_score(family=family, output_text=output_text, gold=gold)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ReportRef(_StrictModel):
    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate_ref(self) -> ReportRef:
        if not self.schema_name.strip() or not self.content_hash.strip():
            raise ValueError("report refs require nonblank schema and hash")
        return self


class EvalRun(_StrictModel):
    run_id: StrictStr
    #: The task family this run evaluated. A persisted-format value: a new
    #: family widens this literal deliberately rather than by inference.
    family: FamilyName
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
    prompt_template: StrictStr

    @model_validator(mode="after")
    def _validate_candidate(self) -> CandidateRecord:
        if not self.name.strip() or not self.candidate_id.strip():
            raise ValueError("candidate name and ID must be nonblank")
        if not self.prompt_template:
            raise ValueError("candidate prompt template must be nonempty")
        if self.payload.get("prompt_template") != self.prompt_template:
            raise ValueError(
                "candidate payload must contain the exact template"
            )
        return self


class TaskRecord(_StrictModel):
    task_id: StrictStr
    task_hash: StrictStr
    seed: StrictInt
    strata: tuple[StrictStr, ...]
    prompt_inputs: dict[StrictStr, StrictStr]
    gold: StrictStr

    @model_validator(mode="after")
    def _validate_task(self) -> TaskRecord:
        if not self.task_id.strip() or not self.task_hash.strip():
            raise ValueError("task ID and hash must be nonblank")
        if not self.strata or any(not label.strip() for label in self.strata):
            raise ValueError("tasks require nonblank strata")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("task strata must be unique and ordered")
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
    #: The row state the *trace* recorded, kept beside ``state`` so the
    #: two can be cross-checked rather than assumed to agree. Its members
    #: are whetstone's ``ExecutedRowState``, restated as a closed literal
    #: because this is a persisted report field: a widened upstream enum
    #: must be a deliberate change here, not one a projection inherits.
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
        # An invalid row's trace says ``invalid`` rather than ``failed``:
        # whetstone-ai 0.1.14 gave ``ExecutedRowState`` its own member for
        # a row that executed and was billed but produced nothing the
        # contract can score. Before it, an invalid row's trace could only
        # say ``failed``, which is why this mapping folded the two.
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
) -> StrictFloat | None:
    """The reported score for ``rows``: per-task mean, then across tasks.

    This is a *contract*, not a style choice. whetstone-ai persists every
    evaluation aggregate through ``unweighted_task_mean``, which reduces
    each task's repeats to one per-task value and then means those values
    across tasks. This package's reporting projection is an independent
    recompute of that same number, and it must be independent in the
    *inputs* it reads -- the rows -- not in the arithmetic it applies.

    The two orders are the same rational number and different floats. A
    flat mean over all rows sums a different sequence of addends than the
    two-stage reduction, and IEEE-754 addition is not associative: at
    ``num_seeds=3`` the two disagree by 1 ULP for roughly half of all
    evaluations (2 tasks scoring 2/3 and 3/3 already diverge). Recomputing
    flat therefore made the check reject evidence that was correct, and
    only ``num_seeds=1`` -- where each task has one row and the two
    reductions are the same sum -- hid it.

    Rows arrive in the persisted matrix order (task-major, seed-minor),
    which the ``EvalReport`` validator enforces, so grouping by
    ``task_index`` reproduces whetstone-ai's within-task addend order
    exactly rather than approximately.

    Returns ``None`` when any row is unscored: an incomplete stratum or
    candidate reports no number at all rather than a mean over whichever
    rows happened to land.
    """
    if not rows:
        return None
    by_task: dict[int, list[float]] = {}
    for row in rows:
        if row.score is None:
            return None
        by_task.setdefault(row.task_index, []).append(row.score)
    task_means = [sum(scores) / len(scores) for scores in by_task.values()]
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


class StratumSummary(_StrictModel):
    stratum: StrictStr
    numerator: StrictInt
    denominator: StrictInt
    accounting: RowAccounting
    score: StrictFloat | None

    @model_validator(mode="after")
    def _validate_summary(self) -> StratumSummary:
        if self.denominator != self.accounting.planned:
            raise ValueError("stratum denominator must equal planned rows")
        if self.numerator < 0 or self.numerator > self.accounting.present:
            raise ValueError("stratum numerator is outside present rows")
        # ``score`` is the two-stage per-task mean (see
        # ``two_stage_task_mean``), so it is *not* ``numerator /
        # denominator``: those stay the honest row-level pass count for
        # display, and at ``num_seeds > 1`` the two-stage mean weights each
        # task equally rather than each row. This validator therefore
        # cannot re-derive the float from its own scalars -- the rows it
        # would need are not here -- and checks the two properties that do
        # survive isolation. ``EvalReport._validate_collections`` performs
        # the full row-level recompute.
        complete = bool(self.denominator) and (
            self.accounting.present == self.denominator
        )
        if complete != (self.score is not None):
            raise ValueError(
                "stratum score must preserve incomplete accounting"
            )
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("stratum score must be a unit-interval mean")
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
    strata: tuple[StratumSummary, ...]


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


class EvalReport(_StrictModel):
    schema_version: Literal["whetstone_envs.eval_report/v2"]
    run: EvalRun
    candidates: tuple[CandidateRecord, ...]
    tasks: tuple[TaskRecord, ...]
    observations: tuple[Observation, ...]
    results: tuple[EvaluationResult, ...]
    #: What this evaluation's provider calls cost, re-derived from the
    #: persisted output rows rather than accumulated while it ran.
    #:
    #: ``None`` when the run evidenced no provider call -- a fake-transport
    #: evaluation reaches no provider, and a paid one whose rows carried no
    #: usage telemetry measured nothing it could bill. Reporting an
    #: all-zero block for either would claim the evaluation was measured
    #: and found free.
    spend: EvalSpend | None = None

    @model_validator(mode="after")
    def _validate_collections(self) -> EvalReport:  # noqa: PLR0912
        names = tuple(candidate.name for candidate in self.candidates)
        if len(set(names)) != len(names) or not names:
            raise ValueError("candidate names must be nonempty and unique")
        candidate_refs = tuple(
            candidate.record_ref for candidate in self.candidates
        )
        if len(set(candidate_refs)) != len(candidate_refs):
            raise ValueError("candidate record refs must be unique")
        task_ids = tuple(task.task_id for task in self.tasks)
        task_hashes = tuple(task.task_hash for task in self.tasks)
        if len(set(task_ids)) != len(task_ids) or len(set(task_hashes)) != len(
            task_hashes
        ):
            raise ValueError("task IDs and hashes must each be unique")
        if tuple(result.candidate_name for result in self.results) != names:
            raise ValueError("results must follow exact candidate order")
        task_by_id = {task.task_id: task for task in self.tasks}
        successful = {
            result.candidate_name
            for result in self.results
            if isinstance(result, EvalSuccess)
        }
        expected = tuple(
            (name, task.task_id, task_index, seed_index)
            for name in names
            if name in successful
            for task_index, task in enumerate(self.tasks)
            for seed_index in range(self.run.repeats)
        )
        actual = tuple(
            (
                row.candidate_name,
                row.task_id,
                row.task_index,
                row.seed_index,
            )
            for row in self.observations
        )
        if actual != expected:
            raise ValueError(
                "observations must cover exact candidate-major matrix order"
            )
        for row in self.observations:
            task = task_by_id.get(row.task_id)
            if task is None or task.task_hash != row.task_hash:
                raise ValueError(
                    "observation references an unknown exact task"
                )
            if row.state is ObservationState.SCORED:
                if row.output_text is None:
                    raise ValueError("scored observations require output text")
                expected_score = _family_score(
                    family=self.run.family,
                    output_text=row.output_text,
                    gold=task.gold,
                )
                if row.score != expected_score:
                    raise ValueError(
                        "observation score disagrees with the "
                        f"{self.run.family} scorer"
                    )
        for result in self.results:
            if not isinstance(result, EvalSuccess):
                continue
            rows = tuple(
                row
                for row in self.observations
                if row.candidate_name == result.candidate_name
            )
            states = {
                state: sum(row.state is state for row in rows)
                for state in ObservationState
            }
            accounting = RowAccounting(
                planned=len(rows),
                present=states[ObservationState.SCORED],
                missing=states[ObservationState.MISSING],
                failed=states[ObservationState.FAILED],
                invalid=states[ObservationState.INVALID],
            )
            numerator = sum(row.score == 1.0 for row in rows)
            # Row-level pass count for display; the score is the two-stage
            # per-task mean and does not equal ``numerator / len(rows)``
            # once a task carries more than one repeat.
            score = two_stage_task_mean(rows)
            if (
                result.accounting != accounting
                or result.evidence.row_accounting != accounting
                or result.numerator != numerator
                or result.denominator != len(rows)
                or result.score != score
            ):
                raise ValueError(
                    "evaluation result disagrees with observation accounting"
                )
            labels = tuple(
                dict.fromkeys(
                    label for task in self.tasks for label in task.strata
                )
            )
            expected_strata: list[StratumSummary] = []
            for label in labels:
                stratum_rows = tuple(
                    row
                    for row in rows
                    if label in task_by_id[row.task_id].strata
                )
                stratum_states = {
                    state: sum(row.state is state for row in stratum_rows)
                    for state in ObservationState
                }
                stratum_accounting = RowAccounting(
                    planned=len(stratum_rows),
                    present=stratum_states[ObservationState.SCORED],
                    missing=stratum_states[ObservationState.MISSING],
                    failed=stratum_states[ObservationState.FAILED],
                    invalid=stratum_states[ObservationState.INVALID],
                )
                stratum_numerator = sum(
                    row.score == 1.0 for row in stratum_rows
                )
                expected_strata.append(
                    StratumSummary(
                        stratum=label,
                        numerator=stratum_numerator,
                        denominator=len(stratum_rows),
                        accounting=stratum_accounting,
                        score=two_stage_task_mean(stratum_rows),
                    )
                )
            if result.strata != tuple(expected_strata):
                raise ValueError(
                    "evaluation strata disagree with observation accounting"
                )
        return self


class RoleSpend(_StrictModel):
    """What one provider role cost a run.

    ``usd`` is present only when every contributing call carried a
    provider-reported price; otherwise the priced/unpriced split shows what a
    total would have covered. Callers render the absent case as
    ``"unpriced"`` rather than as a zero.
    """

    role: RoleSpendName
    calls: StrictInt
    cached_calls: StrictInt
    input_tokens: StrictInt
    output_tokens: StrictInt
    priced_calls: StrictInt
    unpriced_calls: StrictInt
    rows_missing_token_breakdown: StrictInt
    usd: StrictFloat | None

    @model_validator(mode="after")
    def _validate_role_spend(self) -> RoleSpend:
        counts = (
            self.calls,
            self.cached_calls,
            self.input_tokens,
            self.output_tokens,
            self.priced_calls,
            self.unpriced_calls,
            self.rows_missing_token_breakdown,
        )
        if any(value < 0 for value in counts):
            raise ValueError("spend counters must be non-negative")
        if self.priced_calls + self.unpriced_calls != self.calls:
            raise ValueError(
                "priced and unpriced calls must exhaust billable calls"
            )
        if self.usd is not None and self.usd < 0:
            raise ValueError("reported spend must be non-negative")
        if self.usd is not None and self.unpriced_calls:
            raise ValueError(
                "a run with unpriced calls reports no total spend"
            )
        return self


class RunSpend(_StrictModel):
    """The run's cost report, one entry per provider role."""

    schema_version: Literal[1]
    task_model: RoleSpend
    proposer: RoleSpend

    @model_validator(mode="after")
    def _validate_roles(self) -> RunSpend:
        if self.task_model.role != "task_model":
            raise ValueError("task model spend must carry the task_model role")
        if self.proposer.role != "proposer":
            raise ValueError("proposer spend must carry the proposer role")
        return self


class EvalSpend(_StrictModel):
    """What one standalone evaluation's provider calls cost.

    Only the task model appears. A standalone evaluation runs no
    optimizer, so there is no proposer to report -- and an all-zero
    proposer row would claim the evaluation measured one and found it
    free, which is a different and untrue claim. That is why this is not
    :class:`RunSpend`, which requires both roles because an optimizer run
    has both.
    """

    schema_version: Literal[1]
    task_model: RoleSpend

    @model_validator(mode="after")
    def _validate_roles(self) -> EvalSpend:
        if self.task_model.role != "task_model":
            raise ValueError("task model spend must carry the task_model role")
        return self


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
    schema_version: Literal["whetstone_envs.trajectory_report/v1"]
    result_ref: ReportRef
    run_id: StrictStr
    mutation_field: StrictStr
    terminal_status: StrictStr
    candidates: tuple[TrajectoryCandidate, ...]
    steps: tuple[TrajectoryStep, ...]
    resolutions: tuple[TrajectoryResolution, ...]
    terminal_candidate_refs: tuple[ReportRef, ...]
    spend: RunSpend | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> TrajectoryReport:  # noqa: PLR0912
        refs = tuple(candidate.record_ref for candidate in self.candidates)
        if len(set(refs)) != len(refs):
            raise ValueError(
                "trajectory candidates must be unique by exact ref"
            )
        known_refs = set(refs)
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
        if tuple(step.step_index for step in self.steps) != tuple(
            range(len(self.steps))
        ):
            raise ValueError("trajectory step indexes must be contiguous")
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
        for row in self.resolutions:
            embedded = row.eval_report
            if embedded is None:
                continue
            if (
                len(embedded.candidates) != 1
                or embedded.candidates[0].record_ref != row.candidate_ref
            ):
                raise ValueError(
                    "embedded evaluation must cite the exact resolution "
                    "candidate"
                )
            embedded_result = embedded.results[0]
            if row.outcome == "completed":
                if not isinstance(embedded_result, EvalSuccess):
                    raise ValueError(
                        "completed resolution requires successful evaluation "
                        "evidence"
                    )
                if (
                    embedded_result.evidence.evidence_ref
                    != row.eval_result_ref
                    or embedded_result.evidence.reward_ref != row.reward_ref
                ):
                    raise ValueError(
                        "completed resolution refs must match embedded "
                        "evidence"
                    )
                if row.reward_ref is not None and (
                    embedded_result.score != row.reward
                ):
                    raise ValueError(
                        "completed resolution reward must match embedded score"
                    )
            elif row.outcome == "failed":
                if (
                    not isinstance(embedded_result, EvalFailed)
                    or embedded_result.evidence_ref != row.eval_result_ref
                ):
                    raise ValueError(
                        "failed resolution ref must match embedded failure "
                        "evidence"
                    )
        return self
