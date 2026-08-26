"""The reported score is a two-stage per-task mean, bit for bit.

**Fails-before: every site recomputed one flat mean over all rows.**

whetstone-ai persists each evaluation aggregate through
``unweighted_task_mean`` -- each task's repeats reduce to one value, then
those values are meaned across tasks. This package's reporting projection
recomputes that number as an independent check on the persisted evidence,
and it recomputed a *flat* mean over every row instead.

Both orders are the same rational number and different floats, because
IEEE-754 addition is not associative. At ``num_seeds=1`` they are the
identical sum, which is why every pre-existing reporting test passed: the
whole suite ran at one seed. At ``num_seeds=3`` they diverge by 1 ULP for
roughly half of all evaluations, and the check then rejected evidence that
was correct -- ``_success_projection`` raised "recomputed aggregate
disagrees with evidence" and the run failed at publication.

These tests hold ``num_seeds > 1``, the one dimension whose absence let
the bug ship, and they pin an arrangement that *actually* diverges rather
than assuming any multi-seed fixture would.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whetstone_envs.probes import normalize
from whetstone_envs.reporting.schema import (
    EVAL_REPORT_SCHEMA,
    CandidateRecord,
    EvalReport,
    EvalRun,
    EvalSuccess,
    Observation,
    ObservationState,
    ReportedEvidence,
    ReportRef,
    RowAccounting,
    StratumSummary,
    TaskRecord,
    two_stage_task_mean,
)

#: How many repeats each task runs. The divergence this module pins does
#: not exist at 1 -- one row per task makes both reductions the same sum.
REPEATS = 3

#: Passes per task, in task order. Two tasks at 2/3 and two at 3/3 is the
#: smallest arrangement that diverges: it reaches 5/6 as 20/24 flat and as
#: (2/3 + 1 + 2/3 + 1)/4 two-stage, and those are adjacent doubles. It also
#: diverges within each 2-task stratum below, so one fixture closes the
#: overall and the stratum sites at once.
PASSES_BY_TASK = (2, 3, 2, 3)

#: The two floats this fixture reaches. Pinned as literals, not recomputed,
#: so a change to either reduction is a visible edit to this file.
FLAT_MEAN = 0.8333333333333334
TWO_STAGE_MEAN = 0.8333333333333333

#: Each 2-task stratum sees one 2/3 task and one 3/3 task, and diverges
#: the same way -- the values are the same two doubles.
STRATUM_FLAT_MEAN = 0.8333333333333334
STRATUM_TWO_STAGE_MEAN = 0.8333333333333333

GOLD = "yes"
WRONG = "no"
STRATA_BY_TASK = (("left",), ("left",), ("right",), ("right",))


def test_the_fixture_actually_diverges() -> None:
    """The premise: these two reductions are genuinely different floats.

    Asserted rather than assumed. A fixture that happened to reach the
    same double under both orders would let every test below pass against
    the flat recompute and prove nothing.
    """
    rows = [
        1.0 if seed < passes else 0.0
        for passes in PASSES_BY_TASK
        for seed in range(REPEATS)
    ]
    flat = sum(rows) / len(rows)
    two_stage = sum(passes / REPEATS for passes in PASSES_BY_TASK) / len(
        PASSES_BY_TASK
    )

    assert flat == FLAT_MEAN
    assert two_stage == TWO_STAGE_MEAN
    assert flat != two_stage
    # Same rational number, adjacent doubles.
    assert flat == pytest.approx(two_stage)
    assert abs(flat - two_stage) == pytest.approx(1.1e-16, rel=0.2)


def test_two_stage_task_mean_reduces_per_task_then_across_tasks() -> None:
    observations = _observations()

    assert two_stage_task_mean(observations) == TWO_STAGE_MEAN
    assert two_stage_task_mean(observations) != FLAT_MEAN


def test_two_stage_task_mean_reports_nothing_without_a_single_score() -> None:
    """No number exists only when nothing anywhere was scored.

    A *partly* unscored matrix still reports a number -- see the presence
    matrix below. This is the residual case: an empty row tuple, and a
    matrix in which every row failed to produce a score.
    """
    assert two_stage_task_mean(()) is None

    nothing_scored = tuple(
        _unscored(row, ObservationState.MISSING) for row in _observations()
    )

    assert two_stage_task_mean(nothing_scored) is None


#: The presence dimension, crossed with the repeat dimension.
#:
#: #41 closed "no test above ``num_seeds=1``" and left "no test below
#: full presence" open, which is how a single provider refusal took down
#: Stage 2. Both dimensions are now pinned together, so neither can be
#: reopened by a fixture that only varies the other.
PRESENCE_CASES = (
    "complete",
    "one-invalid",
    "one-missing",
    "zero-present-task",
)


def _matrix(
    presence: str, repeats: int
) -> tuple[tuple[Observation, ...], float | None]:
    """The observation matrix for ``presence`` at ``repeats``, and its mean.

    The expected value is built here from the *present* scores only,
    independently of ``two_stage_task_mean``'s grouping, so the assertion
    is a real second opinion rather than the implementation restated.
    """
    rows = _observations(repeats=repeats)
    # Always strike task 0's *last* repeat, so "one row of task 0 is
    # non-present" means the same thing at every repeat count. At
    # ``repeats=1`` that is task 0's only row, which is exactly how the
    # one-row cases reach the zero-present rule.
    struck = repeats - 1
    if presence in {"one-invalid", "one-missing"}:
        state = (
            ObservationState.INVALID
            if presence == "one-invalid"
            else ObservationState.MISSING
        )
        rows = tuple(
            _unscored(row, state)
            if row.task_index == 0 and row.seed_index == struck
            else row
            for row in rows
        )
    elif presence == "zero-present-task":
        # Every row of task 0 is non-present, so task 0 leaves the mean.
        rows = tuple(
            _unscored(row, ObservationState.INVALID)
            if row.task_index == 0
            else row
            for row in rows
        )

    by_task: dict[int, list[float]] = {}
    for row in rows:
        if row.score is not None:
            by_task.setdefault(row.task_index, []).append(row.score)
    means = [sum(v) / len(v) for v in by_task.values() if v]
    return rows, (sum(means) / len(means) if means else None)


@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("presence", PRESENCE_CASES)
def test_two_stage_task_mean_across_the_presence_matrix(
    presence: str, repeats: int
) -> None:
    """Presence selects the addends; completeness does not gate the number.

    **Fails-before for every non-``complete`` case:** the old
    implementation returned ``None`` the moment any row lacked a score,
    at either repeat count.
    """
    rows, expected = _matrix(presence, repeats)

    assert two_stage_task_mean(rows) == expected
    # Only the residual all-unscored case reports nothing, and none of
    # these fixtures is that: every one keeps at least one scored task.
    assert expected is not None
    assert two_stage_task_mean(rows) is not None


@pytest.mark.parametrize("repeats", [1, 3])
def test_a_partly_present_task_contributes_its_present_rows(
    repeats: int,
) -> None:
    """The 2-of-3 task the Stage-2 failure turned on.

    At ``repeats=3`` task 0 loses one row and contributes the mean of the
    two that landed -- at full weight, like every other task. At
    ``repeats=1`` it loses its only row and drops out instead, which is
    the zero-present rule reached from the same edit.
    """
    rows, _ = _matrix("one-invalid", repeats)
    task_zero = [row for row in rows if row.task_index == 0]
    present = [row.score for row in task_zero if row.score is not None]

    assert len(task_zero) == repeats
    assert len(present) == repeats - 1

    others = [
        sum(
            row.score
            for row in rows
            if row.task_index == index and row.score is not None
        )
        / repeats
        for index in range(1, len(PASSES_BY_TASK))
    ]
    # Every task but task 0 kept all its rows.
    assert all(
        sum(1 for row in rows if row.task_index == index) == repeats
        for index in range(1, len(PASSES_BY_TASK))
    )
    if present:
        expected = (sum(present) / len(present) + sum(others)) / (
            len(others) + 1
        )
    else:
        expected = sum(others) / len(others)

    assert two_stage_task_mean(rows) == expected


@pytest.mark.parametrize(
    "state", [ObservationState.INVALID, ObservationState.MISSING]
)
def test_invalid_and_missing_rows_leave_the_mean_identically(
    state: ObservationState,
) -> None:
    """Both non-present kinds are absent from the mean, not zeros.

    A refusal scored as 0.0 would drag the task's mean down; upstream
    drops it instead. Pinning both kinds against the same expected value
    keeps a future edit from special-casing one of them.
    """
    rows = _observations(repeats=3)
    dropped = (*rows[:1], _unscored(rows[1], state), *rows[2:])

    as_zero = (
        *rows[:1],
        rows[1].model_copy(
            update={
                "score": 0.0,
                "output_text": WRONG,
                "normalized_output": normalize(WRONG),
            }
        ),
        *rows[2:],
    )

    assert two_stage_task_mean(dropped) != two_stage_task_mean(as_zero)
    # Task 0 kept 2 of 3 rows, both of which passed.
    assert two_stage_task_mean(dropped) == two_stage_task_mean(
        (*rows[:1], *rows[2:])
    )


def test_eval_report_accepts_the_two_stage_score_at_three_seeds() -> None:
    """The end-to-end shape the failed Stage-1 run rejected.

    ``EvalReport``'s validator recomputes the overall score and every
    stratum score from the observations. Under the flat recompute this
    exact report -- whose scores are what whetstone-ai persisted --
    raised "disagrees with observation accounting".
    """
    report = _report()

    assert report.run.repeats == REPEATS
    result = report.results[0]
    assert isinstance(result, EvalSuccess)
    assert result.score == TWO_STAGE_MEAN
    # The row-level pass count stays honest and row-level: 10 of 12 rows
    # passed, which is not the reported score.
    assert (result.numerator, result.denominator) == (10, 12)
    assert result.numerator / result.denominator != result.score
    assert tuple(item.score for item in result.strata) == (
        STRATUM_TWO_STAGE_MEAN,
        STRATUM_TWO_STAGE_MEAN,
    )
    assert tuple(
        (item.numerator, item.denominator) for item in result.strata
    ) == ((5, 6), (5, 6))


def test_eval_report_rejects_a_flat_mean_overall_score() -> None:
    """The old convention is now a rejected report, not a silent variant."""
    payload = _report().model_dump()
    payload["results"][0]["score"] = FLAT_MEAN

    with pytest.raises(
        ValidationError, match="disagrees with observation accounting"
    ):
        EvalReport.model_validate(payload)


def test_eval_report_rejects_a_flat_mean_stratum_score() -> None:
    payload = _report().model_dump()
    payload["results"][0]["strata"][0]["score"] = STRATUM_FLAT_MEAN

    with pytest.raises(ValidationError, match="strata disagree"):
        EvalReport.model_validate(payload)


def test_stratum_summary_accepts_a_score_its_scalars_cannot_derive() -> None:
    """``numerator / denominator`` is no longer the stratum score.

    ``StratumSummary`` validates in isolation with no rows in reach, so it
    checks the completeness rule and the unit interval and leaves the exact
    float to ``EvalReport``'s row-level recompute.
    """
    summary = StratumSummary(
        stratum="left",
        numerator=5,
        denominator=6,
        accounting=RowAccounting(
            planned=6, present=6, missing=0, failed=0, invalid=0
        ),
        score=STRATUM_TWO_STAGE_MEAN,
    )

    assert summary.score != summary.numerator / summary.denominator


def test_stratum_summary_accepts_a_partially_present_score() -> None:
    """Partial presence reports a number, and must.

    This is the rule the failing Stage-2 evaluation needed: 5 of 6 rows
    scored still yields a mean over those 5, because that is what
    whetstone-ai's aggregation persisted. Requiring full presence here
    rejected correct evidence.
    """
    summary = StratumSummary(
        stratum="left",
        numerator=5,
        denominator=6,
        accounting=RowAccounting(
            planned=6, present=5, missing=0, failed=0, invalid=1
        ),
        score=STRATUM_TWO_STAGE_MEAN,
    )

    assert summary.score == STRATUM_TWO_STAGE_MEAN


@pytest.mark.parametrize(
    ("present", "score", "message"),
    [
        # Nothing scored anywhere, yet a number is reported.
        (0, STRATUM_TWO_STAGE_MEAN, "iff some row was scored"),
        # Rows were scored, yet no number is reported.
        (6, None, "iff some row was scored"),
        (6, 1.5, "unit-interval"),
    ],
)
def test_stratum_summary_still_guards_presence_and_range(
    present: int, score: float | None, message: str
) -> None:
    """Loosening the float check did not loosen the invariants that matter.

    A stratum reports a score iff something in it was scored -- which
    keeps a wholly-unmeasured stratum from being quietly averaged in --
    and a reported score must be a mean.
    """
    with pytest.raises(ValidationError, match=message):
        StratumSummary(
            stratum="left",
            numerator=min(5, present),
            denominator=6,
            accounting=RowAccounting(
                planned=6,
                present=present,
                missing=6 - present,
                failed=0,
                invalid=0,
            ),
            score=score,
        )


def _unscored(
    row: Observation, state: ObservationState, *, code: str = ""
) -> Observation:
    """``row`` as a non-present row of ``state``, keeping its position.

    Non-present is not one thing: an INVALID row executed and was billed
    but produced nothing scoreable (a provider refusal), a MISSING row
    never dispatched. Both are absent from the mean; the presence matrix
    below asserts they are treated identically by it.
    """
    trace_state = {
        ObservationState.MISSING: "missing",
        ObservationState.INVALID: "invalid",
        ObservationState.FAILED: "failed",
    }[state]
    default_code = {
        ObservationState.MISSING: "row-not-dispatched",
        ObservationState.INVALID: "provider-rejection",
        ObservationState.FAILED: "transport-error",
    }[state]
    return row.model_copy(
        update={
            "score": None,
            "output_text": None,
            "normalized_output": None,
            "state": state,
            "trace_state": trace_state,
            "failure_code": code or default_code,
        }
    )


def _observations(repeats: int = REPEATS) -> tuple[Observation, ...]:
    """The 4x``repeats`` matrix, in persisted task-major, seed-minor order."""
    rows: list[Observation] = []
    for task_index, passes in enumerate(PASSES_BY_TASK):
        for seed_index in range(repeats):
            text = GOLD if seed_index < passes else WRONG
            rows.append(
                Observation(
                    candidate_name="naive",
                    task_id=f"t-{task_index}",
                    task_hash=f"{task_index:064x}",
                    task_index=task_index,
                    seed_index=seed_index,
                    rendered_prompt="prompt",
                    output_text=text,
                    normalized_output=normalize(text),
                    score=1.0 if text == GOLD else 0.0,
                    state=ObservationState.SCORED,
                    trace_state="success",
                    failure_code="",
                    finish_reason="stop",
                    provider_error=None,
                    max_budget=None,
                    over_budget=None,
                    submission_result=None,
                    component_trace=(),
                )
            )
    return tuple(rows)


def _report() -> EvalReport:
    observations = _observations()
    accounting = RowAccounting(
        planned=len(observations),
        present=len(observations),
        missing=0,
        failed=0,
        invalid=0,
    )
    stratum_accounting = RowAccounting(
        planned=6, present=6, missing=0, failed=0, invalid=0
    )
    ref = ReportRef(schema_name="record", content_hash="a" * 64)
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA,
        run=EvalRun(
            run_id="two-stage-regression",
            family="c19",
            transport="fake",
            model="fake-model",
            role="internal",
            split_sizes=(len(PASSES_BY_TASK), 0, 0),
            repeats=REPEATS,
            dataset_revision="rev",
            graph_hash="b" * 64,
            eval_config_hash="c" * 64,
            package_version="0.0.0",
        ),
        candidates=(
            CandidateRecord(
                name="naive",
                candidate_id="cand-naive",
                source="naive",
                record_ref=ref,
                identity_hash="d" * 64,
                payload={"prompt_template": "answer"},
                prompt_template="answer",
            ),
        ),
        tasks=tuple(
            TaskRecord(
                task_id=f"t-{index}",
                task_hash=f"{index:064x}",
                seed=index,
                strata=strata,
                prompt_inputs={"question": "q"},
                gold=GOLD,
            )
            for index, strata in enumerate(STRATA_BY_TASK)
        ),
        observations=observations,
        results=(
            EvalSuccess(
                kind="success",
                candidate_name="naive",
                classification="measured",
                message="evaluation completed",
                evidence=ReportedEvidence(
                    evidence_ref=ref,
                    outputs_ref=ref,
                    traces_ref=ref,
                    aggregate_ref=ref,
                    reward_ref=None,
                    aggregate_name="unweighted_task_mean",
                    aggregate_value=TWO_STAGE_MEAN,
                    aggregate_status="ok",
                    row_accounting=accounting,
                ),
                accounting=accounting,
                numerator=sum(PASSES_BY_TASK),
                denominator=len(observations),
                score=TWO_STAGE_MEAN,
                strata=(
                    StratumSummary(
                        stratum="left",
                        numerator=5,
                        denominator=6,
                        accounting=stratum_accounting,
                        score=STRATUM_TWO_STAGE_MEAN,
                    ),
                    StratumSummary(
                        stratum="right",
                        numerator=5,
                        denominator=6,
                        accounting=stratum_accounting,
                        score=STRATUM_TWO_STAGE_MEAN,
                    ),
                ),
            ),
        ),
        spend=None,
    )
