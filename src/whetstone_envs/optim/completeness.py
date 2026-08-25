"""The per-task completeness floor, owned once for every evaluation path.

Two callers reach the provider and then report a number from what came
back: :class:`~whetstone_envs.optim.study.arms.RoleScorer`, inside a
study stage, and :func:`~whetstone_envs.reporting.execution.
run_c19_evaluation`, behind the standalone ``whetstone-eval`` command.
Both are subject to the same loss and the same bias, so both apply the
same floor from here rather than each carrying its own copy.

**In-search evaluations are deliberately not covered.** A task that lost
every repeat reports ``None`` for its per-task value rather than ``0.0``,
and the optimizer's reward policy is what decides how a search treats
that -- refusing mid-search would abort a run over a transient loss the
search itself is entitled to tolerate. This floor guards the numbers a
*claim* is made from: the official selection score, the held-out
measurement, and the standalone report. What a candidate is worth during
a search is the optimizer's question; what an evaluation is fit to
report is this module's.

**The floor bounds a claim; it does not demand a perfect evaluation.**
Losing rows, and even whole tasks, is an infrastructure outcome the study
tolerates: a lost task is carried at zero weight into the reported row,
where it lowers the achieved completeness and the report downgrades the
arm to ``VERDICT_INCOMPLETE`` rather than claiming it. What this module
refuses is only the evaluation too thin to report any number from --
below :data:`~whetstone_envs.optim.experiment.MIN_TASK_COMPLETENESS` of
planned tasks measured to full depth. Aborting on the *first* lost task
was the opposite trade: it discarded a whole stage's paid evidence to
avoid publishing a number the report was already equipped to mark as
incomplete.
"""

from __future__ import annotations

from typing import Protocol

from whetstone_envs.optim.experiment import MIN_TASK_COMPLETENESS


class TaskCompletenessError(RuntimeError):
    """An evaluation is not complete enough to report a number from.

    A distinct type because the two callers raise different things to
    their own layers -- a stage raises ``StageError``, the reporting path
    surfaces a CLI failure -- and each wraps this rather than matching on
    a message.
    """


class TaskCompletenessEvidence(Protocol):
    """Exactly the evidence fields the completeness check reads.

    Narrower than ``EvalEvidence`` on purpose. This check is pure
    arithmetic over the per-task vectors, the repeat count, and the
    planned task list, so naming that surface keeps the dependency
    auditable -- a future field it started relying on would have to be
    added here first -- and lets the tests exercise it without standing
    up a store, a graph, and a persisted aggregate to reach a function
    that touches none of them.
    """

    @property
    def per_task_values(self) -> tuple[float | None, ...]: ...

    @property
    def per_task_counts(self) -> tuple[int, ...]: ...

    @property
    def num_seeds(self) -> int: ...

    @property
    def aggregate_value(self) -> float | None: ...

    @property
    def aggregate_status(self) -> str: ...

    @property
    def task_hashes(self) -> tuple[str, ...]: ...


def fully_lost_task_count(evidence: TaskCompletenessEvidence) -> int:
    """How many tasks produced no present row at all.

    Read directly off the evidence's per-task vectors rather than
    inferred from the two means, because the direct signal is the one
    that survives whetstone's move to present-row per-task reporting.
    Two spellings are accepted, and they are the same question:

    * ``per_task_values`` carrying ``None`` for a task, which is how a
      task with no OK reduction is reported once ``per_task_score``
      aggregates over *present* rows. This is the authoritative spelling.
    * ``per_task_counts`` carrying ``0`` for a task, once
      ``per_task_count`` counts present rows rather than
      ``len(completed_rows(num_seeds))``.

    Both are checked because either alone would be a bet on one release.
    Under the older behaviour neither fires -- ``completed_rows`` pads a
    short task to ``num_seeds``, so counts are uniformly ``num_seeds``
    and a fully-lost task scores ``0.0`` rather than ``None`` -- which is
    why the row-level tolerance was the only bound that could see
    anything at all, and why it was not enough.

    Deliberately *not* inferred from ``aggregate_value`` against the
    per-task mean: that identity only holds while a missing row scores
    ``0.0``, and assuming it would silently stop detecting anything the
    moment a lost task began reporting ``None`` instead.
    """
    lost = sum(1 for value in evidence.per_task_values if value is None)
    counts = evidence.per_task_counts
    if counts:
        lost = max(lost, sum(1 for count in counts if count == 0))
    return lost


def incomplete_task_count(evidence: TaskCompletenessEvidence) -> int:
    """How many tasks measured fewer than ``num_seeds`` present rows.

    A task that ran three of its four repeats is *measured* -- it
    contributes a value, and the zero-present rule correctly leaves it
    alone -- but it is not measured to the design's depth, and a study
    whose tasks are broadly short is reporting a shallower measurement
    than the one it pre-registered.

    This is what makes the 90% floor non-decorative. Counting only
    fully-lost tasks would leave ``achieved`` at exactly 1.0 whenever the
    stricter zero-present rule had already passed, so the floor could
    never fire; counting short tasks gives it the population it was
    written to bound.

    Counts are the only spelling that can express this, because a short
    task's value is a real number either way. An evidence record with no
    ``per_task_counts`` reports nothing here rather than guessing.
    """
    counts = evidence.per_task_counts
    if not counts:
        return 0
    k_repeat = evidence.num_seeds
    if k_repeat < 1:
        return 0
    return sum(1 for count in counts if count < k_repeat)


def require_task_completeness(
    evidence: TaskCompletenessEvidence, *, purpose: str
) -> None:
    """Bound how shallowly an evaluation may be measured and still report.

    **The row tolerance cannot see this.** ``missing_data="skip"`` with a
    10% row bound is a floor against losing an evaluation to a handful of
    scattered 429s, and it works for that. But it counts rows, and a task
    whose every repeat was lost is dropped from the *task mean's
    denominator* rather than counted: whetstone's
    ``unweighted_task_mean`` classifies it ``ZERO_DENOMINATOR``, and the
    outer mean then divides by the tasks that produced a value.

    At the study's own shape those two bounds disagree badly. 76 tasks at
    4 repeats is 304 rows; one task lost entirely is 1.3% of them, so the
    row tolerance passes and the evaluation reports ``status=ok`` with a
    mean over 75 tasks. The error is not noise -- the tasks that lose
    every repeat are the slow, long-generation ones, which are the tasks
    that would have pulled the mean down -- so the reported number is
    biased upward by exactly the tasks whose absence caused it.

    **A fully-lost task degrades the claim; it does not abort the stage.**
    A task with zero present rows is an infrastructure outcome -- the slow,
    long-generation tasks are the ones that lose every repeat -- and the
    study cannot require the provider to be perfect. What it *can* require
    is that a mean over a shrunken population never be presented as though
    it covered the whole one. So a lost task is carried, at zero weight,
    all the way into the reported row: :func:`
    ~whetstone_envs.optim.study.arms.measured_per_task` keeps its position
    in the vector with a count of zero, O7's weighting drives its
    contribution to nothing, and the resulting ``completeness`` falls below
    :data:`~whetstone_envs.optim.study.manifest.COMPLETENESS_BACKSTOP` --
    which the report reads as ``VERDICT_INCOMPLETE``, an arm measured but
    not claimed.

    That is strictly more informative than the abort it replaces. Raising
    here killed the whole stage over one chronically slow task, discarding
    every other arm's paid evaluation with it and leaving no manifest at
    all -- so the reader learned nothing, having paid for everything. The
    degraded verdict says exactly which arm was measured too shallowly to
    claim, beside every arm that was not.

    What remains a refusal is the *bound*: below :data:`
    ~whetstone_envs.optim.experiment.MIN_TASK_COMPLETENESS` of planned
    tasks measured to full depth, the evaluation is too thin to report a
    number from at all, and it says so rather than emitting one. Losing
    whole tasks pushes an evaluation toward that floor -- a fully-lost task
    is incomplete by construction -- so the floor is what bounds the loss,
    rather than a separate rule that fired on the first one.

    Implemented here rather than in the aggregation config because
    whetstone's ``AggregationConfig`` has no per-task completeness
    variable to set: its knobs are ``reduction``, ``missing_data``,
    ``zero_denominator``, and ``max_skip_fraction``, all of which act on
    the flat row vector. So this is an envs-side validator applied to the
    evidence before the evaluation is accepted -- at the seams every
    reporting evaluation already passes through.

    ``max_skip_fraction`` is what makes the row bound tolerant in the
    first place: at ``0.0`` the first skipped row voids the evaluation,
    and the 10% this study sets is the deliberate loosening that this
    task-level floor then backstops.

    **Anchor calibration is not covered by this tolerance and should not
    be.** An anchor is the reference every arm's delta is measured
    against, so a naive anchor missing its hardest tasks rescales the
    whole study rather than degrading one arm's claim. whetstone's own
    calibration keeps its stricter presence requirement, and that
    asymmetry is deliberate.
    """
    planned_tasks = len(evidence.task_hashes)
    if planned_tasks == 0:
        raise TaskCompletenessError(
            f"{purpose}: an evaluation planned no tasks at all"
        )

    incomplete = incomplete_task_count(evidence)
    complete = planned_tasks - incomplete
    achieved = complete / planned_tasks
    if achieved < MIN_TASK_COMPLETENESS:
        lost = fully_lost_task_count(evidence)
        raise TaskCompletenessError(
            f"{purpose}: only {complete} of {planned_tasks} planned tasks "
            f"measured all {evidence.num_seeds} repeats ({achieved:.3f}), "
            f"below the {MIN_TASK_COMPLETENESS:.2f} task-completeness "
            f"floor. {incomplete} task(s) ran short, of which {lost} lost "
            "every repeat, so the split was measured more shallowly than "
            "the design specified."
        )
