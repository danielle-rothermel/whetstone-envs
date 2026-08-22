"""Pre-spend call estimates for each optimizer, derived from its control.

The study's row-derived budget -- runs times tasks times repeats -- prices
only what *selection and reporting* cost. It says nothing about what an
optimizer spends internally, which is where the study's money actually goes:
MIPROv2's bootstrap walk and GEPA's search dwarf the official and held-out
passes.

**Every estimate here is in task-model rows**, because that is the unit
``cost.json`` reports and the unit the Stage-1 gate receives. Keeping one
unit is not a formatting preference: GEPA's budget is denominated in
*metric calls*, and quoting it directly made the gate compare two different
quantities and reject a healthy run. See ``GEPA_TASK_CALL_CEILING``.

This module is the one named place those numbers live. Each constant carries
its derivation in its docstring, because every one of them was wrong at least
once in the protocol drafts and a number without a derivation cannot be
re-checked when a control default moves.

The module has two halves. The first is **estimates** derived from control
defaults, which is all that existed before any run happened; ``plan`` labels
them as estimates for that reason. The second is Wave 3's
**measurements**, taken on fake transport at the protocol's own splits with
zero provider calls, each carrying the run and settings it came from.

Where the two disagree, the measurement is what actually happens and the
estimate is what was wrong. Three of them disagreed materially:

* **F16 / R6 -- retired.** No fan-out. Every evaluation executed exactly the
  task subset it requested; the measured ratio is 1.0, not the feared 2.51x.
* **F10 -- the 28-616 bootstrap bound does not apply.** The runner gives
  MIPROv2 a one-task trainset, so bootstrapping costs 1-2 rows per run.
* **F9 / D3 -- GEPA is pinned to 200 metric calls.** A measured 732-call run
  produced a 1.73 GB store and a 766 MB result document. The pin is also the
  source of GEPA's gated estimate, in rows.

The estimates are kept rather than replaced. The Stage-1 gate divides by an
upper bound, and a loose upper bound cannot false-abort a run, whereas a
tight one tuned to a fake-transport measurement could.
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.optim.study.spec import CODEX_EVALUATE_CALL_CAP

__all__ = [
    "COPRO_DEFAULT_BREADTH",
    "COPRO_DEFAULT_DEPTH",
    "GEPA_MAX_METRIC_CALLS_PINNED",
    "GEPA_MEASURED_FULL_VALSET_PASSES",
    "GEPA_MEASURED_REFLECTION_MINIBATCHES",
    "GEPA_MEASURED_TASK_CALLS_AT_PIN",
    "GEPA_PIN_REASON",
    "GEPA_REFLECTION_MINIBATCH_TASKS",
    "GEPA_RESOLVED_MAX_METRIC_CALLS",
    "GEPA_TASK_CALL_CEILING",
    "MEASURED_FANOUT_RATIO",
    "MEASURED_GEPA_DISTINCT_EVALUATIONS",
    "MEASURED_GEPA_RESULT_JSON_BYTES",
    "MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES",
    "MEASURED_GEPA_SQLITE_BYTES",
    "MEASURED_GEPA_STEPS",
    "MEASURED_GEPA_TASK_CALLS",
    "MEASURED_GEPA_WALL_SECONDS",
    "MEASURED_GEPA_WALL_SECONDS_PER_STEP",
    "MEASURED_MIPROV2_BOOTSTRAP_ROWS_FEWSHOT",
    "MEASURED_MIPROV2_BOOTSTRAP_ROWS_GROUND_ONLY",
    "MEASURED_MIPROV2_BOOTSTRAP_ROWS_ZEROSHOT",
    "MEASURED_MIPROV2_FEWSHOT_TASK_CALLS",
    "MEASURED_MIPROV2_GROUND_ONLY_TASK_CALLS",
    "MEASURED_MIPROV2_MINIBATCH_TASKS",
    "MEASURED_MIPROV2_TRAINSET_TASKS",
    "MEASURED_MIPROV2_ZEROSHOT_TASK_CALLS",
    "MEASUREMENT_N_PER_STRATUM",
    "MEASUREMENT_POOL_SEED_START",
    "MEASUREMENT_SPLIT_SIZES",
    "MIPROV2_BOOTSTRAPPING_PLANS",
    "MIPROV2_BOOTSTRAP_ROWS_BEST_CASE",
    "MIPROV2_BOOTSTRAP_ROWS_WORST_CASE",
    "MIPROV2_FEWSHOT_TASK_CALL_CEILING",
    "MIPROV2_FEWSHOT_TASK_CALL_FLOOR",
    "MIPROV2_FULL_EVAL_CALLS",
    "MIPROV2_MINIBATCH_CALLS",
    "NULL_IDENTITY_HELD_OUT_PASSES",
    "NULL_IDENTITY_OFFICIAL_PASSES",
    "STAGE1_CALL_COUNT_TOLERANCE",
    "OptimizerCallEstimate",
    "estimate_optimizer_calls",
    "null_identity_report_rows",
]

# --------------------------------------------------------------------------
# COPRO
# --------------------------------------------------------------------------

#: The study's COPRO search shape, matching the runner's own defaults.
COPRO_DEFAULT_BREADTH = 2
COPRO_DEFAULT_DEPTH = 1

# --------------------------------------------------------------------------
# MIPROv2 -- the F10 correction
# --------------------------------------------------------------------------

#: Bootstrapping plans in MIPROv2's candidate-seed sweep.
#:
#: ``candidate_seeds = range(-3, upper_bound)`` yields 9 plans, of which
#: ``RESET`` and ``LABELS_ONLY`` bootstrap nothing. Seven plans therefore
#: walk the trainset cursor.
MIPROV2_BOOTSTRAPPING_PLANS = 7

#: The F10 bootstrap-row bound at ``|trainset| = 88``.
#:
#: The protocol's ``6 x 4 x 3 = 72`` was wrong in method:
#: ``runtime.py`` issues ``task_batch_hashes=(attempt.task_hash,)`` -- one row
#: per attempt, with **no** ``K_REPEAT`` factor -- and
#: ``_next_bootstrap_attempt_unchecked`` is a cursor walk that stops at
#: ``max_bootstrapped_demos`` accepted demos or at the end of the trainset.
#: Each plan therefore costs
#: ``min(max_bootstrapped_demos / p_accept, |trainset|)`` rows: 4 rows per
#: plan when every attempt is accepted, and a full 88-row walk when none is.
#:
#: The worst case is the *likely* one here, and correlated with the study's
#: own premise: a low naive anchor means a low acceptance rate, which means
#: more bootstrap rows. A low-accuracy anchor must not read as a budget
#: overrun at the Stage-1 gate.
MIPROV2_BOOTSTRAP_ROWS_BEST_CASE = 28
MIPROV2_BOOTSTRAP_ROWS_WORST_CASE = 616

#: MIPROv2's non-bootstrap evaluation volume at the study's control
#: defaults: full-valset passes of the incumbent, and the minibatch trials.
#: Both are row counts over the 88-task internal split.
MIPROV2_FULL_EVAL_CALLS = 1_050
MIPROV2_MINIBATCH_CALLS = 792

#: The ``fewshot`` per-run task-call range: the two fixed components plus the
#: bootstrap bound at each end. The **ceiling** is what the Stage-1 gate's
#: "within 1.5x" comparison divides by, per F10.
MIPROV2_FEWSHOT_TASK_CALL_FLOOR = (
    MIPROV2_FULL_EVAL_CALLS
    + MIPROV2_MINIBATCH_CALLS
    + MIPROV2_BOOTSTRAP_ROWS_BEST_CASE
)
MIPROV2_FEWSHOT_TASK_CALL_CEILING = (
    MIPROV2_FULL_EVAL_CALLS
    + MIPROV2_MINIBATCH_CALLS
    + MIPROV2_BOOTSTRAP_ROWS_WORST_CASE
)

# --------------------------------------------------------------------------
# GEPA
# --------------------------------------------------------------------------

#: GEPA's resolved metric-call ceiling at the study's control defaults.
#:
#: ``gepa_auto_budget(num_predictors=1, num_candidates=6, valset_size=88,
#: minibatch_size=35, full_eval_steps=5)`` resolves to 732: ``num_trials``
#: is 10, and the total is ``88 + 30 + 350 + 3 * 88``.
#:
#: **This is a metric-call count, not a task-call count.** It is retained
#: for provenance -- it is what ``gepa_auto_budget`` returns -- but it is
#: *not* what the Stage-1 gate compares against; see
#: :data:`GEPA_TASK_CALL_CEILING` for the gated quantity and the unit
#: derivation.
GEPA_RESOLVED_MAX_METRIC_CALLS = 732

# --------------------------------------------------------------------------
# Codex
# --------------------------------------------------------------------------

# The Codex arm's cap is a design decision (D2), so it is owned by
# ``spec.py`` and re-exported here. Unlike the other three arms it is a
# *cap*, enforced by the admission ledger, not an estimate: the agent may
# use fewer. That is why OQ3 excludes Codex from the Stage-1 call-count
# comparison and gates it on capacity respect instead.

#: The Stage-1 gate's tolerance on measured-versus-estimated call counts.
STAGE1_CALL_COUNT_TOLERANCE = 1.5


@dataclass(frozen=True, slots=True)
class OptimizerCallEstimate:
    """One optimizer's estimated evaluation calls for a single run.

    ``low`` and ``high`` bracket the estimate; they are equal when the
    control determines the count exactly. ``gated`` is false for an arm the
    Stage-1 call-count comparison deliberately skips -- Codex, whose agent
    chooses how much of its cap to spend.
    """

    optimizer: str
    low: int
    high: int
    basis: str
    gated: bool = True

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError(
                f"{self.optimizer!r} estimate must be an ordered "
                "non-negative range"
            )
        if not self.basis.strip():
            raise ValueError(
                f"{self.optimizer!r} estimate must state its derivation"
            )


def estimate_optimizer_calls(  # noqa: PLR0913
    optimizer: str,
    *,
    internal_size: int,
    k_repeat: int,
    copro_breadth: int = COPRO_DEFAULT_BREADTH,
    copro_depth: int = COPRO_DEFAULT_DEPTH,
    official_size: int = 0,
    held_out_size: int = 0,
) -> OptimizerCallEstimate:
    """Estimated evaluation calls for one run of ``optimizer``.

    **Every estimate is in task-model rows**, the unit ``cost.json`` reports
    as ``task_model.calls`` and the unit ``call_count_within_estimate``
    receives. An estimate expressed in any other unit -- GEPA's metric calls
    being the one that got this wrong -- is not comparable to the thing it
    is gated against.

    COPRO is the only arm whose count follows from the study's own split
    sizes, because its search shape is fully configured: ``depth + 1`` steps
    of ``breadth`` candidates, each scored over the whole internal split at
    ``K_REPEAT`` repeats. ``null-random`` perturbs the anchor but still
    drives the same selection machinery, so it shares COPRO's shape.
    MIPROv2 and GEPA carry their own internal budgets, so their estimates
    are the pinned constants above rather than a function of the splits.

    ``null-identity`` is the exception: it runs **no optimizer at all**, so
    its estimate is the report harness's official and held-out passes, which
    is why it needs ``official_size`` and ``held_out_size``. See
    :func:`null_identity_report_rows`.
    """
    if internal_size < 0 or k_repeat < 1:
        raise ValueError("internal_size >= 0 and k_repeat >= 1 are required")
    if optimizer == "null-identity":
        rows = null_identity_report_rows(
            official_size=official_size,
            held_out_size=held_out_size,
            k_repeat=k_repeat,
        )
        return OptimizerCallEstimate(
            optimizer=optimizer,
            low=rows,
            high=rows,
            basis=(
                f"no optimizer run; report harness only: "
                f"{NULL_IDENTITY_OFFICIAL_PASSES} official pass x "
                f"{official_size} tasks + {NULL_IDENTITY_HELD_OUT_PASSES} "
                f"held-out pass x {held_out_size} tasks, x {k_repeat} repeats"
            ),
        )
    if optimizer in {"copro", "null-random"}:
        steps = copro_depth + 1
        rows = steps * copro_breadth * internal_size * k_repeat
        return OptimizerCallEstimate(
            optimizer=optimizer,
            low=rows,
            high=rows,
            basis=(
                f"(depth {copro_depth} + 1) steps x breadth {copro_breadth} "
                f"x {internal_size} internal tasks x {k_repeat} repeats"
            ),
        )
    if optimizer == "miprov2":
        return OptimizerCallEstimate(
            optimizer=optimizer,
            low=MIPROV2_FEWSHOT_TASK_CALL_FLOOR,
            high=MIPROV2_FEWSHOT_TASK_CALL_CEILING,
            basis=(
                f"{MIPROV2_FULL_EVAL_CALLS} full-eval + "
                f"{MIPROV2_MINIBATCH_CALLS} minibatch + "
                f"{MIPROV2_BOOTSTRAP_ROWS_BEST_CASE}-"
                f"{MIPROV2_BOOTSTRAP_ROWS_WORST_CASE} bootstrap rows (F10)"
            ),
        )
    if optimizer == "gepa":
        return OptimizerCallEstimate(
            optimizer=optimizer,
            low=GEPA_TASK_CALL_CEILING,
            high=GEPA_TASK_CALL_CEILING,
            basis=(
                f"pinned budget of {GEPA_MAX_METRIC_CALLS_PINNED} metric "
                f"calls bounds task rows at {GEPA_TASK_CALL_CEILING} (D3); "
                f"measured 732-call run scaled to the pin is "
                f"{GEPA_MEASURED_TASK_CALLS_AT_PIN} rows"
            ),
        )
    if optimizer == "codex":
        return OptimizerCallEstimate(
            optimizer=optimizer,
            low=0,
            high=CODEX_EVALUATE_CALL_CAP * internal_size * k_repeat,
            basis=(
                f"up to {CODEX_EVALUATE_CALL_CAP} admitted evaluate-calls "
                f"x {internal_size} internal tasks x {k_repeat} repeats; "
                "the agent chooses how much of its cap to spend"
            ),
            gated=False,
        )
    raise ValueError(f"no call estimate for optimizer {optimizer!r}")


# --------------------------------------------------------------------------
# Wave 3 measurements -- the numbers that replace the estimates above
# --------------------------------------------------------------------------

# Every constant below is a **measurement**, not a derivation. All of them
# come from fake-transport runs at the protocol's own splits, with zero
# provider calls, on 2026-08-22. Provenance is recorded per group: which
# run, which splits, which control settings. Where a measurement disagrees
# with an estimate above, the measurement is what actually happens and the
# estimate is what was wrong.
#
# Reproduce with, from the repository root:
#
#   python -m whetstone_envs.optim.cli --optimizer miprov2 \
#       --demo-mode fewshot --transport fake --split-sizes 88,132,220 \
#       --n-per-stratum 32 --pool-seed-start 1000000 --num-seeds 1 \
#       --seed 2000 --miprov2-minibatch --miprov2-minibatch-size 35 \
#       --miprov2-minibatch-full-eval-steps 5 --run-id <id> --output <dir>
#
#   python -m whetstone_envs.optim.cli --optimizer gepa --transport fake \
#       --split-sizes 88,132,220 --n-per-stratum 32 \
#       --pool-seed-start 1000000 --num-seeds 1 --seed 3000 \
#       --gepa-max-metric-calls 732 --run-id <id> --output <dir>

#: The splits every Wave 3 measurement below was taken at.
MEASUREMENT_SPLIT_SIZES = (88, 132, 220)

#: The c19 pool those splits were drawn from: 22 strata x 32 = 704 tasks.
MEASUREMENT_N_PER_STRATUM = 32
MEASUREMENT_POOL_SEED_START = 1_000_000

# --------------------------------------------------------------------------
# F16 -- the fan-out measurement. R6 is retired.
# --------------------------------------------------------------------------

#: Measured MIPROv2 ``fewshot`` task-model calls for one run at the
#: measurement splits, minibatch on at size 35 with full evaluation every
#: 5 trials. Run ``w3-mipro-fewshot``.
#:
#: The number is confirmed twice over, by two independent paths: summing
#: ``EvalEvidence.row_accounting.planned`` over the run's evaluations, and
#: reading ``cost.json``'s ``task_model.calls``, which is projected from
#: ``OptimResult.cost`` rather than from eval evidence. Both give 245.
MEASURED_MIPROV2_FEWSHOT_TASK_CALLS = 245
MEASURED_MIPROV2_ZEROSHOT_TASK_CALLS = 246
MEASURED_MIPROV2_GROUND_ONLY_TASK_CALLS = 245

#: Measured GEPA task-model calls for one run at the measurement splits
#: with ``max_metric_calls = 732``. Run ``w3-gepa-full``.
#:
#: Far below 732 because a metric call is not a row: GEPA's budget counts
#: *evaluations*, and the effect cache serves a replayed evaluation without
#: re-executing its rows. 732 metric calls resolved to 91 distinct
#: evaluations and 265 executed rows.
MEASURED_GEPA_TASK_CALLS = 265
MEASURED_GEPA_DISTINCT_EVALUATIONS = 91

#: **The F16 finding.** Across all four measured runs, every evaluation's
#: executed rows equalled the task subset its request declared -- the
#: measured fan-out ratio is exactly 1.0, not the feared 2.51x.
#:
#: The platform's deferral row expansion honours per-intent task sets. The
#: per-intent-subset formula predicts the measured rows exactly; the
#: ``intents x tasks x seeds`` formula overpredicts by 1.78x on MIPROv2
#: (435 against 245) and by 30x on GEPA (8,008 against 265). Which formula
#: matches is the evidence, and the subset formula is the one the code
#: implements.
#:
#: R6 -- the risk that minibatch intents silently fan out over the full
#: 88-task split, costing roughly +12,000 calls at ``K_RUN = 5`` -- is
#: therefore **retired before any spend**, which is what F16 was for.
#: ``tests/optim/study/test_fanout.py`` re-measures this mechanically on a
#: small fake run, so a regression fails a test rather than a budget.
MEASURED_FANOUT_RATIO = 1.0

#: Measured minibatch trial size on the ``fewshot`` run: the two
#: ``miprov2_sample`` evaluations each drew exactly 35 of the 87 validation
#: tasks, and neither covered the split.
MEASURED_MIPROV2_MINIBATCH_TASKS = 35

# --------------------------------------------------------------------------
# F10 -- measured bootstrap rows. The 28-616 bound does not apply.
# --------------------------------------------------------------------------

#: Measured bootstrap rows per run, by demo mode, at the measurement
#: splits. These are the ``miprov2_bootstrap`` evaluations' planned rows.
#:
#: **These are not the 28-616 range above, and the difference is not a
#: measurement error.** ``build_miprov2_control`` splits the internal
#: 88-task set as ``trainset=task_hashes[:1]``, ``valset=task_hashes[1:]``
#: -- a **one-task trainset** and an 87-task validation split, at every
#: split size. Bootstrapping is a single cursor walk over the trainset, so
#: one task is a hard ceiling of one row per bootstrapping plan.
#:
#: That partition is now the ``Miprov2Split.SINGLE_TASK`` setting rather
#: than a slice literal, and it remains the default -- so these numbers
#: still describe a default run. ``Miprov2Split.INTERNAL`` gives
#: bootstrapping the whole internal split and would raise them; an arm that
#: selects it has not been measured.
#:
#: Two further corrections to F10's derivation, both verified against the
#: whetstone-ai code the runs executed:
#:
#: 1. The plan count is ``num_candidates - 2`` for the demo modes that
#:    emit a LABELS_ONLY plan, and ``num_candidates - 1`` for ``zeroshot``,
#:    which does not. It is not a fixed 7. This runner's default
#:    ``num_candidates`` is 3, giving 1 bootstrapping plan (2 for
#:    ``zeroshot``) -- which is exactly what was measured. It is now a
#:    setting (``DEFAULT_MIPROV2_NUM_CANDIDATES``); the protocol's
#:    auto-light 6 would give 4 plans.
#: 2. There is no ``/p_accept`` inflation. ``max_rounds`` is 1, so a
#:    rejected attempt still advances the cursor; a low acceptance rate
#:    collects fewer demos but never costs more rows. F10's claim that "the
#:    worst case is the likely one" had the mechanism backwards.
#:
#: The Stage-1 gate consequence is the opposite of F10's concern: the
#: bootstrap term is negligible, not dominant, so the ceiling above is a
#: very loose bound rather than a tight one. It stays as the gate's
#: denominator because a loose upper bound cannot false-abort a run.
MEASURED_MIPROV2_BOOTSTRAP_ROWS_FEWSHOT = 1
MEASURED_MIPROV2_BOOTSTRAP_ROWS_ZEROSHOT = 2
MEASURED_MIPROV2_BOOTSTRAP_ROWS_GROUND_ONLY = 1

#: The one-task MIPROv2 trainset the runner configures, at any split size.
#: Named because it is the reason the measured bootstrap cost is 1 row and
#: not hundreds, and because it is a runner choice that a future change
#: could move without anyone noticing the budget implication.
MEASURED_MIPROV2_TRAINSET_TASKS = 1

# --------------------------------------------------------------------------
# F9 -- GEPA sizing. D3 resolves to the pre-registered fallback.
# --------------------------------------------------------------------------

#: Measured whetstone steps for one GEPA run at ``max_metric_calls = 732``.
#:
#: Not 732. ``run_one_gepa_iteration`` advances the budget by at most one
#: metric call per step, but a step that resolves entirely from the effect
#: cache consumes none, so steps and metric calls are not one-to-one: 556
#: steps consumed the full 732-call budget and terminalized.
MEASURED_GEPA_STEPS = 556

#: Measured wall-clock seconds for that run, fake transport, no provider
#: calls, on the reference development machine.
MEASURED_GEPA_WALL_SECONDS = 1_329
MEASURED_GEPA_WALL_SECONDS_PER_STEP = (
    MEASURED_GEPA_WALL_SECONDS / MEASURED_GEPA_STEPS
)

#: Measured artifact sizes for that run, in bytes. ``runtime.sqlite`` is
#: 1.73 GB and ``result.json`` is 766 MB -- for a run that executed 265
#: task rows.
MEASURED_GEPA_SQLITE_BYTES = 1_728_311_296
MEASURED_GEPA_RESULT_JSON_BYTES = 765_941_729

#: **The F9 root cause**, and the reason the artifacts are three orders of
#: magnitude larger than the work performed.
#:
#: Each whetstone step restarts ``optimize()`` and replays the whole prefix
#: through the effect cache, and every step persists the entire replayed
#: prefix as its own ``search_evidence``. Step *i* carries about *i*
#: entries, so the run's total search evidence is quadratic in its step
#: count: 155,956 entries across 556 steps, addressing just 91 distinct
#: evaluations. The rows do not grow -- the *record of them* does, once per
#: step per replay.
#:
#: This is why ``measure_fanout`` deduplicates by eval-evidence ref. A
#: citation count would report this run as having evaluated 155,956 times.
MEASURED_GEPA_SEARCH_EVIDENCE_ENTRIES = 155_956

#: **The D3 decision, made on the measurement.**
#:
#: The protocol pre-registered the fallback ``max_metric_calls = 200``,
#: to be taken if 732 proved impractical. It did. The measured run stayed
#: inside the wall-clock budget -- 22 minutes, under the 60-minute bar --
#: but blew the store bar by a wide margin: 1.73 GB against a ~1 GB
#: ceiling, plus a 766 MB ``result.json`` that the bar did not anticipate.
#: At ``K_RUN = 5`` that is roughly 12 GB of durable artifacts for one
#: optimizer's arm, and the audit package must load ``result.json`` whole
#: to read a run's evidence.
#:
#: Wall time is also worse than the average suggests, because the cost is
#: not linear: replay cost grows with the prefix, so the last steps are the
#: expensive ones. The final step alone took minutes and peaked above 6 GB
#: of resident memory.
#:
#: So Stage 1 and Stage 2 run GEPA at the pre-registered 200, not 732.
#: This is the value the protocol registered *before* seeing the
#: measurement, which is what keeps it a pre-registration rather than a
#: number chosen to fit.
GEPA_MAX_METRIC_CALLS_PINNED = 200

#: Why the pin was taken, recorded beside it so the manifest can state the
#: reason rather than just the number.
GEPA_PIN_REASON = (
    "measured 732-call run produced a 1.73 GB runtime.sqlite and a "
    "766 MB result.json across 556 steps; per-step prefix replay makes "
    "both superlinear in the step count"
)


# --------------------------------------------------------------------------
# The unit correction: metric calls are not task calls
# --------------------------------------------------------------------------

#: **The unit the Stage-1 gate compares in: task-model rows.**
#:
#: ``call_count_within_estimate`` receives ``observed_task_calls``, which
#: ``_observed_task_calls`` counts as *rows* -- one per task per completed
#: evaluation -- and which ``cost.json`` independently reports as
#: ``task_model.calls``. GEPA's ``max_metric_calls``, by contrast, counts
#: upstream **per-example metric invocations**. The two are different
#: units, and comparing a row count against a metric-call ceiling is what
#: made a real GEPA run trip the 1.5x gate.
#:
#: The Wave 3 measurement decomposes exactly, which is what makes this a
#: derivation rather than a fitted ratio. The ``w3-gepa-full`` run at
#: ``max_metric_calls = 732`` executed **91 distinct evaluations** totalling
#: **265 task rows**, and 91 and 265 have exactly one decomposition into
#: this control's two evaluation shapes:
#:
#:     2 full-valset passes x 88 tasks  +  89 reflection minibatches x 1 task
#:       = 176 + 89
#:       = 265 rows,  across 2 + 89 = 91 evaluations.
#:
#: (``build_gepa_control`` sets ``reflection_minibatch_size=1`` and passes
#: ``valset_task_hashes=None``, so the valset is the whole internal split.
#: Those are the only two shapes a run of this control can produce.)
#:
#: Note that ``2 * 88 + 89`` is *also* 265: upstream charges one metric
#: call per example, so a run's distinct rows and the metric calls those
#: distinct evaluations cost are the same number. The budget consumed 732
#: rather than 265 because ``run_one_gepa_iteration`` restarts ``optimize()``
#: every whetstone step and replays the whole prefix; upstream re-charges
#: each replayed invocation while the durable effect cache serves it
#: without re-executing a row.
#:
#: Two consequences, and the second is the one the gate needs:
#:
#: 1. Distinct task rows can never exceed the metric-call budget, because
#:    every row costs at least one metric call. The budget is therefore a
#:    sound *upper bound* on rows in the gate's own unit.
#: 2. The bound is loose by exactly the replay factor -- 732 / 265 = 2.76x
#:    on the measured run -- and a loose upper bound cannot false-abort,
#:    which is the property the Stage-1 gate is built on.
GEPA_MEASURED_FULL_VALSET_PASSES = 2
GEPA_MEASURED_REFLECTION_MINIBATCHES = 89
GEPA_REFLECTION_MINIBATCH_TASKS = 1

#: GEPA's per-run task-row ceiling at the **pinned** budget, in the gate's
#: unit. This is what ``estimate_optimizer_calls`` returns for GEPA.
#:
#: The source is ``GEPA_MAX_METRIC_CALLS_PINNED`` -- the Wave 3 D3 decision
#: -- and not the 732 the auto budget resolves to, because 200 is what
#: Stage 1 and Stage 2 actually run. Per consequence 1 above, the budget
#: bounds the rows, so the ceiling is the pinned budget itself: a run that
#: is charged 200 metric calls cannot have executed more than 200 distinct
#: task rows.
#:
#: Scaling the measurement to the pin as a cross-check: the measured run
#: executed 265 rows for 732 charged calls, so the same replay factor at
#: 200 predicts ``265 * 200 / 732 = 72.4`` rows -- comfortably inside this
#: ceiling, as an upper bound should be.
GEPA_TASK_CALL_CEILING = GEPA_MAX_METRIC_CALLS_PINNED

#: The measured run's rows scaled to the pinned budget, kept as the
#: cross-check the ceiling is sanity-checked against rather than as the
#: gate's denominator. Ceiling division: a partial row is still a row.
GEPA_MEASURED_TASK_CALLS_AT_PIN = -(
    -MEASURED_GEPA_TASK_CALLS
    * GEPA_MAX_METRIC_CALLS_PINNED
    // GEPA_RESOLVED_MAX_METRIC_CALLS
)


# --------------------------------------------------------------------------
# The null-B correction: a control is not an optimizer run
# --------------------------------------------------------------------------

#: **``null-identity`` runs no optimizer, so it has no optimizer-side cost.**
#:
#: The estimate this replaces gave null-B COPRO's full search shape --
#: ``(depth + 1) x breadth x internal x K_REPEAT`` -- which is the cost of a
#: search null-B never performs. ``StudyOptimizerRunner._run_null`` does not
#: call ``run_optimizer`` at all: it emits the naive anchor unchanged as its
#: terminal candidate and reports ``observed_task_calls=0``.
#:
#: What null-B does cost is the *report harness*, which every arm pays and
#: which ``report_arm`` issues identically for controls and optimizers (L4):
#: one official scoring pass per run, then one held-out pass for the
#: selected representative. Null-B takes ``K_RUN_NULL_B = 1``, so that is
#: one official pass and one held-out pass.
#:
#: This is deliberately *not* folded into the other arms' estimates. The
#: gate compares ``observed_task_calls``, which is projected from the
#: **run's** own evidence and excludes the report harness entirely, so an
#: optimizer arm's estimate must stay optimizer-side to stay in the same
#: unit as the thing it is compared against. Null-B is the one arm whose
#: run-side cost is zero, which is why its estimate is the harness cost
#: instead of a search it does not run.
NULL_IDENTITY_OFFICIAL_PASSES = 1
NULL_IDENTITY_HELD_OUT_PASSES = 1


def null_identity_report_rows(
    *, official_size: int, held_out_size: int, k_repeat: int
) -> int:
    """Task rows one ``null-identity`` arm costs through the report harness.

    One official scoring pass over the official split plus one held-out
    pass over the held-out split, each at ``K_REPEAT`` repeats. See
    :data:`NULL_IDENTITY_OFFICIAL_PASSES` for why this, and not COPRO's
    search shape, is null-B's estimate.
    """
    if official_size < 0 or held_out_size < 0 or k_repeat < 1:
        raise ValueError(
            "official_size >= 0, held_out_size >= 0 and k_repeat >= 1 "
            "are required"
        )
    return k_repeat * (
        NULL_IDENTITY_OFFICIAL_PASSES * official_size
        + NULL_IDENTITY_HELD_OUT_PASSES * held_out_size
    )
