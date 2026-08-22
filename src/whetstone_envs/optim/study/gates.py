"""Pre-spend call estimates for each optimizer, derived from its control.

The study's row-derived budget -- runs times tasks times repeats -- prices
only what *selection and reporting* cost. It says nothing about what an
optimizer spends internally, which is where the study's money actually goes:
GEPA's 732 metric calls and MIPROv2's bootstrap walk dwarf the official and
held-out passes.

This module is the one named place those numbers live. Each constant carries
its derivation in its docstring, because every one of them was wrong at least
once in the protocol drafts and a number without a derivation cannot be
re-checked when a control default moves.

**These are estimates, not measurements.** ``plan`` labels them as such. Wave
3 measures the real per-run counts on fake transport and pins them; where the
two disagree, the measurement wins and this module's constant is the thing
that was wrong. Nothing here gates a stage on its own -- the Stage-1 gate
compares *measured* counts against
:data:`MIPROV2_FEWSHOT_TASK_CALL_CEILING` and its siblings.
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.optim.study.spec import CODEX_EVALUATE_CALL_CAP

__all__ = [
    "COPRO_DEFAULT_BREADTH",
    "COPRO_DEFAULT_DEPTH",
    "GEPA_RESOLVED_MAX_METRIC_CALLS",
    "MIPROV2_BOOTSTRAPPING_PLANS",
    "MIPROV2_BOOTSTRAP_ROWS_BEST_CASE",
    "MIPROV2_BOOTSTRAP_ROWS_WORST_CASE",
    "MIPROV2_FEWSHOT_TASK_CALL_CEILING",
    "MIPROV2_FEWSHOT_TASK_CALL_FLOOR",
    "MIPROV2_FULL_EVAL_CALLS",
    "MIPROV2_MINIBATCH_CALLS",
    "STAGE1_CALL_COUNT_TOLERANCE",
    "OptimizerCallEstimate",
    "estimate_optimizer_calls",
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
#: is 10, and the total is ``88 + 30 + 350 + 3 * 88``. Because
#: ``run_one_gepa_iteration`` advances the budget by one call per whetstone
#: step, 732 metric calls is also **732 whetstone steps** -- which is why
#: F9 measures wall time before Stage 1 rather than after.
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


def estimate_optimizer_calls(
    optimizer: str,
    *,
    internal_size: int,
    k_repeat: int,
    copro_breadth: int = COPRO_DEFAULT_BREADTH,
    copro_depth: int = COPRO_DEFAULT_DEPTH,
) -> OptimizerCallEstimate:
    """Estimated evaluation calls for one run of ``optimizer``.

    COPRO is the only arm whose count follows from the study's own split
    sizes, because its search shape is fully configured: ``depth + 1`` steps
    of ``breadth`` candidates, each scored over the whole internal split at
    ``K_REPEAT`` repeats. MIPROv2 and GEPA carry their own internal budgets,
    so their estimates are the pinned constants above rather than a function
    of the splits. The nulls make no provider proposal call and evaluate
    exactly as COPRO does.
    """
    if internal_size < 0 or k_repeat < 1:
        raise ValueError("internal_size >= 0 and k_repeat >= 1 are required")
    if optimizer in {"copro", "null-random", "null-identity"}:
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
            low=GEPA_RESOLVED_MAX_METRIC_CALLS,
            high=GEPA_RESOLVED_MAX_METRIC_CALLS,
            basis=(
                f"gepa_auto_budget resolves to "
                f"{GEPA_RESOLVED_MAX_METRIC_CALLS} metric calls"
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
