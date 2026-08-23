"""``whetstone-study``: plan, run, report, and check one Step 10 study.

The CLI is a dispatcher, not a harness. Every side of the study that costs
money or takes hours lives behind an injected callable -- ``run`` reaches
the stage harness through :class:`StageRunner` and ``report`` reaches the
report generator through :class:`ReportGenerator` -- so this module is
testable without either, and neither can be reached by accident from a
subcommand that was only meant to print.

``plan`` is the one subcommand that computes something itself, and it
computes only arithmetic over a :class:`StudySpecLike`: the run matrix and
its call budget. That keeps the pre-spend answer ("what is this about to
cost?") available before the harness exists and without touching a
provider.

Injection is by protocol, not by import. All three collaborators default
to the real implementations -- :func:`load_study_spec` over the study's own
manifest, :func:`run_stage` over the stage harness, and
:func:`default_report_generator` over the report package. Tests pass their
own collaborators, which is how the ordering and the wiring are verified
separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from dr_store import DocumentFileError
from dr_store.sync import open_sqlite

from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
)
from whetstone_envs.optim.nulls import NULL_IDENTITY_OPTIMIZER
from whetstone_envs.optim.study.environment import (
    FAKE_TRANSPORT,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_TRANSPORT,
    bound_stage_environment,
)
from whetstone_envs.optim.study.gates import (
    GEPA_MAX_METRIC_CALLS_PINNED,
    GEPA_MEASURED_TASK_CALLS_AT_PIN,
    MEASURED_GEPA_TASK_CALLS,
    MEASURED_MIPROV2_FEWSHOT_TASK_CALLS,
    MEASUREMENT_NUM_SEEDS,
    estimate_optimizer_calls,
)
from whetstone_envs.optim.study.init import init_study
from whetstone_envs.optim.study.leakage import (
    HeldOutObservation,
    LeakageRule,
    OptimizerEvalObservation,
    SearchRepeatObservation,
    SplitIdentity,
    optimizer_observations_for_study,
    study_leakage_check,
)
from whetstone_envs.optim.study.manifest import (
    DISCARD_STALE_RUNS_FLAG,
    STAGE_IDS,
    STUDY_MANIFEST_NAME,
    TRANSPORT_NAMES,
    LeakageCheckEntry,
    LeakageCheckRecord,
    RunSpendRecord,
    SplitName,
    StageId,
    StageRecord,
    TransportName,
    check_manifest_pointers,
    format_pointer_report,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.power import (
    WORST_CASE_SIGMA_SQ,
    minimum_detectable_effect,
)
from whetstone_envs.optim.study.protocols import (
    PROTOCOL_IDS,
    SIZED_FIELDS,
    STEP10_C19_ID,
    StudyProtocol,
    study_protocol,
    without_codex,
)
from whetstone_envs.optim.study.selection import SelectionError
from whetstone_envs.optim.study.spec import load_study_spec
from whetstone_envs.optim.study.stages import StageError
from whetstone_envs.optim.study.stages import run_stage as _run_stage_harness
from whetstone_envs.reporting.study_report import generate_study_report

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from whetstone_envs.optim.study.leakage import LeakageReport
    from whetstone_envs.optim.study.manifest import StudyManifest

#: The console script's name, matching ``[project.scripts]``.
PROGRAM_NAME = "whetstone-study"

#: Exit codes. ``2`` distinguishes "the study says no" -- a failed pointer
#: check or a refused stage -- from ``1``, which is a usage or IO failure.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CHECK_FAILED = 2


class StudySpecLike(Protocol):
    """What ``plan`` needs from Wave 4a's ``StudySpec``.

    Structural, and deliberately minimal: ``plan`` prints a run matrix and
    two budgets, so it needs the arms, their run counts, the split sizes,
    and the repeat count -- nothing about how a stage executes.
    ``StudySpec`` satisfies this by carrying exactly these members.

    An arm and its optimizer are named separately, because the study runs
    one optimizer under more than one arm: MIPROv2's three demo modes are
    three arms of the same optimizer. ``optimizer_by_arm`` is what the
    estimate is keyed on; an optimizer the estimator does not recognize
    reports "no estimate" rather than a number derived from a guess.
    """

    @property
    def study_id(self) -> str: ...

    @property
    def arm_ids(self) -> tuple[str, ...]: ...

    @property
    def k_run_by_arm(self) -> Mapping[str, int]: ...

    @property
    def optimizer_by_arm(self) -> Mapping[str, str]: ...

    @property
    def copro_shape_by_arm(self) -> Mapping[str, tuple[int, int] | None]: ...

    @property
    def k_repeat(self) -> int: ...

    @property
    def split_sizes(self) -> tuple[int, int, int]: ...


class StudySpecLoader(Protocol):
    """Load the study's spec from its directory."""

    def __call__(self, study_dir: Path) -> StudySpecLike: ...


class StageLedgerLoader(Protocol):
    """Load the stage records ``plan`` prints its measured ledger from.

    Separate from :class:`StudySpecLoader` because the two read different
    things for different reasons: the spec is the *design*, from which the
    budget is derived, and the stage records are what has actually been
    bought so far. A caller that stubs one is not thereby asserting
    anything about the other, and ``plan`` must not require a manifest on
    disk merely to print a matrix a caller handed it directly.
    """

    def __call__(self, study_dir: Path) -> tuple[StageRecord, ...]: ...


def default_stage_ledger(study_dir: Path) -> tuple[StageRecord, ...]:
    """The study's own recorded stages, or none when it has not run.

    A study directory with no manifest has bought nothing, which is a
    truthful empty ledger rather than an error: ``plan`` is the command an
    operator runs *before* the first stage, and failing it on the absence
    of a file that the first stage creates would make it unusable exactly
    when it is most useful.
    """
    try:
        return read_study_manifest(study_dir).stages
    except (OSError, ValueError, DocumentFileError):
        return ()


class StageRunner(Protocol):
    """Run one stage of the study and return its updated manifest.

    Wave 4a owns the implementation. It returns the manifest rather than a
    path so the CLI reports what the stage recorded without re-reading a
    file the harness may still be writing.

    ``replace_design`` is part of the contract rather than a Stage-0 detail
    because the CLI cannot know which stage a runner will dispatch to; the
    harness refuses it on the stages that record no design.

    ``allow_real_codex`` is this invocation's authorization to spend on a
    real, billed Codex session. It is a run-time spend authorization, not
    part of the study's design: it never reaches the manifest and never
    enters the pre-registration hash, so two studies that differ only in
    whether the operator authorized Codex spend pre-register identically.

    ``transport`` is the same kind of thing and is treated differently in
    exactly one respect. It is likewise an invocation property and likewise
    outside the pre-registration hash -- but the stage *records* it, because
    a stage run on the fake transport and a stage run against a provider
    are different evidence for whatever the stage measured.
    """

    def __call__(  # noqa: PLR0913
        self,
        *,
        study_dir: Path,
        stage: str,
        replace_design: bool = False,
        allow_real_codex: bool = False,
        discard_stale_runs: bool = False,
        transport: str = FAKE_TRANSPORT,
    ) -> StudyManifest: ...


class ReportGenerator(Protocol):
    """Generate the report packet from a manifest, returning its directory.

    It takes the manifest itself, not the study directory, because the
    report is defined to read only the manifest and the evidence the
    manifest names. :func:`default_report_generator` is the implementation
    this CLI binds.
    """

    def __call__(self, *, manifest: StudyManifest, out_dir: Path) -> Path: ...


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def plan_lines(spec: StudySpecLike) -> tuple[str, ...]:
    """The run matrix and both budgets implied by ``spec``.

    Two budgets, kept visibly separate because they are known to different
    degrees:

    * The **selection and reporting** budget is *derived*: one row per task,
      per repeat, per scored candidate, computed here from the matrix rather
      than read from the spec, so a spec cannot assert a budget its matrix
      does not imply.
    * The **optimizer-side** budget is *estimated*: what each arm spends
      internally, from the control defaults in
      :mod:`whetstone_envs.optim.study.gates`. Every figure is in
      task-model rows, the same unit as the selection budget above, so the
      two columns add up; MIPROv2's bootstrap walk is what dominates it.

      Wave 3 measured two of these arms on fake transport at these very
      splits, so where a measurement exists the plan prints it beside the
      estimate and labels it ``MEASURED``. The labels are the point: an
      estimate and a measurement are known to different degrees, and a
      reader deciding whether to authorize spend needs to see which is
      which rather than a single undifferentiated column.
    """
    internal, official, held_out = spec.split_sizes
    lines = [
        f"study: {spec.study_id}",
        (
            f"splits: internal={internal} official={official} "
            f"held_out={held_out}"
        ),
        f"K_REPEAT: {spec.k_repeat}",
        "",
        "selection and reporting rows (derived from the matrix):",
        (f"{'arm':<24}{'K_RUN':>8}{'official rows':>16}{'held-out rows':>16}"),
    ]
    total_official = 0
    total_held_out = 0
    for arm_id in spec.arm_ids:
        k_run = spec.k_run_by_arm[arm_id]
        # Every run's terminal candidate is scored on official; exactly one
        # representative candidate per arm reaches held-out.
        official_rows = k_run * official * spec.k_repeat
        held_out_rows = held_out * spec.k_repeat
        total_official += official_rows
        total_held_out += held_out_rows
        lines.append(
            f"{arm_id:<24}{k_run:>8}{official_rows:>16}{held_out_rows:>16}"
        )
    lines.extend(
        (
            "",
            f"total official rows: {total_official}",
            f"total held-out rows: {total_held_out}",
            f"total selection+report rows: {total_official + total_held_out}",
            "",
        )
    )
    lines.extend(_mde_lines(held_out_size=held_out, k_repeat=spec.k_repeat))
    lines.extend(
        _optimizer_budget_lines(
            spec,
            internal_size=internal,
            k_repeat=spec.k_repeat,
            official_size=official,
            held_out_size=held_out,
        )
    )
    return tuple(lines)


#: The task-by-arm interaction variances the pre-registered MDE is quoted
#: at. These are the protocol review's own two design points, so they are
#: pinned here rather than swept: a plan that quoted one number would hide
#: how sharply the MDE moves with an assumption nothing has measured yet.
MDE_TAU_SQ_CASES: tuple[float, ...] = (0.05, 0.10)

#: How the MDE row is labelled. It is a *pre-registered* number computed
#: from the design, not a measurement: Stage 0 measures ``tau^2`` and
#: ``sigma^2`` and records the MDE that follows, which may differ.
MDE_HEADING = (
    "pre-registered MDE on the held-out split "
    f"(worst-case sigma^2={WORST_CASE_SIGMA_SQ}, from power.py):"
)


def _mde_lines(*, held_out_size: int, k_repeat: int) -> tuple[str, ...]:
    """The MDE this design can resolve, at each pinned ``tau^2``.

    Computed from :func:`minimum_detectable_effect` at the manifest's own
    held-out size and ``K_REPEAT`` rather than read off a table, so the
    number a reader authorizes spend against is the one the design implies.

    ``sigma^2`` is the worst-case binary within-task variance. Stage 0
    replaces both variances with measurements and records the resulting
    MDE in the design block; until then this is the pre-registration.
    """
    if held_out_size < 1 or k_repeat < 1:
        # A spec with no held-out split or no repeats has no MDE to state,
        # and a fabricated one would be worse than none.
        return ("", f"{MDE_HEADING} not computable at these sizes", "")
    lines = ["", MDE_HEADING]
    lines.extend(
        f"  tau^2={tau_sq:<6} T={held_out_size} K={k_repeat}  "
        f"MDE={
            minimum_detectable_effect(
                tau_sq=tau_sq,
                sigma_sq=WORST_CASE_SIGMA_SQ,
                n_tasks=held_out_size,
                num_seeds=k_repeat,
            ):.4f}"
        for tau_sq in MDE_TAU_SQ_CASES
    )
    lines.append("")
    return tuple(lines)


#: How the optimizer-side budget is labelled. Per-arm rows carry their own
#: label, because some arms are measured and some are not.
OPTIMIZER_BUDGET_HEADING = (
    "optimizer-side calls per run (ESTIMATE from control defaults unless "
    "marked MEASURED):"
)

#: The two per-arm labels. An estimate and a measurement are known to
#: different degrees, so they are never printed in the same voice.
ESTIMATE_LABEL = "ESTIMATE"
MEASURED_LABEL = "MEASURED"

#: What an unrecognised arm reports instead of a fabricated number.
NO_ESTIMATE = "no estimate"

#: Wave 3's fake-transport measurements, per arm, at the study's own
#: splits. Only arms actually measured appear; an arm with no entry prints
#: its estimate and says so. c19 fake transport, ``(88, 132, 220)``,
#: ``num_seeds=1``; see ``gates.py`` for each number's provenance.
MEASURED_TASK_CALLS_BY_ARM = {
    "miprov2": MEASURED_MIPROV2_FEWSHOT_TASK_CALLS,
    # GEPA was measured at ``max_metric_calls = 732``; Stage 1 and Stage 2
    # run the pinned 200 (D3), so the plan prints the measurement scaled to
    # the budget the study actually spends rather than the one it retired.
    "gepa": GEPA_MEASURED_TASK_CALLS_AT_PIN,
}

#: The default provenance line for a measured arm, and the arms whose
#: provenance differs from it. GEPA's number is the measurement *scaled* to
#: the pinned budget rather than a figure read straight off a run, and a
#: label that did not say so would overstate what was measured.
MEASURED_BASIS_DEFAULT = (
    f"measured on fake transport at these splits at "
    f"{MEASUREMENT_NUM_SEEDS} repeat(s), scaled to K_REPEAT (Wave 3)"
)
MEASURED_BASIS_BY_ARM = {
    "gepa": (
        f"measured at max_metric_calls=732 "
        f"({MEASURED_GEPA_TASK_CALLS} rows at {MEASUREMENT_NUM_SEEDS} "
        f"repeat(s)) on fake transport at these splits, scaled to the "
        f"pinned {GEPA_MAX_METRIC_CALLS_PINNED} and to K_REPEAT "
        f"(Wave 3, D3)"
    ),
}


def _optimizer_budget_lines(
    spec: StudySpecLike,
    *,
    internal_size: int,
    k_repeat: int,
    official_size: int = 0,
    held_out_size: int = 0,
) -> tuple[str, ...]:
    """Per-arm optimizer call estimates, plus the study-wide range."""
    lines = [
        OPTIMIZER_BUDGET_HEADING,
        (
            f"{'arm':<24}{'K_RUN':>8}{'per run':>20}"
            f"{'arm total':>22}{'  basis':<10}"
        ),
    ]
    total_low = 0
    total_high = 0
    optimizers = spec.optimizer_by_arm
    shapes = spec.copro_shape_by_arm
    for arm_id in spec.arm_ids:
        k_run = spec.k_run_by_arm[arm_id]
        shape = shapes[arm_id]
        try:
            estimate = estimate_optimizer_calls(
                optimizers[arm_id],
                internal_size=internal_size,
                k_repeat=k_repeat,
                official_size=official_size,
                held_out_size=held_out_size,
                # The arm's own pinned shape, not the estimator's
                # default. COPRO's whole per-run cost is
                # ``breadth x depth x T_int x K_REPEAT``, so an estimate
                # taken at a shape the arm does not run prices a search
                # the study never performs.
                **(
                    {}
                    if shape is None
                    else {
                        "copro_breadth": shape[0],
                        "copro_depth": shape[1],
                    }
                ),
            )
        except ValueError:
            lines.append(f"{arm_id:<24}{k_run:>8}{NO_ESTIMATE:>20}{'':>22}")
            continue
        low = estimate.low * k_run
        high = estimate.high * k_run
        total_low += low
        total_high += high
        suffix = "" if estimate.gated else "  (cap, not gated)"
        lines.append(
            f"{arm_id:<24}{k_run:>8}"
            f"{_range(estimate.low, estimate.high):>20}"
            f"{_range(low, high):>22}  {ESTIMATE_LABEL}{suffix}"
        )
        lines.append(f"{'':<24}basis: {estimate.basis}")
        measured = MEASURED_TASK_CALLS_BY_ARM.get(arm_id)
        if measured is not None:
            # Every Wave 3 measurement was taken at one repeat, and a row
            # count scales with the repeat count: whetstone-ai 0.1.11 bills
            # K_REPEAT rows per evaluation. Printing the raw figure beside
            # an estimate the study runs at K_REPEAT would understate the
            # measured arm by exactly that factor.
            measured = measured * k_repeat // MEASUREMENT_NUM_SEEDS
            lines.append(
                f"{'':<24}{'':>8}{measured:>20}{measured * k_run:>22}"
                f"  {MEASURED_LABEL}"
            )
            lines.append(
                f"{'':<24}basis: "
                + MEASURED_BASIS_BY_ARM.get(arm_id, MEASURED_BASIS_DEFAULT)
            )
    lines.extend(
        ("", f"total optimizer-side calls: {_range(total_low, total_high)}")
    )
    return tuple(lines)


def _range(low: int, high: int) -> str:
    return str(low) if low == high else f"{low}-{high}"


# --------------------------------------------------------------------------
# what each stage actually spent
# --------------------------------------------------------------------------

#: How the per-stage ledger is headed. It is a *measurement* -- every
#: number is read back out of persisted evidence -- which is what
#: distinguishes it from the estimated budget above it.
STAGE_SPEND_HEADING = "recorded spend, per stage (MEASURED from evidence):"

#: What a **fake-transport** stage with no spend records reports. It
#: reached no provider, so there is no bill to measure, and a zero would
#: claim the stage measured its bill and found it free -- a different and
#: untrue statement.
NO_RECORDED_SPEND = "no provider reached (fake transport)"

#: What a **paid** stage with no spend records reports.
#:
#: This is the one case the ledger must not soften. A stage bound to a
#: billed transport called the provider; if nothing came back to record,
#: the study is holding an unknown bill, not a zero one and certainly not
#: a stage that reached no provider. The label is loud on purpose: it is a
#: defect report, and an operator reading the ledger before authorizing
#: the next stage has to see that this row cannot be trusted.
UNLEDGERED_SPEND = (
    "UNLEDGERED -- ran on a paid transport and recorded no spend; "
    "this stage's bill is unknown, not zero"
)

#: What the ledger prints when no stage has run yet.
NO_STAGES_RUN = "no stage has run yet"


def _stage_usd(spend: tuple[RunSpendRecord, ...]) -> str:
    """One stage's total USD, or the honest reason there is none.

    A single unpriced role withholds the whole total, matching the rule
    ``RunSpendRecord`` enforces per role: a sum over the priced roles alone
    would look authoritative while understating what the stage cost.
    """
    if any(entry.usd is None for entry in spend):
        unpriced = sum(entry.unpriced_calls for entry in spend)
        calls = sum(entry.calls for entry in spend)
        return f"unpriced ({unpriced}/{calls} calls)"
    return f"${sum(entry.usd or 0.0 for entry in spend):.6f}"


def _no_spend_label(record: StageRecord) -> str:
    """Why a stage carrying no spend records carries none.

    Three cases share an empty ``spend`` tuple and must never share a
    label, because they call for three different actions:

    * a **fake-transport** stage reached no provider, so there is no bill;
    * a **paid** stage with no records ran against a provider and lost
      track of what it bought, which is a defect an operator must see
      before authorizing the next stage;
    * a stage with records prints them, and does not reach here.

    Collapsing the middle case into the first is the failure this function
    exists to prevent: it would report a fully billed stage as one that
    never called anybody.
    """
    if record.transport == TransportName.FAKE.value:
        return NO_RECORDED_SPEND
    return UNLEDGERED_SPEND


def stage_spend_lines(stages: tuple[StageRecord, ...]) -> tuple[str, ...]:
    """Each recorded stage's transport, rows, calls, and USD.

    Read straight off the manifest's stage records, so ``plan`` before a
    run and ``plan`` after one print the same shape and differ only in
    whether anything has been measured yet. The transport is in every row
    because the spend and the transport answer one question together: what
    did this stage buy, and from whom -- and because it is the transport
    that says whether an empty row means "reached no provider" or
    "reached one and lost the bill".

    A stage row is the whole of what the stage bought: its arms' optimizer
    runs, and the reporting pass -- official-selection scoring, the
    held-out evaluations, and the anchors' re-measurement -- folded onto
    the same row. The two are recorded by different routes because they
    spend by different routes, but neither is left out of the total.
    """
    lines = ["", STAGE_SPEND_HEADING]
    if not stages:
        lines.extend((f"  {NO_STAGES_RUN}", ""))
        return tuple(lines)
    lines.append(
        f"  {'stage':<10}{'transport':<14}{'calls':>10}{'tokens':>14}"
        f"  {'usd':>22}"
    )
    by_stage = {entry.stage: entry for entry in stages}
    for stage in STAGE_IDS:
        record = by_stage.get(stage)
        if record is None:
            continue
        # The stage's whole bill: its runs and its reporting pass. The
        # two are stored apart because they accumulate by opposite rules,
        # but the run-side figure alone understates a paid stage by the
        # entire pass its efficacy claims are made against.
        spend = record.total_spend
        if not spend:
            lines.append(
                f"  {record.stage:<10}{record.transport:<14}"
                f"  {_no_spend_label(record)}"
            )
            continue
        calls = sum(entry.calls for entry in spend)
        tokens = sum(
            entry.input_tokens + entry.output_tokens for entry in spend
        )
        lines.append(
            f"  {record.stage:<10}{record.transport:<14}{calls:>10,}"
            f"{tokens:>14,}  {_stage_usd(spend):>22}"
        )
    lines.append("")
    return tuple(lines)


# --------------------------------------------------------------------------
# subcommand bodies
# --------------------------------------------------------------------------


def _emit(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _require_honest_study_id(
    study_id: str | None,
    *,
    protocol: StudyProtocol,
    toy: bool,
    without_codex_arm: bool,
) -> None:
    """Refuse a study id that claims a design this invocation is not.

    ``--study-id`` exists so a rehearsal can name itself. Nothing stopped
    it naming itself *the study*: a toy or a ``--without-codex``
    projection could be initialised as ``step10-c19``, and every artifact,
    every report headline, and every directory downstream would then cite
    the pre-registration by name while holding a smaller design.

    A protocol id is therefore only allowed on the invocation that is
    actually that protocol -- the full design, at full size, with no
    projection in effect -- where it is also the default and passing it
    changes nothing.
    """
    if study_id is None or study_id not in PROTOCOL_IDS:
        return
    reduced = [
        reason
        for reason, active in (
            ("--toy", toy),
            ("--without-codex", without_codex_arm),
        )
        if active
    ]
    if not reduced and study_id == protocol.study_id:
        return
    detail = (
        f"this invocation is reduced by {' and '.join(reduced)}"
        if reduced
        else f"that id belongs to protocol {study_id!r}, not to "
        f"{protocol.protocol_id!r}"
    )
    raise SystemExit(
        f"refusing --study-id {study_id!r}: it names a registered "
        f"protocol, and {detail}. A study id that claims the "
        "pre-registration while holding a smaller design would make every "
        "artifact downstream cite a design this run is not. Choose an id "
        f"outside {PROTOCOL_IDS}."
    )


def _run_init(  # noqa: PLR0913
    *,
    study_dir: Path,
    protocol_id: str,
    toy: bool,
    protocol_doc: Path | None,
    study_id: str | None,
    without_codex_arm: bool = False,
) -> int:
    """Author a pre-Stage-0 manifest and say where it landed.

    The refusal to overwrite is :func:`write_study_manifest`'s, not a
    check restated here: ``init`` writes without ``replace``, so a second
    initialisation over a study that already holds evidence raises rather
    than resetting a design that evidence refers to.
    """
    protocol = study_protocol(protocol_id, toy=toy)
    if without_codex_arm:
        protocol = without_codex(protocol)
    _require_honest_study_id(
        study_id,
        protocol=protocol,
        toy=toy,
        without_codex_arm=without_codex_arm,
    )
    path = init_study(
        study_dir,
        protocol=protocol,
        study_id=study_id,
        protocol_doc=protocol_doc,
    )
    _emit(
        (
            f"initialised study {(study_id or protocol.study_id)!r} "
            f"from protocol {protocol.protocol_id!r}",
            f"arms: {', '.join(arm.arm_id for arm in protocol.arms)}",
            "splits: internal={} official={} held_out={}".format(
                *protocol.split_sizes
            ),
            f"{path}",
        )
    )
    return EXIT_OK


def _run_plan(
    *,
    study_dir: Path,
    load_spec: StudySpecLoader | None,
    load_stages: StageLedgerLoader,
) -> int:
    if load_spec is None:
        print(
            "plan needs a study spec loader; the study harness is not "
            "wired into this CLI yet",
            file=sys.stderr,
        )
        return EXIT_ERROR
    _emit(plan_lines(load_spec(study_dir)))
    # The estimated budget above, then what has actually been bought. The
    # two are printed together because that is the comparison an operator
    # authorizing the next stage is making.
    _emit(stage_spend_lines(load_stages(study_dir)))
    return EXIT_OK


def _run_stage(  # noqa: PLR0913
    *,
    study_dir: Path,
    stage: str,
    run_stage: StageRunner | None,
    replace_design: bool = False,
    allow_real_codex: bool = False,
    discard_stale_runs: bool = False,
    transport: str = FAKE_TRANSPORT,
) -> int:
    if run_stage is None:
        print(
            f"run --stage {stage} needs a stage runner; the study harness "
            "is not wired into this CLI yet",
            file=sys.stderr,
        )
        return EXIT_ERROR
    manifest = run_stage(
        study_dir=study_dir,
        stage=stage,
        replace_design=replace_design,
        allow_real_codex=allow_real_codex,
        discard_stale_runs=discard_stale_runs,
        transport=transport,
    )
    print(f"{stage} complete for study {manifest.study_id}")
    # The transport is echoed because it decides what the stage's numbers
    # are evidence of, and an operator scripting three stages should see
    # which one each ran on without opening the manifest.
    print(f"transport: {transport}")
    _emit(stage_spend_lines(manifest.stages))
    print(study_dir / STUDY_MANIFEST_NAME)
    return EXIT_OK


def _run_report(
    *,
    study_dir: Path,
    out_dir: Path,
    generate_report: ReportGenerator,
) -> int:
    """Read the manifest and hand it to the generator.

    The study directory is opened once, here, and only the manifest crosses
    into the generator: that is what keeps "the report reads the manifest
    and the evidence it names" a property of the wiring rather than a
    convention the generator is trusted to follow.
    """
    manifest = read_study_manifest(study_dir)
    print(generate_report(manifest=manifest, out_dir=out_dir))
    return EXIT_OK


def _report_generator_for(
    study_dir: Path, generate_report: ReportGenerator
) -> ReportGenerator:
    """Bind the study's own evidence store to ``generate_report``.

    The default generator reads the manifest only. The study directory
    keeps its run store beside that manifest, and resolving it turns the
    audit-finding tables from "did not resolve" into real rows, so the CLI
    supplies it when it is there.

    An injected generator is returned untouched. A test that passes its own
    collaborator is testing the wiring, and silently handing it a store it
    did not ask for would change what it is testing.
    """
    if generate_report is not default_report_generator:
        return generate_report
    store_path = _default_store_path(study_dir)
    if not store_path.is_file():
        return generate_report

    def generate(*, manifest: StudyManifest, out_dir: Path) -> Path:
        with open_sqlite(str(store_path)) as store:
            return generate_study_report(
                manifest=manifest, out_dir=out_dir, store=store
            )

    return generate


def _run_leakage_check(*, study_dir: Path) -> int:
    """Run L1-L6 over the study's manifest and report every rule's verdict.

    The check reads the manifest rather than re-walking run stores: the
    manifest is where the study records what it selected, what it measured
    on held-out, and which task hashes each split holds, and re-deriving
    those from artifacts would check a different set of facts than the ones
    the report will print.

    L1 is the exception, and it is read from the runs themselves. It is a
    rule over each optimizer run's own evaluations, which live in the run
    stores rather than in the manifest, so this command opens every run
    directory the manifest names and extracts the role and the evaluated
    task hashes from all three evaluation surfaces -- resolved intents,
    ``search_evidence``, and ``tool_evidence``. A study whose runs are
    gone -- or which has run none -- yields no observations and L1 is
    reported unchecked. **An unchecked rule fails the command**, exactly
    as a violated one does: from the reader's side, a study whose L1
    nobody checked and one whose L1 failed make the same claim.

    **The verdict is recorded, not only printed.** The report gates its
    headline and every arm verdict on ``manifest.leakage_check``, treating
    an absent block exactly as it treats a failed one, so a study whose
    rules passed on the terminal but were never written back would report
    as though nobody had checked it. Writing the block here is what closes
    that gap, and it is written whether the rules passed or failed --
    a recorded failure is the finding the report must print.
    """
    manifest = read_study_manifest(study_dir)
    report = _leakage_report(
        manifest,
        observations=optimizer_observations_for_study(
            _recorded_run_dirs(manifest)
        ),
    )
    _emit(_format_leakage(report))
    _record_leakage(study_dir, manifest, report)
    unchecked = report.unchecked()
    # L6 is the roll-up of the other rules, so it is excluded from the
    # tally rather than counted a second time alongside what it rolls up.
    failures = tuple(
        finding
        for finding in report.failures()
        if finding.rule is not LeakageRule.L6_CHECK_RAN
    )
    if failures or unchecked:
        print(
            f"{len(failures)} leakage rules failed and {len(unchecked)} "
            "could not be checked from the manifest; this study must not "
            "report",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED
    print(f"all {len(report.findings)} leakage rules passed")
    return EXIT_OK


def _record_leakage(
    study_dir: Path, manifest: StudyManifest, report: LeakageReport
) -> None:
    """Persist this run's verdict into the study's own manifest.

    L6 is the roll-up rather than a rule of its own, so it is not stored
    beside what it rolls up; the record's ``passed`` is the conjunction of
    the rules it holds, which the manifest validates. An **unchecked** rule
    is recorded as failed, because that is what it means to the reader: the
    study did not establish it.
    """
    entries = tuple(
        LeakageCheckEntry(
            check_id=finding.rule.value,
            passed=finding.passed and finding.checked,
            detail=finding.detail,
        )
        for finding in report.findings
        if finding.rule is not LeakageRule.L6_CHECK_RAN
    )
    if not entries:  # pragma: no cover - the rule set is never empty
        return
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "leakage_check": LeakageCheckRecord(
                    passed=all(entry.passed for entry in entries),
                    checks=entries,
                )
            }
        ),
        replace=True,
    )


def _recorded_run_dirs(manifest: StudyManifest) -> tuple[Path, ...]:
    """Every run directory this manifest names, arms and c18 alike.

    Read off the manifest rather than by walking a directory, so the runs
    L1 is checked over are exactly the runs the study recorded -- a stray
    directory beside them is not evidence this study produced.
    """
    recorded = [run for arm in manifest.arms for run in arm.runs]
    if manifest.c18 is not None:
        recorded.extend(manifest.c18.runs)
    return tuple(Path(run.artifact_dir) for run in recorded)


def _search_repeat_observations(
    manifest: StudyManifest,
) -> tuple[SearchRepeatObservation, ...]:
    """Every recorded run's search repeat count, as L7 reads it.

    Arms and c18 alike, and read off the manifest for
    :func:`_recorded_run_dirs`'s reason: the runs the rule is checked over
    are exactly the runs the study recorded.

    Null-identity is excluded by its arm's own recorded optimizer, not by
    its missing count: it runs no optimizer, so it has no search whose
    repeats could disagree with the design, and it records ``None`` by
    construction. Excluding on the optimizer rather than on ``None``
    matters -- a searching run that recorded no count is exactly what the
    rule must catch, and skipping every ``None`` would skip it too.
    """
    recorded = [
        run
        for arm in manifest.arms
        if arm.optimizer != NULL_IDENTITY_OPTIMIZER
        for run in arm.runs
    ]
    if manifest.c18 is not None:
        recorded.extend(manifest.c18.runs)
    return tuple(
        SearchRepeatObservation(
            run_id=run.run_id, search_num_seeds=run.search_num_seeds
        )
        for run in recorded
    )


def _leakage_report(
    manifest: StudyManifest,
    *,
    observations: tuple[OptimizerEvalObservation, ...] = (),
) -> LeakageReport:
    """Run the mechanical checks over what the manifest recorded.

    L1's evidence is per-run and lives in the run stores rather than in the
    manifest, so it is passed in by the caller that opened them. An empty
    set still reports L1 as *not checked* rather than as passed -- a study
    whose runs are gone cannot establish the rule, and saying it passed
    would be the vacuous claim this check exists to avoid.
    """
    splits = manifest.splits
    return study_leakage_check(
        optimizer_observations=observations,
        internal_eval_config_hash=splits.internal.eval_config_hash,
        internal_task_hashes=splits.internal.task_hashes,
        excluded_eval_config_hashes=(
            splits.official.eval_config_hash,
            splits.held_out.eval_config_hash,
        ),
        selected_arm_ids=_selected_arm_ids(manifest),
        expected_arm_ids=[arm.arm_id for arm in manifest.arms],
        held_out_candidate_names=_held_out_claim_names(manifest),
        held_out_observations=_held_out_observations(manifest),
        # L7 reads the manifest's own run records rather than reopening
        # the run directories: the manifest is what the study reports
        # from, so it is the manifest the design is checked against.
        search_repeats=_search_repeat_observations(manifest),
        k_repeat=None if manifest.design is None else manifest.design.k_repeat,
        splits=(
            SplitIdentity(
                role=SplitName.INTERNAL.value,
                task_hashes=splits.internal.task_hashes,
            ),
            SplitIdentity(
                role=SplitName.OFFICIAL.value,
                task_hashes=splits.official.task_hashes,
            ),
            SplitIdentity(
                role=SplitName.HELD_OUT.value,
                task_hashes=splits.held_out.task_hashes,
            ),
        ),
        strict=False,
    )


def _reported_stage(manifest: StudyManifest) -> str:
    """Which stage's records L2 and L3 are checked over.

    A study selects once per arm *per stage*, and the reported result is
    the latest stage that ran: a pilot's selection is not a second
    selection of the study's representative candidate, it is a different
    decision over a smaller run set. Checking both together would report
    every arm as selected twice, which describes the design rather than a
    leak.
    """
    stages = {entry.stage for entry in manifest.selection}
    for candidate in (StageId.STAGE2.value, StageId.STAGE1.value):
        if candidate in stages:
            return candidate
    return StageId.STAGE2.value


def _selected_arm_ids(manifest: StudyManifest) -> list[str]:
    """Every arm the reported stage selected a representative for."""
    stage = _reported_stage(manifest)
    return [
        entry.arm_id for entry in manifest.selection if entry.stage == stage
    ]


def _held_out_observations(
    manifest: StudyManifest,
) -> tuple[HeldOutObservation, ...]:
    """One observation per **completed** held-out evaluation.

    Read from ``held_out_claims`` rather than from ``held_out``, because a
    claim carries the Eval Config and repeat count its own evaluation
    actually used. Reading the rows instead would make L4 a tautology: the
    rows are written by the analysis pass from one study-wide config, so
    they agree by construction whether or not the evaluations did.

    Outstanding claims -- evaluations that were issued and never returned
    -- are excluded, because they carry no config or repeat count to
    compare. Substituting the study's own values for them would rebuild
    exactly the tautology this function exists to avoid; a crashed
    evaluation that used a different config would be unfalsifiable. They
    still count for L3, which
    :func:`_held_out_claim_names` supplies separately.
    """
    stage = _reported_stage(manifest)
    return tuple(
        HeldOutObservation(
            candidate_name=entry.candidate_name,
            eval_config_hash=entry.eval_config_hash or "",
            repeats=entry.repeats or 0,
        )
        for entry in manifest.held_out_claims
        if entry.completed and entry.stage == stage
    )


def _held_out_claim_names(manifest: StudyManifest) -> tuple[str, ...]:
    """Every candidate that consumed a held-out evaluation.

    Outstanding claims are included: what L3 limits is evaluations
    *issued*, so a crashed one has still spent the candidate's one shot.
    """
    stage = _reported_stage(manifest)
    return tuple(
        entry.candidate_name
        for entry in manifest.held_out_claims
        if entry.stage == stage
    )


#: How a rule is reported when the study carries no evidence for it.
NOT_CHECKED = "NOT CHECKED"


def _format_leakage(report: LeakageReport) -> Iterable[str]:
    """One line per rule, plus its offenders when it failed."""
    for finding in report.findings:
        if not finding.checked:
            yield f"{NOT_CHECKED} {finding.rule.value} :: {finding.detail}"
            continue
        mark = "ok" if finding.passed else "FAILED"
        yield f"{mark} {finding.rule.value} :: {finding.detail}"
        for offender in finding.offenders:
            yield f"    {offender}"


def _run_manifest_check(*, path: Path, store_path: Path | None) -> int:
    """Resolve every evidence pointer the manifest at ``path`` cites.

    The store defaults to the run store beside the manifest, which is where
    a study directory keeps it; ``--store`` names another when the evidence
    lives elsewhere.
    """
    manifest = read_study_manifest(path)
    resolved_store = store_path or _default_store_path(path)
    if not resolved_store.is_file():
        print(f"no evidence store at {resolved_store}", file=sys.stderr)
        return EXIT_CHECK_FAILED
    with open_sqlite(str(resolved_store)) as store:
        report = check_manifest_pointers(manifest, store)
    _emit(format_pointer_report(report))
    unresolved = report.unresolved()
    if unresolved:
        print(
            f"{len(unresolved)} of {len(report.checks)} evidence pointers "
            "did not resolve",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED
    print(f"{len(report.checks)} evidence pointers resolved")
    return EXIT_OK


#: The run store a study directory keeps beside its manifest.
DEFAULT_STORE_NAME = "runtime.sqlite"


def _default_store_path(manifest_path: Path) -> Path:
    resolved = manifest_path.resolve()
    directory = resolved.parent if resolved.is_file() else resolved
    return directory / DEFAULT_STORE_NAME


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Plan, run, report, and check a Step 10 validation study."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init",
        help="Write a pre-Stage-0 study.json from a committed protocol.",
    )
    init.add_argument("--study-dir", type=Path, required=True)
    init.add_argument(
        "--protocol",
        choices=(STEP10_C19_ID,),
        required=True,
        help=(
            "Which committed protocol to author. The design is a module, "
            "not a set of flags: every value it pins is in "
            "whetstone_envs.optim.study.protocols, so two initialisations "
            "of the same protocol produce the same manifest."
        ),
    )
    init.add_argument(
        "--toy",
        action="store_true",
        help=(
            "Author the sized-down variant of the same protocol. Only the "
            f"sized fields differ ({', '.join(SIZED_FIELDS)}); everything "
            "else -- the arms, the models, the control pins, the "
            "correction -- is shared with the real design, so a toy "
            "cannot rehearse a study the real one is not."
        ),
    )
    init.add_argument(
        "--protocol-doc",
        type=Path,
        default=None,
        help=(
            "The pre-registration document to digest. Defaults to the "
            "protocol's own durable path; the manifest records the digest "
            "of whatever file is read, so a study names the revision that "
            "was actually in force."
        ),
    )
    init.add_argument(
        "--study-id",
        default=None,
        help=(
            "Override the study id. Defaults to the protocol's own, which "
            "is what a fresh confirmatory run uses; a rehearsal names "
            "itself so its artifacts cannot be mistaken for the study's."
        ),
    )
    init.add_argument(
        "--without-codex",
        action="store_true",
        help=(
            "Drop the Codex arm from the authored design. The Codex arm's "
            "runs spawn a real, billed agent session and the stage "
            "harness refuses a Codex-bearing design before any arm runs, "
            "whatever transport the task model is on -- so a "
            "fake-transport rehearsal of the rest of the study drops the "
            "arm rather than stubbing it. The result is a smaller design, "
            "not the pre-registration."
        ),
    )

    plan = commands.add_parser(
        "plan",
        help="Print the run matrix and evaluation budget for a study spec.",
    )
    plan.add_argument("--study-dir", type=Path, required=True)

    run = commands.add_parser("run", help="Run one stage of the study.")
    run.add_argument("--study-dir", type=Path, required=True)
    run.add_argument(
        "--stage",
        choices=STAGE_IDS,
        required=True,
        help=(
            "stage0 calibrates anchors, stage1 pilots every arm, stage2 "
            "runs the full design."
        ),
    )
    run.add_argument(
        "--allow-real-codex",
        action="store_true",
        help=(
            "Authorize this invocation to spend on a real, billed Codex "
            "session for the study's Codex arm. Half the opt-in: "
            f"{ALLOW_REAL_CODEX_ENV}={ALLOW_REAL_CODEX_ENV_VALUE} must "
            "also be set in the environment. Without both, a stage whose "
            "design names the Codex arm is refused before any arm runs. "
            "This is a spend authorization, not part of the design: it "
            "does not enter the pre-registration hash."
        ),
    )
    run.add_argument(
        "--transport",
        choices=TRANSPORT_NAMES,
        default=FAKE_TRANSPORT,
        help=(
            "Which transport this stage's evaluations run on. "
            f"{FAKE_TRANSPORT!r} (the default) answers from the "
            "experiment's own gold and spends nothing; "
            f"{OPENROUTER_TRANSPORT!r} spends against the provider and "
            f"requires {OPENROUTER_API_KEY_ENV} in the environment. "
            "Recorded per stage: a study whose anchors were calibrated on "
            "one transport refuses to run its arms on the other, because "
            "every held-out delta is paired against those anchors."
        ),
    )
    run.add_argument(
        DISCARD_STALE_RUNS_FLAG,
        action="store_true",
        help=(
            "Discard a run directory whose own artifacts say it is not "
            "this invocation's run, instead of refusing. Run directories "
            "are named deterministically from the arm and seed, so a "
            "cross-transport 'stage0 --replace-design' -- which drops the "
            "stale runs from the manifest but leaves their directories on "
            "disk -- would otherwise leave this stage refusing to reuse "
            "them. Off by default because such a directory may be paid "
            "evidence; a matching directory is still reused either way."
        ),
    )
    run.add_argument(
        "--replace-design",
        action="store_true",
        help=(
            "Re-run stage0 over a study that already pre-registered its "
            "design, recording the replacement as an amendment. Without "
            "this, a second stage0 is refused."
        ),
    )

    report = commands.add_parser(
        "report", help="Generate the report packet from the manifest."
    )
    report.add_argument("--study-dir", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)

    leakage = commands.add_parser(
        "leakage-check",
        help="Run L1-L6 over a study directory before it reports.",
    )
    leakage.add_argument("--study-dir", type=Path, required=True)

    manifest = commands.add_parser(
        "manifest", help="Inspect a study manifest."
    )
    manifest_commands = manifest.add_subparsers(
        dest="manifest_command", required=True
    )
    check = manifest_commands.add_parser(
        "check",
        help="Resolve every evidence pointer the manifest cites.",
    )
    check.add_argument("path", type=Path)
    check.add_argument(
        "--store",
        type=Path,
        default=None,
        help=(
            "The evidence store to resolve against. Defaults to "
            f"{DEFAULT_STORE_NAME} beside the manifest."
        ),
    )
    return parser


def default_stage_runner(  # noqa: PLR0913
    *,
    study_dir: Path,
    stage: str,
    replace_design: bool = False,
    allow_real_codex: bool = False,
    discard_stale_runs: bool = False,
    transport: str = FAKE_TRANSPORT,
) -> StudyManifest:
    """Run one stage on the fake transport, over the study's own directory.

    This is the CLI's default :class:`StageRunner`. It binds the study's
    population and per-role engines through
    :func:`~whetstone_envs.optim.study.environment.bound_stage_environment`
    and hands them to the stage harness, so every resource the stage opens
    is released on every exit path.

    Fake transport is the default because spend authorization attaches at a
    Stage gate and not at a CLI invocation. A paid stage is a deliberate
    act, so it is not reachable by running this command with no flags: it
    takes ``--transport openrouter`` *and* a key in the environment, and
    the binder refuses before it opens anything if either is missing.

    ``allow_real_codex`` is forwarded into the bound environment, where it
    reaches both the study's optimizer runner and the harness's early
    refusal. It travels no further: the manifest this returns records the
    design, and an authorization to spend is not one.
    """
    with bound_stage_environment(
        study_dir,
        transport=transport,
        allow_real_codex=allow_real_codex,
        discard_stale_runs=discard_stale_runs,
    ) as environment:
        return _run_stage_harness(
            study_dir=study_dir,
            stage=stage,
            environment=environment,
            replace_design=replace_design,
        )


def default_report_generator(
    *, manifest: StudyManifest, out_dir: Path
) -> Path:
    """Write the report packet from the manifest alone.

    This is the CLI's default :class:`ReportGenerator` and it deliberately
    opens no store: the :class:`ReportGenerator` contract carries a manifest
    and an output directory, and a generator that went looking for a store
    the contract did not give it would be reading evidence its caller never
    named. ``report`` binds the study directory's own store separately, in
    :func:`_report_generator_for`, which is where the study directory is
    still in scope.
    """
    return generate_study_report(manifest=manifest, out_dir=out_dir)


def _dispatch(
    arguments: argparse.Namespace,
    *,
    load_spec: StudySpecLoader,
    load_stages: StageLedgerLoader,
    run_stage: StageRunner,
    generate_report: ReportGenerator,
) -> int:
    """Route one parsed invocation to its subcommand body.

    Split from :func:`main` so the dispatch and the error translation are
    each readable on their own: this function knows the subcommands, and
    ``main`` knows what each failure means to a caller's exit code.
    """
    if arguments.command == "init":
        return _run_init(
            study_dir=arguments.study_dir,
            protocol_id=arguments.protocol,
            toy=arguments.toy,
            protocol_doc=arguments.protocol_doc,
            study_id=arguments.study_id,
            without_codex_arm=arguments.without_codex,
        )
    if arguments.command == "plan":
        return _run_plan(
            study_dir=arguments.study_dir,
            load_spec=load_spec,
            load_stages=load_stages,
        )
    if arguments.command == "run":
        return _run_stage(
            study_dir=arguments.study_dir,
            stage=arguments.stage,
            run_stage=run_stage,
            replace_design=arguments.replace_design,
            allow_real_codex=arguments.allow_real_codex,
            discard_stale_runs=arguments.discard_stale_runs,
            transport=arguments.transport,
        )
    if arguments.command == "report":
        return _run_report(
            study_dir=arguments.study_dir,
            out_dir=arguments.out,
            generate_report=_report_generator_for(
                arguments.study_dir, generate_report
            ),
        )
    if arguments.command == "leakage-check":
        return _run_leakage_check(study_dir=arguments.study_dir)
    return _run_manifest_check(path=arguments.path, store_path=arguments.store)


def main(
    argv: Sequence[str] | None = None,
    *,
    load_spec: StudySpecLoader | None = None,
    load_stages: StageLedgerLoader | None = None,
    run_stage: StageRunner | None = None,
    generate_report: ReportGenerator | None = None,
) -> int:
    """Dispatch one study subcommand.

    All three collaborators default to the real implementations -- the
    manifest-backed spec loader, the stage harness, and the report package's
    generator -- so the CLI is the study's actual entry point rather than a
    shell around one. Tests pass their own collaborators, which is how the
    ordering is verified independently of the wiring.

    Three exit codes, and the distinction between the last two matters to a
    caller scripting the stages: ``0`` the command did what was asked,
    ``1`` it could not run (a missing directory, an unreadable manifest),
    and ``2`` the study said no -- a refused stage, a refused selection, or
    a check that failed. A refusal must never exit ``0``, because that
    reads as a stage that ran.
    """
    arguments = build_parser().parse_args(argv)
    try:
        return _dispatch(
            arguments,
            load_spec=load_spec or load_study_spec,
            load_stages=load_stages or default_stage_ledger,
            run_stage=run_stage or default_stage_runner,
            generate_report=generate_report or default_report_generator,
        )
    except (StageError, SelectionError) as error:
        # A refused stage and a refused selection are the study saying no:
        # a gate that did not pass, a stage run out of order, a second
        # selection for an arm. They exit ``2`` for the same reason a
        # failed pointer check does -- the command ran correctly and the
        # study declined -- and they must not exit ``0``, because a caller
        # scripting the stages would read that as a stage that spent.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_CHECK_FAILED
    except (OSError, ValueError, DocumentFileError) as error:
        # A missing study directory, an unreadable or non-canonical
        # manifest, and a manifest that fails validation are all
        # operator-facing failures; a traceback would bury the one line
        # that says which. ``DocumentFileError`` is named explicitly
        # because dr-store derives it from ``Exception``, not ``OSError``.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_STORE_NAME",
    "ESTIMATE_LABEL",
    "EXIT_CHECK_FAILED",
    "EXIT_ERROR",
    "EXIT_OK",
    "MDE_HEADING",
    "MDE_TAU_SQ_CASES",
    "MEASURED_LABEL",
    "MEASURED_TASK_CALLS_BY_ARM",
    "NOT_CHECKED",
    "NO_RECORDED_SPEND",
    "NO_STAGES_RUN",
    "OPTIMIZER_BUDGET_HEADING",
    "PROGRAM_NAME",
    "STAGE_SPEND_HEADING",
    "UNLEDGERED_SPEND",
    "ReportGenerator",
    "StageLedgerLoader",
    "StageRunner",
    "StudySpecLike",
    "StudySpecLoader",
    "build_parser",
    "default_report_generator",
    "default_stage_ledger",
    "default_stage_runner",
    "main",
    "plan_lines",
    "stage_spend_lines",
]
