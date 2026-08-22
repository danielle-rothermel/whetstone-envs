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

from whetstone_envs.optim.study.environment import bound_stage_environment
from whetstone_envs.optim.study.gates import estimate_optimizer_calls
from whetstone_envs.optim.study.leakage import (
    HeldOutObservation,
    LeakageRule,
    SplitIdentity,
    study_leakage_check,
)
from whetstone_envs.optim.study.manifest import (
    STAGE_IDS,
    STUDY_MANIFEST_NAME,
    SplitName,
    check_manifest_pointers,
    format_pointer_report,
    read_study_manifest,
)
from whetstone_envs.optim.study.spec import load_study_spec
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

    Arm ids double as optimizer names for the estimate: the study's arms are
    one per optimizer, so the id *is* the optimizer, and an arm whose id the
    estimator does not recognize reports "no estimate" rather than a number
    derived from a guess.
    """

    @property
    def study_id(self) -> str: ...

    @property
    def arm_ids(self) -> tuple[str, ...]: ...

    @property
    def k_run_by_arm(self) -> Mapping[str, int]: ...

    @property
    def k_repeat(self) -> int: ...

    @property
    def split_sizes(self) -> tuple[int, int, int]: ...


class StudySpecLoader(Protocol):
    """Load the study's spec from its directory."""

    def __call__(self, study_dir: Path) -> StudySpecLike: ...


class StageRunner(Protocol):
    """Run one stage of the study and return its updated manifest.

    Wave 4a owns the implementation. It returns the manifest rather than a
    path so the CLI reports what the stage recorded without re-reading a
    file the harness may still be writing.

    ``replace_design`` is part of the contract rather than a Stage-0 detail
    because the CLI cannot know which stage a runner will dispatch to; the
    harness refuses it on the stages that record no design.
    """

    def __call__(
        self,
        *,
        study_dir: Path,
        stage: str,
        replace_design: bool = False,
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
      :mod:`whetstone_envs.optim.study.gates`. It dominates the total --
      GEPA's 732 metric calls and MIPROv2's bootstrap walk dwarf the
      official and held-out passes -- and it is the number Wave 3 replaces
      with a measurement. The output labels it as an estimate for that
      reason.
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
    lines.extend(
        _optimizer_budget_lines(
            spec, internal_size=internal, k_repeat=spec.k_repeat
        )
    )
    return tuple(lines)


#: How the optimizer-side budget is labelled. It is not a measurement, and
#: the label is the thing that keeps it from being read as one.
OPTIMIZER_BUDGET_HEADING = (
    "optimizer-side calls (ESTIMATE from control defaults; "
    "Wave 3 measures these):"
)

#: What an unrecognised arm reports instead of a fabricated number.
NO_ESTIMATE = "no estimate"


def _optimizer_budget_lines(
    spec: StudySpecLike, *, internal_size: int, k_repeat: int
) -> tuple[str, ...]:
    """Per-arm optimizer call estimates, plus the study-wide range."""
    lines = [
        OPTIMIZER_BUDGET_HEADING,
        f"{'arm':<24}{'K_RUN':>8}{'per run':>20}{'arm total':>22}",
    ]
    total_low = 0
    total_high = 0
    for arm_id in spec.arm_ids:
        k_run = spec.k_run_by_arm[arm_id]
        try:
            estimate = estimate_optimizer_calls(
                arm_id, internal_size=internal_size, k_repeat=k_repeat
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
            f"{_range(low, high):>22}{suffix}"
        )
        lines.append(f"{'':<24}basis: {estimate.basis}")
    lines.extend(
        ("", f"total optimizer-side calls: {_range(total_low, total_high)}")
    )
    return tuple(lines)


def _range(low: int, high: int) -> str:
    return str(low) if low == high else f"{low}-{high}"


# --------------------------------------------------------------------------
# subcommand bodies
# --------------------------------------------------------------------------


def _emit(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _run_plan(*, study_dir: Path, load_spec: StudySpecLoader | None) -> int:
    if load_spec is None:
        print(
            "plan needs a study spec loader; the study harness is not "
            "wired into this CLI yet",
            file=sys.stderr,
        )
        return EXIT_ERROR
    _emit(plan_lines(load_spec(study_dir)))
    return EXIT_OK


def _run_stage(
    *,
    study_dir: Path,
    stage: str,
    run_stage: StageRunner | None,
    replace_design: bool = False,
) -> int:
    if run_stage is None:
        print(
            f"run --stage {stage} needs a stage runner; the study harness "
            "is not wired into this CLI yet",
            file=sys.stderr,
        )
        return EXIT_ERROR
    manifest = run_stage(
        study_dir=study_dir, stage=stage, replace_design=replace_design
    )
    print(f"{stage} complete for study {manifest.study_id}")
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

    L1 is the exception. It is a rule over each optimizer run's own intent
    resolutions, which live in the run stores rather than in the manifest,
    so this command cannot check it and reports it as unchecked. **An
    unchecked rule fails the command**, exactly as a violated one does:
    from the reader's side, a study whose L1 nobody checked and one whose
    L1 failed make the same claim. The audit package reads the run evidence
    directly and is what turns this into a pass.
    """
    report = _leakage_report(read_study_manifest(study_dir))
    _emit(_format_leakage(report))
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


def _leakage_report(manifest: StudyManifest) -> LeakageReport:
    """Run the mechanical checks over what the manifest recorded."""
    splits = manifest.splits
    return study_leakage_check(
        # L1's evidence is per-run and not in the manifest, so it is run
        # over an empty observation set: vacuously true, and reported as
        # not checked rather than as passed.
        optimizer_observations=(),
        internal_eval_config_hash=splits.internal.eval_config_hash,
        selected_arm_ids=[entry.arm_id for entry in manifest.selection],
        expected_arm_ids=[arm.arm_id for arm in manifest.arms],
        held_out_candidate_names=_held_out_claim_names(manifest),
        held_out_observations=_held_out_observations(manifest),
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
    return tuple(
        HeldOutObservation(
            candidate_name=entry.candidate_name,
            eval_config_hash=entry.eval_config_hash or "",
            repeats=entry.repeats or 0,
        )
        for entry in manifest.held_out_claims
        if entry.completed
    )


def _held_out_claim_names(manifest: StudyManifest) -> tuple[str, ...]:
    """Every candidate that consumed a held-out evaluation.

    Outstanding claims are included: what L3 limits is evaluations
    *issued*, so a crashed one has still spent the candidate's one shot.
    """
    return tuple(entry.candidate_name for entry in manifest.held_out_claims)


#: How L1 is reported when the manifest carries no per-run evidence for it.
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


def default_stage_runner(
    *, study_dir: Path, stage: str, replace_design: bool = False
) -> StudyManifest:
    """Run one stage on the fake transport, over the study's own directory.

    This is the CLI's default :class:`StageRunner`. It binds the study's
    population and per-role engines through
    :func:`~whetstone_envs.optim.study.environment.bound_stage_environment`
    and hands them to the stage harness, so every resource the stage opens
    is released on every exit path.

    Fake transport is the default because spend authorization attaches at a
    Stage gate and not at a CLI invocation. A paid stage is a deliberate
    act, so it is not reachable by running this command with no flags.
    """
    with bound_stage_environment(study_dir) as environment:
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


def main(
    argv: Sequence[str] | None = None,
    *,
    load_spec: StudySpecLoader | None = None,
    run_stage: StageRunner | None = None,
    generate_report: ReportGenerator | None = None,
) -> int:
    """Dispatch one study subcommand.

    All three collaborators default to the real implementations -- the
    manifest-backed spec loader, the stage harness, and the report package's
    generator -- so the CLI is the study's actual entry point rather than a
    shell around one. Tests pass their own collaborators, which is how the
    ordering is verified independently of the wiring.
    """
    arguments = build_parser().parse_args(argv)
    load_spec = load_spec or load_study_spec
    run_stage = run_stage or default_stage_runner
    generate_report = generate_report or default_report_generator
    try:
        if arguments.command == "plan":
            return _run_plan(
                study_dir=arguments.study_dir, load_spec=load_spec
            )
        if arguments.command == "run":
            return _run_stage(
                study_dir=arguments.study_dir,
                stage=arguments.stage,
                run_stage=run_stage,
                replace_design=arguments.replace_design,
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
        return _run_manifest_check(
            path=arguments.path, store_path=arguments.store
        )
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
    "EXIT_CHECK_FAILED",
    "EXIT_ERROR",
    "EXIT_OK",
    "NOT_CHECKED",
    "OPTIMIZER_BUDGET_HEADING",
    "PROGRAM_NAME",
    "ReportGenerator",
    "StageRunner",
    "StudySpecLike",
    "StudySpecLoader",
    "build_parser",
    "default_report_generator",
    "default_stage_runner",
    "main",
    "plan_lines",
]
