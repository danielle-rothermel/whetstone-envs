"""C3: c18 reaches the optimizers through the identical runner.

The study's generality claim is not "c18 also works" -- it is that the code
path is the same code path, with only the family adapter swapped. A test
that merely ran c18 and asserted it completed would pass just as happily
against a forked runner carrying a c18 branch, which is exactly the domain
leak the study is looking for.

So the assertions here are structural:

* :func:`test_the_two_families_execute_the_same_call_sequence` traces every
  function entered under ``whetstone_envs/optim/`` during a c19 run and a
  c18 run of the same optimizer, and asserts the two traces differ only in
  the family-adapter file set. A c19 branch anywhere in the runner, a c18
  special case in ``provider.py``, or a second GEPA builder would each show
  up as a differing call outside that set.
* :func:`test_no_shared_module_names_or_imports_a_family` delegates to
  :func:`~whetstone_envs.optim.study.adapter_swap.differing_modules`, which
  reads the package source and reports any module outside that set naming
  or importing either family -- so a leak that happens not to execute on
  this fixture is still caught. That function is also what a c18 study
  records into its manifest's C3 block, so the check here and the recorded
  verdict cannot be computed under different rules.
* :func:`test_no_private_whetstone_import_was_needed_for_c18` re-runs the
  public-import guard over the c18 adapter, because "we added c18 by
  reaching into whetstone's internals" would be a finding, not a success.

Every optimizer the runner drives is exercised against c18 on the fake
transport, so the identical-path claim covers the whole optimizer surface
rather than one representative.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.optim.contracts import OptimResult

from whetstone_envs.optim.families import KNOWN_FAMILY_IDS, family_spec
from whetstone_envs.optim.run import OPTIMIZERS, RunSpec, run_optimizer
from whetstone_envs.optim.study.adapter_swap import (
    ALLOWED_FAMILY_DEFAULTS,
    FAMILY_ADAPTER_FILES,
    FAMILY_CONTRACT_FILES,
    OPTIM_ROOT,
    UNREADABLE_DETAIL,
    adapter_swap_record,
    differing_modules,
)
from whetstone_envs.reporting.publication import load_trajectory_report

#: Split sizes small enough to keep a fake-transport run a smoke run, and
#: pool sizes that yield at least four instances in each family.
SMOKE_SPLIT_SIZES = (2, 2, 0)
POOL_SIZE_BY_FAMILY = {"c19": 2, "c18": 1}

#: MIPROv2's demonstration regimes, all of which must drive c18.
DEMO_MODES = ("fewshot", "zeroshot", "ground_only")

#: Optimizers whose traced runs are fast enough to compare call-for-call.
#: MIPROv2 under ``sys.setprofile`` takes about a minute per run, so its
#: identical-path evidence is the untraced end-to-end runs below plus the
#: source-level assertions, which do not depend on execution at all.
TRACED_OPTIMIZERS = ("copro", "gepa")


#: ``miprov2`` and ``gepa`` require an explicit disjoint train/val split of
#: the internal split and the others refuse one, so the smoke spec supplies
#: it per optimizer. At the smoke internal 2 the only partition is 1/1.
TRAIN_VAL_OPTIMIZER_IDS = ("gepa", "miprov2")


def _smoke_spec(*, family: str, optimizer: str, output: Path, **overrides):
    split = (
        {"train_size": 1, "val_size": 1}
        if optimizer in TRAIN_VAL_OPTIMIZER_IDS
        else {}
    )
    return RunSpec(
        optimizer=optimizer,
        transport="fake",
        family=family,
        split_sizes=SMOKE_SPLIT_SIZES,
        n_per_stratum=POOL_SIZE_BY_FAMILY[family],
        run_id=f"{family}-{optimizer}-swap",
        output_dir=output,
        **split,
        **overrides,
    )


def _traced_run(spec: RunSpec) -> tuple[tuple[str, str], ...]:
    """Every function entered under ``optim/`` during one run, in order.

    Consecutive repeats collapse to one entry: a comprehension frame per
    task row is data volume, not structure, and the two families' pools are
    deliberately different sizes.
    """
    calls: list[tuple[str, str]] = []

    def record(frame, event, _arg):
        if event != "call":
            return
        code = frame.f_code
        if code.co_filename.startswith(str(OPTIM_ROOT)):
            calls.append((Path(code.co_filename).name, code.co_qualname))
        return

    sys.setprofile(record)
    try:
        run_optimizer(spec)
    finally:
        sys.setprofile(None)

    collapsed: list[tuple[str, str]] = []
    for call in calls:
        if not collapsed or collapsed[-1] != call:
            collapsed.append(call)
    return tuple(collapsed)


@pytest.mark.parametrize("optimizer", TRACED_OPTIMIZERS)
def test_the_two_families_execute_the_same_call_sequence(
    tmp_path, optimizer: str
) -> None:
    """Only the family adapter differs between a c19 and a c18 run.

    This is the adapter-swap assertion. It is deliberately not a claim that
    the two traces are equal -- they cannot be, because each family's own
    builder must appear in its own trace -- but that every call present in
    one and absent from the other lives in the permitted file set.
    """
    c19_trace = _traced_run(
        _smoke_spec(family="c19", optimizer=optimizer, output=tmp_path / "c19")
    )
    c18_trace = _traced_run(
        _smoke_spec(family="c18", optimizer=optimizer, output=tmp_path / "c18")
    )
    assert c19_trace
    assert c18_trace

    only_c19 = Counter(c19_trace) - Counter(c18_trace)
    only_c18 = Counter(c18_trace) - Counter(c19_trace)
    differing = {
        (filename, qualname) for filename, qualname in (*only_c19, *only_c18)
    }
    offenders = sorted(
        f"{filename}:{qualname}"
        for filename, qualname in differing
        if filename not in FAMILY_ADAPTER_FILES | FAMILY_CONTRACT_FILES
    )
    assert offenders == [], (
        "a c19 run and a c18 run entered different code outside the family "
        f"adapter: {offenders}"
    )

    # The differing calls really are each family's own builder, so this is
    # an adapter swap rather than two traces that happen not to disagree.
    differing_qualnames = {qualname for _file, qualname in differing}
    assert "prepare_c19_experiment" in differing_qualnames
    assert "prepare_c18_experiment" in differing_qualnames


@pytest.mark.parametrize("optimizer", TRACED_OPTIMIZERS)
def test_the_shared_runner_is_the_single_entry_point(
    tmp_path, optimizer: str
) -> None:
    """Both families enter through ``run_optimizer`` and nothing else."""
    entry = ("run.py", "run_optimizer")
    for family in KNOWN_FAMILY_IDS:
        trace = _traced_run(
            _smoke_spec(
                family=family,
                optimizer=optimizer,
                output=tmp_path / f"{family}-{optimizer}",
            )
        )
        assert trace[0] == entry
        assert trace.count(entry) == 1


def test_no_shared_module_names_or_imports_a_family() -> None:
    """The source-level half of the assertion, over the shipped package.

    A branch that never executes on this fixture would slip past the traced
    comparison, so the source is checked too. Two signatures count as
    family-specific knowledge: importing ``whetstone_envs.c18`` or
    ``whetstone_envs.c19``, and spelling either name as a bare string --
    an ``if family == "c18"`` branch, a c19 template inlined into the
    runner, or a c18 special case in the fake transport would each show up
    as one.

    Delegated to :func:`differing_modules` rather than recomputed here.
    That function is what a c18 study *records* into its manifest, so a
    check that kept its own copy of the rule could pass while the recorded
    verdict was computed under a different one -- and the manifest's claim,
    not this test's, is the study's evidence.
    """
    assert differing_modules() == ()


def test_the_recorded_verdict_is_the_one_this_test_checks() -> None:
    """The manifest's C3 block says what the guard above found.

    The record is the artifact a reader trusts, so its ``passed`` must be
    the guard's own conjunction and its module list the guard's own
    output. A record that could report a pass while the guard found a leak
    would make the study's generality claim unfalsifiable from the
    artifact.
    """
    record = adapter_swap_record()
    assert record.differing_modules == differing_modules()
    assert record.passed is (record.differing_modules == ())
    assert record.passed


def test_the_guard_reports_a_planted_leak(tmp_path) -> None:
    """Fails-loudly evidence: the guard is not vacuously green.

    A guard that returned ``()`` unconditionally would pass every test
    above on every tree. Planting each signature in a throwaway package
    shows the guard reads them, and that an exempt filename really is
    exempt rather than merely absent.
    """
    (tmp_path / "leaky_branch.py").write_text(
        'FAMILY = "c18"\n', encoding="utf-8"
    )
    (tmp_path / "leaky_import.py").write_text(
        "from whetstone_envs.c19 import PROBES\n", encoding="utf-8"
    )
    # An adapter file may do both; that is what makes it the adapter.
    (tmp_path / "c18_experiment.py").write_text(
        'from whetstone_envs.c18 import PROBES\nF = "c18"\n', encoding="utf-8"
    )
    # A docstring naming a family is prose, not a branch.
    (tmp_path / "prose.py").write_text('"""About c18."""\n', encoding="utf-8")
    assert differing_modules(root=tmp_path) == (
        "leaky_branch.py",
        "leaky_import.py",
    )


def test_a_docstring_cannot_exempt_a_files_own_literals(tmp_path) -> None:
    """A one-word docstring must not switch the guard off for its file.

    Fails before this change. The exclusion compared each string's
    *value* against the set of docstring values, so a module whose
    docstring was the single word ``c18`` exempted every bare ``"c18"``
    literal in it -- including a live dispatch. The two files here carry
    the identical branch and differ only in that docstring, so a guard
    that reads values gives them opposite verdicts.
    """
    (tmp_path / "honest.py").write_text('F = "c18"\n', encoding="utf-8")
    (tmp_path / "poisoned.py").write_text(
        '"""c18"""\nF = "c18"\n', encoding="utf-8"
    )
    assert differing_modules(root=tmp_path) == ("honest.py", "poisoned.py")


def test_an_unreadable_module_fails_the_verdict_rather_than_raising(
    tmp_path,
) -> None:
    """A file the guard cannot parse is named, not raised past.

    This runs at the end of an arm stage, after the last paid operation.
    Raising would abandon the c18 block over a syntax error in an
    unrelated module, losing the recorded C3 evidence for runs the study
    had already paid for.
    """
    (tmp_path / "broken.py").write_text("def (\n", encoding="utf-8")
    modules = differing_modules(root=tmp_path)
    assert len(modules) == 1
    assert modules[0].startswith("broken.py")
    assert UNREADABLE_DETAIL in modules[0]
    # And the record built from it is a *failed* verdict, not an error.
    record = adapter_swap_record(root=tmp_path)
    assert record.passed is False
    assert record.differing_modules == modules


def test_every_shared_path_exemption_is_still_in_use() -> None:
    """An exemption for a file that no longer needs one is dead licence.

    Each entry in :data:`ALLOWED_FAMILY_DEFAULTS` widens what the shared
    path may spell, so an entry that stopped being load-bearing should be
    deleted rather than left standing to cover a future leak silently.
    """
    for name, allowed in ALLOWED_FAMILY_DEFAULTS.items():
        matches = [
            path for path in OPTIM_ROOT.rglob("*.py") if path.name == name
        ]
        assert matches, f"{name} is exempted but not on the shared path"
        assert allowed, f"{name} is exempted for no identifier"


def test_the_runner_default_family_is_a_registered_family() -> None:
    """The default is a value the registry resolves, not a literal branch."""
    assert RunSpec(optimizer="copro", transport="fake").family == "c19"
    assert family_spec(RunSpec(optimizer="copro", transport="fake").family)


def test_no_private_whetstone_import_was_needed_for_c18() -> None:
    """Admitting c18 required no reach into whetstone-ai's internals.

    Delegating to the package-wide guard keeps one owner for what counts as
    a private import; what this test adds is the C3 reading of a failure --
    a private import introduced by the second family is a domain leak in
    whetstone-ai's public surface, not a local style problem.
    """
    from tests.optim.test_public_imports import (
        test_optim_package_imports_no_private_whetstone_members,
    )

    test_optim_package_imports_no_private_whetstone_members()


#: Optimizers this test can drive with nothing but the fake transport.
#:
#: Codex is excluded because it is not one of them: it spawns a real
#: foreign agent, so driving it needs the scripted CLI and the test seam
#: that reaches it. Its c18 evidence is
#: ``test_codex_runs_the_second_family_unchanged`` in ``test_e2e.py``,
#: which drives the identical ``run_optimizer`` path over c18 and audits
#: the result -- the same C3 claim, made where the fake CLI is available.
#:
#: The exclusion is now a statement about *coverage*, not a safety
#: measure. It used to be both: parametrizing over ``OPTIMIZERS`` here
#: spawned the real Codex binary. ``run_optimizer`` refuses an unseamed
#: Codex run outright now, which
#: ``test_the_excluded_arm_would_refuse_rather_than_spend`` pins -- so
#: re-adding Codex to a parametrization can only fail a test, never buy a
#: session.
SELF_DRIVING_OPTIMIZERS = tuple(
    optimizer for optimizer in OPTIMIZERS if optimizer != "codex"
)


@pytest.mark.parametrize(
    "optimizer", [o for o in OPTIMIZERS if o not in SELF_DRIVING_OPTIMIZERS]
)
def test_the_excluded_arm_would_refuse_rather_than_spend(
    tmp_path, optimizer: str
) -> None:
    """Excluding Codex is a choice here, not the thing keeping us safe.

    Parametrized over the complement of ``SELF_DRIVING_OPTIMIZERS`` so it
    covers whatever this module excludes rather than re-naming "codex":
    if a future arm is excluded for the same reason, this asserts that
    arm is also refused rather than silently billed.
    """
    from whetstone_envs.optim.codex import RealCodexRefusedError

    with pytest.raises(RealCodexRefusedError):
        run_optimizer(
            _smoke_spec(
                family="c18",
                optimizer=optimizer,
                output=tmp_path / optimizer,
            )
        )
    assert not (tmp_path / optimizer).exists()


@pytest.mark.parametrize("optimizer", SELF_DRIVING_OPTIMIZERS)
def test_every_optimizer_drives_c18_to_a_complete_run(
    tmp_path, optimizer: str
) -> None:
    """C3's runtime evidence: c18 completes on every optimizer.

    MIPROv2's demonstration regimes are covered separately, because the
    demo mode changes which search MIPROv2 runs and one mode completing
    says nothing about the other two.
    """
    output = tmp_path / optimizer
    assert (
        run_optimizer(
            _smoke_spec(family="c18", optimizer=optimizer, output=output)
        )
        == output.resolve()
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None
    assert result.step_results
    assert result.step_results[-1].record.status.value == "complete"

    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    assert trajectory.mutation_field == family_spec("c18").mutation_field
    # Every embedded eval report labels itself c18, not c19: the report
    # schema now derives the family from the evidence rather than assuming
    # one, which is what let the second family through it unchanged.
    embedded = [
        row.eval_report
        for row in trajectory.resolutions
        if row.eval_report is not None
    ]
    assert embedded
    for report in embedded:
        assert report.run.family == "c18"
        assert report.run.dataset_revision == "c18/v1"


@pytest.mark.parametrize("demo_mode", DEMO_MODES)
def test_every_miprov2_demo_mode_drives_c18(tmp_path, demo_mode: str) -> None:
    """Each MIPROv2 regime completes on c18 through the shared runner.

    C18's internal split is smaller than MIPROv2's default minibatch size,
    so a run that turned minibatching on would be refused by
    ``configure_miprov2``. The runner keeps ``minibatch=False``, which is
    what lets the same control builder serve both families.
    """
    output = tmp_path / demo_mode
    run_optimizer(
        _smoke_spec(
            family="c18",
            optimizer="miprov2",
            output=output,
            demo_mode=demo_mode,
        )
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"


def test_c18_candidates_carry_c18_templates_not_c19s(tmp_path) -> None:
    """A c18 run optimizes a c18 template, so the swap really swapped.

    Without this, a runner that silently kept the c19 probe pair would
    still complete and still trace identically.
    """
    output = tmp_path / "c18-payload"
    run_optimizer(_smoke_spec(family="c18", optimizer="copro", output=output))
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    c18 = family_spec("c18")
    c19 = family_spec("c19")
    templates = {
        proposal.candidate.record.payload[c18.mutation_field]
        for proposal in result.proposals
    }
    assert templates
    for template in templates:
        assert isinstance(template, str)
        # Every proposal satisfies c18's contract and none satisfies c19's.
        assert set(c18.render_contract().validate_template(template)) == set(
            c18.prompt_fields
        )
        with pytest.raises(ValueError):
            c19.render_contract().validate_template(template)
