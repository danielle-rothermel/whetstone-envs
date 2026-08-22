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
* :func:`test_only_the_family_adapter_names_a_family` reads the package
  source and asserts no module outside that set names either family, so a
  leak that happens not to execute on this fixture is still caught.
* :func:`test_no_private_whetstone_import_was_needed_for_c18` re-runs the
  public-import guard over the c18 adapter, because "we added c18 by
  reaching into whetstone's internals" would be a finding, not a success.

Every optimizer the runner drives is exercised against c18 on the fake
transport, so the identical-path claim covers the whole optimizer surface
rather than one representative.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.optim.contracts import OptimResult

from whetstone_envs.optim import run as run_module
from whetstone_envs.optim.families import KNOWN_FAMILY_IDS, family_spec
from whetstone_envs.optim.run import OPTIMIZERS, RunSpec, run_optimizer
from whetstone_envs.reporting.publication import load_trajectory_report

OPTIM_ROOT = Path(run_module.__file__).parent

#: The only two files a second family is permitted to add or change. The
#: registry entry lives in ``families.py`` by construction -- that module is
#: the registry -- and everything else about c18 lives in its own adapter.
#: A differing call or a family name outside this set is the C3 finding.
FAMILY_ADAPTER_FILES = frozenset({"c18_experiment.py", "families.py"})

#: Files that legitimately name a family for a reason other than driving it.
#: ``experiment.py`` owns C19's own contract, mirroring ``c18_experiment.py``.
FAMILY_CONTRACT_FILES = frozenset({"experiment.py"})

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


def _smoke_spec(*, family: str, optimizer: str, output: Path, **overrides):
    return RunSpec(
        optimizer=optimizer,
        transport="fake",
        family=family,
        split_sizes=SMOKE_SPLIT_SIZES,
        n_per_stratum=POOL_SIZE_BY_FAMILY[family],
        run_id=f"{family}-{optimizer}-swap",
        output_dir=output,
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


def _family_package_imports(path: Path) -> set[str]:
    """Every ``whetstone_envs.<family>`` module one file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return {
        module
        for module in modules
        if any(
            module == f"whetstone_envs.{family}"
            or module.startswith(f"whetstone_envs.{family}.")
            for family in KNOWN_FAMILY_IDS
        )
    }


def test_only_the_family_adapter_imports_a_family_package() -> None:
    """No module on the shared path reaches into a family's own package.

    A branch that never executes on this fixture would slip past the traced
    comparison, so the source is checked too. Importing ``whetstone_envs
    .c18`` or ``whetstone_envs.c19`` is the mechanical signature of
    family-specific knowledge, and it is confined to the three adapter
    files: each family's own contract module plus the registry that binds
    them.
    """
    exempt = FAMILY_ADAPTER_FILES | FAMILY_CONTRACT_FILES
    offenders = {
        str(path.relative_to(OPTIM_ROOT)): sorted(imported)
        for path in sorted(OPTIM_ROOT.rglob("*.py"))
        if path.name not in exempt
        and (imported := _family_package_imports(path))
    }
    assert offenders == {}, (
        "the shared optimizer path imports a task family's package; every "
        f"family import belongs in its own adapter: {offenders}"
    )


def _family_string_literals(path: Path) -> set[str]:
    """Family identifiers appearing as bare string values in one module.

    Docstrings and comments are excluded: prose naming a family explains
    the shared path, it does not branch on one.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
        and node.value in set(KNOWN_FAMILY_IDS)
    }


#: Where the shared path may name a family, and why each is not a branch.
#:
#: ``cli.py``: the CLI's default choice of which family an unparameterised
#: run drives. A default is not a branch -- ``run_optimizer`` resolves it
#: through ``family_spec`` like any other value -- but it is still a family
#: literal, so it is enumerated here rather than tolerated by a loose rule.
#: ``RunSpec.family`` defaults to ``FamilyId.C19.value``, a reference to the
#: registry's own enumeration rather than an inlined string, so it does not
#: appear here.
#:
#: ``manifest.py``: the study manifest's ``c18`` block, which records the
#: C3 generalization evidence. This is a *persisted wire key*, not a
#: dispatch on family: nothing in the manifest branches on it, and it names
#: the block rather than selecting an adapter. It is spelled as a literal
#: on purpose -- ``tests/optim/study/test_manifest.py`` golden-pins the
#: block names, and deriving a persisted key from an enum elsewhere is how
#: a stored format drifts silently when that enum is renamed.
ALLOWED_FAMILY_DEFAULTS = {"cli.py": {"c19"}, "manifest.py": {"c18"}}


def test_the_shared_path_names_a_family_only_as_a_default() -> None:
    """Beyond that one default, no shared module carries a family literal.

    A ``if family == "c18"`` branch, a c19 template inlined into the
    runner, or a c18 special case in the fake transport would each add a
    literal here and fail.
    """
    exempt = FAMILY_ADAPTER_FILES | FAMILY_CONTRACT_FILES
    offenders = {
        name: sorted(found - ALLOWED_FAMILY_DEFAULTS.get(name, set()))
        for path in sorted(OPTIM_ROOT.rglob("*.py"))
        if path.name not in exempt
        and (found := _family_string_literals(path))
        and found - ALLOWED_FAMILY_DEFAULTS.get(path.name, set())
        for name in (path.name,)
    }
    assert offenders == {}, (
        "the shared optimizer path names a task family outside its "
        f"documented defaults: {offenders}"
    )


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
