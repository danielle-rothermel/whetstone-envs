"""The C3 adapter-swap assertion, as a function the study can record.

Section 4.1 of the protocol document asks for a *mechanical* generality
claim: the second family reaches the optimizers through the identical
``run_optimizer``, differing only in the family adapter. "c18 also ran" is
not that claim -- it would pass just as happily against a forked runner
carrying a c18 branch, which is precisely the domain leak the study exists
to find.

This module computes the assertion over the shared optimizer package's
**source**, and returns the answer in the shape the manifest persists. It
lives in ``src`` rather than in the test that used to own it because the
study has to *record* the verdict: a manifest whose ``c18`` block asserted
generality on the strength of a test that ran in someone's CI, at some
unrecorded commit, would be citing evidence the artifact does not carry.

What is checked here, and why it is the source rather than a trace:

* **Family-package imports.** Importing :mod:`whetstone_envs.c18` or
  :mod:`whetstone_envs.c19` is the mechanical signature of family-specific
  knowledge, and it belongs to the adapter files alone.
* **Family literals.** A ``family == "c18"`` branch, a c19 template inlined
  into the runner, or a c18 special case in the fake transport each spell a
  family's name as a bare string on the shared path.

Both are properties of the code as shipped, so they hold for branches that
did not happen to execute -- which a call trace cannot say. The trace-level
comparison stays in ``tests/optim/test_c18_adapter_swap.py``, where it can
afford to run both families under ``sys.setprofile``: it is a stronger
check on a weaker sample, and a paid stage can neither afford to re-run c19
for the diff nor profile a provider-backed run without measuring something
other than the assertion. The two are complementary, and the test asserts
this module agrees with it on the shipped tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

from whetstone_envs.optim import run as _run_module
from whetstone_envs.optim.families import KNOWN_FAMILY_IDS
from whetstone_envs.optim.study.manifest import AdapterSwapRecord

__all__ = [
    "ALLOWED_FAMILY_DEFAULTS",
    "FAMILY_ADAPTER_FILES",
    "FAMILY_CONTRACT_FILES",
    "OPTIM_ROOT",
    "UNREADABLE_DETAIL",
    "adapter_swap_record",
    "differing_modules",
]

#: The shared optimizer package: everything a family runs through.
OPTIM_ROOT = Path(_run_module.__file__).parent

#: The only two files a second family is permitted to add or change.
#:
#: The registry entry lives in ``families.py`` by construction -- that
#: module *is* the registry -- and everything else about c18 lives in its
#: own adapter. A family import or a family literal outside this set is the
#: C3 finding.
FAMILY_ADAPTER_FILES = frozenset({"c18_experiment.py", "families.py"})

#: Files that legitimately name a family for a reason other than driving
#: one. ``experiment.py`` owns C19's own contract, mirroring
#: ``c18_experiment.py``, and ``scoring_runner.py`` owns C19's eval-node
#: runner the same way ``c18_experiment.py`` owns C18's. They are family
#: adapter code that predates the c18 split and kept its own module.
FAMILY_CONTRACT_FILES = frozenset({"experiment.py", "scoring_runner.py"})

#: Where the shared path may name a family, and why each is not a branch.
#:
#: ``cli.py``: the CLI's default choice of which family an unparameterised
#: run drives. A default is not a branch -- ``run_optimizer`` resolves it
#: through ``family_spec`` like any other value -- but it is still a family
#: literal, so it is enumerated rather than tolerated by a loose rule.
#:
#: ``manifest.py``: the study manifest's ``c18`` block, which records this
#: very verdict. A persisted wire key, not a dispatch on family: nothing in
#: the manifest branches on it, and it names the block rather than
#: selecting an adapter.
#: ``adapter_swap.py``: this module, which names both families because it
#: is the guard. A guard that could not spell what it guards could not
#: state its own exemption table, and the names here select nothing --
#: they are the *subject* of the check, not a dispatch on family. The
#: exemption is narrow on purpose: it admits the two identifiers and
#: nothing else, so a real branch added to this file would still fail the
#: import half of the check, which has no exemption at all.
ALLOWED_FAMILY_DEFAULTS: dict[str, frozenset[str]] = {
    "cli.py": frozenset({"c19"}),
    "manifest.py": frozenset({"c18"}),
    "adapter_swap.py": frozenset(KNOWN_FAMILY_IDS),
}

#: The families this guard looks for, as a set for membership tests.
_FAMILY_IDS = frozenset(KNOWN_FAMILY_IDS)

#: How an unreadable or unparseable module is named in the verdict. A file
#: the guard could not clear is reported as an offender rather than raised
#: past a stage that has already spent, so the reason is part of the name.
UNREADABLE_DETAIL = "unreadable"


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
            for family in _FAMILY_IDS
        )
    }


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """The identity of every node that *is* a docstring in ``tree``.

    Identified by position in the tree rather than by value, which is the
    whole point. Excluding by value asks "does this string equal some
    docstring in this file", and a module whose docstring is the single
    word ``c18`` then exempts every bare ``"c18"`` literal in it --
    including a live ``if family == "c18"`` branch. One line of prose
    would switch the guard off for the file it is guarding.

    Excluding by identity asks "is this string node the docstring", which
    is the question the exemption was always meant to ask: prose naming a
    family explains the shared path, and a literal in an expression
    branches on one, no matter that the two spell the same characters.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return docstrings


def _family_string_literals(path: Path) -> set[str]:
    """Family identifiers appearing as bare string values in one module.

    Docstrings are excluded -- by node identity, never by value; see
    :func:`_docstring_nodes` for why that distinction is load-bearing.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    docstrings = _docstring_nodes(tree)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value in _FAMILY_IDS
    }


def differing_modules(*, root: Path | None = None) -> tuple[str, ...]:
    """Shared-path modules that carry family-specific knowledge.

    Empty on a clean tree, which is the C3 pass. A non-empty result is the
    study's generality *finding* rather than an error, so it is returned
    rather than raised: section 4.1 says a leak is reported with the module
    named, not silently absorbed.

    Named by their path relative to the optimizer package, so a reader of
    the manifest can open the offending file without knowing where the
    package was installed.
    """
    base = root or OPTIM_ROOT
    exempt = FAMILY_ADAPTER_FILES | FAMILY_CONTRACT_FILES
    offenders: set[str] = set()
    for path in sorted(base.rglob("*.py")):
        if path.name in exempt:
            continue
        name = str(path.relative_to(base))
        allowed = ALLOWED_FAMILY_DEFAULTS.get(path.name, frozenset())
        try:
            # Both halves, unioned rather than short-circuited: a module
            # that leaks an import and a literal is one offender either
            # way, but a module whose import leak masked its literal leak
            # would be reported as clean the moment the import was removed.
            leaked = _family_package_imports(path) | (
                _family_string_literals(path) - allowed
            )
        except (OSError, SyntaxError, ValueError) as error:
            # A file this guard cannot read or parse is a file it cannot
            # clear, and it is named as such rather than raised.
            #
            # This runs at the end of an arm stage, after the last paid
            # operation. An exception here would abandon the c18 block
            # entirely -- the study would lose its recorded C3 evidence
            # over a syntax error in some unrelated module, having already
            # paid for every run the block was going to cite. Recording
            # "this module could not be cleared" is both honest and
            # non-destructive: the verdict fails, the reason is legible,
            # and the runs are still written.
            offenders.add(
                f"{name} ({UNREADABLE_DETAIL}: {type(error).__name__})"
            )
            continue
        if leaked:
            offenders.add(name)
    return tuple(sorted(offenders))


def adapter_swap_record(*, root: Path | None = None) -> AdapterSwapRecord:
    """The C3 assertion in the shape the manifest persists.

    ``passed`` is the conjunction the report reads: no shared-path module
    names or imports a family outside the adapter set. The modules are
    persisted rather than reduced to the boolean because a failure names
    *which* module leaked, and that name is the finding.
    """
    modules = differing_modules(root=root)
    return AdapterSwapRecord(passed=not modules, differing_modules=modules)
