"""Base-install modules must not reach an optional dependency.

``whetstone-envs`` ships two console scripts, and only one of them --
``whetstone-study`` -- belongs to the ``optim`` extra. ``whetstone-eval``
is a base-install entry point: an install that took no extra must still
reach it, which means every module beneath it must import without
``whetstone``, ``dr_providers``, or anything else the base dependency set
does not name.

That property is invisible to the rest of the suite, because the
development environment installs every extra: a module-scope
``from dr_providers import ...`` beneath ``reporting.cli`` imports
perfectly here and fails only in ``scripts/check_distributions.py``'s
isolated wheel install, or in a user's. This test spends a subprocess to
catch it in the suite instead.

The module list is derived exactly as ``check_distributions.py`` derives
its ``SMOKE_MODULES`` -- every package directly under ``whetstone_envs``
-- so the two cannot drift: a new subpackage is covered here the moment
it exists, without being added to a list.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import whetstone_envs

PACKAGE_DIR = Path(whetstone_envs.__file__).parent

#: Modules the base install must import, mirroring the wheel smoke test's
#: ``SMOKE_MODULES``: the package itself plus each of its subpackages.
BASE_INSTALL_MODULES = (
    "whetstone_envs",
    *sorted(
        f"whetstone_envs.{path.name}"
        for path in PACKAGE_DIR.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    ),
)

#: Entry points reachable without the ``optim`` extra. ``optim.study.cli``
#: is deliberately absent: ``whetstone-study`` is the extra's own command
#: and may import the optimizer stack at module scope.
BASE_INSTALL_ENTRY_POINTS = ("whetstone_envs.reporting.cli",)

#: Distributions the base install does not depend on. Importing one from
#: a base-install module is the failure this test exists to catch.
OPTIONAL_DISTRIBUTION_MODULES = ("dr_providers", "whetstone", "rich")


def _import_without_optional_dependencies(
    module: str,
    *,
    extra: str = "",
) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a subprocess where the optim stack is missing.

    Rather than build an extra-free virtualenv -- slow, and dependent on
    a network -- this blocks the optional distributions at the import
    system, which reproduces exactly what the base install presents: the
    module is not there, and reaching for it raises ``ModuleNotFoundError``.

    ``extra`` runs after the import, in the same blocked interpreter, so a
    caller can exercise more than import time -- building a parser, say --
    under the base install's constraints.
    """
    blocker = ",".join(repr(name) for name in OPTIONAL_DISTRIBUTION_MODULES)
    code = f"""
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return None

    def find_spec(self, name, path=None, target=None):
        root = name.partition(".")[0]
        if root in {{{blocker}}}:
            raise ModuleNotFoundError(
                f"No module named {{root!r}}", name=root
            )
        return None

sys.meta_path.insert(0, _Blocked())
for name in list(sys.modules):
    if name.partition(".")[0] in {{{blocker}}}:
        del sys.modules[name]
import {module}
{extra}
"""
    return subprocess.run(  # noqa: S603 - fixed interpreter, derived module name
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("module", BASE_INSTALL_MODULES)
def test_base_install_module_imports_without_the_optim_extra(
    module: str,
) -> None:
    completed = _import_without_optional_dependencies(module)
    assert completed.returncode == 0, (
        f"{module} is imported by the wheel smoke test on a base install, "
        f"but it reaches an optional dependency at import time. Import it "
        f"lazily, inside the function that needs it, or move the code that "
        f"needs it into a module the base install does not import.\n"
        f"stderr:\n{completed.stderr}"
    )


def test_the_base_install_entry_point_parses_help_without_the_extra() -> None:
    """``--help`` must work on a base install, not merely import.

    The wheel smoke test in ``scripts/check_distributions.py`` invokes
    ``whetstone-eval --help`` on a real extra-free install, so a parser
    that reached the ``optim`` extra while *building* -- an argparse
    ``type=`` or ``choices=`` naming a ``dr_providers`` enum, say -- broke
    the packaged console script while every import-time check still
    passed. This closes that gap at unit speed.
    """
    completed = _import_without_optional_dependencies(
        "whetstone_envs.reporting.cli",
        extra=(
            "import whetstone_envs.reporting.cli as cli\n"
            "parser = cli.build_parser()\n"
            "try:\n"
            "    parser.parse_args(['--help'])\n"
            "except SystemExit as error:\n"
            "    raise SystemExit(error.code or 0)\n"
        ),
    )
    assert completed.returncode == 0, (
        "whetstone-eval --help must work on a base install; the parser "
        "reached an optional dependency while being built.\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("module", BASE_INSTALL_ENTRY_POINTS)
def test_base_install_entry_point_imports_without_the_optim_extra(
    module: str,
) -> None:
    completed = _import_without_optional_dependencies(module)
    assert completed.returncode == 0, (
        f"{module} backs a console script that a base install must reach, "
        f"but it reaches an optional dependency at import time.\n"
        f"stderr:\n{completed.stderr}"
    )
