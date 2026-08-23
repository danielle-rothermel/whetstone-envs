"""Why a machine can or cannot host the real-Codex ladder.

Split out of ``tests/real_codex/conftest.py`` so the decision itself is
importable without whetstone-ai. The conftest reaches into the optimizer
stack -- the containment module, the admission table, the run path -- which
the ``optim`` extra installs only on Python 3.13+. Importing this decision
through the conftest therefore hard-errored at *collection* on a base
install, so the Python 3.12 CI job never ran the gate's own tests at all:
the coverage that stops an all-skipped ladder being reported as green was
itself invisible on the job most likely to lose it.

Nothing here imports whetstone-ai or ``whetstone_envs.optim``. The two
facts it would otherwise borrow from them -- the spend opt-in variable and
the credential filenames -- are parameters, supplied by the conftest from
their real owners so this module cannot drift into a second source of
truth for either.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "REAL_CODEX_BINARY_ENV",
    "REAL_CODEX_ENV",
    "SANDBOX_EXEC_PATH",
    "real_codex_precondition_failure",
]

#: The opt-in for the ladder package. Absent or not "1", every rung is
#: skipped. Owned here because the decision below is written in terms of
#: it and the conftest re-exports it.
REAL_CODEX_ENV = "WHETSTONE_ENVS_REAL_CODEX"
#: Where the real binary is expected. Overridable for a non-brew install.
REAL_CODEX_BINARY_ENV = "WHETSTONE_ENVS_REAL_CODEX_BINARY"

#: Where macOS keeps the only process-isolation mechanism this ladder will
#: run under. Mirrors the runner's own ``_MACOS_SANDBOX_EXEC``: a rung that
#: reaches the runner without it fails opaquely, so the precondition names
#: the same file rather than trusting the platform tag alone.
SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")


def real_codex_precondition_failure(  # noqa: PLR0911, PLR0913
    *,
    opted_in: bool,
    platform: str,
    binary_found: bool,
    binary: str,
    sandbox_exec_found: bool,
    auth_found: bool,
    auth_home: Path,
    auth_filenames: Sequence[str],
    spend_opt_in: str | None,
    spend_opt_in_env: str,
    spend_opt_in_value: str,
) -> str | None:
    """Why this machine cannot host the ladder, or ``None`` if it can.

    A pure function of the environment, so the whole decision matrix is
    testable without a macOS host, a Codex binary, or a live session.

    The opt-in is what makes an unmet precondition an *error*. Without
    ``WHETSTONE_ENVS_REAL_CODEX=1`` the ladder is simply not requested and
    the collection hook skips it. With the opt-in, the operator asked for
    a real run, and every one of these conditions means they will not get
    one -- so the answer is a message the caller raises, never a skip.
    Skipping here is what would let a Linux host, a missing binary, or an
    absent ``sandbox-exec`` produce an all-skipped session that
    ``scripts/check-real-codex.sh`` reports as "all rungs passed": pytest
    exits 0 on a fully skipped session.

    ``spend_opt_in`` is checked here for the same reason. Every rung
    drives the production ``run_optimizer`` path, which refuses a real
    Codex run unless the spend opt-in variable names its exact value.
    Without it every rung raises ``RealCodexRefusedError`` -- a wall of
    identical failures whose real cause is one missing variable. Naming it
    once, before any rung runs, is the difference between a diagnosis and
    a pile of tracebacks.

    ``auth_filenames``, ``spend_opt_in_env``, and ``spend_opt_in_value``
    are parameters rather than imports so this module stays free of the
    optimizer stack. Their owners are whetstone-ai's containment module
    and :mod:`whetstone_envs.optim.codex`, and the conftest passes the
    real ones, so this cannot become a second spelling of either.
    """
    if not opted_in:
        return None
    if platform != "darwin":
        return (
            f"{REAL_CODEX_ENV}=1 was set on {platform!r}, but the Codex "
            "sandbox is macOS sandbox-exec only. Run the ladder on macOS "
            f"or unset {REAL_CODEX_ENV}."
        )
    if not sandbox_exec_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but {SANDBOX_EXEC_PATH} is not "
            "present; the ladder refuses to drive the real CLI without "
            "kernel-enforced process isolation."
        )
    if not binary_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but the real Codex binary was "
            f"not found at {binary!r}; set {REAL_CODEX_BINARY_ENV} to "
            "its path."
        )
    if not auth_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but no Codex session was found "
            f"under {auth_home} ({'/'.join(auth_filenames)}); "
            "run `codex login` first."
        )
    if spend_opt_in != spend_opt_in_value:
        return (
            f"{REAL_CODEX_ENV}=1 was set but "
            f"{spend_opt_in_env}={spend_opt_in_value} was "
            "not: every rung drives the production run path, which "
            "refuses a real Codex run without the deliberate spend "
            "opt-in. Set both, or unset "
            f"{REAL_CODEX_ENV}."
        )
    return None
