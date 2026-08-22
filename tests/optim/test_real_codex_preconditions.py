"""The real-Codex opt-in gate decides *fail*, never *skip*, once opted in.

``scripts/check-real-codex.sh`` reports "all rungs passed" from pytest's
exit status, and pytest exits 0 on a session where every test skipped. So a
precondition that skipped -- wrong platform, missing binary, missing
``sandbox-exec``, no session, no spend opt-in -- would turn a ladder that
never drove the real CLI at all into a green report. Once
``WHETSTONE_ENVS_REAL_CODEX=1`` is set the operator has asked for a real
run, and every unmet precondition is an error.

These exercise the pure decision function directly, so the whole matrix is
testable on any host without a Codex binary, a live session, or macOS. They
carry no ``real_codex`` marker: this is ordinary CI coverage of the gate
itself, not a rung.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.real_codex.conftest import (
    REAL_CODEX_BINARY_ENV,
    REAL_CODEX_ENV,
    real_codex_precondition_failure,
)
from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
)

_AUTH_HOME = Path("/home/someone/.codex")


def _decide(  # noqa: PLR0913
    *,
    opted_in: bool = True,
    platform: str = "darwin",
    binary_found: bool = True,
    binary: str = "/opt/homebrew/bin/codex",
    sandbox_exec_found: bool = True,
    auth_found: bool = True,
    auth_home: Path = _AUTH_HOME,
    spend_opt_in: str | None = ALLOW_REAL_CODEX_ENV_VALUE,
) -> str | None:
    """The decision for a hostable machine, with named deviations.

    The defaults *are* the hostable machine, so each test names only what
    it breaks. Spelling them as real keyword parameters rather than a
    dict of overrides keeps the call sites type-checked -- a
    heterogeneous mapping splatted into this signature type-checks as
    nothing at all, which is how a renamed field would go unnoticed here.
    """
    return real_codex_precondition_failure(
        opted_in=opted_in,
        platform=platform,
        binary_found=binary_found,
        binary=binary,
        sandbox_exec_found=sandbox_exec_found,
        auth_found=auth_found,
        auth_home=auth_home,
        spend_opt_in=spend_opt_in,
    )


def test_a_hostable_machine_reports_no_failure() -> None:
    assert _decide() is None


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        pytest.param(
            {"platform": "linux"},
            "macOS sandbox-exec only",
            id="non_macos_host",
        ),
        pytest.param(
            {"sandbox_exec_found": False},
            "sandbox-exec",
            id="sandbox_exec_missing",
        ),
        pytest.param(
            {"binary_found": False},
            REAL_CODEX_BINARY_ENV,
            id="binary_missing",
        ),
        pytest.param(
            {"auth_found": False},
            "codex login",
            id="no_session",
        ),
        pytest.param(
            {"spend_opt_in": None},
            ALLOW_REAL_CODEX_ENV,
            id="no_spend_opt_in",
        ),
    ],
)
def test_an_unmet_precondition_fails_when_opted_in(
    overrides, expected_fragment
) -> None:
    """Each unmet precondition names itself, and names the opt-in."""
    failure = _decide(**overrides)

    assert failure is not None, (
        f"{overrides} produced no failure while opted in; an unhostable "
        "machine would run zero rungs and still report success"
    )
    assert expected_fragment in failure
    # The message has to say why this became an error rather than a skip.
    assert REAL_CODEX_ENV in failure


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"platform": "linux"}, id="non_macos_host"),
        pytest.param({"sandbox_exec_found": False}, id="sandbox_exec_missing"),
        pytest.param({"binary_found": False}, id="binary_missing"),
        pytest.param({"auth_found": False}, id="no_session"),
        pytest.param({"spend_opt_in": None}, id="no_spend_opt_in"),
        pytest.param({}, id="fully_hostable"),
    ],
)
def test_nothing_fails_when_not_opted_in(overrides) -> None:
    """Not opting in is not an error: the collection hook skips the ladder.

    The gate must stay silent here even on a machine that *could* host the
    ladder -- an ordinary CI run sets no opt-in and must not be failed by a
    package it never asked to run.
    """
    assert _decide(opted_in=False, **overrides) is None


@pytest.mark.parametrize(
    "value",
    ["", "0", "true", "TRUE", "yes", "11"],
    ids=["empty", "zero", "true", "upper_true", "yes", "suffixed"],
)
def test_only_the_exact_spend_value_satisfies_the_gate(value) -> None:
    """A half-remembered spelling refuses rather than spends.

    This mirrors ``refuse_unauthorized_real_codex``'s own exact-value
    check. If the gate accepted anything truthy it would pass a machine
    the production path then refuses, turning one clear precondition
    message into a rung-shaped wall of ``RealCodexRefusedError``.
    """
    failure = _decide(spend_opt_in=value)

    assert failure is not None
    assert ALLOW_REAL_CODEX_ENV in failure


def test_the_platform_check_precedes_the_macos_only_checks() -> None:
    """A Linux host is told the platform, not that sandbox-exec is absent.

    ``/usr/bin/sandbox-exec`` is necessarily missing off macOS, so
    reporting it first would hand a Linux operator a file path to chase
    instead of the actual reason.
    """
    failure = _decide(
        platform="linux", sandbox_exec_found=False, binary_found=False
    )

    assert failure is not None
    assert "macOS sandbox-exec only" in failure
    assert str(_AUTH_HOME) not in failure


def test_the_session_check_precedes_the_spend_check() -> None:
    """No session is the more basic failure, so it is reported first.

    Telling an operator to set the spend variable on a machine with no
    Codex login would send them to buy something they cannot use.
    """
    failure = _decide(auth_found=False, spend_opt_in=None)

    assert failure is not None
    assert "codex login" in failure


def test_the_failure_message_never_carries_credential_material() -> None:
    """The gate names locations, never contents.

    It only ever stats these paths, and this pins that the message stays a
    location -- a regression that read a session file to explain itself
    would put credential bytes into a pytest report and CI log.
    """
    failure = _decide(auth_found=False)

    assert failure is not None
    assert str(_AUTH_HOME) in failure
    for secret_ish in ("token", "OPENAI_API_KEY", "Bearer", "eyJ"):
        assert secret_ish not in failure
