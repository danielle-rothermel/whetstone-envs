"""Opt-in gate and shared world for the envs Codex arm's real-CLI ladder.

Every rung in this package drives the *real* Codex CLI against a live
subscription session, so the whole package is skipped unless
``WHETSTONE_ENVS_REAL_CODEX=1`` is set. The *task* model stays fake
throughout -- every rung runs ``--transport fake``, so a ladder run spends
Codex agent turns and no eval-provider credit.

Unlike whetstone-ai's ladder, which wires a harness by hand, this one
drives the arm the way the study does: through
:func:`~whetstone_envs.optim.run.run_optimizer` with ``optimizer="codex"``.
That is the point of the envs ladder -- the out-of-process MCP server
rebuilds the c19/c18 experiment from :class:`EnvsCodexRuntimeConfig`, and
only a real run proves that rebuild lands on the same Eval Config the
agent's calls are admitted against.

Because the real CLI is reached through the production path rather than a
seam, each rung must also carry the real-Codex opt-in that
``refuse_unauthorized_real_codex`` demands: ``allow_real_codex=True`` on
the spec *and* ``WHETSTONE_ENVS_ALLOW_REAL_CODEX=1`` in the environment.
The two are deliberately separate from this package's own opt-in, so
collecting the ladder is not by itself authority to spend.

This package is also the sole owner of the one exception to the root
conftest's session tripwire: ``tests/conftest.py`` arms
``WHETSTONE_ENVS_FORBID_REAL_CODEX`` for every session and defers to the
claim :func:`pytest_collection_modifyitems` below makes for a ladder
session. See that hook for why the exception is decided here.

No test here ever reads, copies, or prints credential material. The
runner's own ``stage_auth`` copies ``~/.codex/auth.json`` into each run's
scratch ``CODEX_HOME``; that is production code and the bytes never enter
the test process.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from whetstone.optim.codex.containment import CODEX_AUTH_FILENAMES
from whetstone.optim.tools.admission import _ENTRY_TABLE, ToolCallState
from whetstone.optim.tools.contracts import RefusalClass

from tests.conftest import REAL_CODEX_LADDER_SESSION
from tests.real_codex.preconditions import (
    REAL_CODEX_BINARY_ENV,
    REAL_CODEX_ENV,
    SANDBOX_EXEC_PATH,
    real_codex_precondition_failure,
)
from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
)
from whetstone_envs.optim.run import RunSpec, run_optimizer

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The opt-in, the binary override, and the sandbox path live in
#: :mod:`tests.real_codex.preconditions` beside the decision written in
#: terms of them, and are re-exported here so every rung keeps importing
#: them from the conftest.
DEFAULT_REAL_CODEX_BINARY = "/opt/homebrew/bin/codex"

#: Every rung's wall budget. Generous enough that a real session finishing
#: normally is never cut short, so a wall-budget stop is only ever the
#: property rung 4 asserts deliberately.
RUNG_WALL_SECONDS = 300.0

#: The splits the cheap rungs use. Small on purpose: the arm's fidelity
#: does not depend on split size, and each admitted call evaluates the
#: whole internal split. Rung 7 deliberately uses the real §6 size instead.
LADDER_SPLIT_SIZES = (2, 2, 0)


def real_codex_binary() -> str:
    return os.environ.get(REAL_CODEX_BINARY_ENV) or DEFAULT_REAL_CODEX_BINARY


#: This conftest is nested, but pytest still calls its collection hook with
#: every collected item in the session -- including the ordinary CI suites.
#: Skipping unconditionally would silently disable all Codex coverage in
#: CI, so the hook filters to items that live under this directory.
_LADDER_ROOT = Path(__file__).resolve().parent


def _ladder_items(items: list[pytest.Item]) -> list[pytest.Item]:
    """The collected items that live under this package."""
    selected: list[pytest.Item] = []
    for item in items:
        try:
            path = Path(str(item.fspath)).resolve()
        except (AttributeError, OSError):
            continue
        if path.is_relative_to(_LADDER_ROOT):
            selected.append(item)
    return selected


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the ladder -- and only the ladder -- unless opted into.

    This hook also owns the *one* exception to the root conftest's
    session tripwire. ``tests/conftest.py`` arms
    ``WHETSTONE_ENVS_FORBID_REAL_CODEX`` for every session, and
    ``refuse_unauthorized_real_codex`` honours it above every other
    input -- so a ladder session that let it be armed would have every
    rung raise ``RealCodexRefusedError`` instead of driving the CLI.

    The exception is claimed here rather than decided there, for two
    reasons. Ordering: pytest runs a nested conftest's collection hook
    before the root's, and both before any session fixture, so a claim
    made here is visible to the root fixture that reads it. And
    narrowness: this hook knows something the root cannot, namely that
    the session *actually collected ladder items*. Requiring a collected
    rung on top of :data:`REAL_CODEX_ENV` is what keeps a stray export in
    a developer's shell from disarming the tripwire for an ordinary
    ``pytest tests/`` run -- that session collects no rung, claims
    nothing, and is armed exactly as before. The ``real_codex`` marker
    (deselected by default via ``addopts``) is the third condition, since
    a deselected rung is not a collected item.

    The spend opt-in itself is not checked here. That is
    ``_real_codex_preconditions``' job, and it fails the session loudly
    rather than quietly proceeding; claiming the exception without it
    would simply mean the tripwire is disarmed and the production gate
    refuses on the missing allow variable, which is the correct outcome
    either way.
    """
    ladder_items = _ladder_items(items)
    if os.environ.get(REAL_CODEX_ENV) == "1":
        # The claim also requires that the session collected *nothing but*
        # the ladder. A mixed session -- ``-m ""``, or an explicit path
        # alongside the ladder -- would otherwise disarm the tripwire for
        # every ordinary test running beside the rungs, which is exactly
        # the state in which an authorization test that monkeypatches the
        # allow variable can reach the real CLI. The ladder's exception is
        # for the ladder, so a mixed session keeps the tripwire armed and
        # the rungs refuse instead.
        if ladder_items and len(ladder_items) == len(items):
            config.stash[REAL_CODEX_LADDER_SESSION] = True
        return
    skip = pytest.mark.skip(
        reason=(
            f"real-Codex ladder is opt-in: set {REAL_CODEX_ENV}=1 "
            "(drives the real CLI against a live subscription session)"
        )
    )
    for item in ladder_items:
        item.add_marker(skip)


def _observe_real_codex_preconditions() -> str | None:
    """Read the machine, then let the pure function decide."""
    binary = real_codex_binary()
    home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    return real_codex_precondition_failure(
        opted_in=os.environ.get(REAL_CODEX_ENV) == "1",
        platform=sys.platform,
        binary_found=(
            shutil.which(binary) is not None or Path(binary).is_file()
        ),
        binary=binary,
        sandbox_exec_found=SANDBOX_EXEC_PATH.is_file(),
        # Existence only. The ladder never opens these files.
        auth_found=any(
            (home / name).is_file() for name in CODEX_AUTH_FILENAMES
        ),
        auth_home=home,
        # Passed from their real owners so the split module cannot become
        # a second spelling of either fact.
        auth_filenames=CODEX_AUTH_FILENAMES,
        spend_opt_in=os.environ.get(ALLOW_REAL_CODEX_ENV),
        spend_opt_in_env=ALLOW_REAL_CODEX_ENV,
        spend_opt_in_value=ALLOW_REAL_CODEX_ENV_VALUE,
    )


@pytest.fixture(scope="session", autouse=True)
def _real_codex_preconditions() -> None:
    """Fail loudly, before any rung runs, if the machine cannot host one."""
    failure = _observe_real_codex_preconditions()
    if failure is not None:
        # exit, not fail: an unhostable machine makes every remaining rung
        # meaningless, and a session-scoped fail would be reported once
        # per rung as an error rather than once as a refusal.
        pytest.exit(failure, returncode=1)


@pytest.fixture(autouse=True)
def _no_task_model_key(monkeypatch) -> None:
    """The agent's env must never carry an eval-provider key.

    The ladder's whole cost claim is "Codex turns and nothing else". The
    task model is fake, so an ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY``
    leaking into this process would not change what the fake transport
    does -- but it would travel into the runner's environment allowlist
    and make the claim unverifiable by inspection. Removing them here
    makes the absence a property of every rung rather than of the shell
    that happened to launch it.
    """
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def real_codex_run_spec(  # noqa: PLR0913
    *,
    output_dir: Path,
    run_id: str,
    family: str = "c19",
    split_sizes: tuple[int, int, int] = LADDER_SPLIT_SIZES,
    capacity: int | None = None,
    n_per_stratum: int | None = None,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
    wall_seconds: float | None = RUNG_WALL_SECONDS,
) -> RunSpec:
    """The spec every rung runs: real Codex agent, fake task model.

    ``transport="fake"`` is what keeps a rung's cost to Codex turns alone,
    and ``allow_real_codex=True`` is one half of the production spend
    opt-in -- the other half is the environment variable the session
    precondition already proved is set.
    """
    kwargs: dict[str, Any] = {
        "optimizer": "codex",
        "transport": "fake",
        "family": family,
        "split_sizes": split_sizes,
        "output_dir": output_dir,
        "run_id": run_id,
        "codex_capacity": capacity,
        "n_per_stratum": n_per_stratum,
        "codex_binary": real_codex_binary(),
        "codex_wall_seconds": wall_seconds,
        "allow_real_codex": True,
    }
    if codex_model is not None:
        kwargs["codex_model"] = codex_model
    if codex_reasoning_effort is not None:
        kwargs["codex_reasoning_effort"] = codex_reasoning_effort
    return RunSpec(**kwargs)


def run_real_codex(spec: RunSpec) -> Path:
    """One real Codex run through the production entry point.

    No ``codex_test_seam``: that is the whole point. The seam is what CI
    uses to reach the scripted fake CLI, and passing one here would mean
    the ladder proved nothing about the real binary.
    """
    return run_optimizer(spec)


def run_namespace_key(result: Any) -> str:
    """The one store namespace this run's Tool admitted calls in.

    Read from ``OptimRun.tool_configs`` rather than from the Step's
    reported evidence, mirroring the audit's own ``_tool_scope``: a run
    that reported nothing still names its granted Tool, and scoping by
    what the agent reported would make an under-reporting run look
    empty rather than wrong.
    """
    configs = result.run.record.tool_configs
    assert len(configs) == 1, (
        f"a Codex run is granted exactly one Tool, found {len(configs)}"
    )
    return str(configs[0].record.store_namespace_key)


def capacity_refusals(
    *, sqlite_path: Path, namespace_key: str
) -> tuple[dict[str, Any], ...]:
    """Every durable CAPACITY refusal this run's namespace recorded.

    A capacity refusal debits no capacity, so it is deliberately absent
    from the Step's tool evidence -- that absence is what makes the
    evidence count agree with the budget debit. There is therefore no
    public API that enumerates refusals, and ``find_entry`` would need a
    ``call_id`` the real agent chose for itself and never reported.

    So this reads the admission entry table directly. It is a test-only
    assertion helper: a real rung has to distinguish "the agent was
    refused" from "the agent never tried a second call", and only the
    durable ledger can tell those apart. Widening the production surface
    to enumerate refusals would add an API no production caller wants.
    """
    connection = sqlite3.connect(str(sqlite_path))
    try:
        rows = connection.execute(
            f"SELECT entry_json FROM {_ENTRY_TABLE} "  # noqa: S608
            "WHERE store_namespace_key = ?",
            (namespace_key,),
        ).fetchall()
    finally:
        connection.close()
    entries = [json.loads(row[0]) for row in rows]
    return tuple(
        entry
        for entry in entries
        if entry.get("state") == ToolCallState.REFUSED.value
        and (entry.get("refusal") or {}).get("refusal_class")
        == RefusalClass.CAPACITY.value
    )


@pytest.fixture
def ladder_output(tmp_path) -> Iterator[Path]:
    """A scratch root for one rung's run artifacts."""
    return tmp_path
