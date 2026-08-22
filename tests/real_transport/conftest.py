"""Opt-in gating and shared fixtures for the real-transport smoke rungs.

These tests spend real money on the OpenRouter transport, so they are
deselected by default and reached only through
``WHETSTONE_ENVS_REAL_TRANSPORT=1``. The two halves are deliberately
different in kind: the variable is the *opt-in*, and
``OPENROUTER_API_KEY`` is the *credential*. Opting in without the
credential is an operator error rather than a reason to skip, so it FAILs
loudly -- a silent skip there would let a "green" run report that the real
transport was exercised when nothing was ever called.

Every rung asserts on persisted state under an isolated run directory, so
no rung can read another's artifacts and no rung writes into the git tree.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from whetstone_envs.optim.audit import AuditStatus, audit_run
from whetstone_envs.optim.run_cost import read_run_cost

if TYPE_CHECKING:
    from pathlib import Path

#: The real-transport opt-in. Spelled here and re-exported so the check
#: script and the workflow name exactly one string.
REAL_TRANSPORT_ENV = "WHETSTONE_ENVS_REAL_TRANSPORT"
REAL_TRANSPORT_ENV_VALUE = "1"

#: The credential the OpenRouter transport reads. Named by
#: ``run_optimizer`` when ``--transport openrouter`` is selected.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

#: The toy split every rung runs on: four internal, two official, two
#: held-out. Small enough to stay a smoke run, and the held-out entry is
#: positive so the held-out evaluation rung has tasks to score.
SMOKE_SPLIT_SIZES = (4, 2, 2)

#: The task model for every rung. Cheap, and the model the study's own
#: defaults already name.
SMOKE_MODEL = "openai/gpt-4.1-nano"


def _opted_in() -> bool:
    return os.environ.get(REAL_TRANSPORT_ENV) == REAL_TRANSPORT_ENV_VALUE


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Deselect the real-transport rungs unless the operator opted in.

    Deselection, not skipping: an un-opted-in run should not report these
    as skipped tests it might have run, because running them costs money.
    """
    del config
    if _opted_in():
        return
    skip_real = pytest.mark.skip(
        reason=(
            f"requires {REAL_TRANSPORT_ENV}={REAL_TRANSPORT_ENV_VALUE}; "
            "these rungs spend real money on the OpenRouter transport"
        ),
    )
    for item in items:
        if "real_transport" in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture(scope="session", autouse=True)
def _require_credential_when_opted_in() -> None:
    """Opted in without a key is a loud failure, never a skip.

    A skip here would let a run that called nothing report success, which
    is precisely the mock-only outcome these rungs exist to rule out.
    """
    if not _opted_in():
        return
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key or not key.strip():
        message = (
            f"{REAL_TRANSPORT_ENV}={REAL_TRANSPORT_ENV_VALUE} opts in to "
            f"real provider spend, but {OPENROUTER_KEY_ENV} is unset or "
            "empty, so no rung could reach the real transport. Provide "
            "the credential or drop the opt-in."
        )
        raise RuntimeError(message)


@pytest.fixture(scope="session", autouse=True)
def _never_real_codex() -> None:
    """The real-transport rungs never reach the billed Codex CLI.

    D2 owns the Codex ladder. Asserting the flag is absent here means a
    shell that exported it cannot turn a provider-spend run into an agent
    -spend run.
    """
    present = os.environ.get("WHETSTONE_ENVS_ALLOW_REAL_CODEX")
    if present is not None:
        message = (
            "WHETSTONE_ENVS_ALLOW_REAL_CODEX is set in this process. The "
            "real-transport rungs spend on the provider only and must "
            "never be able to spawn the billed Codex CLI."
        )
        raise RuntimeError(message)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """An isolated, off-repo output directory for one rung."""
    return tmp_path / "run"


def assert_persisted_run(directory: Path, *, optimizer: str) -> None:
    """Every rung's shared assertion on one run's persisted state.

    Named once rather than repeated per rung so a rung cannot quietly
    assert less than its siblings: the point of the smoke ladder is that
    every arm is held to the same evidence.
    """
    result_path = directory / "result.json"
    sqlite_path = directory / "runtime.sqlite"
    assert result_path.is_file(), f"{optimizer}: result.json was not written"
    assert sqlite_path.is_file(), (
        f"{optimizer}: runtime.sqlite was not written"
    )
    assert sqlite_path.stat().st_size > 0, (
        f"{optimizer}: runtime.sqlite is empty"
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result, f"{optimizer}: result.json is empty"

    report = audit_run(directory)
    failed = [
        finding
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    ]
    assert report.passed, f"{optimizer}: audit_run failed: " + "; ".join(
        f"{finding.invariant_id.value}: {finding.detail}" for finding in failed
    )
    assert report.findings, (
        f"{optimizer}: audit_run judged no invariants at all"
    )

    assert_priced_spend(directory, optimizer=optimizer)


def assert_priced_spend(directory: Path, *, optimizer: str) -> None:
    """The run really called the provider, and the calls carried prices.

    This is the assertion that separates a real rung from a mocked one: a
    fake-transport run reports ``priced_calls == 0`` for the task model,
    so a rung that silently fell back to the fake transport fails here
    rather than passing quietly.
    """
    cost_path = directory / "cost.json"
    assert cost_path.is_file(), (
        f"{optimizer}: cost.json was not written, so the run reported no "
        "provider spend at all"
    )
    document = read_run_cost(directory)
    assert document.spend, f"{optimizer}: cost document reports no roles"

    task_rows = [
        record for record in document.spend if record.role == "task_model"
    ]
    assert task_rows, (
        f"{optimizer}: cost document has no task_model role, so no task "
        "evaluation reached the provider"
    )
    for record in task_rows:
        assert record.calls > 0, (
            f"{optimizer}: task_model made {record.calls} calls; the real "
            "transport was never reached"
        )
        assert record.priced_calls > 0, (
            f"{optimizer}: task_model recorded {record.priced_calls} "
            f"priced calls out of {record.calls}; a real OpenRouter call "
            "carries a provider-reported price, so this run did not go "
            "through the real transport"
        )
        assert record.input_tokens > 0, (
            f"{optimizer}: task_model reported {record.input_tokens} "
            "input tokens; a real call always consumes prompt tokens"
        )


def total_usd(directory: Path) -> float:
    """Ledgered USD across every priced role, for the rung table."""
    document = read_run_cost(directory)
    return sum(
        record.usd for record in document.spend if record.usd is not None
    )


def rung_ledger(directory: Path, *, rung: str) -> dict[str, Any]:
    """One rung's ledger line, emitted for the transcript."""
    document = read_run_cost(directory)
    return {
        "rung": rung,
        "run_id": document.run_id,
        "usd": total_usd(directory),
        "spend": [
            {
                "role": record.role,
                "calls": record.calls,
                "priced_calls": record.priced_calls,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "usd": record.usd,
            }
            for record in document.spend
        ],
    }
