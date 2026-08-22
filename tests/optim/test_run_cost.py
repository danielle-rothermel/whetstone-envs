"""``cost.json``: its pinned wire format, its projection, and its storage.

The golden test at the top is the persisted-format contract: every literal
is written out rather than derived from a field name, so a rename that
changed stored identity fails here instead of in a report six weeks later.

The end-to-end tests run a real fake-transport optimizer, because the whole
point of ``cost.json`` is that the runner writes it -- a projection that
only works on a hand-built ``OptimResult`` would not prove the artifact
lands beside the result it projects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from pydantic import ValidationError

from whetstone_envs.optim.run_cost import (
    MAX_RUN_COST_BYTES,
    RUN_COST_NAME,
    RUN_COST_SCHEMA,
    RUN_COST_SCHEMA_NAME,
    RUN_COST_SCHEMA_VERSION,
    RunCostDocument,
    read_run_cost,
    write_run_cost,
)
from whetstone_envs.optim.study.manifest import RunSpendRecord

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------
# Golden literals
# --------------------------------------------------------------------------


def test_persisted_literals_are_pinned() -> None:
    assert RUN_COST_NAME == "cost.json"
    assert RUN_COST_SCHEMA_NAME == "whetstone_envs.run_cost"
    assert RUN_COST_SCHEMA_VERSION == 1
    assert RUN_COST_SCHEMA == "whetstone_envs.run_cost/v1"
    assert MAX_RUN_COST_BYTES == 256 * 1024


def _spend(role: str) -> RunSpendRecord:
    return RunSpendRecord(
        role=role,
        calls=10,
        cached_calls=2,
        input_tokens=100,
        output_tokens=20,
        priced_calls=10,
        unpriced_calls=0,
        rows_missing_token_breakdown=1,
        usd=0.25,
    )


def _document() -> RunCostDocument:
    return RunCostDocument(
        run_id="c19-copro-1",
        cost_report_schema_version=1,
        spend=(_spend("task_model"), _spend("proposer")),
    )


def test_document_wire_keys_are_pinned() -> None:
    payload = _document().model_dump(mode="json")
    assert list(payload) == [
        "schema_version",
        "run_id",
        "cost_report_schema_version",
        "spend",
    ]
    assert list(payload["spend"][0]) == [
        "role",
        "calls",
        "cached_calls",
        "input_tokens",
        "output_tokens",
        "priced_calls",
        "unpriced_calls",
        "rows_missing_token_breakdown",
        "usd",
    ]


# --------------------------------------------------------------------------
# The record's honesty rules
# --------------------------------------------------------------------------


def test_a_priced_total_beside_unpriced_calls_is_refused() -> None:
    """An absent total means "not knowable", so a present one must be."""
    with pytest.raises(ValidationError, match="no total spend"):
        RunSpendRecord(
            role="task_model",
            calls=10,
            cached_calls=0,
            input_tokens=1,
            output_tokens=1,
            priced_calls=9,
            unpriced_calls=1,
            rows_missing_token_breakdown=0,
            usd=0.5,
        )


def test_priced_and_unpriced_must_exhaust_the_billable_calls() -> None:
    with pytest.raises(ValidationError, match="exhaust"):
        RunSpendRecord(
            role="task_model",
            calls=10,
            cached_calls=0,
            input_tokens=1,
            output_tokens=1,
            priced_calls=4,
            unpriced_calls=4,
            rows_missing_token_breakdown=0,
            usd=None,
        )


def test_missing_token_breakdowns_cannot_exceed_the_calls() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        RunSpendRecord(
            role="task_model",
            calls=2,
            cached_calls=0,
            input_tokens=1,
            output_tokens=1,
            priced_calls=2,
            unpriced_calls=0,
            rows_missing_token_breakdown=3,
            usd=0.1,
        )


def test_a_role_is_reported_at_most_once() -> None:
    with pytest.raises(ValidationError, match="reported once"):
        RunCostDocument(
            run_id="r",
            cost_report_schema_version=1,
            spend=(_spend("task_model"), _spend("task_model")),
        )


def test_a_foreign_schema_version_is_refused() -> None:
    with pytest.raises(ValidationError, match="expected schema version"):
        RunCostDocument(
            schema_version=RUN_COST_SCHEMA_VERSION + 1,
            run_id="r",
            cost_report_schema_version=1,
            spend=(_spend("task_model"),),
        )


# --------------------------------------------------------------------------
# Writing and reading
# --------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path, reference = write_run_cost(tmp_path, _document())
    assert path.name == RUN_COST_NAME
    assert reference is None
    assert read_run_cost(tmp_path) == _document()
    assert read_run_cost(path) == _document()


def test_a_store_copy_is_the_same_bytes_as_the_file(tmp_path: Path) -> None:
    """``manifest check`` resolves the store copy, so it must be the file."""
    with open_sqlite(str(tmp_path / "runtime.sqlite")) as store:
        path, reference = write_run_cost(tmp_path, _document(), store=store)
        assert reference is not None
        assert reference.schema == RUN_COST_SCHEMA_NAME
        stored = store.get(reference)
    assert stored == json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# End to end, through a real fake-transport run
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_run_dir(tmp_path_factory) -> Path:
    from whetstone_envs.optim.run import RunSpec, run_optimizer

    output = tmp_path_factory.mktemp("run-cost") / "copro-run"
    return run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            run_id="c19-copro-run-cost",
            output_dir=output,
        )
    )


def test_the_runner_writes_cost_json_beside_result_json(
    fake_run_dir: Path,
) -> None:
    assert (fake_run_dir / "result.json").is_file()
    document = read_run_cost(fake_run_dir)
    assert document.run_id == "c19-copro-run-cost"
    assert {entry.role for entry in document.spend} == {
        "task_model",
        "proposer",
    }


def test_cost_json_is_a_projection_not_a_recomputation(
    fake_run_dir: Path,
) -> None:
    """Every number copies from ``result.json``, which stays the authority."""
    result = json.loads(
        (fake_run_dir / "result.json").read_text(encoding="utf-8")
    )
    cost = result["cost"]
    document = read_run_cost(fake_run_dir)
    by_role = {entry.role: entry for entry in document.spend}
    for role in ("task_model", "proposer"):
        source = cost[role]
        projected = by_role[role]
        assert projected.calls == source["calls"]
        assert projected.cached_calls == source["cached_calls"]
        assert projected.input_tokens == source["input_tokens"]
        assert projected.output_tokens == source["output_tokens"]
        assert projected.priced_calls == source["priced_calls"]
        assert projected.unpriced_calls == source["unpriced_calls"]
        assert (
            projected.rows_missing_token_breakdown
            == source["rows_missing_token_breakdown"]
        )
        assert projected.usd == source["usd"]
    assert document.cost_report_schema_version == cost["schema_version"]


def test_the_runs_cost_document_resolves_in_its_own_store(
    fake_run_dir: Path,
) -> None:
    """The pointer a manifest would cite is already in the run's store."""
    payload = json.loads(
        (fake_run_dir / RUN_COST_NAME).read_text(encoding="utf-8")
    )
    with open_sqlite(str(fake_run_dir / "runtime.sqlite")) as store:
        reference, status = store.put(RUN_COST_SCHEMA_NAME, payload)
        # A second put of identical bytes is a no-op, which is how we know
        # the runner already stored exactly this record.
        assert status.name != "CREATED"
        assert store.get(reference) == payload


def test_the_honesty_split_survives_a_real_unpriced_run(
    fake_run_dir: Path,
) -> None:
    """A fake transport reports no price, so the split is the whole story."""
    by_role = {
        entry.role: entry for entry in read_run_cost(fake_run_dir).spend
    }
    task_model = by_role["task_model"]
    assert task_model.calls > 0
    assert task_model.unpriced_calls == task_model.calls
    assert task_model.usd is None
