from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.core.identity import TypedRef
from whetstone.eval import EvalEvidence
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import OptimResult

from tests.optim.codex_support import (
    CODEX_SPLIT_SIZES,
    codex_output_artifact,
    codex_test_seam,
    codex_tool_steps,
)
from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.cli import main
from whetstone_envs.optim.codex import CODEX_EVALUATE_CALL_CAP
from whetstone_envs.optim.experiment import C19_MUTATION_FIELD
from whetstone_envs.optim.run import RunSpec, run_optimizer
from whetstone_envs.reporting.publication import (
    DurableRunError,
    load_trajectory_report,
)

#: ``miprov2`` and ``gepa`` require an explicit disjoint train/val split of
#: the internal split; the others refuse one. At these fixtures' internal 2
#: the only valid partition is 1/1.
TRAIN_VAL_FLAGS = ("--train-size", "1", "--val-size", "1")


def _split_flags(optimizer: str) -> list[str]:
    """The split flags this optimizer requires, or none."""
    if optimizer in {"gepa", "miprov2"}:
        return list(TRAIN_VAL_FLAGS)
    return []


@pytest.mark.parametrize("optimizer", ["copro", "gepa"])
def test_fake_transport_completes(  # noqa: PLR0915
    tmp_path, optimizer: str
) -> None:
    output = tmp_path / f"{optimizer}-run"
    code = main(
        [
            "--family",
            "c19",
            "--optimizer",
            optimizer,
            "--transport",
            "fake",
            "--split-sizes",
            "2,2,0",
            *_split_flags(optimizer),
            "--run-id",
            f"c19-{optimizer}-e2e",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"
    assert result.proposals
    assert (output / "trajectory-report.json").is_file()
    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    assert all(row.eval_report is not None for row in trajectory.resolutions)

    if optimizer == "gepa":
        from whetstone.optim.gepa.contracts import (
            GepaEffectTranscript,
            GepaEvaluationEffectResult,
        )
        from whetstone.optim.gepa.harness_adapter import (
            GEPA_TERMINAL_ARTIFACT_KEY,
        )
        from whetstone.optim.gepa.result_artifact import GepaRunResultArtifact

        template = result.proposals[0].candidate.record.payload[
            C19_MUTATION_FIELD
        ]
        assert template == PROBES.ceiling_template
        assert template != PROBES.naive_template
        final_step = result.step_results[-1].record
        assert final_step.history_ref is not None
        expected = []
        with open_sqlite(str(output / "runtime.sqlite")) as store:
            history = store.get(final_step.history_ref.reference)
            assert isinstance(history, dict)
            artifact_ref = TypedRef.model_validate(
                history[GEPA_TERMINAL_ARTIFACT_KEY]
            )
            artifact = GepaRunResultArtifact.model_validate_json(
                json.dumps(store.get(artifact_ref.reference))
            )
            transcript = GepaEffectTranscript.model_validate_json(
                json.dumps(store.get(artifact.effect_transcript_ref.reference))
            )
            for entry in transcript.entries:
                if entry.effect_kind != "evaluate":
                    continue
                effect_result = GepaEvaluationEffectResult.model_validate_json(
                    json.dumps(store.get(entry.result_ref.reference))
                )
                assert effect_result.resolution is not None
                resolution = effect_result.resolution
                assert resolution.eval_result_ref is not None
                evidence = EvalEvidence.model_validate_json(
                    json.dumps(store.get(resolution.eval_result_ref.reference))
                )
                expected.append(
                    (
                        int(resolution.optim_eval_request.optim_step_index),
                        entry.invocation_ordinal,
                        resolution.optim_eval_request.eval_request.request_id,
                        resolution.optim_eval_request.eval_request.candidate,
                        evidence.task_hashes,
                    )
                )
        assert expected
        actual = []
        for row in trajectory.resolutions:
            assert row.eval_report is not None
            actual.append(
                (
                    row.step_index,
                    row.resolution_index,
                    row.request_id,
                    row.candidate_ref.content_hash,
                    tuple(task.task_hash for task in row.eval_report.tasks),
                )
            )
        assert tuple(actual) == tuple(
            (
                step_index,
                invocation_ordinal,
                request_id,
                candidate_reference(candidate).record_ref.content_hash,
                task_hashes,
            )
            for (
                step_index,
                invocation_ordinal,
                request_id,
                candidate,
                task_hashes,
            ) in expected
        )
        candidate_counts = Counter(
            candidate_reference(candidate).record_ref.content_hash
            for _step, _ordinal, _request, candidate, _tasks in expected
        )
        assert any(count > 1 for count in candidate_counts.values())
        # Per-step ``search_evidence`` is incremental: a Step records only
        # the evaluations no ancestor Step already recorded, so the run's
        # evidence is linear in evaluations rather than quadratic in Steps.
        # The sum across Steps therefore counts each evaluation exactly
        # once, and the trajectory projects exactly those evaluations as
        # resolutions -- same requests, same order.
        evidence_request_ids = [
            evidence.eval_request_id
            for step in result.step_results
            for evidence in step.record.search_evidence
        ]
        assert len(set(evidence_request_ids)) == len(evidence_request_ids)
        assert [row.request_id for row in trajectory.resolutions] == (
            evidence_request_ids
        )
        # Each entry is bound to the Step that caused it, so a Step never
        # re-reports a prefix its predecessors already paid for.
        for step in result.step_results:
            assert all(
                evidence.optim_step_index == step.record.step_index
                for evidence in step.record.search_evidence
            )
    else:
        assert tuple(step.step_index for step in trajectory.steps) == (0, 1)
        assert tuple(
            (row.step_index, row.resolution_index)
            for row in trajectory.resolutions
        ) == ((0, 0), (0, 1))
        assert any(
            candidate.base_candidate_ref is not None
            for candidate in trajectory.candidates
        )
        assert all(row.gains is None for row in trajectory.resolutions)
        assert all(row.regressions is None for row in trajectory.resolutions)
        assert trajectory.steps[0].budget_delta_consumed == {
            "proposal_calls": 1
        }
        assert trajectory.steps[0].budget_cumulative_consumed == {
            "proposal_calls": 1
        }
        assert trajectory.steps[0].budget_remaining == {"proposal_calls": 0}


def test_projection_failure_preserves_runtime_and_terminal_result(
    tmp_path,
) -> None:
    output = tmp_path / "projection-failure"
    with (
        patch(
            "whetstone_envs.optim.run.project_trajectory_report",
            side_effect=RuntimeError("projection boom"),
        ),
        pytest.raises(DurableRunError, match="projection boom") as captured,
    ):
        run_optimizer(
            RunSpec(
                optimizer="copro",
                transport="fake",
                split_sizes=(2, 2, 0),
                output_dir=output,
                run_id="c19-projection-failure",
            )
        )

    assert captured.value.directory == output.resolve()
    assert (output / "runtime.sqlite").is_file()
    OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert not (output / "trajectory-report.json").exists()


def test_run_refuses_in_repo_output() -> None:
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_optimizer(
            RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=Path("artifacts") / "c19-run",
            )
        )


def test_run_refuses_in_repo_output_when_cwd_is_elsewhere(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_optimizer(
            RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=repo_root / "artifacts" / "c19-run",
            )
        )


@pytest.mark.parametrize("demo_mode", ["fewshot", "zeroshot", "ground_only"])
def test_miprov2_fake_transport_completes(tmp_path, demo_mode: str) -> None:
    """MIPROv2 completes on the fake transport in every demonstration mode.

    Demonstrations reach the candidate through MIPROv2's own composed
    ``### Demonstrations`` section: ``fewshot`` renders the selected demo set
    there, while ``zeroshot`` and ``ground_only`` leave it empty.
    """
    output = tmp_path / f"miprov2-{demo_mode}-run"
    code = main(
        [
            "--family",
            "c19",
            "--optimizer",
            "miprov2",
            "--demo-mode",
            demo_mode,
            "--transport",
            "fake",
            "--split-sizes",
            "2,2,0",
            *TRAIN_VAL_FLAGS,
            "--run-id",
            f"c19-miprov2-{demo_mode}-e2e",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.terminal_failure is None
    assert result.step_results[-1].record.status.value == "complete"
    # MIPROv2 drives its search through Steps, not SearchEvidence.
    assert all(not step.record.search_evidence for step in result.step_results)
    assert any(step.record.proposer_usage for step in result.step_results)

    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    spend = trajectory.spend
    assert spend is not None
    assert spend.proposer.calls > 0
    assert spend.task_model.calls > 0


def test_runner_rejects_unknown_demo_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported demo mode"):
        run_optimizer(
            RunSpec(
                optimizer="miprov2",
                transport="fake",
                output_dir=tmp_path / "bad-demo-mode",
                demo_mode="handful",
                train_size=1,
                val_size=1,
            )
        )


def test_runner_rejects_non_positive_num_seeds(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_seeds must be at least 1"):
        run_optimizer(
            RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=tmp_path / "bad-num-seeds",
                num_seeds=0,
            )
        )


@pytest.mark.parametrize("optimizer", ["copro", "gepa"])
def test_fake_transport_reports_run_spend(tmp_path, optimizer: str) -> None:
    """Every optimizer's trajectory report carries the run's spend."""
    output = tmp_path / f"{optimizer}-spend"
    assert (
        main(
            [
                "--optimizer",
                optimizer,
                "--transport",
                "fake",
                *_split_flags(optimizer),
                "--run-id",
                f"c19-{optimizer}-spend",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    trajectory = load_trajectory_report(output)
    spend = trajectory.spend
    assert spend is not None
    assert spend.task_model.role == "task_model"
    assert spend.proposer.role == "proposer"
    for role in (spend.task_model, spend.proposer):
        assert role.priced_calls + role.unpriced_calls == role.calls
        # A fake transport reports no price, so no total is presented.
        if role.unpriced_calls:
            assert role.usd is None


#: Tables the effect lease authority owns inside the run's ``runtime.sqlite``.
#: The object store owns ``objects``/``bindings`` in the same file; the two
#: components never collide, so one database carries both.
LEASE_TABLE = "dr_store_lease_authority"
LEASE_METADATA_TABLE = "dr_store_lease_authority_metadata"


def _lease_rows(database: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(
            f"SELECT * FROM {LEASE_TABLE} ORDER BY semantic_key"  # noqa: S608
        ).fetchall()


def _table_names(database: Path) -> set[str]:
    with closing(sqlite3.connect(database)) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


def test_fake_run_persists_effect_leases_beside_the_store(tmp_path) -> None:
    """The run's effect leases outlive the process that created them.

    A memory authority would drop every terminal at process exit; the sqlite
    authority writes them into the run directory's own ``runtime.sqlite``.
    """
    output = tmp_path / "lease-durability"
    assert (
        main(
            [
                "--optimizer",
                "copro",
                "--transport",
                "fake",
                "--split-sizes",
                "2,2,0",
                "--run-id",
                "c19-lease-durability",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    database = output / "runtime.sqlite"
    assert database.is_file()
    # Both components share the file without colliding.
    assert {
        LEASE_TABLE,
        LEASE_METADATA_TABLE,
        "objects",
        "bindings",
    } <= _table_names(database)
    assert _lease_rows(database), "the run recorded no effect leases"


def test_rerun_against_the_same_output_replays_recorded_effects(
    tmp_path,
) -> None:
    """A second run over a completed directory replays instead of re-running.

    Every lease row -- including ``attempt_id`` and ``fence`` -- is unchanged,
    which a re-execution could not produce: acquiring an effect afresh mints a
    new attempt and bumps the fence. The replay itself is whetstone-ai's
    guarantee; what this pins is that the runner hands it a durable authority
    so the guarantee survives a process boundary.
    """
    output = tmp_path / "lease-replay"
    argv = [
        "--optimizer",
        "copro",
        "--transport",
        "fake",
        "--split-sizes",
        "2,2,0",
        "--run-id",
        "c19-lease-replay",
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    database = output / "runtime.sqlite"
    before = _lease_rows(database)
    assert before
    first_result = (output / "result.json").read_text(encoding="utf-8")

    assert main(argv) == 0
    assert _lease_rows(database) == before
    assert (output / "result.json").read_text(encoding="utf-8") == first_result


# --------------------------------------------------------------------------
# The Codex arm, driven by the scripted fake CLI
# --------------------------------------------------------------------------

requires_codex_sandbox = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)


def _codex_run(  # noqa: PLR0913
    *,
    tmp_path: Path,
    run_id: str,
    templates: tuple[str, ...],
    selected: str | None,
    capacity: int | None = CODEX_EVALUATE_CALL_CAP,
    family: str = "c19",
    n_per_stratum: int | None = None,
) -> Path:
    """One Codex arm run over one family, agent decisions scripted."""
    return run_optimizer(
        RunSpec(
            optimizer="codex",
            transport="fake",
            family=family,
            split_sizes=CODEX_SPLIT_SIZES,
            output_dir=tmp_path / run_id,
            run_id=run_id,
            codex_capacity=capacity,
            n_per_stratum=n_per_stratum,
        ),
        codex_test_seam=codex_test_seam(
            steps=codex_tool_steps(
                templates=templates,
                selected=selected,
                scratch=tmp_path,
                family_id=family,
                split_sizes=CODEX_SPLIT_SIZES,
            ),
            binary_dir=tmp_path / "codex-bin",
        ),
    )


@requires_codex_sandbox
def test_codex_fake_cli_completes(tmp_path) -> None:
    """The Codex arm end to end, on the real admission and ledger path.

    Only the agent's decisions are scripted: the fake CLI is a real
    subprocess that speaks real MCP over HTTP to the whetstone-hosted
    evaluation server, so admission, leasing, evaluation, and the ledger
    are the production ones.
    """
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e",
        templates=(PROBES.ceiling_template,),
        selected="c1",
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    # Codex is TOOL_USING: its paid evaluations are cited from tool
    # evidence, and it resolves no intent and mints no search evidence.
    assert step.resolved_intents == ()
    assert step.search_evidence == ()
    assert len(step.tool_evidence) == 1
    assert step.budget_delta.consumed["tool_calls"] == 1
    # The accepted candidate is rebuilt from the recorded tool call's
    # arguments, not from anything the agent's artifact asserted.
    accepted = result.proposals[0].candidate.record.payload
    assert accepted[C19_MUTATION_FIELD] == PROBES.ceiling_template


@requires_codex_sandbox
def test_codex_two_evaluations_debit_the_whole_budget(tmp_path) -> None:
    """Every admitted call is on the ledger and debits the step budget.

    Under-reporting is a terminal failure upstream, so a completing run
    is itself the proof that what the agent reported and what the run
    durably admitted agree.
    """
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-two",
        templates=(PROBES.naive_template, PROBES.ceiling_template),
        selected="c2",
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    assert len(step.tool_evidence) == 2
    assert step.budget_delta.consumed["tool_calls"] == 2
    assert {
        str(entry.store_entry.call_id) for entry in step.tool_evidence
    } == {"c1", "c2"}
    accepted = result.proposals[0].candidate.record.payload
    assert accepted[C19_MUTATION_FIELD] == PROBES.ceiling_template


@requires_codex_sandbox
def test_codex_selecting_nothing_retains_the_seed(tmp_path) -> None:
    """An honest "nothing beat the seed" keeps the seed and its evidence."""
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-seed",
        templates=(PROBES.naive_template,),
        selected=None,
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    assert step.seed_retained is True
    assert step.accepted_candidates == ()
    assert (
        step.retained_candidate_ref == result.run.record.initial_candidate_ref
    )
    # The evaluation it did buy is still reachable and still debited.
    assert len(step.tool_evidence) == 1
    assert step.budget_delta.consumed["tool_calls"] == 1


@requires_codex_sandbox
def test_codex_trajectory_report_renders_the_tool_evaluations(
    tmp_path,
) -> None:
    """A Codex run's report must show the evaluations it bought.

    Codex resolves no intent, so a projection reading only the intent
    path renders a Codex run as having evaluated nothing -- a terminal
    candidate with no measurement behind it, and a report that silently
    understates what the arm did. Its paid evaluations are cited from
    tool evidence instead, and each becomes a trajectory row with its own
    embedded evaluation.
    """
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-report",
        templates=(PROBES.naive_template, PROBES.ceiling_template),
        selected="c2",
    )
    trajectory = load_trajectory_report(output)

    assert trajectory.terminal_status == "complete"
    assert len(trajectory.resolutions) == 2
    assert [row.request_id for row in trajectory.resolutions] == [
        "tool:c1",
        "tool:c2",
    ]
    assert all(row.outcome == "completed" for row in trajectory.resolutions)
    # Every row carries its own embedded evaluation, and each is
    # attributed to the candidate that tool call actually built -- the
    # ceiling template scores above the naive one.
    assert all(row.eval_report is not None for row in trajectory.resolutions)
    rewards = [row.reward for row in trajectory.resolutions]
    assert all(reward is not None for reward in rewards)
    naive, ceiling = rewards
    assert naive is not None
    assert ceiling is not None
    assert ceiling > naive
    assert trajectory.steps[0].resolution_indexes == (1, 2)


@requires_codex_sandbox
def test_codex_reports_run_spend_from_tool_evidence(tmp_path) -> None:
    """A Codex run's whole spend is task-model spend, driven through tools.

    Nothing is attributed to a proposer -- the agent proposes, and its own
    model spend is not on the study's key (OQ1). An aggregator reading
    only the intent path would report a Codex run as having cost nothing,
    so this asserts the task-model side is non-zero.
    """
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-spend",
        templates=(PROBES.ceiling_template,),
        selected="c1",
    )
    cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
    by_role = {row["role"]: row for row in cost["spend"]}
    # The two roles the cost report knows. Per OQ1 there is no
    # ``codex_agent`` role: the agent's own model runs on the Codex
    # subscription rather than the study's key, so whetstone has no
    # evidence to price it with and does not invent a field it cannot
    # populate.
    assert set(by_role) == {"task_model", "proposer"}
    assert by_role["task_model"]["calls"] > 0
    assert by_role["proposer"]["calls"] == 0


@requires_codex_sandbox
def test_codex_audit_passes_on_a_fresh_run(tmp_path) -> None:
    """Every Codex invariant, plus the shared one, on a run made just now.

    The committed fixtures prove the invariants against evidence produced
    once; this proves them against evidence produced by the code under
    test, which is what keeps them regression tests rather than a
    snapshot of one historical run.
    """
    from whetstone_envs.optim.audit.registry import audit_run

    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-audit",
        templates=(PROBES.ceiling_template,),
        selected="c1",
    )
    report = audit_run(output)
    assert report.optimizer == "codex"
    assert report.passed, [
        (finding.invariant_id, finding.detail)
        for finding in report.findings
        if finding.status.value == "fail"
    ]


@requires_codex_sandbox
def test_codex_runs_the_second_family_unchanged(tmp_path) -> None:
    """C3 generality: the Codex arm names no family of its own.

    Everything family-specific -- the render contract, the mutation
    field, the task set the one Tool evaluates, and the experiment the
    out-of-process MCP server rebuilds -- is read from the family
    registry, so c18 runs through the identical path with no code change
    and audits the same way.

    The *projected report* is asserted here alongside the audit, because
    the two check different things and a passing audit does not imply a
    publishable report. The report's embedded ``EvalReport`` re-derives
    every observation's score to validate it, and doing that with a
    family-agnostic rule is what once made a c18 Codex run unpublishable
    while its audit still passed.
    """
    from whetstone_envs.c18 import PROBES as C18_PROBES
    from whetstone_envs.optim.audit.registry import audit_run
    from whetstone_envs.reporting.publication import load_trajectory_report

    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c18-codex-e2e",
        templates=(C18_PROBES.ceiling_template,),
        selected="c1",
        family="c18",
        n_per_stratum=1,
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None, result.terminal_failure
    assert len(result.step_results[-1].record.tool_evidence) == 1
    report = audit_run(output)
    assert report.optimizer == "codex"
    assert report.passed, [
        (finding.invariant_id, finding.detail)
        for finding in report.findings
        if finding.status.value == "fail"
    ]

    # Reading it back re-validates it: publication writes the document,
    # and the loader re-runs every schema invariant over the persisted
    # bytes, so a report that only validated in memory does not pass.
    trajectory = load_trajectory_report(output)
    embedded = [
        resolution.eval_report
        for resolution in trajectory.resolutions
        if resolution.eval_report is not None
    ]
    assert embedded, (
        "the c18 Codex run published no embedded evaluation report, so "
        "nothing exercised the schema's per-family score check"
    )
    assert all(eval_report.run.family == "c18" for eval_report in embedded), [
        eval_report.run.family for eval_report in embedded
    ]
    assert any(eval_report.observations for eval_report in embedded), (
        "no embedded report carried a scored observation"
    )


@requires_codex_sandbox
def test_codex_cli_end_to_end(tmp_path) -> None:
    """The CLI admits ``--optimizer codex`` and its flags reach the run.

    The CLI cannot build a test seam, by design, so this drives the real
    argument path as far as it goes and stops where a real Codex session
    would be required -- the preflight, which is exactly the gate that
    must not be bypassable from the command line.
    """
    from whetstone_envs.optim.codex import CODEX_DEFAULT_BINARY
    from whetstone_envs.optim.run import RunSpec as _RunSpec

    captured: list[_RunSpec] = []

    def capture(spec, **kwargs):
        captured.append(spec)
        assert kwargs == {}, "the CLI must pass no test seam"
        return tmp_path / "unused"

    with patch("whetstone_envs.optim.cli.run_optimizer", capture):
        code = main(
            [
                "--optimizer",
                "codex",
                "--transport",
                "fake",
                "--codex-capacity",
                "4",
                "--run-id",
                "c19-codex-cli",
                "--output",
                str(tmp_path / "cli-run"),
            ]
        )

    assert code == 0
    spec = captured[0]
    assert spec.optimizer == "codex"
    assert spec.codex_capacity == 4
    assert spec.codex_binary == CODEX_DEFAULT_BINARY


# --------------------------------------------------------------------------
# The prompt-builder diagnostic seam
# --------------------------------------------------------------------------


@requires_codex_sandbox
def test_codex_prompt_builder_replaces_the_agents_instruction(
    tmp_path,
) -> None:
    """The ladder's steering seam actually reaches the spawned CLI.

    The real-CLI ladder's capacity and no-tool-call rungs cannot observe
    what they assert under the truthful production prompt: an agent
    correctly told it may make one call makes one call, and the durable
    refusal path is never exercised. So they replace the prompt -- and a
    seam that silently did not reach the process would make those rungs
    pass while observing an obedient agent instead of the behavior under
    test.

    The fake CLI echoes the prompt it received back through
    ``conversation_evidence``, so this asserts on what the runner actually
    emitted rather than on the builder having been called.
    """
    from whetstone.testing.fake_codex_cli import FAKE_CODEX_PROMPT_EVIDENCE_KEY

    from whetstone_envs.optim.audit.registry import audit_run

    marker = "LADDER-STEERING-MARKER"
    seen: list[str] = []

    def prompt_builder(context) -> str:
        # A builder inherits the default's obligations: model_route and
        # base_ref are the two values the agent can derive from nothing
        # it can see, so a builder that dropped them would have every
        # call refused after admission.
        seen.append(context.model_route)
        return (
            f"{marker}\n"
            f"Use only the external {context.tool_name} MCP tool.\n"
            "The model_route argument is a fixed string and must be "
            f"exactly {context.model_route!r}.\n"
            "The base_ref argument must be copied verbatim as "
            f"{context.base_ref}.\n"
            "Copy lease_token_hash verbatim as "
            f"{context.lease_token_hash!r}.\n"
        )

    output = run_optimizer(
        RunSpec(
            optimizer="codex",
            transport="fake",
            family="c19",
            split_sizes=CODEX_SPLIT_SIZES,
            output_dir=tmp_path / "prompt-seam",
            run_id="c19-codex-prompt-seam",
            codex_capacity=CODEX_EVALUATE_CALL_CAP,
        ),
        codex_test_seam=codex_test_seam(
            steps=codex_tool_steps(
                templates=(PROBES.ceiling_template,),
                selected="c1",
                scratch=tmp_path,
            ),
            binary_dir=tmp_path / "codex-bin",
        ),
        codex_prompt_builder=prompt_builder,
    )

    assert seen, "the prompt builder was never invoked"
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    # The steered run still completed on the production path.
    assert len(step.tool_evidence) == 1
    # The prompt the runner actually emitted, echoed back by the fake CLI
    # and carried into the artifact's process evidence under "agent".
    artifact = codex_output_artifact(output)
    assert artifact is not None, "the run recorded no Codex output artifact"
    agent_evidence = artifact.conversation_evidence["agent"]
    emitted = agent_evidence[FAKE_CODEX_PROMPT_EVIDENCE_KEY]
    assert marker in emitted, (
        "the replaced prompt did not reach the spawned CLI; the ladder's "
        "steering rungs would silently observe the production prompt"
    )
    # A steered run is still a well-formed run: the seam changes what the
    # agent is told, not what the arm is held to.
    report = audit_run(output)
    assert report.passed, [
        (finding.invariant_id, finding.detail)
        for finding in report.findings
        if finding.status.value == "fail"
    ]


def test_the_prompt_builder_applies_only_to_the_codex_optimizer(
    tmp_path,
) -> None:
    """A steering seam on another arm is a mistake, not a no-op.

    Mirrors ``codex_test_seam``'s own guard: an argument that silently did
    nothing would let a caller believe it had changed a COPRO run's
    instruction.
    """
    with pytest.raises(ValueError, match="only to --optimizer codex"):
        run_optimizer(
            RunSpec(
                optimizer="copro",
                transport="fake",
                family="c19",
                split_sizes=CODEX_SPLIT_SIZES,
                output_dir=tmp_path / "wrong-arm",
                run_id="c19-copro-prompt-seam",
            ),
            codex_prompt_builder=lambda _context: "unused",
        )


@requires_codex_sandbox
def test_codex_a_call_rejected_after_admission_still_publishes(
    tmp_path,
) -> None:
    """A paid-but-unevaluated call must not take down the whole run.

    Found by the real-CLI ladder. The evaluator admits a call and *then*
    validates it, so an agent that submits a template the family's render
    contract does not accept -- here a field c19 does not expose -- gets
    ``tool_evaluation_rejected`` after admission: capacity is debited and
    no ``EvalEvidence`` is ever minted.

    The projection called that ``failed``, which the trajectory schema
    defines as "an evaluation ran and ended badly" and therefore requires
    an eval result for. The row could not validate, publication raised,
    and ``durable_run_boundary`` turned one wasted tool call into a lost
    run -- no ``result.json``, no report, nothing to audit. A real §6 run
    would lose everything the agent had already bought.

    The fake CLI is *handed* its tool arguments, so no scripted transcript
    reaches this state by accident; this one constructs it deliberately.
    """
    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-rejected",
        # ``{prompt}`` is not a c19 field: the render contract exposes
        # grid/command/question. Admitted, then rejected.
        templates=("Answer {prompt} briefly.",),
        selected=None,
    )

    # The run published at all -- this is the regression.
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    step = result.step_results[-1].record
    # The call was paid for: it debited capacity despite evaluating
    # nothing, which is exactly why it must appear in the report.
    assert len(step.tool_evidence) == 1
    assert step.budget_delta.consumed["tool_calls"] == 1
    evidence = step.tool_evidence[0]
    assert evidence.result.record.terminal_failure is not None
    assert evidence.result.record.evaluation_evidence_refs == ()

    trajectory = load_trajectory_report(output)
    assert len(trajectory.resolutions) == 1
    row = trajectory.resolutions[0]
    # ``rejected``, not ``failed``: no evaluation happened, so the row
    # carries no eval result -- the same distinction the intent path
    # draws between the two outcomes.
    assert row.outcome == "rejected"
    assert row.eval_result_ref is None
    assert row.eval_report is None
    assert row.reward is None
    assert row.terminal_failure is None
    # The *reason* survives. The structured failure is evidence the
    # schema forbids on a rejected row, but the message is not, and it is
    # the only account of why a paid-for call scored nothing that reaches
    # the projected trajectory. Dropping it left the row saying a call was
    # rejected and giving the reader no way to find out what went wrong.
    assert row.message
    assert row.message == evidence.result.record.terminal_failure.message


@requires_codex_sandbox
def test_codex_run_supplies_l1_leakage_evidence(tmp_path) -> None:
    """A Codex run must be visible to the study's L1 leakage rule.

    Found by the real-CLI ladder's study rung. L1 asks "did the optimizer
    see anything but the internal split?", and it read its evidence from
    each run's ``resolved_intents``. Codex is ``TOOL_USING``: it resolves
    no intent and mints no search evidence *by design*, so that read
    returned nothing for a Codex run.

    L1 then reports itself **unchecked** rather than passed -- correctly,
    since a vacuous pass is not a check -- which fails ``leakage-check``
    and blocks the whole study from reporting. A study whose only real
    optimizer arm is Codex could therefore never publish, even though the
    evidence L1 wants was sitting in ``tool_evidence`` the whole time.

    The two fields come from different records on this path: the role
    from the ``EvalEvidence`` the Tool Result cites, and the Eval Config
    from the Tool Config the call was admitted against.
    """
    from whetstone_envs.optim.study.leakage import (
        INTERNAL_ROLE,
        check_l1_optimizer_internal_only,
        optimizer_observations_from_run,
    )

    output = _codex_run(
        tmp_path=tmp_path,
        run_id="c19-codex-e2e-l1",
        templates=(PROBES.naive_template, PROBES.ceiling_template),
        selected="c2",
    )

    observations = optimizer_observations_from_run(output)
    assert observations, (
        "a Codex run supplied no L1 evidence, so L1 reports itself "
        "unchecked and the study cannot report"
    )
    # One observation per admitted, scored evaluation.
    assert len(observations) == 2
    assert {row.eval_role for row in observations} == {INTERNAL_ROLE}

    # And the rule itself passes against the config identity the *study
    # manifest* records, rather than merely finding rows.
    #
    # Which hash that is matters, and getting it wrong fails in the
    # direction that looks like a leak. The manifest records
    # ``EvalConfigRef.config_hash`` -- the config's own identity -- while
    # the Tool Config carries a ``TypedRef`` content hash over the stored
    # record. They are two hashes of the same config, so reading the
    # convenient one reports every honest evaluation as having left the
    # internal split. This asserts against the identity a real study
    # compares to.
    from whetstone_envs.optim.audit._evidence import load_run_evidence

    evidence = load_run_evidence(output)
    first = next(
        found
        for step in evidence.steps
        for entry in step.tool_evidence
        for reference in entry.result.record.evaluation_evidence_refs
        if (found := evidence.eval_evidence(reference)) is not None
    )
    internal_config = str(first.eval_config_ref.config_hash)
    assert all(
        row.resolved_eval_config_hash == internal_config
        for row in observations
    )
    # Containment is checked against the internal split the evidence
    # itself names, which is what a manifest records for a run whose
    # evaluations cover the whole split.
    finding = check_l1_optimizer_internal_only(
        observations,
        internal_eval_config_hash=internal_config,
        internal_task_hashes=first.task_hashes,
        excluded_eval_config_hashes=(),
    )
    assert finding.passed, finding.detail
