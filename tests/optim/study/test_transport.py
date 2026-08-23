"""The transport is an invocation property the study records and checks.

Everything here runs on the fake transport or refuses before it reaches a
provider, so the whole module is free. That is deliberate: the refusals are
what a paid run depends on, and a guard only exercised by paying for it is
not a guard.

Three properties, each with its own failure mode if it drifts:

* ``--transport openrouter`` without a key is refused **before** anything is
  opened -- no store, no pool, no engine, no provider call -- so a study
  directory is never left half-initialized by an unauthorized run.
* Every stage records the transport it ran on, because a number measured
  against the experiment's own gold and a number measured against a
  provider are different evidence for the same claim.
* A stage whose transport disagrees with the anchors' is refused, because
  every held-out delta is paired against those anchors and a cross-transport
  subtraction is not a comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_providers import ProviderKind
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.optim.study.cli import (
    EXIT_CHECK_FAILED,
    EXIT_ERROR,
    EXIT_OK,
    NO_RECORDED_SPEND,
    NO_STAGES_RUN,
    STAGE_SPEND_HEADING,
    UNLEDGERED_SPEND,
    main,
    stage_spend_lines,
)
from whetstone_envs.optim.study.environment import (
    FAKE_TRANSPORT,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_TRANSPORT,
    SPLIT_ROLE_BY_EVAL_ROLE,
    bound_stage_environment,
    require_transport_credentials,
)
from whetstone_envs.optim.study.manifest import (
    AMENDMENT_REASON_TRANSPORT_CHANGE,
    DISCARD_STALE_RUNS_FLAG,
    PROVIDER_CONTROL_UNSET,
    STUDY_MANIFEST_NAME,
    STUDY_STORE_NAME,
    ArmRecord,
    EvidencePointer,
    LeakageCheckEntry,
    LeakageCheckRecord,
    RunRecord,
    RunSpendRecord,
    StageId,
    StageRecord,
    StudyManifest,
    TransportName,
    read_study_manifest,
    recorded_transport,
    write_study_manifest,
)
from whetstone_envs.optim.study.spec import StageId as SpecStageId
from whetstone_envs.optim.study.spend import (
    ReportSpendLedger,
    run_spend_records,
)
from whetstone_envs.optim.study.stages import (
    StageEnvironment,
    StageError,
    _arm_stage_record,
    _executed_run_spend,
    _stage_record,
    require_matching_transport,
)
from whetstone_envs.reporting.study_report import (
    NO_PROVIDER_STAGE_DETAIL,
    STAGE_SPEND_COVERAGE_NOTE,
    UNLEDGERED_STAGE_DETAIL,
    study_leakage_failed,
)

from .conftest import toy_manifest

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.schema import EvalEvidence
    from whetstone.experiment.candidate import Candidate

    from whetstone_envs.optim.study.anchors import EngineBinder

#: One real optimizer and one control, matching the Stage-1/2 end-to-end
#: fixture: the cross-transport refusal has to fire on a stage that would
#: otherwise really run arms.
TRANSPORT_ARMS = ("copro", "null-identity")


def _arms() -> tuple[ArmRecord, ...]:
    return tuple(
        ArmRecord(
            arm_id=arm_id,
            optimizer=arm_id,
            demo_mode=None,
            train_size=None,
            val_size=None,
            control_identity_hash=chr(ord("a") + index) * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        )
        for index, arm_id in enumerate(TRANSPORT_ARMS)
    )


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    """A study directory holding a pre-Stage-0 toy manifest."""
    directory = tmp_path / "study"
    write_study_manifest(directory, toy_manifest(arms=_arms()))
    return directory


@pytest.fixture
def no_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the paid transport has no credential in this test.

    Deleted rather than assumed absent: this suite must behave the same
    whether or not the developer running it has a key exported, and a test
    that silently passed because a key happened to be missing would stop
    testing anything on the machine that matters.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)


# --------------------------------------------------------------------------
# The credential refusal
# --------------------------------------------------------------------------


def test_a_paid_transport_without_a_key_is_refused(
    no_openrouter_key: None,  # noqa: ARG001
) -> None:
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        require_transport_credentials(OPENROUTER_TRANSPORT)


def test_a_blank_key_counts_as_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty variable is the shell's usual way to unset one.

    Treating it as present would send an empty bearer token to the provider
    and turn a local configuration mistake into a remote 401.
    """
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "   ")
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        require_transport_credentials(OPENROUTER_TRANSPORT)


def test_the_fake_transport_needs_no_key(
    no_openrouter_key: None,  # noqa: ARG001
) -> None:
    require_transport_credentials(FAKE_TRANSPORT)


def test_an_unknown_transport_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unsupported transport"):
        require_transport_credentials("anthropic")


def test_the_refusal_never_echoes_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong key is the provider's refusal to make, not this one's.

    The guard checks presence only, so there is no path on which a
    credential could reach an error message -- and an unknown transport
    must not print one either.
    """
    secret = "sk-or-v1-not-a-real-key"  # noqa: S105
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, secret)
    with pytest.raises(ValueError) as excinfo:
        require_transport_credentials("anthropic")
    assert secret not in str(excinfo.value)


def test_binding_refuses_before_it_writes_anything(
    study_dir: Path,
    no_openrouter_key: None,  # noqa: ARG001
) -> None:
    """The whole point of checking first: nothing is left behind.

    ``bound_stage_environment`` opens a sqlite store in the study
    directory. A refusal that happened after that would leave a store file
    for a stage that never ran, which is the state hardest to tell apart
    from a crashed one.
    """
    store = study_dir / STUDY_STORE_NAME
    assert not store.exists()
    with (
        pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV),
        bound_stage_environment(study_dir, transport=OPENROUTER_TRANSPORT),
    ):
        pytest.fail("the binder must refuse before it yields")
    assert not store.exists()


def test_the_cli_refuses_a_keyless_paid_stage_nonzero(
    study_dir: Path,
    capsys: pytest.CaptureFixture[str],
    no_openrouter_key: None,  # noqa: ARG001
) -> None:
    """Exit non-zero and say why, with no manifest written.

    A caller scripting the three stages reads a zero as "this stage ran",
    so a refused paid stage must never produce one.
    """
    code = main(
        [
            "run",
            "--study-dir",
            str(study_dir),
            "--stage",
            StageId.STAGE0.value,
            "--transport",
            OPENROUTER_TRANSPORT,
        ]
    )
    assert code == EXIT_ERROR
    assert OPENROUTER_API_KEY_ENV in capsys.readouterr().err
    assert read_study_manifest(study_dir).design is None


# --------------------------------------------------------------------------
# What a stage records
# --------------------------------------------------------------------------


def _run_stage(study_dir: Path, stage: str, transport: str) -> int:
    return main(
        [
            "run",
            "--study-dir",
            str(study_dir),
            "--stage",
            stage,
            "--transport",
            transport,
        ]
    )


def test_stage0_records_the_transport_it_ran_on(study_dir: Path) -> None:
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    manifest = read_study_manifest(study_dir)
    assert recorded_transport(manifest.stages, StageId.STAGE0) == (
        FAKE_TRANSPORT
    )


def test_the_transport_defaults_to_fake_with_no_flag(
    study_dir: Path,
) -> None:
    """No flag reaches no provider. The default is the safety property."""
    assert (
        main(
            [
                "run",
                "--study-dir",
                str(study_dir),
                "--stage",
                StageId.STAGE0.value,
            ]
        )
        == EXIT_OK
    )
    manifest = read_study_manifest(study_dir)
    assert recorded_transport(manifest.stages, StageId.STAGE0) == (
        FAKE_TRANSPORT
    )


def test_every_stage_records_its_own_transport(study_dir: Path) -> None:
    for stage in (StageId.STAGE0, StageId.STAGE1, StageId.STAGE2):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    manifest = read_study_manifest(study_dir)
    assert {entry.stage for entry in manifest.stages} == {
        stage.value for stage in StageId
    }
    assert {entry.transport for entry in manifest.stages} == {FAKE_TRANSPORT}


def test_a_fake_stage_records_no_spend(study_dir: Path) -> None:
    """A stage that reached no provider owes no bill.

    The fake transport's rows are real rows -- a generation happened and
    the row proves it -- so the shared row rule counts them as billable and
    unpriced, which is right for a provider row and wrong for a stage that
    never called one. Recording "N unpriced calls" here would be a bill
    nobody owes.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    stages = read_study_manifest(study_dir).stages
    assert [entry.spend for entry in stages] == [()]


def test_a_stage_is_recorded_once_however_often_it_runs(
    study_dir: Path,
) -> None:
    """A re-run replaces its record rather than appending a second one.

    Two records for one stage would leave the study unable to say which
    transport its numbers came from, which is exactly the ambiguity the
    block exists to remove.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    assert (
        main(
            [
                "run",
                "--study-dir",
                str(study_dir),
                "--stage",
                StageId.STAGE0.value,
                "--replace-design",
            ]
        )
        == EXIT_OK
    )
    stages = read_study_manifest(study_dir).stages
    assert [entry.stage for entry in stages] == [StageId.STAGE0.value]


def test_the_manifest_refuses_two_records_for_one_stage(
    study_dir: Path,
) -> None:
    """The structural half of the same rule, stated on the model itself."""
    payload = read_study_manifest(study_dir).model_dump(
        mode="json", by_alias=True
    )
    payload["stages"] = [
        {
            "stage": StageId.STAGE0.value,
            "transport": FAKE_TRANSPORT,
            "spend": [],
        },
        {
            "stage": StageId.STAGE0.value,
            "transport": OPENROUTER_TRANSPORT,
            "spend": [],
        },
    ]
    with pytest.raises(ValueError, match="each stage is recorded at most"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_a_stage_record_refuses_an_unknown_transport() -> None:
    with pytest.raises(ValueError, match="a stage record names one of"):
        StageRecord(stage=StageId.STAGE0.value, transport="anthropic")


def test_a_stage_record_refuses_an_unknown_stage() -> None:
    with pytest.raises(ValueError, match="a stage record names one of"):
        StageRecord(stage="stage3", transport=FAKE_TRANSPORT)


# --------------------------------------------------------------------------
# The cross-stage refusal
# --------------------------------------------------------------------------


def _manifest_anchored_on(transport: str) -> StudyManifest:
    return toy_manifest(arms=_arms()).model_copy(
        update={
            "stages": (
                StageRecord(stage=StageId.STAGE0.value, transport=transport),
            )
        }
    )


@pytest.mark.parametrize(
    ("anchored", "requested"),
    [
        (FAKE_TRANSPORT, OPENROUTER_TRANSPORT),
        (OPENROUTER_TRANSPORT, FAKE_TRANSPORT),
    ],
)
@pytest.mark.parametrize("stage", [StageId.STAGE1, StageId.STAGE2])
def test_an_arm_stage_refuses_the_other_transport(
    anchored: str, requested: str, stage: StageId
) -> None:
    """Both directions, because both produce the same meaningless delta."""
    with pytest.raises(StageError) as excinfo:
        require_matching_transport(
            _manifest_anchored_on(anchored),
            stage=stage,
            transport=requested,
        )
    message = str(excinfo.value)
    # The message names the *recorded* transport, which is the fact the
    # operator does not have in front of them.
    assert anchored in message
    assert stage.value in message


def test_a_matching_transport_is_allowed() -> None:
    require_matching_transport(
        _manifest_anchored_on(FAKE_TRANSPORT),
        stage=StageId.STAGE1,
        transport=FAKE_TRANSPORT,
    )


def test_a_study_with_no_recorded_anchors_has_nothing_to_disagree_with() -> (
    None
):
    """An arm stage on a pre-v5 manifest is not refused for silence.

    The stage still refuses for want of a design; being unable to name the
    anchors' transport is not itself a contradiction, and treating it as
    one would make a manifest written before this block existed unusable.
    """
    require_matching_transport(
        toy_manifest(arms=_arms()),
        stage=StageId.STAGE1,
        transport=OPENROUTER_TRANSPORT,
    )


def test_stage0_is_not_checked_against_itself() -> None:
    """Re-calibrating on the other transport is the sanctioned path."""
    require_matching_transport(
        _manifest_anchored_on(FAKE_TRANSPORT),
        stage=StageId.STAGE0,
        transport=OPENROUTER_TRANSPORT,
    )


def test_the_cli_refuses_a_mismatched_arm_stage(
    study_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end, and *before* any arm runs.

    The key is set so the refusal under test is the transport mismatch and
    not the credential check; nothing is spent, because the stage never
    reaches an arm.
    """
    monkeypatch.setenv(
        OPENROUTER_API_KEY_ENV,
        "sk-or-v1-unused-by-this-test",
    )
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    code = _run_stage(study_dir, StageId.STAGE1.value, OPENROUTER_TRANSPORT)
    assert code == EXIT_CHECK_FAILED
    error = capsys.readouterr().err
    assert FAKE_TRANSPORT in error
    # No arm ran: the refusal is before the spend, not after it.
    assert all(not arm.runs for arm in read_study_manifest(study_dir).arms)


# --------------------------------------------------------------------------
# The per-stage spend ledger
# --------------------------------------------------------------------------


def _spend(*, calls: int, usd: float | None) -> RunSpendRecord:
    priced = calls if usd is not None else 0
    return RunSpendRecord(
        role="task_model",
        calls=calls,
        cached_calls=0,
        input_tokens=600 * calls,
        output_tokens=30 * calls,
        priced_calls=priced,
        unpriced_calls=calls - priced,
        rows_missing_token_breakdown=0,
        usd=usd,
    )


def test_the_ledger_says_so_when_no_stage_has_run() -> None:
    text = "\n".join(stage_spend_lines(()))
    assert STAGE_SPEND_HEADING in text
    assert NO_STAGES_RUN in text


def test_the_ledger_prints_calls_tokens_and_usd() -> None:
    record = StageRecord(
        stage=StageId.STAGE0.value,
        transport=OPENROUTER_TRANSPORT,
        spend=(_spend(calls=112, usd=0.001234),),
    )
    text = "\n".join(stage_spend_lines((record,)))
    assert OPENROUTER_TRANSPORT in text
    assert "112" in text
    # 112 * 630 tokens, thousands-separated as the ledger prints it.
    assert "70,560" in text
    assert "$0.001234" in text


def test_an_unpriced_stage_reports_no_total() -> None:
    """An absent total is "not knowable", never "zero".

    The same rule ``RunSpendRecord`` enforces per role, applied to the
    stage: a sum over the priced share alone would look authoritative while
    understating what the stage cost.
    """
    record = StageRecord(
        stage=StageId.STAGE0.value,
        transport=OPENROUTER_TRANSPORT,
        spend=(_spend(calls=8, usd=None),),
    )
    text = "\n".join(stage_spend_lines((record,)))
    assert "unpriced (8/8 calls)" in text
    assert "$" not in text


def test_a_stage_with_no_spend_records_says_so_rather_than_zero() -> None:
    record = StageRecord(stage=StageId.STAGE1.value, transport=FAKE_TRANSPORT)
    text = "\n".join(stage_spend_lines((record,)))
    assert NO_RECORDED_SPEND in text
    assert "0" not in text.replace(STAGE_SPEND_HEADING, "")


def test_the_ledger_is_ordered_by_stage_not_by_storage() -> None:
    """A re-run Stage 0 must not sort after the stages that followed it."""
    out_of_order = (
        StageRecord(stage=StageId.STAGE1.value, transport=FAKE_TRANSPORT),
        StageRecord(stage=StageId.STAGE0.value, transport=FAKE_TRANSPORT),
    )
    stage_lines = [
        line.split()[0]
        for line in stage_spend_lines(out_of_order)
        if line.split()[:1] in ([StageId.STAGE0.value], [StageId.STAGE1.value])
    ]
    assert stage_lines == [StageId.STAGE0.value, StageId.STAGE1.value]


def test_plan_prints_the_measured_ledger_beside_the_estimate(
    study_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The comparison an operator authorizing the next stage is making."""
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    assert main(["plan", "--study-dir", str(study_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "optimizer-side calls per run" in out
    assert STAGE_SPEND_HEADING in out
    assert StageId.STAGE0.value in out


def test_plan_before_any_stage_needs_no_recorded_ledger(
    study_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``plan`` is the command run *before* the first stage.

    Failing it because nothing has been bought yet would make it unusable
    exactly when it is most useful.
    """
    assert main(["plan", "--study-dir", str(study_dir)]) == EXIT_OK
    assert NO_STAGES_RUN in capsys.readouterr().out


def test_run_echoes_the_transport_and_the_ledger(
    study_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    out = capsys.readouterr().out
    assert f"transport: {FAKE_TRANSPORT}" in out
    assert STAGE_SPEND_HEADING in out
    assert str(study_dir / STUDY_MANIFEST_NAME) in out


# --------------------------------------------------------------------------
# The fake path is unchanged
# --------------------------------------------------------------------------


def test_the_transport_stays_out_of_the_pre_registration_hash(
    study_dir: Path, tmp_path: Path
) -> None:
    """Two studies differing only in transport pre-register identically.

    The transport changes what a stage's numbers are evidence *of*; it does
    not change what the study is designed to measure. Hashing it would make
    an otherwise-identical design read as a different pre-registration.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    fake_hash = read_study_manifest(study_dir).pre_registration
    assert fake_hash is not None

    other = tmp_path / "other-study"
    write_study_manifest(
        other,
        _manifest_anchored_on(OPENROUTER_TRANSPORT).model_copy(
            update={"stages": ()}
        ),
    )
    assert _run_stage(other, StageId.STAGE0.value, FAKE_TRANSPORT) == EXIT_OK
    other_hash = read_study_manifest(other).pre_registration
    assert other_hash is not None
    assert other_hash.design_hash == fake_hash.design_hash


def test_one_transport_serves_every_role() -> None:
    """The assumption behind binding one live client per stage.

    A stage binds an engine per (role, repeat count) and rebinds on every
    scored candidate, so the paid path builds one transport and reuses it.
    That is only sound because the three roles' execution policies are
    identical -- they differ in ``split_role``, which the engine carries,
    not in anything the transport is built from. If a role ever gained its
    own policy, the shared client would silently serve it under another
    role's, and this is the assertion that would catch it.
    """
    policies = [
        ReferenceEvalRuntimeConfig(
            split_role=SPLIT_ROLE_BY_EVAL_ROLE[role],
            transport_api_key_env=OPENROUTER_API_KEY_ENV,
            provider_kind=ProviderKind.OPENROUTER,
        ).execution_policy
        for role in SPLIT_ROLE_BY_EVAL_ROLE
    ]
    assert all(policy == policies[0] for policy in policies)


def test_transport_names_are_the_persisted_vocabulary() -> None:
    """The CLI's choices and the manifest's values are one list."""
    assert TransportName.FAKE.value == FAKE_TRANSPORT
    assert TransportName.OPENROUTER.value == OPENROUTER_TRANSPORT


# --------------------------------------------------------------------------
# An arm stage's spend reaches its stage record
# --------------------------------------------------------------------------


def test_an_arm_stage_records_the_spend_its_runs_reported(
    study_dir: Path,
) -> None:
    """The stage row is the fold of its runs' per-role records.

    An arm stage spends through optimizer runs rather than through the
    engine, so its bill lives on the runs. A stage row that ignored them
    reported a fully paid stage as one with nothing recorded, which the
    ledger then rendered as "reached no provider".

    Run on the fake transport, where the *record* is still empty by the
    fake-stage rule -- so the fold itself is asserted on the aggregator,
    and this test pins the wiring: the runs the stage executed are what
    reaches ``_stage_record``.
    """
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    manifest = read_study_manifest(study_dir)
    executed = tuple(
        entry
        for arm in manifest.arms
        for run in arm.runs
        for entry in run.spend
    )
    # The fixture's arms really do spend, so the fold has something to
    # fold; a fixture that recorded nothing would pass this vacuously.
    assert executed
    folded = run_spend_records(executed)
    assert {entry.role for entry in folded} == {
        entry.role for entry in executed
    }
    assert sum(entry.calls for entry in folded) == sum(
        entry.calls for entry in executed
    )


def test_a_paid_arm_stage_records_the_fold_and_the_ledger_prints_it() -> None:
    """The finding, end to end on the record and the rendering.

    A paid arm stage whose runs spent must record that spend and print the
    totals -- not the empty tuple that made a fully paid stage render, under
    a MEASURED heading, as one that reached no provider.

    Reaches no provider: ``_stage_record`` is handed the run records
    directly, which is exactly what ``run_arm_stage`` hands it.
    """
    runs = tuple(
        _run_on(f"r{index}", transport=OPENROUTER_TRANSPORT).model_copy(
            update={
                "spend": (
                    _spend(calls=24, usd=0.04),
                    RunSpendRecord(
                        role="proposer",
                        calls=1,
                        cached_calls=0,
                        input_tokens=100,
                        output_tokens=10,
                        priced_calls=1,
                        unpriced_calls=0,
                        rows_missing_token_breakdown=0,
                        usd=0.002,
                    ),
                )
            }
        )
        for index in range(2)
    )
    record = _stage_record(
        stage=SpecStageId.STAGE1,
        environment=_paid_environment(),
        run_spend=_executed_run_spend(runs),
    )
    assert record.transport == OPENROUTER_TRANSPORT
    by_role = {entry.role: entry for entry in record.spend}
    assert by_role["task_model"].calls == 48
    assert by_role["proposer"].calls == 2
    assert by_role["task_model"].usd == pytest.approx(0.08)

    text = "\n".join(stage_spend_lines((record,)))
    assert "50" in text
    assert "$0.084000" in text
    # And never either of the empty-spend labels.
    assert UNLEDGERED_SPEND not in text
    assert NO_RECORDED_SPEND not in text


def test_a_paid_arm_stage_with_no_run_spend_stays_unledgered() -> None:
    """The other half: a paid stage whose runs recorded nothing.

    The fold has nothing to fold, so the record stays empty -- and the
    ledger must say the bill is unknown rather than absent.
    """
    record = _stage_record(
        stage=SpecStageId.STAGE1,
        environment=_paid_environment(),
        run_spend=(),
    )
    assert record.spend == ()
    assert UNLEDGERED_SPEND in "\n".join(stage_spend_lines((record,)))


def test_a_resumed_arm_stage_keeps_the_spend_it_already_measured() -> None:
    """The defect: a resume with nothing to run erased the stage's bill.

    An arm stage that crashed after its manifest write has already paid
    for its runs and already recorded what they cost. Resuming it re-runs
    nothing -- every seed is recorded, so ``executed_runs`` is empty --
    and the replacement stage record was built from that empty tuple, so
    the measured spend was overwritten with nothing and the ledger
    reported a fully paid stage as UNLEDGERED.

    Reaches no provider: the existing record and the (empty) executed runs
    are handed to the merge directly, which is what ``run_arm_stage``
    does.
    """
    measured = _stage_record(
        stage=SpecStageId.STAGE1,
        environment=_paid_environment(),
        run_spend=(_spend(calls=24, usd=0.04),),
    )
    assert measured.spend  # the fixture really did measure something
    manifest = toy_manifest(arms=_arms()).model_copy(
        update={"stages": (measured,)}
    )

    merged = _arm_stage_record(
        manifest,
        _stage_record(
            stage=SpecStageId.STAGE1,
            environment=_paid_environment(),
            run_spend=_executed_run_spend(()),
        ),
    )

    by_stage = {entry.stage: entry for entry in merged}
    kept = by_stage[SpecStageId.STAGE1.value]
    assert kept.spend == measured.spend
    assert UNLEDGERED_SPEND not in "\n".join(stage_spend_lines((kept,)))


def test_a_resumed_arm_stage_adds_the_runs_it_did_execute() -> None:
    """Merged, not merely preserved: a partial resume bills both halves.

    The spend already on the row is what an earlier invocation paid; the
    executed runs are what this one paid. The stage's bill is the sum, and
    taking either alone would under-report the study's cost.
    """
    existing = _stage_record(
        stage=SpecStageId.STAGE1,
        environment=_paid_environment(),
        run_spend=(_spend(calls=24, usd=0.04),),
    )
    manifest = toy_manifest(arms=_arms()).model_copy(
        update={"stages": (existing,)}
    )

    merged = _arm_stage_record(
        manifest,
        _stage_record(
            stage=SpecStageId.STAGE1,
            environment=_paid_environment(),
            run_spend=(_spend(calls=6, usd=0.01),),
        ),
    )

    by_role = {
        entry.role: entry
        for entry in {entry.stage: entry for entry in merged}[
            SpecStageId.STAGE1.value
        ].spend
    }
    assert by_role["task_model"].calls == 30
    assert by_role["task_model"].usd == pytest.approx(0.05)


def test_a_fresh_arm_stage_still_records_only_what_it_ran() -> None:
    """The normal path is untouched: no prior record, no merge.

    A first run of a stage has nothing to preserve, so its row is exactly
    the runs this invocation executed -- which is what keeps the ledger's
    rows summing to what the study spent rather than double-billing a run
    Stage 1 already paid for.
    """
    manifest = toy_manifest(arms=_arms())
    assert not manifest.stages

    fresh = _stage_record(
        stage=SpecStageId.STAGE1,
        environment=_paid_environment(),
        run_spend=(_spend(calls=6, usd=0.01),),
    )
    merged = _arm_stage_record(manifest, fresh)

    assert merged == (fresh,)


def test_the_fold_sums_counters_and_withholds_an_unknown_total() -> None:
    """Two runs' records add, and one unpriced run withholds the total.

    The honesty rule is re-applied to the fold rather than carried from
    the parts: a sum over the priced runs alone would look authoritative
    while understating the role's bill.
    """
    priced = _spend(calls=10, usd=0.5)
    also_priced = _spend(calls=4, usd=0.25)
    folded = run_spend_records((priced, also_priced))
    assert len(folded) == 1
    assert folded[0].calls == 14
    assert folded[0].usd == pytest.approx(0.75)

    unpriced = _spend(calls=6, usd=None)
    mixed = run_spend_records((priced, unpriced))
    assert len(mixed) == 1
    assert mixed[0].calls == 16
    assert mixed[0].usd is None


def test_the_fold_keeps_roles_apart() -> None:
    """Two roles fold to two records, in first-seen order."""
    task = _spend(calls=10, usd=0.5)
    proposer = RunSpendRecord(
        role="proposer",
        calls=2,
        cached_calls=0,
        input_tokens=20,
        output_tokens=4,
        priced_calls=2,
        unpriced_calls=0,
        rows_missing_token_breakdown=0,
        usd=0.125,
    )
    folded = run_spend_records((task, proposer, task))
    assert [entry.role for entry in folded] == ["task_model", "proposer"]
    assert folded[0].calls == 20
    assert folded[1].calls == 2


# --------------------------------------------------------------------------
# The three renderings of an empty spend tuple
# --------------------------------------------------------------------------


def test_a_fake_stage_reads_as_having_reached_no_provider() -> None:
    record = StageRecord(stage=StageId.STAGE1.value, transport=FAKE_TRANSPORT)
    text = "\n".join(stage_spend_lines((record,)))
    assert NO_RECORDED_SPEND in text
    assert "no provider reached (fake transport)" in text
    assert UNLEDGERED_SPEND not in text


def test_a_paid_stage_with_no_records_reads_as_unledgered() -> None:
    """The rendering this finding exists to force apart.

    A paid stage that recorded nothing called a provider and lost track of
    what it bought. Reporting it as "reached no provider" described a
    fully billed stage as a free one, under a MEASURED heading.
    """
    record = StageRecord(
        stage=StageId.STAGE1.value, transport=OPENROUTER_TRANSPORT
    )
    text = "\n".join(stage_spend_lines((record,)))
    assert UNLEDGERED_SPEND in text
    assert (
        "UNLEDGERED -- ran on a paid transport and recorded no spend; "
        "this stage's bill is unknown, not zero"
    ) in text
    assert NO_RECORDED_SPEND not in text
    assert "no provider" not in text


def test_a_paid_stage_with_records_prints_the_totals() -> None:
    record = StageRecord(
        stage=StageId.STAGE1.value,
        transport=OPENROUTER_TRANSPORT,
        spend=(_spend(calls=112, usd=0.001234),),
    )
    text = "\n".join(stage_spend_lines((record,)))
    assert "112" in text
    assert "$0.001234" in text
    assert UNLEDGERED_SPEND not in text
    assert NO_RECORDED_SPEND not in text


def test_the_ledger_no_longer_disclaims_the_reporting_pass() -> None:
    """The omission the note described is closed, so the note is gone.

    Fails-before: every rendering of the ledger carried
    ``UNLEDGERED_SCORING_NOTE`` -- "official-selection scoring and held-out
    evaluation calls are not yet ledgered; every total below excludes
    them." That was true and is now false: the reporting pass is folded
    onto the stage's own row, so a total that still disclaimed it would
    understate its own completeness.
    """
    for stages in (
        (),
        (StageRecord(stage=StageId.STAGE0.value, transport=FAKE_TRANSPORT),),
    ):
        text = "\n".join(stage_spend_lines(stages))
        assert "not yet ledgered" not in text
        assert "excludes them" not in text


def test_the_report_states_what_the_ledger_covers() -> None:
    """Both routes named, so a reader knows the row is the whole bill."""
    assert "not yet ledgered" not in STAGE_SPEND_COVERAGE_NOTE
    assert "lower bound" not in STAGE_SPEND_COVERAGE_NOTE
    for phrase in (
        "official-selection scoring",
        "held-out evaluations",
        "optimizer runs",
        "persisted output rows",
    ):
        assert phrase in STAGE_SPEND_COVERAGE_NOTE, phrase


def test_the_report_labels_the_two_empty_cases_apart() -> None:
    """The literals themselves; the rendering is pinned in the report suite."""
    assert UNLEDGERED_STAGE_DETAIL == (
        "UNLEDGERED -- ran on a paid transport and recorded no spend; this "
        "stage's bill is unknown, not zero"
    )
    assert NO_PROVIDER_STAGE_DETAIL == "no provider reached (fake transport)"
    assert UNLEDGERED_STAGE_DETAIL != NO_PROVIDER_STAGE_DETAIL


# --------------------------------------------------------------------------
# The guard reads the evidence, not just Stage 0's summary
# --------------------------------------------------------------------------


def _paid_environment() -> StageEnvironment:
    """A paid stage environment bound to nothing that could call out.

    ``_stage_record`` reads only ``transport`` and ``store`` off it, and a
    ``None`` store is what keeps the evidence route from being taken --
    which is the point here, since the spend must come from the runs.
    """
    return StageEnvironment(
        bind_engine=cast("EngineBinder", None),
        naive_candidate=cast("Candidate", None),
        ceiling_candidate=cast("Candidate", None),
        task_ids_by_role={},
        pool_ceiling=1,
        transport=OPENROUTER_TRANSPORT,
        store=None,
    )


def _run_on(run_id: str, *, transport: str) -> RunRecord:
    pointer = EvidencePointer(schema_name="s/v1", content_hash="f" * 64)
    return RunRecord(
        run_id=run_id,
        seed=1000,
        artifact_dir=f"/tmp/runs/{run_id}",  # noqa: S108
        result_ref=pointer,
        audit_ref=pointer,
        cost_ref=pointer,
        audit_passed=True,
        spend=(),
        transport=transport,
    )


def test_the_guard_refuses_a_stage_whose_own_record_disagrees() -> None:
    """Stage 0 agreeing is not enough: the target stage ran elsewhere.

    A stage that already ran on the other transport still holds the runs
    that invocation produced, so resuming it here would select across two
    transports inside one stage.
    """
    manifest = _manifest_anchored_on(OPENROUTER_TRANSPORT).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE0.value,
                    transport=OPENROUTER_TRANSPORT,
                ),
                StageRecord(
                    stage=StageId.STAGE1.value, transport=FAKE_TRANSPORT
                ),
            )
        }
    )
    with pytest.raises(StageError) as excinfo:
        require_matching_transport(
            manifest, stage=StageId.STAGE1, transport=OPENROUTER_TRANSPORT
        )
    message = str(excinfo.value)
    assert FAKE_TRANSPORT in message
    assert "already ran on" in message


def test_the_guard_refuses_surviving_runs_from_another_transport() -> None:
    """The check that reads the evidence rather than the summary.

    Both stage rows can say ``openrouter`` while the runs beneath them
    were measured on ``fake``, because a resumed stage keeps runs an
    earlier invocation produced. Those runs are selected over alongside
    this stage's, so the arg-max would span two experiments.
    """
    arms = tuple(
        arm.model_copy(
            update={
                "runs": (_run_on(f"{arm.arm_id}-1", transport=FAKE_TRANSPORT),)
            }
        )
        for arm in _arms()
    )
    manifest = _manifest_anchored_on(OPENROUTER_TRANSPORT).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE0.value,
                    transport=OPENROUTER_TRANSPORT,
                ),
            ),
            "arms": arms,
        }
    )
    with pytest.raises(StageError) as excinfo:
        require_matching_transport(
            manifest, stage=StageId.STAGE1, transport=OPENROUTER_TRANSPORT
        )
    message = str(excinfo.value)
    assert "copro-1" in message
    assert "another" in message


def test_the_guard_passes_when_every_run_agrees() -> None:
    """No refusal when the evidence is all from one transport."""
    arms = tuple(
        arm.model_copy(
            update={
                "runs": (
                    _run_on(f"{arm.arm_id}-1", transport=OPENROUTER_TRANSPORT),
                )
            }
        )
        for arm in _arms()
    )
    manifest = _manifest_anchored_on(OPENROUTER_TRANSPORT).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE0.value,
                    transport=OPENROUTER_TRANSPORT,
                ),
            ),
            "arms": arms,
        }
    )
    require_matching_transport(
        manifest, stage=StageId.STAGE1, transport=OPENROUTER_TRANSPORT
    )


def test_a_run_records_the_transport_it_ran_on(study_dir: Path) -> None:
    """Every run the harness records carries its own transport."""
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    manifest = read_study_manifest(study_dir)
    recorded = [run.transport for arm in manifest.arms for run in arm.runs]
    assert recorded
    assert set(recorded) == {FAKE_TRANSPORT}


# --------------------------------------------------------------------------
# --replace-design across transports drops the stale evidence
# --------------------------------------------------------------------------


def _replace_design_on(study_dir: Path, transport: str) -> int:
    return main(
        [
            "run",
            "--study-dir",
            str(study_dir),
            "--stage",
            StageId.STAGE0.value,
            "--replace-design",
            "--transport",
            transport,
        ]
    )


def test_replace_design_onto_the_same_transport_keeps_the_evidence(
    study_dir: Path,
) -> None:
    """A re-calibration that does not change transport drops nothing.

    The arm stages ran against the design being replaced, but their
    evidence is still from this study's transport, so the ordinary
    amendment path applies and nothing is recorded as dropped.
    """
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    before = read_study_manifest(study_dir)
    assert before.arms[0].runs

    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_OK
    after = read_study_manifest(study_dir)
    assert after.amendments == ()
    assert [(arm.arm_id, len(arm.runs)) for arm in after.arms] == [
        (arm.arm_id, len(arm.runs)) for arm in before.arms
    ]
    assert {entry.stage for entry in after.stages} == {
        entry.stage for entry in before.stages
    }


def test_replace_design_across_transports_drops_the_stale_evidence(
    study_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: a design change *and* evidence from another transport.

    Before this, ``stage0 --replace-design --transport openrouter`` left
    ``stages[stage1]`` saying ``fake`` and every fake arm run in place, so
    a Stage 2 on the paid transport reused them against freshly bought
    anchors. Now they are dropped, and what was dropped is recorded.

    Driven the other way round -- the study is calibrated on ``openrouter``
    in the manifest, then re-calibrated on ``fake`` -- so the assertion
    needs no key and reaches no provider.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    assert ran.arms[0].runs
    assert ran.selection
    assert ran.held_out_claims
    assert ran.call_count_gate is not None

    # Relabel every record as the paid transport, then re-calibrate on
    # fake: the direction that needs no credential. The runs stay fake, so
    # the paid-evidence refusal does not fire and the drop does.
    relabelled = ran.model_copy(
        update={
            "stages": tuple(
                entry.model_copy(update={"transport": OPENROUTER_TRANSPORT})
                for entry in ran.stages
            )
        }
    )
    write_study_manifest(study_dir, relabelled, replace=True)

    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_OK
    after = read_study_manifest(study_dir)

    # The stale stage rows, runs, selections, claims, rows, and the pilot's
    # gate are all gone.
    assert {entry.stage for entry in after.stages} == {StageId.STAGE0.value}
    assert all(arm.runs == () for arm in after.arms)
    assert after.selection == ()
    assert after.held_out_claims == ()
    assert after.held_out == ()
    assert after.call_count_gate is None
    # The arms themselves survive: they are design, not evidence.
    assert [arm.arm_id for arm in after.arms] == [
        arm.arm_id for arm in ran.arms
    ]

    # And the drop is recorded rather than silent.
    assert len(after.amendments) == 1
    amendment = after.amendments[0]
    assert amendment.reason == AMENDMENT_REASON_TRANSPORT_CHANGE
    assert amendment.from_transport == OPENROUTER_TRANSPORT
    assert amendment.to_transport == FAKE_TRANSPORT
    assert amendment.dropped_stages == (StageId.STAGE1.value,)
    assert set(amendment.dropped_run_ids) == {
        run.run_id for arm in ran.arms for run in arm.runs
    }
    assert amendment.dropped_selections == len(ran.selection)
    assert amendment.dropped_held_out_claims == len(ran.held_out_claims)
    assert amendment.dropped_held_out_rows == len(ran.held_out)
    assert amendment.dropped_call_count_gate is True
    # The manifest drops the runs; the disk keeps their directories, and
    # the next stage to compute one of those names refuses to reuse what
    # it finds. The amendment names them so the operator resolving that
    # refusal does not have to reconstruct the naming rule by hand.
    assert set(amendment.dropped_run_directories) == {
        run.artifact_dir for arm in ran.arms for run in arm.runs
    }
    # A real optimizer arm's directory is really on disk, which is what
    # makes this list actionable. A control writes no run directory, so
    # the list names what each dropped run *claimed* as its artifacts
    # rather than asserting every entry survives -- the operator needs the
    # names the next stage will compute, not a filtered subset.
    optimizer_runs = tuple(
        run for arm in ran.arms if arm.optimizer == "copro" for run in arm.runs
    )
    assert optimizer_runs
    assert all(
        Path(run.artifact_dir).is_dir()
        and run.artifact_dir in amendment.dropped_run_directories
        for run in optimizer_runs
    )


def test_the_amendment_leaves_directories_the_next_stage_refuses(
    study_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The two halves meet: the drop orphans directories, the stage refuses.

    End to end through the CLI, because the refusal is only useful if it
    survives the wiring. After the amendment the replacement Stage 1
    recomputes the dropped runs' directory names, finds the old runs still
    there, and must refuse rather than re-record them -- and
    ``--discard-stale-runs`` must then actually get the study moving.

    Reaches no provider: everything runs on the fake transport, and the
    amendment is driven by relabelling the manifest rather than by
    calibrating on a paid one.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        ran.model_copy(
            update={
                "stages": tuple(
                    entry.model_copy(
                        update={"transport": OPENROUTER_TRANSPORT}
                    )
                    for entry in ran.stages
                )
            }
        ),
        replace=True,
    )
    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_OK
    amendment = read_study_manifest(study_dir).amendments[0]
    orphaned = tuple(
        directory
        for directory in amendment.dropped_run_directories
        if Path(directory).is_dir()
    )
    # The drop cleared the manifest and not the disk, which is the whole
    # reason the next stage needs a check at all.
    assert orphaned

    # The stale directories now claim a *different design*: the amendment
    # replaced the one they ran against. Their transport still matches, so
    # this drives the check on run identity rather than on transport --
    # the same refusal, reached without a paid calibration.
    for directory in orphaned:
        report = Path(directory) / "trajectory-report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["run_id"] = "a-run-from-somewhere-else"
        report.write_text(json.dumps(payload), encoding="utf-8")

    assert _run_stage(study_dir, StageId.STAGE1.value, FAKE_TRANSPORT) == (
        EXIT_CHECK_FAILED
    )
    error = capsys.readouterr().err
    assert orphaned[0] in error
    assert DISCARD_STALE_RUNS_FLAG in error
    # It refused rather than overwriting, so the artifacts are still there
    # for the operator to inspect.
    assert all(Path(directory).is_dir() for directory in orphaned)

    # And the recovery the message names actually recovers.
    assert (
        main(
            [
                "run",
                "--study-dir",
                str(study_dir),
                "--stage",
                StageId.STAGE1.value,
                "--transport",
                FAKE_TRANSPORT,
                DISCARD_STALE_RUNS_FLAG,
            ]
        )
        == EXIT_OK
    )
    assert read_study_manifest(study_dir).arms[0].runs


def test_replace_design_refuses_to_discard_paid_evidence(
    study_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paid evidence is never dropped automatically, and never silently.

    A fake run costs nothing and re-running it is the obvious recovery. A
    paid run is money already spent, and a command whose stated purpose is
    re-calibrating Stage 0 must not delete it as a side effect.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    paid = ran.model_copy(
        update={
            "stages": tuple(
                entry.model_copy(update={"transport": OPENROUTER_TRANSPORT})
                for entry in ran.stages
            ),
            "arms": tuple(
                arm.model_copy(
                    update={
                        "runs": tuple(
                            run.model_copy(
                                update={"transport": OPENROUTER_TRANSPORT}
                            )
                            for run in arm.runs
                        )
                    }
                )
                for arm in ran.arms
            ),
        }
    )
    write_study_manifest(study_dir, paid, replace=True)

    # A refused check, not an operator error: the study is intact and the
    # command declined to change it.
    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_CHECK_FAILED
    # Nothing was dropped, and nothing was amended.
    after = read_study_manifest(study_dir)
    assert after.amendments == ()
    assert all(arm.runs for arm in after.arms if arm.arm_id == "copro")
    assert after.call_count_gate is not None


def test_the_paid_evidence_refusal_names_the_recovery(
    study_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal an operator cannot act on is a refusal that gets forced."""
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        ran.model_copy(
            update={
                "stages": tuple(
                    entry.model_copy(
                        update={"transport": OPENROUTER_TRANSPORT}
                    )
                    for entry in ran.stages
                ),
                "arms": tuple(
                    arm.model_copy(
                        update={
                            "runs": tuple(
                                run.model_copy(
                                    update={"transport": OPENROUTER_TRANSPORT}
                                )
                                for run in arm.runs
                            )
                        }
                    )
                    for arm in ran.arms
                ),
            }
        ),
        replace=True,
    )
    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "Paid evidence is never discarded automatically" in err
    assert "fresh one" in err


def test_the_amendment_lets_the_next_stage_run(
    study_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the drop, the guard is satisfied and Stage 1 runs again.

    The two halves of this finding are one property: the drop is what
    makes the strengthened guard passable rather than a permanent refusal.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    write_study_manifest(
        study_dir,
        ran.model_copy(
            update={
                "stages": tuple(
                    entry.model_copy(
                        update={"transport": OPENROUTER_TRANSPORT}
                    )
                    for entry in ran.stages
                )
            }
        ),
        replace=True,
    )
    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_OK
    assert _run_stage(study_dir, StageId.STAGE1.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    after = read_study_manifest(study_dir)
    assert after.arms[0].runs
    # The amendment survives the re-run: it records what the study once
    # held, so a later write must not quietly drop it.
    assert len(after.amendments) == 1


def test_the_amendment_clears_every_verdict_over_dropped_evidence(
    study_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verdict computed over dropped runs goes with the runs.

    ``leakage_check`` is L6's mechanical pass over the run artifacts the
    amendment is dropping. Leaving it behind would let a regenerated
    report inherit a pass that was established over evidence the study no
    longer holds -- which reads exactly like a study whose leakage checks
    were run against its current runs and passed.
    """
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    assert ran.arms[0].runs

    # A passing L6 over the runs that are about to be dropped, and the
    # paid relabelling that makes this a cross-transport amendment.
    passing_leakage = LeakageCheckRecord(
        passed=True,
        checks=(
            LeakageCheckEntry(
                check_id="L1",
                passed=True,
                detail="every optimizer evaluation ran on the internal split",
            ),
        ),
    )
    write_study_manifest(
        study_dir,
        ran.model_copy(
            update={
                "leakage_check": passing_leakage,
                "stages": tuple(
                    entry.model_copy(
                        update={"transport": OPENROUTER_TRANSPORT}
                    )
                    for entry in ran.stages
                ),
            }
        ),
        replace=True,
    )

    assert _replace_design_on(study_dir, FAKE_TRANSPORT) == EXIT_OK
    after = read_study_manifest(study_dir)

    # The evidence L6 was computed over is gone, so the verdict is too.
    assert all(arm.runs == () for arm in after.arms)
    assert after.leakage_check is None
    # And the report reads an absent block as not-checked rather than as a
    # pass, which is what makes the clearing sufficient.
    assert study_leakage_failed(after)


# --------------------------------------------------------------------------
# The fake path's Eval Config hashes, pinned
# --------------------------------------------------------------------------

#: The three role Eval Config hashes the fake toy study binds.
#:
#: These are stored identity: L1 compares each optimizer evaluation's
#: resolved config against the ``internal`` hash recorded here, so a change
#: to how the fake path builds its call config -- most obviously, letting a
#: ``provider_call_config`` leak onto it the way the paid path carries one
#: -- would silently rebase every study's recorded config and turn L1 into
#: a check that always agrees with whatever just ran.
#:
#: Pinned rather than derived, for the reason every persisted-format
#: literal is: a hash recomputed by the same code that produced it cannot
#: catch that code changing. Recompute deliberately and update these if the
#: fake path's config is meant to change.
FAKE_TOY_EVAL_CONFIG_HASHES = {
    "internal": (
        "58d7f579f007870d14598ebc023540043d855028f1c79a647cb647c56a9f2bfb"
    ),
    "official": (
        "9fecb9025af7dd99618ec6c5f281c416e27a83edd47f0da11462d1f68d61f068"
    ),
    "held_out": (
        "df7978daf73a12643e7ce8e5b0349fa516e919c1bdf5aaa0305a1cd4e313f45e"
    ),
}


def test_the_fake_paths_eval_config_hashes_are_pinned(
    study_dir: Path,
) -> None:
    """The fake transport binds no provider call config, and these prove it.

    The paid path seeds an OpenRouter call config onto every role engine.
    The fake path must not, because its Eval Config hashes are what L1
    checks against and what every study recorded before a paid path
    existed. A hash that moved would mean the fake path had started
    carrying provider configuration it has no use for.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    splits = read_study_manifest(study_dir).splits
    assert {
        name: getattr(splits, name).eval_config_hash
        for name in FAKE_TOY_EVAL_CONFIG_HASHES
    } == FAKE_TOY_EVAL_CONFIG_HASHES


def test_the_three_roles_bind_three_distinct_configs(
    study_dir: Path,
) -> None:
    """Each role evaluates its own split, so no two hashes may agree.

    A collision would mean two roles resolved to one configuration, which
    is the state that makes a leakage check pass by construction.
    """
    assert _run_stage(study_dir, StageId.STAGE0.value, FAKE_TRANSPORT) == (
        EXIT_OK
    )
    splits = read_study_manifest(study_dir).splits
    hashes = [
        getattr(splits, name).eval_config_hash
        for name in FAKE_TOY_EVAL_CONFIG_HASHES
    ]
    assert len(set(hashes)) == len(hashes)


# --------------------------------------------------------------------------
# The reporting pass is ledgered too (D3 defect (c))
# --------------------------------------------------------------------------


class _StubEvidence:
    """The two fields the ledger reads off an evaluation's evidence."""

    def __init__(self, key: tuple[str, str]) -> None:
        schema_name, content_hash = key
        self.outputs_ref = SimpleNamespace(
            schema_name=schema_name,
            content_hash=content_hash,
            reference=key,
        )


def _ledger_with(records: dict[tuple[str, str], tuple[RunSpendRecord, ...]]):
    """A ledger whose pricing is stubbed, so the *collection* is under test.

    The projection from rows to records is ``stage_spend_records``' own
    contract and is exercised where it lives; what this pins is that every
    evaluation is collected exactly once, attributed to its evidence, and
    folded under the honesty rules.
    """
    ledger = ReportSpendLedger(cast("ObjectStore", object()))

    def _priced(*, store, evidence):  # noqa: ARG001
        (record,) = evidence
        key = (
            record.outputs_ref.schema_name,
            record.outputs_ref.content_hash,
        )
        return records.get(key, ())

    return ledger, _priced


def test_the_ledger_records_one_entry_per_evaluation(monkeypatch) -> None:
    """One record per role per evaluation, cited by its own evidence.

    Fails-before: there was no ledger. Official-selection scoring and
    held-out evaluation reached the provider through the evaluation engine
    outside any optimizer run, and nothing collected their rows -- so the
    stage row stopped at the run-side total and a study's reported cost
    understated its spend by the whole reporting pass.
    """
    first = ("outputs", "a" * 64)
    second = ("outputs", "b" * 64)
    ledger, priced = _ledger_with(
        {
            first: (_spend(calls=10, usd=0.01),),
            second: (_spend(calls=4, usd=0.004),),
        }
    )
    monkeypatch.setattr(
        "whetstone_envs.optim.study.spend.stage_spend_records", priced
    )

    ledger.record(
        evidence=cast("EvalEvidence", _StubEvidence(first)),
        purpose="study-official-selection",
        candidate_name="copro-run1",
    )
    ledger.record(
        evidence=cast("EvalEvidence", _StubEvidence(second)),
        purpose="study-held-out-report",
        candidate_name="copro-run1",
    )

    records = ledger.records()
    assert len(records) == 2
    assert [record.purpose for record in records] == [
        "study-official-selection",
        "study-held-out-report",
    ]
    # Each record cites the evidence its number was derived from, so the
    # total is checkable rather than taken on the ledger's word.
    assert [record.evidence_key for record in records] == [first, second]

    (folded,) = ledger.folded()
    assert folded.role == "task_model"
    assert folded.calls == 14
    assert folded.usd == pytest.approx(0.014)


def test_one_unpriced_reporting_evaluation_withholds_the_total(
    monkeypatch,
) -> None:
    """The honesty rule survives the fold, as it does for runs.

    A sum over the priced evaluations alone would look authoritative while
    understating the pass, which is the same failure ``RunSpendRecord``
    forbids within one run.
    """
    priced_key = ("outputs", "c" * 64)
    unpriced_key = ("outputs", "d" * 64)
    ledger, priced = _ledger_with(
        {
            priced_key: (_spend(calls=10, usd=0.01),),
            unpriced_key: (_spend(calls=2, usd=None),),
        }
    )
    monkeypatch.setattr(
        "whetstone_envs.optim.study.spend.stage_spend_records", priced
    )
    for key in (priced_key, unpriced_key):
        ledger.record(
            evidence=cast("EvalEvidence", _StubEvidence(key)),
            purpose="study-held-out-report",
            candidate_name="arm",
        )

    (folded,) = ledger.folded()
    assert folded.calls == 12
    assert folded.usd is None


def test_an_evaluation_that_evidenced_no_call_appends_nothing(
    monkeypatch,
) -> None:
    """ "Not measured" and "free" stay distinct, as everywhere else here."""
    key = ("outputs", "e" * 64)
    ledger, priced = _ledger_with({})
    monkeypatch.setattr(
        "whetstone_envs.optim.study.spend.stage_spend_records", priced
    )
    ledger.record(
        evidence=cast("EvalEvidence", _StubEvidence(key)),
        purpose="study-held-out-report",
        candidate_name="arm",
    )
    assert ledger.records() == ()
    assert ledger.folded() == ()


def test_a_ledger_with_no_store_collects_nothing() -> None:
    """A caller supplying its own collaborators prices nothing.

    The rows are read back out of the store, so with none bound there is
    nothing to read -- and inventing a bill would be worse than omitting one.
    """
    ledger = ReportSpendLedger(None)
    ledger.record(
        evidence=cast("EvalEvidence", _StubEvidence(("outputs", "f" * 64))),
        purpose="study-held-out-report",
        candidate_name="arm",
    )
    assert ledger.records() == ()
    assert ledger.folded() == ()


# --------------------------------------------------------------------------
# The effective provider call config is recorded (Phase E item 4)
# --------------------------------------------------------------------------


def test_binding_a_stage_records_what_the_transport_bound(
    study_dir: Path,
) -> None:
    """The manifest gains the config the transport actually resolved.

    Fails-before: `models` named which model the study meant to run and
    `stages` named which transport it ran on, but nothing recorded the
    *effective* call config -- the resolved route and the request controls
    -- so neither the spend model nor the claim that two stages ran "the
    same experiment" could be checked against the manifest.
    """
    with bound_stage_environment(study_dir):
        pass

    (record,) = read_study_manifest(study_dir).models.provider_calls
    assert record.transport == FAKE_TRANSPORT
    assert record.protocol
    assert record.model_route
    # Every control is stated, set or not: an omitted control would read
    # as one the study chose, and "provider default" is the state that
    # explains this study's per-call bill.
    assert record.reasoning == PROVIDER_CONTROL_UNSET
    assert record.temperature == PROVIDER_CONTROL_UNSET
    assert record.seed == PROVIDER_CONTROL_UNSET


def test_rebinding_the_same_transport_rewrites_no_manifest(
    study_dir: Path,
) -> None:
    """An ordinary resume does not touch the block it would restate."""
    with bound_stage_environment(study_dir):
        pass
    first = read_study_manifest(study_dir)
    with bound_stage_environment(study_dir):
        pass
    assert read_study_manifest(study_dir) == first


def test_the_paid_route_records_the_route_it_would_bind() -> None:
    """The paid config's route is the manifest's task model, not a default.

    Asserted on the projection rather than through a bound stage: binding
    the paid transport needs a key, and what is under test here is that
    the recorded route is the one the study named.
    """
    from whetstone_envs.optim.provider import openrouter_seeded_call_config
    from whetstone_envs.optim.study.environment import _provider_call_record

    record = _provider_call_record(
        transport=OPENROUTER_TRANSPORT,
        config=openrouter_seeded_call_config(model="openai/gpt-5-nano"),
    )
    assert record.transport == OPENROUTER_TRANSPORT
    assert record.provider == "openrouter"
    assert record.model_route == "openai/gpt-5-nano"
    # Recorded verbatim and never set from here: whether the design pins a
    # task-model reasoning effort is an open decision, and this block
    # states what was bound rather than choosing it.
    assert record.reasoning == PROVIDER_CONTROL_UNSET
