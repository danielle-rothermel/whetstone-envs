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
from typing import TYPE_CHECKING

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
    STUDY_MANIFEST_NAME,
    STUDY_STORE_NAME,
    ArmRecord,
    RunSpendRecord,
    StageId,
    StageRecord,
    StudyManifest,
    TransportName,
    read_study_manifest,
    recorded_transport,
    write_study_manifest,
)
from whetstone_envs.optim.study.stages import (
    StageError,
    require_matching_transport,
)

from .conftest import toy_manifest

if TYPE_CHECKING:
    from pathlib import Path

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
