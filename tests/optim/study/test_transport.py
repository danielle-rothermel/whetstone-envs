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

from whetstone_envs.optim.study.arms import arm_run_directory
from whetstone_envs.optim.study.cli import (
    EXIT_CHECK_FAILED,
    EXIT_ERROR,
    EXIT_OK,
    MIXED_RUN_WIDTHS,
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
    ALLOW_WIDTH_CHANGE_FLAG,
    AMENDMENT_REASON_TRANSPORT_CHANGE,
    DISCARD_STALE_RUNS_FLAG,
    PROVIDER_CONTROL_UNSET,
    PROVIDER_SEED_DERIVED_PER_CALL,
    STUDY_MANIFEST_NAME,
    STUDY_STORE_NAME,
    AmendmentRecord,
    ArmKind,
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
    run_widths,
    write_study_manifest,
)
from whetstone_envs.optim.study.spec import NULL_ARM_IDS
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
    refuse_resumed_width_change,
    require_matching_transport,
)
from whetstone_envs.reporting.study_report import (
    MIXED_RUN_WIDTHS_DETAIL,
    NO_PROVIDER_STAGE_DETAIL,
    STAGE_SPEND_COVERAGE_NOTE,
    UNLEDGERED_STAGE_DETAIL,
    _mixed_width_detail,
    study_leakage_failed,
)

from .conftest import toy_manifest

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig
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
            kind=(ArmKind.NULL if arm_id in NULL_ARM_IDS else ArmKind.REAL),
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


def _rebuilt_task_identity(study_dir: Path, *, effort) -> str:
    """One engine's task-model identity at a given effort.

    Built through the Codex runtime config, which is the one path in this
    package that rebuilds an engine from parameters alone -- so the
    comparison is against a real engine rather than a hand-recomputed hash.
    """
    from dr_store.sync import open_sqlite

    from whetstone_envs.optim.codex_runtime import EnvsCodexRuntimeConfig

    manifest = read_study_manifest(study_dir)
    config = EnvsCodexRuntimeConfig(
        family_id=manifest.population.family,
        split_sizes=(
            manifest.splits.internal.size,
            manifest.splits.official.size,
            manifest.splits.held_out.size,
        ),
        n_per_stratum=manifest.population.n_per_stratum,
        pool_seed_start=manifest.population.pool_seed_start,
        num_seeds=1,
        transport="openrouter",
        model=manifest.models.task_model,
        reasoning_effort=effort,
    )
    name = "pinned" if effort is not None else "unpinned"
    with open_sqlite(str(study_dir / f"identity-{name}.sqlite")) as store:
        engine = config.build_engine(cast("ObjectStore", store))
        return engine.task_model_identity_hash()


def test_every_paid_role_binds_the_manifests_reasoning_effort(
    study_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pin reaches every arm, because it reaches the shared route.

    Every arm evaluates through one bound provider call config, so this is
    where "every arm carries the effort" is decided: a pin that reached
    only some arms would mean arms measured against different task models
    under one pre-registration.

    Read off the manifest rather than off the protocol module, so a study
    initialised at one effort cannot be run at another.

    Fails-before: the binder called ``openrouter_seeded_call_config`` with
    no effort at all, so the bound config was always unpinned.
    """
    from whetstone.core.roles import EvalRole

    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "test-key-not-a-real-secret")
    manifest = read_study_manifest(study_dir)
    assert manifest.models.task_reasoning_effort == "minimal"

    with bound_stage_environment(
        study_dir, transport=OPENROUTER_TRANSPORT
    ) as environment:
        identities = {
            role: environment.bind_engine(
                role=role, num_seeds=1
            ).task_model_identity_hash()
            for role in EvalRole
        }

    # One route, so every role lands on one task-model identity: an arm
    # that bound a different effort would show up here as a second value.
    assert len(set(identities.values())) == 1
    bound = next(iter(identities.values()))

    # And that one identity is the *pinned* one, not the unpinned route it
    # would otherwise be. Compared against a rebuild rather than a literal
    # so the assertion survives an unrelated identity-payload change.
    from dr_providers import ReasoningEffort

    unpinned = _rebuilt_task_identity(study_dir, effort=None)
    pinned = _rebuilt_task_identity(study_dir, effort=ReasoningEffort.MINIMAL)
    assert pinned != unpinned
    assert bound == pinned


def test_the_in_search_route_binds_the_pinned_effort(
    study_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimizers' own evaluations run pinned, not just the report pass.

    This is the path the engines bound by ``bound_stage_environment`` do
    **not** cover. ``StudyOptimizerRunner`` builds its own ``RunSpec`` per
    arm, and the in-search evaluations that spec drives are the
    K_REPEAT-multiplied majority of the study's paid calls -- so an effort
    that reached the reporting engines and not the runner would leave most
    of the study measuring a task model the pre-registration does not name.

    Asserted on the route the runner reports for a real arm spec, which is
    the same config ``run_optimizer`` builds from that spec.

    Fails-before: ``StudyOptimizerRunner`` had no effort field at all, so
    every ``RunSpec`` it built carried ``task_reasoning_effort=None`` and
    the bound control was ``None``.
    """
    from dr_providers import ReasoningEffort

    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "test-key-not-a-real-secret")
    manifest = read_study_manifest(study_dir)

    from whetstone_envs.optim.study.arms import StudyOptimizerRunner
    from whetstone_envs.optim.study.spec import ArmKind, ArmSpec

    recorded: list[ProviderCallConfig] = []
    with bound_stage_environment(
        study_dir, transport=OPENROUTER_TRANSPORT
    ) as environment:
        # The concrete runner, not the ``OptimizerRunner`` protocol the
        # environment is typed against: this test is about the study's own
        # runner carrying the pin into the specs it builds.
        runner = cast("StudyOptimizerRunner", environment.run_optimizer)
        assert runner is not None
        # The runner carries the design's effort, not a default.
        assert runner.task_reasoning_effort == ReasoningEffort(
            manifest.models.task_reasoning_effort
        )

        # And it reaches the ``RunSpec`` every arm actually runs from --
        # which is what ``run_optimizer`` binds the in-search route out of.
        arm = ArmSpec(
            arm_id="copro",
            optimizer="copro",
            kind=ArmKind.REAL,
            k_run=1,
            seeds=(1,),
            copro_breadth=2,
            copro_depth=1,
        )
        spec = runner._spec_for(arm, seed=1, run_dir=study_dir / "run")
        assert spec.task_reasoning_effort is ReasoningEffort.MINIMAL

        # The route that spec reports into the manifest's witness carries
        # it too, which is what puts the in-search bind under the refusal.
        from dataclasses import replace

        probe = replace(
            runner,
            record_provider_call=lambda config, _policy: recorded.append(
                config
            ),
        )
        probe._record_in_search_route(spec)

    (bound,) = recorded
    assert bound.controls.reasoning is ReasoningEffort.MINIMAL


def test_a_paid_bind_at_an_unpinned_effort_is_refused(
    study_dir: Path,
) -> None:
    """The disagreement is a refusal, not a note for a reader.

    Recording the bound effort makes a mismatch *visible*; it does not
    make it *safe*, because seeing it requires a reader and the reading
    happens after the stage has spent. This turns the same comparison into
    a gate: a paid stage whose task route does not carry the
    pre-registered effort fails before it bills.

    Fails-before: ``_record_provider_call_config`` wrote the disagreeing
    record and returned normally.
    """
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.provider import (
        hardened_execution_policy,
        openrouter_seeded_call_config,
    )
    from whetstone_envs.optim.study.environment import (
        _record_provider_call_config,
    )
    from whetstone_envs.optim.study.stages import StageError

    policy = hardened_execution_policy(
        ReferenceEvalRuntimeConfig(
            transport_api_key_env="OPENROUTER_API_KEY",
        ).execution_policy
    )
    manifest = read_study_manifest(study_dir)
    with pytest.raises(StageError, match="pre-registered reasoning effort"):
        _record_provider_call_config(
            study_dir,
            transport=OPENROUTER_TRANSPORT,
            # The defect the gate exists to catch: a task route bound
            # without the design's effort.
            config=openrouter_seeded_call_config(
                model=manifest.models.task_model
            ),
            policy=policy,
        )
    # Refused before the write, so the manifest is untouched.
    assert read_study_manifest(study_dir) == manifest


def test_the_fake_transport_is_not_held_to_the_pin(
    study_dir: Path,
) -> None:
    """The refusal is about paid routes, not about every bind.

    The fake transport binds whetstone's reference default and never
    reaches a provider, so its recorded effort is not a claim about the
    study's treatment. Holding it to the pin would refuse every free
    rehearsal of a study that pre-registers one.
    """
    with bound_stage_environment(study_dir):
        pass
    (record,) = read_study_manifest(study_dir).models.provider_calls
    assert record.transport == FAKE_TRANSPORT
    assert record.reasoning == PROVIDER_CONTROL_UNSET


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
#:
#: **Rebased 2026-08-23** by protocol Revision 2 item 18, which moved the
#: aggregation config from ``missing_data="propagate"`` to ``"skip"`` with
#: ``max_skip_fraction = 0.10``. The aggregation config is part of the Eval
#: Config, so all three hashes move with it; this is the deliberate
#: recompute the note above calls for, and not a fake path that started
#: carrying provider configuration. The superseded values were
#: ``58d7f579...`` / ``9fecb902...`` / ``df7978da...``, which are what a
#: study initialised before that change recorded.
FAKE_TOY_EVAL_CONFIG_HASHES = {
    "internal": (
        "6e8a5bc3522e167cd3bcc6297e07cec6b8706c971e955a94cd1d76433cfacbba"
    ),
    "official": (
        "3e53c0606186b881958ffeaf4219320e4da1c81ee16d35ab8653a331b562eb21"
    ),
    "held_out": (
        "887b20c7b59928002582b1f375e51c4a4779b3319f04cb09c2f00505915a9e65"
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
    # The seed is the exception, and it is not "provider default": the
    # eval contract puts a derived seed on every call, so the statically
    # bound control is not what reaches the wire.
    assert record.seed == PROVIDER_SEED_DERIVED_PER_CALL


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
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.provider import (
        TASK_CALL_MAX_ATTEMPTS,
        TASK_CALL_TIMEOUT_SECONDS,
        hardened_execution_policy,
        openrouter_seeded_call_config,
    )
    from whetstone_envs.optim.study.environment import _provider_call_record

    record = _provider_call_record(
        transport=OPENROUTER_TRANSPORT,
        config=openrouter_seeded_call_config(model="openai/gpt-5-nano"),
        policy=hardened_execution_policy(
            ReferenceEvalRuntimeConfig(
                transport_api_key_env="OPENROUTER_API_KEY",
            ).execution_policy
        ),
    )
    assert record.transport == OPENROUTER_TRANSPORT
    assert record.provider == "openrouter"
    assert record.model_route == "openai/gpt-5-nano"
    # Recorded verbatim and never set from here. This call binds no effort,
    # so the record says so: the design pins the effort in
    # ``ModelsRecord.task_reasoning_effort`` and the study path passes it
    # in, while this block only ever states what was bound.
    assert record.reasoning == PROVIDER_CONTROL_UNSET
    # The execution settings that were actually in force, so a stage that
    # lost rows to timeouts and one that did not are distinguishable.
    assert record.timeout_seconds == repr(TASK_CALL_TIMEOUT_SECONDS)
    # The *effective* count, not the driver policy's. The driver is pinned
    # to one attempt so the two retry loops cannot multiply; the five
    # attempts are spent inside the transport wrapper, and five is what an
    # operator reconciles billed calls against.
    assert record.max_attempts == str(TASK_CALL_MAX_ATTEMPTS)
    assert "exponential" in record.retry_backoff
    # Only the delta-seconds form is honoured; the HTTP-date form is
    # ignored by design, and the record says so rather than implying the
    # header is honoured in full.
    assert "Retry-After delta honoured" in record.retry_backoff
    assert "HTTP-date ignored" in record.retry_backoff


def test_the_paid_route_records_a_pinned_reasoning_effort() -> None:
    """The manifest's request-side proof that the pin reached the wire.

    A reasoning effort is not checkable from billed tokens -- a provider
    may spend what it likes at any effort, and OpenRouter is known to
    ignore controls on nano routes. What *is* checkable is the config the
    transport was bound with, and this record is where a reader of a live
    study's manifest sees it. A stage that bound the task route without the
    pin records ``PROVIDER_CONTROL_UNSET`` here while
    ``models.task_reasoning_effort`` still says ``minimal``, and the
    disagreement is visible without rerunning anything.

    Fails-before: ``openrouter_seeded_call_config`` took no effort, so this
    record could only ever read "unset".
    """
    from dr_providers import ReasoningEffort
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.provider import (
        hardened_execution_policy,
        openrouter_seeded_call_config,
    )
    from whetstone_envs.optim.study.environment import _provider_call_record

    record = _provider_call_record(
        transport=OPENROUTER_TRANSPORT,
        config=openrouter_seeded_call_config(
            model="openai/gpt-5-nano",
            reasoning_effort=ReasoningEffort.MINIMAL,
        ),
        policy=hardened_execution_policy(
            ReferenceEvalRuntimeConfig(
                transport_api_key_env="OPENROUTER_API_KEY",
            ).execution_policy
        ),
    )
    assert record.reasoning != PROVIDER_CONTROL_UNSET
    assert "minimal" in record.reasoning


# --------------------------------------------------------------------------
# The width is an invocation property recorded per run, and a resume that
# changes it is refused
# --------------------------------------------------------------------------


def _run_at(run_id: str, *, seed: int, width: int) -> RunRecord:
    """One recorded run at a given seed and provider concurrency."""
    pointer = EvidencePointer(schema_name="s/v1", content_hash="f" * 64)
    return RunRecord(
        run_id=run_id,
        seed=seed,
        artifact_dir=f"/tmp/runs/{run_id}",  # noqa: S108
        result_ref=pointer,
        audit_ref=pointer,
        cost_ref=pointer,
        audit_passed=True,
        spend=(),
        transport=OPENROUTER_TRANSPORT,
        provider_concurrency=width,
    )


def _manifest_run_at(
    width: int, *, stage_width: int, seed: int = 1000
) -> StudyManifest:
    """A study whose Stage 1 row and whose surviving run name two widths."""
    arms = tuple(
        arm.model_copy(
            update={
                "runs": (
                    (_run_at(f"{arm.arm_id}-{seed}", seed=seed, width=width),)
                    if arm.arm_id == "copro"
                    else ()
                )
            }
        )
        for arm in _arms()
    )
    return toy_manifest(arms=arms).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE1.value,
                    transport=OPENROUTER_TRANSPORT,
                    provider_concurrency=stage_width,
                ),
            )
        }
    )


def test_a_run_records_the_width_it_ran_at() -> None:
    """**Fails-before: ``RunRecord`` had no width at all.**

    The stage row carried one width for the whole stage, so a resume at a
    new width left every reused run described by a width it never ran at
    -- and there was no field on the run that could have said otherwise.
    """
    run = _run_at("copro-1000", seed=1000, width=32)
    assert run.provider_concurrency == 32
    payload = run.model_dump(mode="json", by_alias=True)
    assert payload["provider_concurrency"] == 32


def test_a_run_record_refuses_a_width_below_one() -> None:
    """A width below one names no run at all, on a run as on a stage."""
    with pytest.raises(ValueError, match="provider concurrency is at least"):
        _run_at("copro-1000", seed=1000, width=0)


def test_a_pre_width_run_record_reads_as_the_recorded_default() -> None:
    """The historical default, pinned as a literal rather than tracked.

    A run written before the field existed ran at whetstone's default of
    the day, which this package pins so a record keeps meaning what it
    said even if the dependency's default moves.
    """
    from whetstone_envs.optim.provider import DEFAULT_PROVIDER_CONCURRENCY

    assert (
        _run_on("copro-1000", transport=FAKE_TRANSPORT).provider_concurrency
        == DEFAULT_PROVIDER_CONCURRENCY
    )


def test_a_resume_at_a_new_width_is_refused_before_dispatch() -> None:
    """**Fails-before: the resume ran and silently rewrote the width.**

    ``StudyOptimizerRunner.__call__`` reuses a run directory it can claim
    rather than re-running it, so a resume at a new width re-ran none of
    the survivors at it -- while ``_arm_stage_record`` overwrote the
    stage's single ``provider_concurrency`` with the new value. The row
    then named a width most of its runs never ran at, and a stage's wall
    time and its rate-limit failures are read against nothing else.
    """
    with pytest.raises(StageError) as excinfo:
        refuse_resumed_width_change(
            _manifest_run_at(16, stage_width=16),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
        )
    message = str(excinfo.value)
    # Both widths, because "the width changed" is not actionable without
    # knowing in which direction.
    assert "16" in message
    assert "64" in message
    # The run that would have been misdescribed, named rather than counted.
    assert "copro-1000" in message
    # Both recoveries, and the override.
    assert "--provider-concurrency 16" in message
    assert "fresh study directory" in message
    assert ALLOW_WIDTH_CHANGE_FLAG in message
    # The width is not design, and the refusal says so rather than
    # leaving an operator to fear they are amending a pre-registration.
    assert "amends no design" in message
    assert "pre-registration hash" in message


def test_the_same_width_resumes_without_a_refusal() -> None:
    """The ordinary resume is untouched: same width, nothing to reconcile."""
    assert (
        refuse_resumed_width_change(
            _manifest_run_at(16, stage_width=16),
            stage=SpecStageId.STAGE1,
            provider_concurrency=16,
        )
        is None
    )


def test_a_stage_with_no_surviving_runs_may_change_width_freely() -> None:
    """Nothing to misdescribe, so nothing to refuse.

    A stage whose runs an amendment dropped, or one that recorded its row
    and died before its first run, has no evidence a new width could
    misdescribe -- and refusing it would strand the study behind a guard
    protecting nothing.
    """
    manifest = toy_manifest(arms=_arms()).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE1.value,
                    transport=OPENROUTER_TRANSPORT,
                    provider_concurrency=16,
                ),
            )
        }
    )
    assert not any(arm.runs for arm in manifest.arms)
    assert (
        refuse_resumed_width_change(
            manifest, stage=SpecStageId.STAGE1, provider_concurrency=64
        )
        is None
    )


def test_a_first_run_of_a_stage_has_no_width_to_disagree_with() -> None:
    """No recorded row, so no earlier width and no refusal."""
    assert (
        refuse_resumed_width_change(
            toy_manifest(arms=_arms()),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
        )
        is None
    )


def _crashed_study(tmp_path: Path, *, runs: int = 1) -> Path:
    """A study directory in the state a crash before the row write leaves.

    Run directories on disk, no stage record in the manifest. Built by
    making the directories directly rather than by running a stage and
    killing it: the state under test is entirely "directories exist, row
    does not", and constructing it is what makes the test assert on that
    state rather than on a process's timing.
    """
    for index in range(runs):
        arm_run_directory(tmp_path, f"copro-seed{1000 + index}").mkdir(
            parents=True
        )
    return tmp_path


def test_run_directories_with_no_stage_row_refuse_the_resume(
    tmp_path: Path,
) -> None:
    """**Fails-before: a missing row returned early and the resume ran.**

    The guard read a missing ``StageRecord`` as "first run of this stage,
    nothing recorded to disagree with" and returned ``None``. But a stage
    writes its row *after* its arms finish, so a crash between the last
    run and ``write_study_manifest`` leaves the run directories on disk
    with no row to say what width produced them -- a state the manifest
    cannot tell apart from a first run. The resume then claimed those
    directories, re-ran nothing, and wrote a row naming a width they may
    never have run at. Unlike the recorded-width case, nothing survives
    that could say which width to re-run at, so this is the worse state
    and it was the one that passed.
    """
    study_dir = _crashed_study(tmp_path)
    manifest = toy_manifest(arms=_arms())
    assert not any(
        entry.stage == SpecStageId.STAGE1.value for entry in manifest.stages
    )
    with pytest.raises(StageError) as excinfo:
        refuse_resumed_width_change(
            manifest,
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            study_dir=study_dir,
        )
    message = str(excinfo.value)
    # The requested width, and the reason the recorded one is absent.
    assert "64" in message
    assert "no stage record" in message
    # The directory that would have been misdescribed, named.
    assert "copro-seed1000" in message
    # Both escapes, plus the fresh-directory recovery.
    assert ALLOW_WIDTH_CHANGE_FLAG in message
    assert DISCARD_STALE_RUNS_FLAG in message
    assert "fresh study directory" in message
    # The width is not design, on this refusal as on the recorded one.
    assert "pre-registration hash" in message


def test_a_crashed_stage_with_no_run_directories_is_a_first_run(
    tmp_path: Path,
) -> None:
    """No row and no directories is the ordinary first run, still allowed.

    The refusal is about surviving evidence of unknown width. A study
    that produced none has nothing a new width could misdescribe, and
    refusing it would strand every first run behind a guard.
    """
    assert (
        refuse_resumed_width_change(
            toy_manifest(arms=_arms()),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            study_dir=tmp_path,
        )
        is None
    )


def test_allow_width_change_accepts_run_directories_of_unknown_width(
    tmp_path: Path,
) -> None:
    """The first escape: record the requested width over them deliberately.

    The note says what the recorded-width note cannot -- that the reused
    runs' width is not merely different but unrecoverable -- so a reader
    of the stage row knows the width describes this invocation and not
    necessarily the runs beneath it.
    """
    note = refuse_resumed_width_change(
        toy_manifest(arms=_arms()),
        stage=SpecStageId.STAGE1,
        provider_concurrency=64,
        allow_width_change=True,
        study_dir=_crashed_study(tmp_path),
    )
    assert note is not None
    assert "64" in note
    assert "no stage record" in note
    assert ALLOW_WIDTH_CHANGE_FLAG in note
    # No amendment, for the same reason the recorded-width note says so.
    assert "does not enter the pre-registration hash" in note
    assert "design is unchanged" in note


def test_discard_stale_runs_accepts_run_directories_of_unknown_width(
    tmp_path: Path,
) -> None:
    """The second escape: the directories are not evidence to preserve.

    An operator who has authorized discarding directories this
    invocation cannot claim has said those runs may go, so the resume
    re-runs rather than reuses them and every run of the stage ends up at
    the requested width. Nothing survives to be misdescribed, so there is
    no refusal and nothing to note.
    """
    assert (
        refuse_resumed_width_change(
            toy_manifest(arms=_arms()),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            discard_stale_runs=True,
            study_dir=_crashed_study(tmp_path),
        )
        is None
    )


def test_the_unrecorded_width_refusal_summarises_beyond_the_shown_bound(
    tmp_path: Path,
) -> None:
    """Actionable without printing every run of the full design.

    The same bound the recorded-width refusal uses, applied to directory
    names rather than run ids.
    """
    from whetstone_envs.optim.study.stages import _WIDTH_RUNS_SHOWN

    extra = 3
    study_dir = _crashed_study(tmp_path, runs=_WIDTH_RUNS_SHOWN + extra)
    with pytest.raises(StageError) as excinfo:
        refuse_resumed_width_change(
            toy_manifest(arms=_arms()),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            study_dir=study_dir,
        )
    message = str(excinfo.value)
    assert f"holds {_WIDTH_RUNS_SHOWN + extra} run" in message
    assert f"(+{extra} more)" in message


def test_a_recorded_runs_directory_is_not_crash_residue(
    tmp_path: Path,
) -> None:
    """**Fails-before: every Stage 2 was refused.**

    ``runs/`` is one directory for the whole study rather than one per
    stage, so a stage's genuine first run finds the previous stage's
    directories in it as a matter of course. Reading any surviving
    directory as unattributable width made the check refuse exactly the
    ordinary case it was meant to leave alone: Stage 2, whose whole
    design is to stand on Stage 1's runs. A recorded run carries its
    width on its own ``RunRecord``, so there is nothing to recover.
    """
    study_dir = _crashed_study(tmp_path)
    # A manifest with the run recorded but no stage row, which is what a
    # Stage 2 standing on Stage 1's runs looks like to this check.
    recorded = toy_manifest(
        arms=tuple(
            arm.model_copy(
                update={
                    "runs": (
                        (_run_at("copro-seed1000", seed=1000, width=16),)
                        if arm.arm_id == "copro"
                        else ()
                    )
                }
            )
            for arm in _arms()
        )
    )
    # The manifest records exactly the directory that is on disk.
    assert {run.run_id for arm in recorded.arms for run in arm.runs} == {
        "copro-seed1000"
    }
    assert (
        refuse_resumed_width_change(
            recorded,
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            study_dir=study_dir,
        )
        is None
    )


def test_a_directory_an_amendment_orphaned_is_left_to_the_stale_refusal(
    tmp_path: Path,
) -> None:
    """**Fails-before: the width refusal pre-empted the stale-run one.**

    ``--replace-design`` drops runs from the manifest and records exactly
    which directories it left behind, so the study already says why they
    are unrecorded. They belong to the stale-run refusal, which reads
    each directory's own identity and names ``--discard-stale-runs``
    against it; a width refusal firing first would replace that specific
    message with a vaguer one about an unknown width.
    """
    study_dir = _crashed_study(tmp_path)
    orphaned = str(arm_run_directory(study_dir, "copro-seed1000"))
    manifest = toy_manifest(arms=_arms()).model_copy(
        update={
            "amendments": (
                AmendmentRecord(
                    at="2026-08-24T12:00:00+00:00",
                    amended_stage=StageId.STAGE0.value,
                    reason=AMENDMENT_REASON_TRANSPORT_CHANGE,
                    from_transport=FAKE_TRANSPORT,
                    to_transport=OPENROUTER_TRANSPORT,
                    dropped_stages=(StageId.STAGE1.value,),
                    dropped_run_ids=("copro-seed1000",),
                    dropped_run_directories=(orphaned,),
                    dropped_selections=1,
                    dropped_held_out_claims=0,
                    dropped_held_out_rows=0,
                    dropped_call_count_gate=False,
                    dropped_official_scores=1,
                    dropped_report_spend=0,
                ),
            )
        }
    )
    assert (
        refuse_resumed_width_change(
            manifest,
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
            study_dir=study_dir,
        )
        is None
    )


def test_the_crash_residue_refusal_survives_the_cli_wiring(
    study_dir: Path, no_openrouter_key: None, capsys
) -> None:
    """End to end, because the refusal is only useful if it is reached.

    The state is built by running Stage 0 and Stage 1 for real and then
    rewinding the manifest to what it held before Stage 1's write, while
    leaving Stage 1's run directories on disk -- which is exactly what a
    crash between the last run and ``write_study_manifest`` leaves,
    reconstructed by editing the ledger rather than by killing a process
    mid-write. The rewind drops the stage row, the runs, the selections,
    and the held-out entries together, because one
    ``write_study_manifest`` call records them all: a surviving run entry
    would account for its own directory, a surviving selection would name
    a run the manifest no longer holds, and a surviving held-out claim
    would make the resume refuse for an unrelated reason.

    Reaches no provider: everything runs on the fake transport.
    """
    del no_openrouter_key
    for stage in (StageId.STAGE0, StageId.STAGE1):
        assert _run_stage(study_dir, stage.value, FAKE_TRANSPORT) == EXIT_OK
    ran = read_study_manifest(study_dir)
    crashed = ran.model_copy(
        update={
            "stages": tuple(
                entry
                for entry in ran.stages
                if entry.stage != StageId.STAGE1.value
            ),
            "arms": tuple(
                arm.model_copy(update={"runs": ()}) for arm in ran.arms
            ),
            "selection": (),
            "held_out_claims": (),
            "held_out": (),
        }
    )
    write_study_manifest(study_dir, crashed, replace=True)
    # The directories the dropped runs left behind are still there, which
    # is the whole reason the manifest alone cannot see this state.
    assert any(
        arm_run_directory(study_dir, run.run_id).is_dir()
        for arm in ran.arms
        for run in arm.runs
    )

    assert _run_stage(study_dir, StageId.STAGE1.value, FAKE_TRANSPORT) == (
        EXIT_CHECK_FAILED
    )
    error = capsys.readouterr().err
    assert "no stage record" in error
    assert ALLOW_WIDTH_CHANGE_FLAG in error
    assert DISCARD_STALE_RUNS_FLAG in error
    # It refused before dispatch, so the manifest is still the crashed one.
    assert not read_study_manifest(study_dir).arms[0].runs

    # And the escape the message names actually recovers.
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
                ALLOW_WIDTH_CHANGE_FLAG,
            ]
        )
        == EXIT_OK
    )
    resumed = read_study_manifest(study_dir)
    assert resumed.arms[0].runs
    # The authorized change is recorded rather than silent.
    stage1 = {entry.stage: entry for entry in resumed.stages}[
        StageId.STAGE1.value
    ]
    assert stage1.width_change_notes


def test_without_a_study_dir_the_guard_reads_only_the_manifest(
    tmp_path: Path,
) -> None:
    """A caller that passes no directory cannot be refused for one.

    The directory check is an addition to a manifest-only guard, and a
    caller holding only a manifest -- the reporting paths, and every
    existing test of the recorded-width case -- keeps the behaviour it
    had. The refusal is reached from the arm stage, which always has the
    study directory.
    """
    _crashed_study(tmp_path)
    assert (
        refuse_resumed_width_change(
            toy_manifest(arms=_arms()),
            stage=SpecStageId.STAGE1,
            provider_concurrency=64,
        )
        is None
    )


def test_the_override_records_the_change_instead_of_refusing() -> None:
    """**Fails-before: there was no override, because there was no refusal.**

    ``--allow-width-change`` is the operator asserting the change is
    deliberate. It returns the note the stage records rather than raising,
    and the note states that no design was amended: the width never
    entered the pre-registration hash, so a study that changed it
    pre-registers exactly as it did before.
    """
    note = refuse_resumed_width_change(
        _manifest_run_at(16, stage_width=16),
        stage=SpecStageId.STAGE1,
        provider_concurrency=64,
        allow_width_change=True,
    )
    assert note is not None
    assert "16" in note
    assert "64" in note
    assert ALLOW_WIDTH_CHANGE_FLAG in note
    # No amendment: the width is an invocation property, so the study's
    # pre-registration is untouched and the note says so.
    assert "does not enter the pre-registration hash" in note
    assert "design is unchanged" in note


def test_an_authorized_change_is_appended_to_the_stages_notes() -> None:
    """Append-only, so a stage narrowed and re-widened records both.

    Assigning would report the latest change as the stage's whole
    history, which is the same defect the attempt counters avoid by
    folding rather than replacing.
    """
    first = "narrowed from 64 to 16"
    manifest = _manifest_run_at(16, stage_width=16).model_copy(
        update={
            "stages": (
                StageRecord(
                    stage=StageId.STAGE1.value,
                    transport=OPENROUTER_TRANSPORT,
                    provider_concurrency=16,
                    width_change_notes=(first,),
                ),
            )
        }
    )
    merged = _arm_stage_record(
        manifest,
        _stage_record(
            stage=SpecStageId.STAGE1,
            environment=_paid_environment(),
            run_spend=(),
        ),
        width_change_note="widened from 16 to 64",
    )
    kept = {entry.stage: entry for entry in merged}[SpecStageId.STAGE1.value]
    assert kept.width_change_notes == (first, "widened from 16 to 64")


def test_the_stage_row_records_this_invocations_width() -> None:
    """A single field can only name one width, and this is the honest one.

    It is what the runs this invocation executed ran at and what its
    reporting pass ran at. The runs it kept carry their own widths on
    their own records, which is the fact the stage-level field
    structurally cannot hold.
    """
    from dataclasses import replace

    manifest = _manifest_run_at(16, stage_width=16)
    merged = _arm_stage_record(
        manifest,
        _stage_record(
            stage=SpecStageId.STAGE1,
            environment=replace(_paid_environment(), provider_concurrency=64),
            run_spend=(),
        ),
        width_change_note="widened from 16 to 64",
    )
    kept = {entry.stage: entry for entry in merged}[SpecStageId.STAGE1.value]
    assert kept.provider_concurrency == 64


def test_the_ledger_names_the_per_run_widths_when_they_differ() -> None:
    """**Fails-before: the ledger printed one width and nothing else.**

    A study resumed under ``--allow-width-change`` really does hold runs
    from two widths, and a reader seeing only the stage column would take
    it for the width every run beneath it ran at.
    """
    manifest = _manifest_run_at(16, stage_width=64)
    mixed = tuple(
        arm.model_copy(
            update={
                "runs": (
                    *arm.runs,
                    _run_at("copro-1001", seed=1001, width=64),
                )
            }
        )
        if arm.arm_id == "copro"
        else arm
        for arm in manifest.arms
    )
    text = "\n".join(stage_spend_lines(manifest.stages, arms=mixed))
    assert MIXED_RUN_WIDTHS in text
    assert "16, 64" in text


def test_the_ledger_stays_quiet_when_every_run_ran_at_one_width() -> None:
    """The exception is stated; the ordinary case is not annotated.

    Appending "recorded at widths 64" to every study would turn the
    exception into noise and hide the case the note exists for.
    """
    manifest = _manifest_run_at(16, stage_width=16)
    text = "\n".join(stage_spend_lines(manifest.stages, arms=manifest.arms))
    assert MIXED_RUN_WIDTHS not in text


def test_the_report_names_the_per_run_widths_when_they_differ() -> None:
    """The report packet asks the same question the ledger does."""
    manifest = _manifest_run_at(16, stage_width=64)
    mixed = tuple(
        arm.model_copy(
            update={
                "runs": (
                    *arm.runs,
                    _run_at("copro-1001", seed=1001, width=64),
                )
            }
        )
        if arm.arm_id == "copro"
        else arm
        for arm in manifest.arms
    )
    assert _mixed_width_detail(
        manifest.model_copy(update={"arms": mixed})
    ) == (f" ({MIXED_RUN_WIDTHS_DETAIL} 16, 64)")
    assert _mixed_width_detail(manifest) == ""


def test_run_widths_is_the_distinct_set_the_runs_ran_at() -> None:
    """Sorted and deduplicated: the question is a set, not a sequence."""
    manifest = _manifest_run_at(16, stage_width=16)
    assert run_widths(manifest.arms) == (16,)
    assert run_widths(()) == ()
