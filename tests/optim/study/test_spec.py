"""The pre-registered design, and the rules that keep it runnable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    HOLM_FAMILY_SIZE,
    K_CAL_CAP,
    K_CAL_INITIAL,
    NULL_ARM_IDS,
    PROTOCOL_SPLIT_SIZES,
    PROTOCOL_TRAIN_SIZE,
    PROTOCOL_VAL_SIZE,
    REAL_OPTIMIZER_ARM_IDS,
    SEED_RANGE_BY_OPTIMIZER,
    ArmKind,
    ArmSpec,
    SplitSpec,
    StageId,
    StudySpec,
    arm_seeds,
    default_arms,
    k_run_for,
    next_k_cal,
    spec_from_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _spec(
    *,
    k_cal: int = K_CAL_INITIAL,
    held_out: SplitSpec | None = None,
    arms: tuple[ArmSpec, ...] | None = None,
) -> StudySpec:
    """The study's real design, with one field optionally mutated."""
    return StudySpec(
        study_id="step10",
        family="c19",
        n_per_stratum=32,
        pool_seed_start=1_000_000,
        internal=SplitSpec("internal", 88),
        official=SplitSpec("official", 132),
        held_out=SplitSpec("held_out", 220) if held_out is None else held_out,
        task_model="openai/gpt-5-nano",
        proposer_model="openai/gpt-5.4-nano",
        k_cal=k_cal,
        arms=default_arms(stage=StageId.STAGE2) if arms is None else arms,
    )


def test_k_cal_defaults_to_four_with_a_cap_of_sixteen() -> None:
    """D1: the calibration count, distinct from the design repeat count."""
    assert K_CAL_INITIAL == 4
    assert K_CAL_CAP == 16
    spec = _spec()
    assert spec.k_cal == K_CAL_INITIAL
    assert spec.k_cal != spec.k_repeat


def test_the_doubling_rule_runs_4_8_16_then_refuses() -> None:
    assert next_k_cal(4) == 8
    assert next_k_cal(8) == 16
    with pytest.raises(ValueError, match="capped at 16"):
        next_k_cal(16)


def test_an_odd_k_cal_is_refused_because_split_half_needs_halves() -> None:
    with pytest.raises(ValueError, match="split-half"):
        _spec(k_cal=5)


def test_k_cal_outside_the_pre_registered_range_is_refused() -> None:
    with pytest.raises(ValueError, match="k_cal must be between"):
        _spec(k_cal=2)
    with pytest.raises(ValueError, match="k_cal must be between"):
        _spec(k_cal=32)


def test_run_counts_follow_the_adopted_decisions() -> None:
    """D4: null-A gets the full repeat count, null-B runs exactly once."""
    assert k_run_for("copro", stage=StageId.STAGE1) == 2
    assert k_run_for("copro", stage=StageId.STAGE2) == 5
    assert k_run_for("null-random", stage=StageId.STAGE2) == 5
    assert k_run_for("null-identity", stage=StageId.STAGE2) == 1
    assert k_run_for("null-identity", stage=StageId.STAGE1) == 1


def test_stage0_runs_no_optimizers() -> None:
    with pytest.raises(ValueError, match="runs no optimizers"):
        k_run_for("copro", stage=StageId.STAGE0)


def test_seed_ranges_are_disjoint_across_arms() -> None:
    """Disjointness is what lets a seed identify its arm."""
    assigned: list[int] = []
    for optimizer in SEED_RANGE_BY_OPTIMIZER:
        stage = StageId.STAGE2
        assigned.extend(arm_seeds(optimizer, stage=stage))
    assert len(set(assigned)) == len(assigned)


def test_stage2_seeds_extend_stage1_rather_than_replacing_them() -> None:
    """Stage 1's runs count toward Stage 2: same code, same seeds."""
    pilot = arm_seeds("copro", stage=StageId.STAGE1)
    full = arm_seeds("copro", stage=StageId.STAGE2)
    assert full[: len(pilot)] == pilot
    assert pilot == (1000, 1001)
    assert full == (1000, 1001, 1002, 1003, 1004)


def test_an_unknown_optimizer_has_no_seed_range() -> None:
    with pytest.raises(ValueError, match="unknown optimizer"):
        arm_seeds("copro-v2", stage=StageId.STAGE2)


def test_the_holm_family_is_the_four_real_optimizers() -> None:
    spec = _spec()
    assert len(REAL_OPTIMIZER_ARM_IDS) == HOLM_FAMILY_SIZE
    assert tuple(arm.arm_id for arm in spec.real_arms) == (
        REAL_OPTIMIZER_ARM_IDS
    )
    assert tuple(arm.arm_id for arm in spec.null_arms) == NULL_ARM_IDS
    assert all(arm.kind is ArmKind.NULL for arm in spec.null_arms)


def test_the_codex_cap_is_the_adopted_eight() -> None:
    """D2, named here so no caller re-decides it per run."""
    assert CODEX_EVALUATE_CALL_CAP == 8


def test_the_protocol_split_sizes_are_pinned() -> None:
    """The study's three sizes, as literals.

    Pinned rather than derived, and deliberately *not* the c19 generation
    default: ``whetstone_envs.c19.generation.DEFAULT_SPLIT_SIZES`` is
    ``(88, 132, 132)``, which is what the generator returns when nobody
    asks. The protocol pre-registered a held-out split of 440 because the
    design's MDE depends on it, so the two must be allowed to differ and a
    test that derived one from the other would hide the difference.
    """
    from whetstone_envs.c19.generation import DEFAULT_SPLIT_SIZES

    assert PROTOCOL_SPLIT_SIZES == (88, 132, 440)
    assert PROTOCOL_SPLIT_SIZES != DEFAULT_SPLIT_SIZES


def test_the_protocol_partition_covers_the_protocol_internal_split() -> None:
    """GEPA's coverage rule, checked against the protocol's own numbers.

    GEPA requires ``train + val == internal`` exactly. If the protocol's
    44/44 ever stopped covering its internal 88, every GEPA arm in the
    study would refuse at spec validation -- so the relationship is pinned
    here rather than discovered at Stage 1.
    """
    assert PROTOCOL_SPLIT_SIZES[0] == (PROTOCOL_TRAIN_SIZE + PROTOCOL_VAL_SIZE)


def test_split_sizes_reach_the_runner_as_one_triple() -> None:
    assert _spec().split_sizes == (88, 132, 220)


def test_a_study_without_a_held_out_split_is_refused() -> None:
    """The study reports from held-out; a zero-size one has nothing to say."""
    with pytest.raises(ValueError, match="reports from held-out"):
        _spec(held_out=SplitSpec("held_out", 0))


def test_a_split_whose_hashes_contradict_its_size_is_refused() -> None:
    with pytest.raises(ValueError, match="task hashes for a declared size"):
        SplitSpec("held_out", 3, ("a", "b"))


def test_a_split_that_repeats_a_task_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats a task hash"):
        SplitSpec("held_out", 2, ("a", "a"))


def test_an_arm_whose_seeds_disagree_with_its_run_count_is_refused() -> None:
    with pytest.raises(ValueError, match="declares 2 runs but 1 seeds"):
        ArmSpec(
            arm_id="copro",
            optimizer="copro",
            kind=ArmKind.REAL,
            k_run=2,
            seeds=(1000,),
        )


def test_duplicate_arm_ids_are_refused() -> None:
    arm = ArmSpec(
        arm_id="copro",
        optimizer="copro",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(1000,),
    )
    with pytest.raises(ValueError, match="unique by arm_id"):
        _spec(arms=(arm, arm))


def test_default_arms_never_drop_the_nulls() -> None:
    for stage in (StageId.STAGE1, StageId.STAGE2):
        arm_ids = tuple(arm.arm_id for arm in default_arms(stage=stage))
        assert set(NULL_ARM_IDS) <= set(arm_ids)


# --------------------------------------------------------------------------
# Recovering the design from a manifest
# --------------------------------------------------------------------------


def test_an_unstarted_study_recovers_its_pre_registered_run_counts() -> None:
    """The bug this guards: reading `K_RUN` off `len(arm.runs)` makes a
    study that has not started look like a one-run-per-arm study, and
    under-reports the whole budget by a factor of `K_RUN`."""
    from .conftest import toy_manifest

    spec = spec_from_manifest(toy_manifest())
    assert spec.k_run_by_arm == {"copro": 5, "null-identity": 1}
    assert spec.arm_ids == ("copro", "null-identity")


def test_the_pilot_prices_the_pilot_not_the_full_design() -> None:
    from .conftest import toy_manifest

    pilot = spec_from_manifest(toy_manifest(), stage=StageId.STAGE1)
    assert pilot.k_run_by_arm == {"copro": 2, "null-identity": 1}


def test_a_recorded_design_wins_over_the_table() -> None:
    """Stage 0 may have recorded one permitted adjustment; that is the
    study's design, not whatever the table said before it ran."""
    from whetstone_envs.optim.study.manifest import DesignRecord

    from .conftest import toy_manifest

    manifest = toy_manifest()
    adjusted = manifest.model_copy(
        update={
            "design": DesignRecord(
                k_cal=4,
                k_repeat=3,
                k_run_by_arm={"copro": 3, "null-identity": 1},
                ci_level=0.95,
                resamples=10_000,
                bootstrap_seed=0,
                correction="holm-bonferroni",
                m=4,
                mde_formula="MDE(T, K) = ...",
                mde_measured=0.1,
                tau_sq=0.01,
                sigma_sq=0.02,
                completeness_rule="achieved-count-weighted-per-task-delta",
                completeness_backstop=0.90,
            )
        }
    )
    assert spec_from_manifest(adjusted).k_run_by_arm == {
        "copro": 3,
        "null-identity": 1,
    }


def test_an_arm_naming_an_unseeded_optimizer_is_refused() -> None:
    """Every other read of the seed table raises on an unknown optimizer;
    seeding one from zero would collide with anything else that defaulted
    the same way."""
    from whetstone_envs.optim.study.manifest import ArmRecord

    from .conftest import toy_manifest

    manifest = toy_manifest(
        arms=(
            ArmRecord(
                arm_id="mystery",
                optimizer="not-an-optimizer",
                demo_mode=None,
                train_size=None,
                val_size=None,
                control_identity_hash="f" * 64,
                seed_note="provider-seed-control-only",
                runs=(),
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown optimizer"):
        spec_from_manifest(manifest)


# --------------------------------------------------------------------------
# Per-arm MIPROv2 settings
# --------------------------------------------------------------------------


def test_an_arm_carries_no_miprov2_settings_by_default() -> None:
    """Unset means "keep the runner's default", not "pin it here twice"."""
    arm = ArmSpec(
        arm_id="miprov2",
        optimizer="miprov2",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(2000,),
        train_size=2,
        val_size=2,
    )
    assert arm.miprov2_num_trials is None
    assert arm.miprov2_num_candidates is None


def test_an_arm_can_request_the_protocol_search_shape() -> None:
    arm = ArmSpec(
        arm_id="miprov2",
        optimizer="miprov2",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(2000,),
        miprov2_num_trials=10,
        miprov2_num_candidates=6,
        train_size=2,
        val_size=2,
    )
    assert arm.miprov2_num_trials == 10
    assert arm.miprov2_num_candidates == 6


@pytest.mark.parametrize(
    ("trials", "candidates"),
    [(10, None), (None, 6)],
)
def test_miprov2_settings_are_refused_on_another_arms_optimizer(
    trials: int | None, candidates: int | None
) -> None:
    """A setting nothing reads must not look honoured on a COPRO arm."""
    with pytest.raises(ValueError, match="sets MIPROv2 settings"):
        ArmSpec(
            arm_id="copro",
            optimizer="copro",
            kind=ArmKind.REAL,
            k_run=1,
            seeds=(1000,),
            miprov2_num_trials=trials,
            miprov2_num_candidates=candidates,
        )


@pytest.mark.parametrize(
    ("trials", "candidates", "message"),
    [
        (0, None, "miprov2_num_trials must be at least 1"),
        (None, 0, "miprov2_num_candidates must be at least 1"),
    ],
)
def test_an_arm_refuses_an_invalid_miprov2_setting(
    trials: int | None,
    candidates: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ArmSpec(
            arm_id="miprov2",
            optimizer="miprov2",
            kind=ArmKind.REAL,
            k_run=1,
            seeds=(2000,),
            miprov2_num_trials=trials,
            miprov2_num_candidates=candidates,
            train_size=2,
            val_size=2,
        )


@pytest.mark.parametrize("optimizer", ["miprov2", "gepa"])
def test_an_arm_with_a_train_val_concept_must_declare_the_split(
    optimizer: str,
) -> None:
    """A design field: the arm states what it trained and scored on."""
    with pytest.raises(ValueError, match="must declare train_size"):
        ArmSpec(
            arm_id=optimizer,
            optimizer=optimizer,
            kind=ArmKind.REAL,
            k_run=1,
            seeds=(2000,),
        )


def test_an_arm_without_a_train_val_concept_refuses_a_split() -> None:
    with pytest.raises(ValueError, match="sets a train/val split"):
        ArmSpec(
            arm_id="copro",
            optimizer="copro",
            kind=ArmKind.REAL,
            k_run=1,
            seeds=(1000,),
            train_size=2,
            val_size=2,
        )


# --------------------------------------------------------------------------
# The rebuilt spec must match the pinned split
# --------------------------------------------------------------------------


def _pinned_study(tmp_path: Path) -> Path:
    """A study whose Stage 0 pinned a pre-registration.

    The arm carrying the split is MIPROv2 rather than the toy manifest's
    COPRO: COPRO has no train/val concept, so ``ArmSpec`` refuses a split
    on it outright and a mismatch test built on it would pass on that
    unrelated refusal whether or not the pinned-split check existed.
    """
    pytest.importorskip("whetstone.experiment.env")
    from whetstone_envs.optim.study.environment import bound_stage_environment
    from whetstone_envs.optim.study.manifest import (
        ArmRecord,
        write_study_manifest,
    )
    from whetstone_envs.optim.study.stages import run_stage0_into_manifest

    from .conftest import TOY_TRAIN_SIZE, TOY_VAL_SIZE, toy_arms, toy_manifest

    miprov2 = ArmRecord(
        arm_id="miprov2",
        optimizer="miprov2",
        demo_mode=None,
        train_size=TOY_TRAIN_SIZE,
        val_size=TOY_VAL_SIZE,
        control_identity_hash="a" * 64,
        seed_note="provider-seed-control-only",
        runs=(),
    )
    study_dir = tmp_path / "study"
    write_study_manifest(study_dir, toy_manifest(arms=(*toy_arms(), miprov2)))
    with bound_stage_environment(study_dir) as environment:
        run_stage0_into_manifest(study_dir=study_dir, environment=environment)
    return study_dir


def test_arm_records_disagreeing_with_the_pinned_split_are_refused(
    tmp_path: Path,
) -> None:
    """The pinned block is the truth; the arm record is not protected.

    Fails-before: Stages 1 and 2 rebuilt each arm's runnable spec from
    ``ArmRecord.train_size``/``val_size`` -- ordinary mutable fields,
    rewritten every time a stage merges runs -- while
    ``pre_registration.split_by_arm`` is immutable and hashed, and the two
    were never compared. An edited record therefore ran MIPROv2 or GEPA at
    a partition the design never registered, under a design hash that
    still validated.
    """
    from whetstone_envs.optim.study.manifest import (
        PreRegistrationViolationError,
        read_study_manifest,
    )

    manifest = read_study_manifest(_pinned_study(tmp_path))
    pinned = manifest.pre_registration
    assert pinned is not None
    # Rewrite the MIPROv2 arm's recorded partition without touching the
    # pinned block, which is exactly the drift the check exists to catch.
    # The rewritten split is still a legal one for this optimizer, so the
    # only thing that can refuse it is the comparison under test.
    assert pinned.split_by_arm["miprov2"] == (2, 2)
    edited = tuple(
        arm.model_copy(update={"train_size": 1, "val_size": 3})
        if arm.arm_id == "miprov2"
        else arm
        for arm in manifest.arms
    )
    with pytest.raises(PreRegistrationViolationError, match="split_by_arm"):
        spec_from_manifest(manifest.model_copy(update={"arms": edited}))


def test_an_arm_the_pre_registration_never_named_is_refused(
    tmp_path: Path,
) -> None:
    """An arm added after pinning would spend on an unregistered design.

    ``split_by_arm`` names exactly the arms the design declared, so an arm
    absent from it has no pinned partition, run count, or place in the
    correction family. Checked separately from the split comparison
    because Stage 0 legitimately sees this state while writing an
    amendment.
    """
    from whetstone_envs.optim.study.manifest import (
        PreRegistrationViolationError,
        read_study_manifest,
    )
    from whetstone_envs.optim.study.spec import require_pinned_arms

    manifest = read_study_manifest(_pinned_study(tmp_path))
    added = manifest.model_copy(
        update={
            "arms": (
                *manifest.arms,
                manifest.arms[0].model_copy(update={"arm_id": "gepa"}),
            )
        }
    )
    with pytest.raises(PreRegistrationViolationError, match="not named"):
        require_pinned_arms(added)
    # The loader itself stays permissive, which is what keeps
    # ``stage0 --replace-design`` able to rebuild a spec over the new arm
    # before it writes the block that pins it.
    assert spec_from_manifest(added).arms


def test_a_manifest_agreeing_with_its_pinned_split_still_loads(
    tmp_path: Path,
) -> None:
    """The control: the check refuses drift, not every pinned study."""
    from whetstone_envs.optim.study.manifest import read_study_manifest

    manifest = read_study_manifest(_pinned_study(tmp_path))
    assert spec_from_manifest(manifest).arms
