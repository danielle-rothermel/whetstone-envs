"""The pre-registered design, and the rules that keep it runnable."""

from __future__ import annotations

import pytest

from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    HOLM_FAMILY_SIZE,
    K_CAL_CAP,
    K_CAL_INITIAL,
    NULL_ARM_IDS,
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
)


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
