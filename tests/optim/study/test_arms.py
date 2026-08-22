"""How a study arm's settings reach the runner's ``RunSpec``.

``StudyOptimizerRunner`` is the seam between the study's arms and the shared
optimizer runner. An arm setting that never reaches ``RunSpec`` would look
honoured in the manifest while the run ignored it, which is the failure the
per-arm validation in ``spec.py`` exists to prevent -- so the forwarding
itself is pinned here rather than assumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
)
from whetstone_envs.optim.study.arms import StudyOptimizerRunner
from whetstone_envs.optim.study.spec import ArmKind, ArmSpec

if TYPE_CHECKING:
    from pathlib import Path


def _runner(tmp_path: Path) -> StudyOptimizerRunner:
    return StudyOptimizerRunner(
        study_dir=tmp_path,
        family_id="c19",
        transport="fake",
        split_sizes=(2, 2, 0),
        n_per_stratum=2,
        pool_seed_start=1,
        task_model="openai/gpt-4.1-nano",
        proposer_model="openai/gpt-4.1-nano",
        num_seeds=1,
        naive_template="naive {input}",
        store_path=tmp_path / "store.sqlite",
    )


def _arm(
    *,
    miprov2_num_trials: int | None = None,
    miprov2_num_candidates: int | None = None,
    train_size: int = 1,
    val_size: int = 1,
) -> ArmSpec:
    return ArmSpec(
        arm_id="miprov2",
        optimizer="miprov2",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(2000,),
        miprov2_num_trials=miprov2_num_trials,
        miprov2_num_candidates=miprov2_num_candidates,
        train_size=train_size,
        val_size=val_size,
    )


def test_an_unset_arm_leaves_the_runner_defaults_in_place(
    tmp_path: Path,
) -> None:
    """Not forwarding an unset field is what keeps one default, not two."""
    spec = _runner(tmp_path)._spec_for(
        _arm(), seed=2000, run_dir=tmp_path / "run"
    )
    assert spec.miprov2_num_trials == DEFAULT_MIPROV2_NUM_TRIALS
    assert spec.miprov2_num_candidates == DEFAULT_MIPROV2_NUM_CANDIDATES
    # The split is not a runner default: the arm always states it.
    assert (spec.train_size, spec.val_size) == (1, 1)


def test_an_arms_miprov2_settings_reach_the_run_spec(
    tmp_path: Path,
) -> None:
    """The protocol's auto-light shape, requested by the arm."""
    arm = _arm(miprov2_num_trials=10, miprov2_num_candidates=6)
    spec = _runner(tmp_path)._spec_for(
        arm, seed=2000, run_dir=tmp_path / "run"
    )
    assert spec.miprov2_num_trials == 10
    assert spec.miprov2_num_candidates == 6
    # And the resulting spec is one the runner accepts.
    assert spec.optimizer == "miprov2"


def test_a_partially_set_arm_forwards_only_what_it_set(
    tmp_path: Path,
) -> None:
    spec = _runner(tmp_path)._spec_for(
        _arm(miprov2_num_trials=10), seed=2000, run_dir=tmp_path / "run"
    )
    assert spec.miprov2_num_trials == 10
    assert spec.miprov2_num_candidates == DEFAULT_MIPROV2_NUM_CANDIDATES


# --------------------------------------------------------------------------
# The Codex arm
# --------------------------------------------------------------------------


def _codex_arm() -> ArmSpec:
    return ArmSpec(
        arm_id="codex",
        optimizer="codex",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(4000,),
    )


def test_the_codex_arm_runs_through_the_shared_runner(
    tmp_path: Path,
) -> None:
    """The Codex arm is a ``RunSpec`` like any other, not a side path.

    Every arm reaches the same ``run_optimizer``; what makes Codex
    different is its adapter, not its dispatch. Its arm id is admitted by
    ``OPTIMIZER_ARM_IDS``, so it is not refused as a control.
    """
    from whetstone_envs.optim.study.arms import OPTIMIZER_ARM_IDS

    assert "codex" in OPTIMIZER_ARM_IDS
    spec = _runner(tmp_path)._spec_for(
        _codex_arm(), seed=4000, run_dir=tmp_path / "run"
    )
    assert spec.optimizer == "codex"
    assert spec.seed == 4000
    assert spec.transport == "fake"


def test_the_studys_capacity_cap_reaches_the_codex_arm(
    tmp_path: Path,
) -> None:
    """D2's cap is the study's to set, and it must actually be carried."""
    from dataclasses import replace

    from whetstone_envs.optim.study.spec import CODEX_EVALUATE_CALL_CAP

    runner = replace(_runner(tmp_path), codex_capacity=CODEX_EVALUATE_CALL_CAP)
    spec = runner._spec_for(_codex_arm(), seed=4000, run_dir=tmp_path / "run")
    assert spec.codex_capacity == CODEX_EVALUATE_CALL_CAP


def test_the_capacity_cap_is_not_forwarded_to_another_arm(
    tmp_path: Path,
) -> None:
    """A cap on a COPRO spec would be refused at validation, rightly.

    Forwarding it anyway would turn a study-level setting into a run
    failure for every non-Codex arm, so the arm is what gates it.
    """
    from dataclasses import replace

    runner = replace(_runner(tmp_path), codex_capacity=8)
    spec = runner._spec_for(_arm(), seed=2000, run_dir=tmp_path / "run")
    assert spec.codex_capacity is None


def test_the_study_capacity_cap_agrees_with_the_arms_own() -> None:
    """One number, named in the study spec and defaulted in the builder.

    Two constants that could drift would let a study advertise one cap
    and a run enforce another.
    """
    from whetstone_envs.optim.codex import (
        CODEX_EVALUATE_CALL_CAP as BUILDER_CAP,
    )
    from whetstone_envs.optim.study.spec import (
        CODEX_EVALUATE_CALL_CAP as STUDY_CAP,
    )

    assert BUILDER_CAP == STUDY_CAP == 8


def test_the_study_cannot_dispatch_a_paid_codex_run(tmp_path: Path) -> None:
    """An unauthorized runner refuses the Codex arm rather than billing it.

    ``StudyOptimizerRunner`` calls ``run_optimizer`` with a ``RunSpec``
    and nothing else -- there is no ``codex_test_seam`` parameter on the
    stage-harness protocol and no field on ``ArmSpec`` that could build
    one. Without an authorization the runner therefore forwards
    ``allow_real_codex=False``, and ``run_optimizer`` refuses before any
    preflight, adapter, or subprocess exists.

    **The authorization is the runner's, not the arm's.** It is a run-time
    permission to spend on this invocation rather than a design choice, so
    it lives on ``StudyOptimizerRunner`` (set from ``whetstone-study run
    --allow-real-codex``) and deliberately not on ``ArmSpec``: putting it
    on the arm would push it into the manifest and the pre-registration
    hash, making two runs of one design pre-register differently. The
    environment variable is the other half of the gate and neither half
    authorizes spend alone.
    """
    from whetstone_envs.optim.codex import RealCodexRefusedError

    runner = _runner(tmp_path)
    arm = _codex_arm()
    with pytest.raises(RealCodexRefusedError):
        runner(arm=arm, seed=4000, study_dir=tmp_path)


def test_an_authorized_runner_forwards_the_opt_in_to_the_codex_arm(
    tmp_path: Path,
) -> None:
    """The flag reaches ``RunSpec``, which is what lifts the refusal.

    Fails-before: ``_spec_for`` forwarded no ``allow_real_codex`` at all
    and ``ArmSpec`` had no such field, so a Stage 1 or Stage 2 whose design
    names the Codex arm could not be authorized by any means -- it aborted
    at the Codex arm's turn, after the earlier arms had already been paid
    for.
    """
    from dataclasses import replace

    runner = replace(_runner(tmp_path), allow_real_codex=True)
    spec = runner._spec_for(_codex_arm(), seed=4000, run_dir=tmp_path / "run")
    assert spec.allow_real_codex is True


def test_the_opt_in_is_not_forwarded_to_another_arm(tmp_path: Path) -> None:
    """Codex settings are refused on other optimizers, this one included.

    Forwarding it unconditionally would turn one authorized stage into a
    ``RunSpec`` validation failure on every non-Codex arm.
    """
    from dataclasses import replace

    runner = replace(_runner(tmp_path), allow_real_codex=True)
    spec = runner._spec_for(_arm(), seed=2000, run_dir=tmp_path / "run")
    assert spec.allow_real_codex is False


def test_the_authorization_is_not_an_arm_field() -> None:
    """The design and the permission to spend stay separate types.

    An ``allow_real_codex`` on ``ArmSpec`` would travel into ``ArmRecord``
    and from there into the pre-registration payload, which is exactly what
    the payload must not carry: whether an operator was allowed to bill a
    Codex session says nothing about what the study pre-registered.
    """
    from dataclasses import fields

    assert "allow_real_codex" not in {field.name for field in fields(ArmSpec)}


def test_an_authorized_runner_still_needs_the_environment_half(
    tmp_path: Path, monkeypatch
) -> None:
    """One half of the opt-in authorizes nothing.

    The runner carries the flag and the spec carries it onward, and the
    run is still refused: a serialized spec or a copied command line
    cannot buy a session on a machine that never opted in.
    """
    from dataclasses import replace

    from whetstone_envs.optim.codex import (
        ALLOW_REAL_CODEX_ENV,
        RealCodexRefusedError,
    )

    monkeypatch.delenv(ALLOW_REAL_CODEX_ENV, raising=False)
    runner = replace(_runner(tmp_path), allow_real_codex=True)
    with pytest.raises(RealCodexRefusedError):
        runner(arm=_codex_arm(), seed=4000, study_dir=tmp_path)
