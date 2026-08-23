"""How a study arm's settings reach the runner's ``RunSpec``.

``StudyOptimizerRunner`` is the seam between the study's arms and the shared
optimizer runner. An arm setting that never reaches ``RunSpec`` would look
honoured in the manifest while the run ignored it, which is the failure the
per-arm validation in ``spec.py`` exists to prevent -- so the forwarding
itself is pinned here rather than assumed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.completeness import (
    TaskCompletenessError,
    require_task_completeness,
)
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
)
from whetstone_envs.optim.study.arms import (
    StudyOptimizerRunner,
    arm_run_directory,
)
from whetstone_envs.optim.study.manifest import DISCARD_STALE_RUNS_FLAG
from whetstone_envs.optim.study.spec import ArmKind, ArmSpec
from whetstone_envs.optim.study.stages import StageError

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


def _shaped_copro_arm(*, breadth: int, depth: int = 3) -> ArmSpec:
    return ArmSpec(
        arm_id="copro",
        optimizer="copro",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(1000,),
        copro_breadth=breadth,
        copro_depth=depth,
    )


def test_a_fake_copro_arm_is_handed_drafts_enough_for_its_breadth(
    tmp_path: Path,
) -> None:
    """The pinned breadth 6 needs more drafts than the family scripts.

    A family scripts two bodies -- the ceiling draft and the naive seed --
    and the naive one fills a slot COPRO never requests, so an unaided fake
    round can land exactly one draft. At the protocol's breadth the round
    cannot be filled and the run dies inside the durable boundary with
    ``copro_proposal_cardinality``, before it evaluates anything. That made
    a fake-transport rehearsal of the registered search shape impossible.
    """
    spec = _runner(tmp_path)._spec_for(
        _shaped_copro_arm(breadth=6), seed=1000, run_dir=tmp_path / "run"
    )
    # One is the family's own ceiling draft, so the arm supplies the rest.
    assert len(spec.extra_proposal_bodies) == 5
    assert len(set(spec.extra_proposal_bodies)) == 5
    contract = family_spec("c19").render_contract()
    for body in spec.extra_proposal_bodies:
        contract.validate_template(body)


def test_a_real_transport_copro_arm_is_handed_no_drafts(
    tmp_path: Path,
) -> None:
    """On a paid transport the proposer writes its own bodies.

    ``run_optimizer`` refuses ``extra_proposal_bodies`` outside the fake
    transport, so forwarding them unconditionally would make every paid
    COPRO arm unrunnable.
    """
    runner = replace(_runner(tmp_path), transport="openrouter")
    spec = runner._spec_for(
        _shaped_copro_arm(breadth=6), seed=1000, run_dir=tmp_path / "run"
    )
    assert spec.extra_proposal_bodies == ()


def test_a_null_random_arm_is_handed_no_drafts(tmp_path: Path) -> None:
    """null-A generates its own drafts, so scripted bodies would go unread.

    It shares COPRO's search shape and its breadth, but binds a
    ``NullRandomTransport`` that mints a fresh draft per slot.
    """
    arm = replace(
        _shaped_copro_arm(breadth=6),
        arm_id="null-random",
        optimizer="null-random",
    )
    spec = _runner(tmp_path)._spec_for(
        arm, seed=1000, run_dir=tmp_path / "run"
    )
    assert spec.extra_proposal_bodies == ()


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


# --------------------------------------------------------------------------
# A run directory is only reusable when it is this invocation's own run
# --------------------------------------------------------------------------


def _copro_arm() -> ArmSpec:
    """A real optimizer arm whose run actually produces artifacts."""
    return ArmSpec(
        arm_id="copro",
        optimizer="copro",
        kind=ArmKind.REAL,
        k_run=1,
        seeds=(2000,),
    )


def _fake_run_directory(tmp_path: Path) -> tuple[StudyOptimizerRunner, Path]:
    """One real fake-transport run, left on disk under its own name.

    This is the state a cross-transport ``--replace-design`` leaves: the
    manifest dropped the run, its deterministically named directory did
    not go with it.
    """
    runner = _runner(tmp_path)
    arm = _copro_arm()
    result = runner(arm=arm, seed=2000, study_dir=tmp_path)
    run_dir = arm_run_directory(tmp_path, result.record.run_id)
    assert run_dir.is_dir()
    assert result.record.transport == "fake"
    return runner, run_dir


def test_a_run_directory_from_another_transport_is_refused(
    tmp_path: Path,
) -> None:
    """The defect: a paid stage silently re-recorded a free run as paid.

    Run ids are deterministic on arm and seed, so after a cross-transport
    amendment the replacement stage computes the same directory name the
    dropped run already occupies. It found the directory, skipped
    ``run_optimizer``, and recorded the fake run under its own transport --
    a manifest that reads as a paid study whose numbers came from the free
    transport.

    Nothing here reaches a provider: the openrouter runner refuses on the
    directory's own evidence, before any run is dispatched.
    """
    _, run_dir = _fake_run_directory(tmp_path)

    paid = replace(_runner(tmp_path), transport="openrouter")
    with pytest.raises(StageError) as refusal:
        paid(arm=_copro_arm(), seed=2000, study_dir=tmp_path)

    message = str(refusal.value)
    # The refusal names the directory and both recoveries, because the
    # operator is the one who knows which of their runs is the real one.
    assert str(run_dir) in message
    assert "transport" in message
    assert DISCARD_STALE_RUNS_FLAG in message
    # And it refused rather than overwriting: the artifacts are untouched.
    assert (run_dir / "result.json").is_file()


def test_a_matching_run_directory_is_still_reused(tmp_path: Path) -> None:
    """The resume path survives: a run this invocation owns is not re-run.

    Reuse is what makes a crashed stage resumable without paying twice, so
    the check must refuse the stale without refusing the ordinary.
    """
    runner, run_dir = _fake_run_directory(tmp_path)
    before = (run_dir / "result.json").read_bytes()

    with patch(
        "whetstone_envs.optim.study.arms.run_optimizer",
        side_effect=AssertionError("a reusable run must not be re-run"),
    ):
        again = runner(arm=_copro_arm(), seed=2000, study_dir=tmp_path)

    assert again.record.run_id
    assert (run_dir / "result.json").read_bytes() == before


def test_a_directory_with_no_readable_identity_is_refused(
    tmp_path: Path,
) -> None:
    """A directory that cannot vouch for itself is not evidence it matches.

    An empty or half-written directory is a different fact from a
    mismatch, and it is resolved the same way: not silently. Re-running
    would overwrite artifacts that may be paid evidence.
    """
    run_dir = arm_run_directory(tmp_path, "copro-seed2000")
    run_dir.mkdir(parents=True)

    with pytest.raises(StageError) as refusal:
        _runner(tmp_path)(arm=_copro_arm(), seed=2000, study_dir=tmp_path)

    message = str(refusal.value)
    assert str(run_dir) in message
    assert DISCARD_STALE_RUNS_FLAG in message


def test_discard_stale_runs_replaces_the_directory_it_cannot_claim(
    tmp_path: Path,
) -> None:
    """The named recovery works, and produces this invocation's own run.

    The refusal is only honest if the recovery it names does something, so
    the authorized path is exercised rather than merely advertised.
    """
    _, run_dir = _fake_run_directory(tmp_path)
    authorized = replace(_runner(tmp_path), discard_stale_runs=True)
    # Re-run on the same transport after corrupting the directory's
    # identity, so the discard path runs without needing a provider.
    (run_dir / "trajectory-report.json").unlink()
    result = authorized(arm=_copro_arm(), seed=2000, study_dir=tmp_path)

    assert result.record.transport == "fake"
    # The directory was rebuilt rather than left half-stale: the report
    # the discard removed is back, written by the run this invocation made.
    assert (run_dir / "trajectory-report.json").is_file()
    assert (run_dir / "result.json").is_file()


# --------------------------------------------------------------------------
# Null-A is a real run, because a control for selection must select
# --------------------------------------------------------------------------


def _null_a_arm() -> ArmSpec:
    return ArmSpec(
        arm_id="null-random",
        optimizer="null-random",
        kind=ArmKind.NULL,
        k_run=1,
        seeds=(5000,),
    )


def _null_b_arm() -> ArmSpec:
    return ArmSpec(
        arm_id="null-identity",
        optimizer="null-identity",
        kind=ArmKind.NULL,
        k_run=1,
        seeds=(6000,),
    )


def test_null_a_dispatches_through_the_shared_runner(
    tmp_path: Path,
) -> None:
    """Null-A's spec is an ordinary run spec, dispatched like any arm.

    Fails-before: ``__call__`` matched ``null-random`` alongside
    ``null-identity`` and returned ``_run_null``, which never built a
    ``RunSpec`` and never called ``run_optimizer`` at all. The arm's
    "perturbation" was a ``(variant N)`` suffix on the naive template and
    its record carried ``observed_task_calls=0`` and ``spend=()``.
    """
    spec = _runner(tmp_path)._spec_for(
        _null_a_arm(), seed=5000, run_dir=tmp_path / "run"
    )
    assert spec.optimizer == "null-random"
    assert spec.seed == 5000
    # COPRO's search shape, so it has no train/val partition of its own --
    # the same fields the COPRO arm leaves unset.
    assert spec.train_size is None
    assert spec.val_size is None
    assert spec.transport == "fake"


def test_null_a_evaluates_and_records_spend_like_every_other_arm(
    tmp_path: Path,
) -> None:
    """The control's evidence is an arm's evidence, produced the same way.

    Fails-before: the arm evaluated nothing, so ``observed_task_calls``
    was 0, its ``result_ref``/``audit_ref``/``cost_ref`` all addressed one
    synthesized ``study_null_run`` record, and there was no run directory
    at all. Selection-on-noise cannot be controlled for by an arm that
    never selected, which is what makes this the item most likely to
    invalidate an efficacy claim.
    """
    result = _runner(tmp_path)(
        arm=_null_a_arm(), seed=5000, study_dir=tmp_path
    )

    run_dir = arm_run_directory(tmp_path, result.record.run_id)
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "trajectory-report.json").is_file()
    # It evaluated candidates on the internal split rather than nothing.
    assert result.observed_task_calls > 0
    # And it produced the same three distinct evidence records an
    # optimizer arm does, rather than one record standing in for all three.
    pointers = {
        result.record.result_ref,
        result.record.audit_ref,
        result.record.cost_ref,
    }
    assert len(pointers) == 3
    assert result.record.audit_passed
    assert result.record.transport == "fake"


def test_null_a_terminal_template_comes_from_its_own_search(
    tmp_path: Path,
) -> None:
    """The candidate is what the run selected, not a suffixed anchor.

    Fails-before: the template was ``f"{naive}\\n\\n(variant {seed})"`` --
    a meaning-free suffix rather than the protocol's placeholder-preserving
    perturbation, and never a candidate any evaluation ranked.
    """
    runner = _runner(tmp_path)
    result = runner(arm=_null_a_arm(), seed=5000, study_dir=tmp_path)

    assert "(variant 5000)" not in result.candidate.template
    assert result.candidate.template != runner.naive_template


def test_null_a_is_resumable_from_its_recorded_run(tmp_path: Path) -> None:
    """A recorded null-A is re-read off disk, not re-synthesized.

    Fails-before: ``load_recorded_run`` rebuilt null-A's template from the
    seed, so a resumed stage produced a candidate no run had evaluated.
    """
    runner = _runner(tmp_path)
    arm = _null_a_arm()
    first = runner(arm=arm, seed=5000, study_dir=tmp_path)

    reloaded = runner.load_recorded_run(arm=arm, run=first.record)
    assert reloaded is not None
    assert reloaded.candidate.template == first.candidate.template
    assert reloaded.observed_task_calls == first.observed_task_calls


def test_null_b_still_runs_no_optimizer(tmp_path: Path) -> None:
    """Null-B controls for pipeline overhead, so it proposes nothing.

    It stays the seed through the report harness (note 13): a byte-identical
    proposal is unreachable under COPRO's proposal-cardinality contract, so
    there is no search to drive.
    """
    runner = _runner(tmp_path)
    result = runner(arm=_null_b_arm(), seed=6000, study_dir=tmp_path)

    assert result.candidate.template == runner.naive_template
    assert result.observed_task_calls == 0
    assert result.record.spend == ()
    # One synthesized record standing in for all three, which is honest
    # here precisely because there is no optimizer result to point at.
    assert (
        result.record.result_ref
        == result.record.audit_ref
        == result.record.cost_ref
    )


# --------------------------------------------------------------------------
# A fully-lost task voids the evaluation rather than biasing its mean
# --------------------------------------------------------------------------


class _Evidence:
    """The four evidence fields the completeness check reads.

    A stub rather than a real ``EvalEvidence`` because the check is pure
    arithmetic over the two means, and constructing real evidence would
    require a store, a graph, and a persisted aggregate to exercise a
    function that touches none of them.
    """

    def __init__(
        self,
        *,
        per_task_values: tuple[float | None, ...],
        aggregate_value: float | None,
        aggregate_status: str = "ok",
        per_task_counts: tuple[int, ...] | None = None,
        num_seeds: int = 4,
    ) -> None:
        self.per_task_values = per_task_values
        self.aggregate_value = aggregate_value
        self.aggregate_status = aggregate_status
        self.num_seeds = num_seeds
        self.task_hashes = tuple(
            f"h{index:064x}" for index in range(len(per_task_values))
        )
        self.per_task_counts = (
            per_task_counts
            if per_task_counts is not None
            else tuple(
                0 if value is None else num_seeds for value in per_task_values
            )
        )


def _evidence_with_lost_tasks(
    *, tasks: int, lost: int, score: float = 1.0
) -> _Evidence:
    """Evidence as whetstone builds it when ``lost`` tasks vanish entirely.

    A task with no present row reports ``None`` for its per-task score
    and ``0`` for its per-task count, which is what per-task reporting
    over *present* rows yields; the aggregate drops it and averages over
    the tasks that produced a value.
    """
    values: tuple[float | None, ...] = tuple(
        None if index >= tasks - lost else score for index in range(tasks)
    )
    contributing = tasks - lost
    aggregate = score if contributing else None
    return _Evidence(per_task_values=values, aggregate_value=aggregate)


def test_a_fully_lost_task_refuses_the_evaluation() -> None:
    """One task losing every repeat voids the number, whatever the fraction.

    **Fails-before: accepted, reporting 1.0 where the truth is 0.9868.**
    At the study's own shape -- 76 tasks, 4 repeats -- one fully-lost task
    is 4 of 304 rows, or 1.3%, which sits well inside the 10%
    ``max_skip_fraction``. So the row tolerance passes it, whetstone's
    ``unweighted_task_mean`` drops the task from the denominator, and the
    evaluation reports ``status=ok`` with a mean over 75 tasks presented
    as though it covered 76.

    The bias is upward and systematic rather than noise: a task that
    loses *every* repeat is a slow, long-generation one, which is
    precisely the task that would have scored low.
    """
    evidence = _evidence_with_lost_tasks(tasks=76, lost=1)
    # The aggregate really does read as a clean 1.0 while a task is gone,
    # and the row tolerance never sees it: 4 of 304 rows is 1.3%.
    assert evidence.aggregate_value == 1.0
    assert evidence.per_task_values[-1] is None
    present = [v for v in evidence.per_task_values if v is not None]
    assert len(present) == 75
    # Counting the lost task at its worst case is what the mean hides.
    assert sum(present) / 76 == pytest.approx(0.98684, abs=1e-5)

    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(evidence, purpose="official:cand")


def test_a_partially_lost_task_is_accepted() -> None:
    """Losing some repeats of a task is the case the row tolerance is for.

    Nothing here is refused: every task still contributed a value, so the
    mean is over the population it claims, and the shortfall shows up as
    reduced completeness the way the protocol pre-registered.
    """
    # Four tasks, four repeats each; one task lost one of its four. Its
    # per-task value drops to 0.75 and it still contributes.
    evidence = _Evidence(
        per_task_values=(1.0, 1.0, 1.0, 0.75),
        aggregate_value=(1.0 + 1.0 + 1.0 + 0.75) / 4,
    )
    require_task_completeness(evidence, purpose="official:cand")


def test_losing_a_tenth_of_the_tasks_trips_the_floor() -> None:
    """The task floor is the same 90% the row bound complements."""
    # Refused by the zero-present rule first, which is stricter; the floor
    # exists for the case where tasks vanish before they are counted.
    evidence = _evidence_with_lost_tasks(tasks=76, lost=8)
    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(evidence, purpose="official:cand")


def _evidence_with_short_tasks(
    *, tasks: int, short: int, num_seeds: int = 4, score: float = 1.0
) -> _Evidence:
    """Evidence where ``short`` tasks ran fewer than every repeat.

    Every task still contributes a value, so the zero-present rule passes
    and only the completeness floor can see the shortfall.
    """
    counts = tuple(
        num_seeds - 1 if index < short else num_seeds for index in range(tasks)
    )
    return _Evidence(
        per_task_values=tuple(score for _ in range(tasks)),
        aggregate_value=score,
        per_task_counts=counts,
        num_seeds=num_seeds,
    )


def test_the_floor_fires_on_short_tasks_the_zero_rule_passes() -> None:
    """The 90% floor bounds a population the zero-present rule cannot see.

    **Fails-before: accepted.** ``achieved`` was computed as
    ``(planned - lost) / planned``, but the zero-present rule above it
    already refused whenever ``lost`` was nonzero -- so by the time the
    floor was evaluated ``lost`` was always 0, ``achieved`` was always
    exactly 1.0, and the comparison against 0.90 could never be true. The
    floor was decorative: it could not fire on any input.

    Counting *incomplete* tasks -- present, but measured to less than
    ``k_repeat`` -- gives it a real population. Here 20 of 76 tasks ran
    three of their four repeats: every task contributes a value, nothing
    is fully lost, and the split is still measured more shallowly than
    the design pre-registered.
    """
    evidence = _evidence_with_short_tasks(tasks=76, short=20)
    # Nothing is fully lost, so the stricter rule genuinely passes.
    assert all(value is not None for value in evidence.per_task_values)
    assert all(count > 0 for count in evidence.per_task_counts)
    # 56 of 76 complete is 0.737, below the 0.90 floor.
    with pytest.raises(TaskCompletenessError, match="task-completeness floor"):
        require_task_completeness(evidence, purpose="official:cand")


def test_a_few_short_tasks_stay_inside_the_floor() -> None:
    """The floor tolerates the scattered shortfall it was set to allow.

    Seven of 76 tasks short leaves 69 complete, or 0.908 -- just above
    the bound -- so a handful of dropped repeats does not void an
    otherwise sound evaluation. This is the tolerance the row bound and
    the retry schedule are both aimed at preserving.
    """
    evidence = _evidence_with_short_tasks(tasks=76, short=7)
    require_task_completeness(evidence, purpose="official:cand")


def test_the_zero_present_rule_stays_unconditional() -> None:
    """A fully-lost task refuses even when completeness is otherwise fine.

    75 of 76 tasks complete is 0.987, far above the floor, so only the
    unconditional rule can catch the one task that vanished. The two
    conditions are independent and neither subsumes the other.
    """
    evidence = _evidence_with_lost_tasks(tasks=76, lost=1)
    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(evidence, purpose="official:cand")


def test_evidence_without_counts_reports_no_incomplete_tasks() -> None:
    """An evidence record carrying no counts is not guessed at.

    A short task's value is a real number either way, so ``per_task_counts``
    is the only spelling that can express the shortfall. Absent it, the
    floor reports nothing rather than inventing a population.
    """
    from whetstone_envs.optim.completeness import incomplete_task_count

    evidence = _Evidence(
        per_task_values=(1.0, 1.0, 1.0, 1.0),
        aggregate_value=1.0,
        per_task_counts=(),
    )
    assert incomplete_task_count(evidence) == 0
    require_task_completeness(evidence, purpose="official:cand")


def test_a_complete_evaluation_is_untouched() -> None:
    """The check is a floor, not a tax on the normal path."""
    evidence = _evidence_with_lost_tasks(tasks=76, lost=0, score=0.62)
    require_task_completeness(evidence, purpose="official:cand")


def test_an_all_zero_evaluation_is_not_falsely_refused() -> None:
    """A genuinely zero-scoring evaluation has no upward bias to catch.

    With every contributing task at zero the ratio the check infers from
    is undefined, and a lost task is indistinguishable from a task that
    scored zero -- but both leave the mean at zero, so nothing is being
    hidden.
    """
    evidence = _Evidence(
        per_task_values=(0.0, 0.0, 0.0, 0.0), aggregate_value=0.0
    )
    require_task_completeness(evidence, purpose="official:cand")


def test_a_lost_task_is_caught_by_either_spelling() -> None:
    """Presence is read off both per-task vectors, not just one.

    ``per_task_values`` carrying ``None`` and ``per_task_counts``
    carrying ``0`` are the same fact reported two ways, and which one an
    evidence record uses depends on the whetstone release it was written
    by. Checking only one would make this floor silently stop working
    across a dependency bump -- the exact failure mode it exists to
    prevent.
    """
    # Value says None, count disagrees (still refused).
    by_value = _Evidence(
        per_task_values=(1.0, 1.0, 1.0, None),
        aggregate_value=1.0,
        per_task_counts=(4, 4, 4, 4),
    )
    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(by_value, purpose="official:cand")

    # Count says zero, value disagrees (still refused).
    by_count = _Evidence(
        per_task_values=(1.0, 1.0, 1.0, 0.0),
        aggregate_value=1.0,
        per_task_counts=(4, 4, 4, 0),
    )
    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(by_count, purpose="official:cand")


def test_a_zero_scoring_task_is_not_a_lost_task() -> None:
    """Scoring zero is a measurement; losing every repeat is not.

    The distinction matters because the floor must not refuse a genuinely
    hard task that was fully measured and simply got everything wrong --
    that number is real evidence, and the study needs it.
    """
    evidence = _Evidence(
        per_task_values=(1.0, 1.0, 1.0, 0.0),
        aggregate_value=0.75,
        per_task_counts=(4, 4, 4, 4),
    )
    require_task_completeness(evidence, purpose="official:cand")
