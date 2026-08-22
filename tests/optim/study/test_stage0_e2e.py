"""Stage 0 end to end on toy c19 splits, fake transport, zero provider calls.

This is the dry run that proves the stage is wired: three roles calibrated
through one procedure, real ``EvalEvidence`` behind every number, and the
gate reading held-out. The splits are tiny, so the gate's verdict here is
about plumbing rather than about the study's design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema import EvalEvidence

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    c19_candidate,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.provider import (
    c19_fake_gold_by_prompt,
    c19_fake_transport_factory,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner
from whetstone_envs.optim.study.anchors import ANCHOR_ROLES, run_stage0
from whetstone_envs.optim.study.leakage import (
    HeldOutObservation,
    SplitIdentity,
    check_l4_identical_held_out_procedure,
    check_l5_splits_disjoint,
)
from whetstone_envs.optim.study.spec import SplitSpec, StudySpec
from whetstone_envs.reporting.schema import SPLIT_ROLE_BY_REPORT_ROLE

if TYPE_CHECKING:
    from dr_store import ObjectStore

#: Toy sizes: enough tasks per role for a real calibration, small enough to
#: stay a unit test. The study's own sizes are (88, 132, 220).
TOY_SPLIT_SIZES = (4, 4, 6)

#: The split role each evaluation role is bound to, spelled as whetstone
#: spells it. Reading it from the reporting map would force a cast at every
#: call site; naming it once here keeps the binder readable.
SPLIT_ROLE_BY_EVAL_ROLE = {
    EvalRole.INTERNAL: SPLIT_ROLE_BY_REPORT_ROLE["internal"],
    EvalRole.OFFICIAL: SPLIT_ROLE_BY_REPORT_ROLE["official"],
    EvalRole.HELD_OUT: SPLIT_ROLE_BY_REPORT_ROLE["held_out"],
}


@pytest.fixture
def toy_study(tmp_path):
    """A prepared toy c19 experiment plus a per-role engine binder."""
    pool = generate_pool(n_per_stratum=1, seed_start=765_432)
    prepared = prepare_c19_experiment(
        pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=1
    )
    gold_by_prompt = c19_fake_gold_by_prompt(prepared.experiment)

    with open_sqlite(str(tmp_path / "runtime.sqlite")) as store:
        built: list[object] = []

        def bind_engine(*, role: EvalRole, num_seeds: int):
            # Each role gets its own experiment at the calibration repeat
            # count, which is what "one procedure, three roles" means: the
            # only thing that differs between the three engines is which
            # split they are bound to.
            role_prepared = prepare_c19_experiment(
                pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=num_seeds
            )
            config = ReferenceEvalRuntimeConfig(
                split_role=SPLIT_ROLE_BY_EVAL_ROLE[role],
                transport_api_key_env="WHETSTONE_TOY_API_KEY",
            )
            engine = config.build_engine(
                cast("ObjectStore", store),
                experiment=role_prepared.experiment,
                eval_runner=ExactMatchEvalProcedureRunner(),
                mutation_field=C19_MUTATION_FIELD,
                render_contract=c19_render_contract(),
                transport_factory=c19_fake_transport_factory(
                    gold_by_prompt=c19_fake_gold_by_prompt(
                        role_prepared.experiment
                    )
                ),
            )
            built.append(engine)
            return engine

        yield prepared, bind_engine, gold_by_prompt


def _task_ids_by_role(prepared) -> dict[EvalRole, tuple[str, ...]]:
    split = prepared.split
    return {
        EvalRole.INTERNAL: tuple(
            instance.id for instance in split.internal_eval
        ),
        EvalRole.OFFICIAL: tuple(instance.id for instance in split.official),
        EvalRole.HELD_OUT: tuple(instance.id for instance in split.held_out),
    }


def _spec(k_cal: int = 4) -> StudySpec:
    internal, official, held_out = TOY_SPLIT_SIZES
    return StudySpec(
        study_id="stage0-dry-run",
        family="c19",
        n_per_stratum=1,
        pool_seed_start=765_432,
        internal=SplitSpec("internal", internal),
        official=SplitSpec("official", official),
        held_out=SplitSpec("held_out", held_out),
        task_model="fake",
        proposer_model="fake",
        k_cal=k_cal,
        k_repeat=1,
    )


def test_stage0_calibrates_all_three_roles_on_fake_transport(
    toy_study,
) -> None:
    prepared, bind_engine, _ = toy_study
    task_ids = _task_ids_by_role(prepared)

    result = run_stage0(
        spec=_spec(),
        bind_engine=bind_engine,
        naive_candidate=c19_candidate(
            candidate_id="c19-naive", template=PROBES.naive_template
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling", template=PROBES.ceiling_template
        ),
        task_ids_by_role=task_ids,
        pool_ceiling=sum(TOY_SPLIT_SIZES),
    )

    assert (
        tuple(calibration.role for calibration in result.calibrations)
        == ANCHOR_ROLES
    )
    for calibration in result.calibrations:
        expected = len(task_ids[calibration.role])
        assert len(calibration.naive_per_task) == expected
        assert len(calibration.ceiling_per_task) == expected
        assert len(calibration.task_hashes) == expected


def test_stage0_calibrates_at_k_cal_not_at_the_design_repeat_count(
    toy_study,
) -> None:
    """K_CAL is a measurement input; conflating it with K biases the gate."""
    prepared, bind_engine, _ = toy_study
    spec = _spec(k_cal=4)
    result = run_stage0(
        spec=spec,
        bind_engine=bind_engine,
        naive_candidate=c19_candidate(
            candidate_id="c19-naive", template=PROBES.naive_template
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling", template=PROBES.ceiling_template
        ),
        task_ids_by_role=_task_ids_by_role(prepared),
        pool_ceiling=sum(TOY_SPLIT_SIZES),
    )
    assert result.k_cal == 4
    assert result.inputs.k_cal == 4
    assert result.inputs.k_repeat == spec.k_repeat == 1
    for calibration in result.calibrations:
        for anchor in (
            calibration.calibration.baseline,
            calibration.calibration.ceiling,
        ):
            evidence = anchor.evidence
            assert isinstance(evidence, EvalEvidence)
            assert evidence.num_seeds == 4


def test_the_gate_reads_held_out_and_reports_every_condition(
    toy_study,
) -> None:
    """Held-out is the split the study reports from, so it is the one gated."""
    prepared, bind_engine, _ = toy_study
    result = run_stage0(
        spec=_spec(),
        bind_engine=bind_engine,
        naive_candidate=c19_candidate(
            candidate_id="c19-naive", template=PROBES.naive_template
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling", template=PROBES.ceiling_template
        ),
        task_ids_by_role=_task_ids_by_role(prepared),
        pool_ceiling=sum(TOY_SPLIT_SIZES),
    )
    held_out = result.held_out
    assert result.inputs.naive_mean == pytest.approx(held_out.naive_mean)
    assert result.inputs.ceiling_mean == pytest.approx(held_out.ceiling_mean)
    assert result.inputs.held_out_size == len(held_out.task_hashes)
    assert len(result.gate.outcomes) == 4
    assert result.gate.mde_measured >= 0.0
    assert result.null_b_expected_delta >= 0.0

    # On a fake transport both anchors land on the same score, so the
    # calibration is degenerate: no headroom and no variance. The gate must
    # refuse that rather than read the resulting near-zero MDE as excellent
    # power -- an undetectable design and a perfectly powered one both have
    # a tiny MDE, and only the headroom conditions tell them apart. This is
    # the plumbing assertion the dry run exists for: a real Stage 0 buys
    # real anchors, and this one proves the gate is wired to them.
    assert held_out.naive_mean == pytest.approx(held_out.ceiling_mean)
    assert not result.passed
    assert {failure.name for failure in result.gate.failures()} == {
        "headroom",
        "ceiling_not_floored",
    }


def test_a_missing_role_stops_stage0_before_it_spends(toy_study) -> None:
    prepared, bind_engine, _ = toy_study
    task_ids = _task_ids_by_role(prepared)
    del task_ids[EvalRole.HELD_OUT]
    with pytest.raises(ValueError, match="missing task ids"):
        run_stage0(
            spec=_spec(),
            bind_engine=bind_engine,
            naive_candidate=c19_candidate(
                candidate_id="c19-naive", template=PROBES.naive_template
            ),
            ceiling_candidate=c19_candidate(
                candidate_id="c19-ceiling", template=PROBES.ceiling_template
            ),
            task_ids_by_role=task_ids,
            pool_ceiling=sum(TOY_SPLIT_SIZES),
        )


def test_stage0_splits_are_disjoint_by_task_hash(toy_study) -> None:
    """L5 over real derived splits, not over a hand-built fixture."""
    prepared, bind_engine, _ = toy_study
    result = run_stage0(
        spec=_spec(),
        bind_engine=bind_engine,
        naive_candidate=c19_candidate(
            candidate_id="c19-naive", template=PROBES.naive_template
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling", template=PROBES.ceiling_template
        ),
        task_ids_by_role=_task_ids_by_role(prepared),
        pool_ceiling=sum(TOY_SPLIT_SIZES),
    )
    finding = check_l5_splits_disjoint(
        SplitIdentity(
            role=calibration.role.value,
            task_hashes=calibration.task_hashes,
        )
        for calibration in result.calibrations
    )
    assert finding.passed, finding.offenders


def test_each_role_calibrates_under_its_own_eval_config(toy_study) -> None:
    """Three roles, three configs -- and one config within held-out (L4)."""
    prepared, bind_engine, _ = toy_study
    result = run_stage0(
        spec=_spec(),
        bind_engine=bind_engine,
        naive_candidate=c19_candidate(
            candidate_id="c19-naive", template=PROBES.naive_template
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling", template=PROBES.ceiling_template
        ),
        task_ids_by_role=_task_ids_by_role(prepared),
        pool_ceiling=sum(TOY_SPLIT_SIZES),
    )
    hashes = {
        calibration.role: calibration.eval_config_hash
        for calibration in result.calibrations
    }
    assert len(set(hashes.values())) == len(ANCHOR_ROLES)

    held_out = result.held_out
    finding = check_l4_identical_held_out_procedure(
        (
            HeldOutObservation("naive", held_out.eval_config_hash, 4),
            HeldOutObservation("ceiling", held_out.eval_config_hash, 4),
        )
    )
    assert finding.passed
