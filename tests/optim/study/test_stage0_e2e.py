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
from whetstone.eval.protocol import EvalRequest
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
    fake_gold_by_prompt,
    fake_transport_factory,
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
    gold_by_prompt = fake_gold_by_prompt(
        prepared.experiment,
        render_contract=c19_render_contract(),
        ceiling_template=PROBES.ceiling_template,
    )

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
                transport_factory=fake_transport_factory(
                    gold_by_prompt=fake_gold_by_prompt(
                        role_prepared.experiment,
                        render_contract=c19_render_contract(),
                        ceiling_template=PROBES.ceiling_template,
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
        task_reasoning_effort="minimal",
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


# --------------------------------------------------------------------------
# A genuinely fully-lost task refuses the stage, on real evidence
# --------------------------------------------------------------------------


class _LosesOneTask:
    """A fake transport that permanently refuses one task's calls.

    Real loss rather than a stubbed vector: the engine runs, the rows are
    persisted, whetstone aggregates them, and the task that never
    succeeded reports `None` for its per-task value under `EvalEvidence`
    v6. That is the evidence the floor has to recognise, and only driving
    it end to end shows that it does.

    The failure is permanent so the retrying transport does not spend its
    budget re-asking; what is under test is a lost task, not a retry.
    """

    def __init__(
        self, inner, *, doomed_question: str, doomed_grid: str
    ) -> None:
        self._inner = inner
        self._doomed_question = doomed_question
        self._doomed_grid = doomed_grid
        self.refusals = 0

    def __call__(self, request):
        from dr_providers import (
            ProviderInvocationEvidence,
            ProviderTransportFailure,
            RecoverabilityClass,
        )
        from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence

        messages = getattr(
            getattr(request, "transcript", None), "messages", ()
        )
        prompt = str(messages[-1].content) if messages else ""
        if self._doomed_question in prompt and self._doomed_grid in prompt:
            self.refusals += 1
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self._inner._policy.transport_policy,
                http_request=ProviderHttpRequestEvidence(
                    method="POST",
                    url="http://whetstone.fake/llm",
                    headers={},
                    body={},
                    body_bytes=0,
                ),
                outcome=ProviderTransportFailure(
                    recoverability=RecoverabilityClass.PERMANENT,
                    message="this task is refused for the whole evaluation",
                    status_code=400,
                ),
            )
        return self._inner(request)


def test_a_fully_lost_task_refuses_stage0_on_real_evidence(
    tmp_path,
) -> None:
    """A task losing every repeat cannot become a Stage-0 anchor.

    Real loss, driven end to end: one task's calls are permanently
    refused, the engine runs, the rows persist, and whetstone reports the
    task with no per-task value at all under ``EvalEvidence`` v6.

    **Stage 0 refuses in whetstone's own calibration**, which requires a
    full ``per_task_counts`` for every anchor -- an anchor defines the
    achievable range, so a partially measured one is not a weaker anchor
    but a wrong one. That refusal is upstream's and fires before this
    package sees the evidence, so the assertion here is that the loss is
    *caught*, not that this package's floor is what catches it.

    The floor is what covers the case calibration does not: an arm's
    scored and held-out evaluations, which are not anchors and which
    whetstone is content to aggregate over a shrunken denominator. This
    test pins both halves against the same real evidence -- and pins that
    the 0.1.13 upgrade did not quietly stop the floor from seeing a lost
    task, since v6 spells it ``None`` where v5 spelled it ``0.0``.
    """
    from whetstone_envs.optim.completeness import (
        TaskCompletenessError,
        fully_lost_task_count,
    )

    pool = generate_pool(n_per_stratum=1, seed_start=765_432)
    prepared = prepare_c19_experiment(
        pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=1
    )
    # Doom one held-out task by what identifies it inside the prompt. The
    # task id never reaches the prompt, and the grid alone is shared by
    # the instances built from one seed, so the question is what
    # distinguishes them.
    doomed = prepared.split.held_out[0]
    doomed_question = str(doomed.prompt_inputs["question"])
    doomed_grid = str(doomed.prompt_inputs["grid"])
    assert (
        sum(
            1
            for instance in prepared.split.held_out
            if str(instance.prompt_inputs["question"]) == doomed_question
            and str(instance.prompt_inputs["grid"]) == doomed_grid
        )
        == 1
    )

    refusers: list[_LosesOneTask] = []

    with open_sqlite(str(tmp_path / "runtime.sqlite")) as store:

        def bind_engine(*, role: EvalRole, num_seeds: int):
            role_prepared = prepare_c19_experiment(
                pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=num_seeds
            )
            config = ReferenceEvalRuntimeConfig(
                split_role=SPLIT_ROLE_BY_EVAL_ROLE[role],
                transport_api_key_env="WHETSTONE_TOY_API_KEY",
            )
            inner_factory = fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    role_prepared.experiment,
                    render_contract=c19_render_contract(),
                    ceiling_template=PROBES.ceiling_template,
                )
            )

            def transport_factory(policy):
                refuser = _LosesOneTask(
                    inner_factory(policy),
                    doomed_question=doomed_question,
                    doomed_grid=doomed_grid,
                )
                refusers.append(refuser)
                return refuser

            return config.build_engine(
                cast("ObjectStore", store),
                experiment=role_prepared.experiment,
                eval_runner=ExactMatchEvalProcedureRunner(),
                mutation_field=C19_MUTATION_FIELD,
                render_contract=c19_render_contract(),
                transport_factory=transport_factory,
            )

        from whetstone_envs.optim.completeness import (
            require_task_completeness,
        )

        # Stage 0 refuses rather than anchoring on a shrunken population.
        with pytest.raises(
            ValueError, match="per-task sample counts"
        ) as refused:
            run_stage0(
                spec=_spec(),
                bind_engine=bind_engine,
                naive_candidate=c19_candidate(
                    candidate_id="c19-naive", template=PROBES.naive_template
                ),
                ceiling_candidate=c19_candidate(
                    candidate_id="c19-ceiling",
                    template=PROBES.ceiling_template,
                ),
                task_ids_by_role=_task_ids_by_role(prepared),
                pool_ceiling=sum(TOY_SPLIT_SIZES),
            )
        assert refused.value is not None

        # The loss was real: the doomed task's calls were actually refused.
        assert refusers
        assert any(refuser.refusals for refuser in refusers)

        # And the same real loss, evaluated the way an arm evaluates rather
        # than the way an anchor calibrates, is what the floor refuses. This
        # is the path calibration does not cover: whetstone aggregates it
        # happily, over a denominator one task smaller than it reports.
        config = ReferenceEvalRuntimeConfig(
            split_role=SPLIT_ROLE_BY_EVAL_ROLE[EvalRole.HELD_OUT],
            transport_api_key_env="WHETSTONE_TOY_API_KEY",
        )
        engine = bind_engine(role=EvalRole.HELD_OUT, num_seeds=1)
        assert config is not None
        subset = engine.for_task_ids(
            _task_ids_by_role(prepared)[EvalRole.HELD_OUT]
        )
        outcome = subset.evaluate(
            EvalRequest(
                request_id="held_out:naive",
                candidate=c19_candidate(
                    candidate_id="c19-naive", template=PROBES.naive_template
                ),
                metadata={},
            )
        )
        lost_evidence = outcome.evidence

    assert isinstance(lost_evidence, EvalEvidence)
    # v6 spells a fully-lost task as an absent per-task value.
    assert any(value is None for value in lost_evidence.per_task_values)
    assert fully_lost_task_count(lost_evidence) == 1

    with pytest.raises(TaskCompletenessError, match="lost every"):
        require_task_completeness(lost_evidence, purpose="held_out:naive")
