from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY
from whetstone.optim.miprov2.render import compose_user_prompt_template

from whetstone_envs.c19 import generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_RESPONSE_FIELD,
    c19_render_contract,
    gold_by_task_hash,
    prepare_c19_experiment,
    provider_call_config_ref,
)
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    DEMO_MODES,
    MIPROV2_COMPONENT_ID,
    Miprov2DemoMode,
    build_miprov2_adapter,
    build_miprov2_control,
    build_miprov2_state,
    miprov2_budget,
    miprov2_run_ref,
)
from whetstone_envs.optim.provider import (
    fake_gold_by_prompt,
    fake_transport_factory,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.core.identity import ImmutableJsonObject

C19 = family_spec("c19")


@pytest.fixture
def prepared():
    return prepare_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(2, 2, 0),
        num_seeds=1,
    )


@pytest.fixture
def engine_and_store(tmp_path, prepared):
    runtime_config = ReferenceEvalRuntimeConfig(
        transport_api_key_env="WHETSTONE_TOY_API_KEY",
    )
    with open_sqlite(str(tmp_path / "runtime.sqlite")) as store:
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=prepared.experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    prepared.experiment,
                    render_contract=C19.render_contract(),
                    ceiling_template=C19.probes.ceiling_template,
                )
            ),
        )
        yield engine, store


class _ShapedControl:
    """Only the control fields :func:`miprov2_budget` reads.

    Building a real ``Miprov2Control`` needs 43 required fields and a bound
    engine; the derivation reads seven of them, so a stand-in states
    exactly what the ceiling depends on and nothing else.
    """

    trainset_task_hashes = tuple(str(i) for i in range(44))
    valset_task_hashes = tuple(str(i) for i in range(44, 88))
    minibatch = True
    minibatch_size = 35
    num_trials = 10
    num_candidates = 3

    def __init__(self, num_seeds: int = 3) -> None:
        self.num_seeds = num_seeds


def _halves(engine) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The one valid partition of this fixture's two-task internal split.

    The control now requires an explicit disjoint train/val split, so every
    call that is not itself testing the partition takes the trivial one.
    """
    task_hashes = tuple(engine.sampling.task_hashes)
    return task_hashes[:1], task_hashes[1:2]


def test_demo_modes_cover_the_whetstone_enumeration() -> None:
    assert set(DEMO_MODES) == {mode.value for mode in Miprov2DemoMode}
    assert DEMO_MODES == ("fewshot", "zeroshot", "ground_only")


@pytest.mark.parametrize("demo_mode", list(Miprov2DemoMode))
def test_control_binds_the_c19_mutation_surface(
    engine_and_store, prepared, demo_mode: Miprov2DemoMode
) -> None:
    engine, _store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        demo_mode=demo_mode,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    assert control.mutation_field == C19_MUTATION_FIELD
    assert control.template_render_contract == c19_render_contract()
    assert control.demo_mode is demo_mode
    assert control.component_ids == (MIPROV2_COMPONENT_ID,)
    # Train and validation partition the engine's tasks with no overlap.
    train = set(control.trainset_task_hashes)
    val = set(control.valset_task_hashes)
    assert train
    assert val
    assert not train & val
    assert train | val == set(engine.sampling.task_hashes)


@pytest.fixture
def repeated_engine_and_store(tmp_path):
    """An engine bound at three repeats per task, like the study's design."""
    prepared = prepare_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(2, 2, 0),
        num_seeds=3,
    )
    runtime_config = ReferenceEvalRuntimeConfig(
        transport_api_key_env="WHETSTONE_TOY_API_KEY",
    )
    with open_sqlite(str(tmp_path / "repeated.sqlite")) as store:
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=prepared.experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    prepared.experiment,
                    render_contract=C19.render_contract(),
                    ceiling_template=C19.probes.ceiling_template,
                )
            ),
        )
        yield engine, store, prepared


def test_the_control_takes_its_repeat_count_from_the_engine(
    repeated_engine_and_store,
) -> None:
    """The bound engine's seed plan is the authority on repeats.

    ``Miprov2Control.num_seeds`` defaults to 1 upstream, so a control built
    without reading the engine ran every in-search evaluation at one repeat
    while the engine was bound at ``K_REPEAT``. ``engine_binding.resolve``
    refuses that disagreement outright -- "engine sampling repeats (3) do
    not match the requested num_seeds (1)" -- which killed every MIPROv2
    arm of a ``K_REPEAT = 3`` study inside the durable run boundary.
    """
    engine, _store, prepared = repeated_engine_and_store
    assert engine.sampling.num_seeds == 3
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    assert control.num_seeds == 3


def test_the_budget_scales_with_the_shape_the_control_pins() -> None:
    """A fixed ceiling cannot bound a shape the design chooses.

    The ceiling exists to catch runaway fan-out, so it is a signal only
    while it sits above what the trial schedule costs. At the Step 10
    design -- a 44/44 partition, minibatch 35, 10 trials, ``K_REPEAT = 3``
    -- the schedule plans roughly 2,900 rows against the old fixed 256-row
    ceiling, and every bootstrap attempt bills one row per repeat against
    the old fixed 32. Both were exhausted mid-run, inside the durable run
    boundary, before the schedule was: the MIPROv2 arms of a
    ``--without-codex`` rehearsal died with "MIPROv2 bootstrap_generations
    budget exhausted".
    """
    budget = miprov2_budget(_ShapedControl())
    # What the design's own schedule plans, in rows: the trials' batches,
    # the baseline and periodic full-valset passes, and the bootstrap
    # walk -- each billed once per repeat.
    planned_rows = (10 * 35 + 11 * 44 + 44 * 3) * 3
    assert budget.task_rows > planned_rows
    # A bootstrap plan walks at most the trainset, once per candidate
    # plan. This ceiling is in *attempts*, not rows: upstream debits
    # ``bootstrap_generations`` by one per attempt regardless of repeats.
    assert budget.bootstrap_generations > 44 * 3
    # And the ceiling is a bound, not a price: it stays clear of the
    # schedule rather than tracking it.
    assert budget.task_rows > 2 * planned_rows


@pytest.mark.parametrize("num_seeds", [1, 3, 5])
def test_the_budget_ceilings_scale_in_their_own_units(num_seeds: int) -> None:
    """The two bootstrap ceilings are in different units.

    A bootstrap attempt debits ``bootstrap_generations`` by one and
    ``task_rows`` by the repeat count (``adapter.py``: ``{"bootstrap_
    generations": 1, "task_rows": num_seeds}``), and upstream's
    ``effect_counts`` compares the former against a *count of effects*.
    So the attempt ceiling must bound ``trainset x plans`` and must NOT
    be inflated by ``num_seeds``, while the row ceiling must scale with
    it. Reading ``bootstrap_generations`` as a row budget makes it look
    exhausted at ``num_seeds = 5`` (132 attempts x 5 = 660 > 528) when no
    such run can exhaust it -- and inflating it to match would blind the
    guard to runaway fan-out by a factor of the repeat count.
    """
    budget = miprov2_budget(_ShapedControl(num_seeds))
    max_attempts = 44 * 3
    # In attempts: bounds the walk, and is repeat-independent.
    assert budget.bootstrap_generations == 4 * max_attempts
    # In rows: every planned row billed once per repeat, with headroom.
    planned_rows = (10 * 35 + 11 * 44 + max_attempts) * num_seeds
    assert budget.task_rows == 4 * planned_rows


def test_the_budget_without_a_control_keeps_the_small_run_ceilings() -> None:
    """The runner's own small shape is unaffected by the derivation."""
    budget = miprov2_budget()
    assert budget.task_rows == 256
    assert budget.bootstrap_generations == 32


def test_zeroshot_carries_zero_demo_maxima(engine_and_store, prepared) -> None:
    engine, _store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        demo_mode=Miprov2DemoMode.ZEROSHOT,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    assert control.max_bootstrapped_demos == 0
    assert control.max_labeled_demos == 0


@pytest.mark.parametrize(
    ("demo_mode", "searches_demos"),
    [
        (Miprov2DemoMode.FEWSHOT, True),
        (Miprov2DemoMode.ZEROSHOT, False),
        (Miprov2DemoMode.GROUND_ONLY, False),
    ],
)
def test_only_fewshot_searches_over_demo_sets(
    demo_mode: Miprov2DemoMode, *, searches_demos: bool
) -> None:
    assert demo_mode.searches_demos is searches_demos


def test_scripted_proposal_bodies_keep_required_placeholders() -> None:
    """MIPROv2 rejects an instruction that drops a required placeholder."""
    contract = c19_render_contract()
    for body in C19.proposal_bodies():
        assert set(contract.validate_template(body)) == set(
            contract.required_fields
        )


@pytest.mark.parametrize("with_demos", [True, False])
def test_composed_template_satisfies_the_c19_render_contract(
    *,
    with_demos: bool,
) -> None:
    """The demonstrations slot lives in MIPROv2's composed template.

    The composer emits a ``### Demonstrations`` section itself and escapes
    its JSON, so a fewshot candidate renders demos while zeroshot and
    ground_only render an empty section -- and both still satisfy the C19
    ``{grid}``/``{command}``/``{question}`` contract.
    """
    contract = c19_render_contract()
    demos = (
        [{"grid": "  ", "command": "F", "question": "heading?"}]
        if with_demos
        else None
    )
    component = {
        "component_id": MIPROV2_COMPONENT_ID,
        "instruction": C19.proposal_bodies()[0],
        "instruction_identity_hash": "a" * 64,
        "instruction_index": 0,
        "demo_index": 0 if with_demos else None,
        "demo_identity_hash": ("d" * 64) if with_demos else None,
        "demo_set": demos,
    }
    composed = compose_user_prompt_template(
        [component], template_render_contract=contract
    )
    assert "### Demonstrations" in composed
    assert set(contract.validate_template(composed)) == set(
        contract.required_fields
    )
    if with_demos:
        assert "heading?" in composed
    else:
        assert composed.endswith("### Demonstrations\n[]")


def test_adapter_and_opening_state_bind_the_exact_run(
    engine_and_store, prepared
) -> None:
    engine, store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    adapter = build_miprov2_adapter(
        store=store, engine=engine, control=control, family=C19
    )
    assert adapter.key == MIPROV2_ADAPTER_KEY
    run_ref = miprov2_run_ref(
        run_id="c19-miprov2-unit",
        control=control,
        experiment=prepared.experiment,
    )
    state = build_miprov2_state(
        run=run_ref,
        control=control,
        engine=engine,
        experiment=prepared.experiment,
        adapter=adapter,
        family=C19,
    )
    assert state.run == run_ref
    assert state.control.identity_hash() == control.identity_hash()
    assert (
        state.bindings.proposal_executor_policy_identity_hash
        == adapter.proposal_executor_policy_identity_hash
    )
    assert (
        state.bindings.proposal_transport_durability_identity_hash
        == adapter.proposal_transport_durability_identity_hash
    )
    assert state.labeled_trainset
    assert state.proposal_trainset
    assert state.proposal_components[0].component_id == MIPROV2_COMPONENT_ID


def test_control_requires_two_tasks_to_split(tmp_path) -> None:
    """A single internal task cannot split into a trainset and a valset."""
    single = prepare_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(1, 1, 0),
        num_seeds=1,
    )
    runtime_config = ReferenceEvalRuntimeConfig(
        transport_api_key_env="WHETSTONE_TOY_API_KEY",
    )
    with open_sqlite(str(tmp_path / "single.sqlite")) as store:
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=single.experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    single.experiment,
                    render_contract=C19.render_contract(),
                    ceiling_template=C19.probes.ceiling_template,
                )
            ),
        )
        task_hashes = tuple(engine.sampling.task_hashes)
        assert len(task_hashes) == 1
        with pytest.raises(ValueError, match="at least two tasks"):
            build_miprov2_control(
                engine=engine,
                experiment=single.experiment,
                family=C19,
                trainset_task_hashes=task_hashes,
                valset_task_hashes=task_hashes,
            )


def test_prompt_model_binds_the_experiment_provider_call_config(
    engine_and_store, prepared
) -> None:
    engine, _store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    expected = provider_call_config_ref(prepared.experiment)
    assert control.prompt_model.provider_call_config == expected


def test_labeled_demos_carry_the_task_gold(engine_and_store, prepared) -> None:
    """FAILS-BEFORE probe for empty labeled demo outputs."""
    engine, store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    adapter = build_miprov2_adapter(
        store=store, engine=engine, control=control, family=C19
    )
    run_ref = miprov2_run_ref(
        run_id="c19-miprov2-gold",
        control=control,
        experiment=prepared.experiment,
    )
    state = build_miprov2_state(
        run=run_ref,
        control=control,
        engine=engine,
        experiment=prepared.experiment,
        adapter=adapter,
        family=C19,
    )
    gold_by_hash = gold_by_task_hash(prepared.experiment)
    assert C19.response_field == C19_RESPONSE_FIELD
    assert state.labeled_trainset
    for demo in state.labeled_trainset:
        outputs = cast(
            "ImmutableJsonObject",
            demo.outputs_by_component[MIPROV2_COMPONENT_ID],
        ).to_json()
        assert outputs == {
            C19.response_field: gold_by_hash[demo.source_task_hash]
        }
        assert outputs[C19.response_field]


# --------------------------------------------------------------------------
# Search shape and split
# --------------------------------------------------------------------------


def test_the_default_search_shape_is_this_runners_own(
    engine_and_store, prepared
) -> None:
    """Wave 3's measured call counts are the cost of *these* numbers.

    Both are below the protocol's auto-light 10 and 6, which is exactly the
    caveat Wave 3 recorded; the defaults stay put so the fake-transport
    end-to-end tests remain fast.
    """
    engine, _store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    assert DEFAULT_MIPROV2_NUM_TRIALS == 2
    assert DEFAULT_MIPROV2_NUM_CANDIDATES == 3
    assert control.num_trials == DEFAULT_MIPROV2_NUM_TRIALS
    assert control.num_candidates == DEFAULT_MIPROV2_NUM_CANDIDATES


def test_the_protocol_shape_reaches_the_control(
    engine_and_store, prepared
) -> None:
    """The point of the setting: request auto-light without editing code."""
    engine, _store = engine_and_store
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        num_trials=10,
        num_candidates=6,
        trainset_task_hashes=_halves(engine)[0],
        valset_task_hashes=_halves(engine)[1],
    )
    assert control.num_trials == 10
    assert control.num_candidates == 6


@pytest.mark.parametrize(
    ("num_trials", "num_candidates", "message"),
    [
        (0, 3, "num_trials must be at least 1"),
        (-1, 3, "num_trials must be at least 1"),
        (2, 0, "num_candidates must be at least 1"),
    ],
)
def test_a_non_positive_search_shape_is_refused(
    engine_and_store,
    prepared,
    num_trials: int,
    num_candidates: int,
    message: str,
) -> None:
    """Refused outside the durable run boundary, like the other settings."""
    engine, _store = engine_and_store
    with pytest.raises(ValueError, match=message):
        build_miprov2_control(
            engine=engine,
            experiment=prepared.experiment,
            family=C19,
            num_trials=num_trials,
            num_candidates=num_candidates,
            trainset_task_hashes=_halves(engine)[0],
            valset_task_hashes=_halves(engine)[1],
        )


def test_the_control_carries_the_partition_it_was_given(
    engine_and_store, prepared
) -> None:
    """The two sets reach the control exactly as the caller named them."""
    engine, _store = engine_and_store
    task_hashes = tuple(engine.sampling.task_hashes)
    control = build_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        family=C19,
        trainset_task_hashes=task_hashes[:1],
        valset_task_hashes=task_hashes[1:2],
    )
    assert control.trainset_task_hashes == task_hashes[:1]
    assert control.valset_task_hashes == task_hashes[1:2]
    assert not set(control.trainset_task_hashes) & set(
        control.valset_task_hashes
    )


def test_an_overlapping_partition_is_refused(
    engine_and_store, prepared
) -> None:
    """DSPy's trainset = valset default is exactly what this refuses.

    Bootstrapped demonstrations would then be scored on their own tasks,
    so an in-search gain could be memorization rather than search.
    """
    engine, _store = engine_and_store
    task_hashes = tuple(engine.sampling.task_hashes)
    with pytest.raises(ValueError, match="must be disjoint"):
        build_miprov2_control(
            engine=engine,
            experiment=prepared.experiment,
            family=C19,
            trainset_task_hashes=task_hashes,
            valset_task_hashes=task_hashes,
        )


def test_a_partition_outside_the_internal_split_is_refused(
    engine_and_store, prepared
) -> None:
    """A valset the engine cannot evaluate is refused, not silently kept."""
    engine, _store = engine_and_store
    task_hashes = tuple(engine.sampling.task_hashes)
    with pytest.raises(ValueError, match="subset of the internal split"):
        build_miprov2_control(
            engine=engine,
            experiment=prepared.experiment,
            family=C19,
            trainset_task_hashes=task_hashes[:1],
            valset_task_hashes=("not-a-task-in-the-split",),
        )
