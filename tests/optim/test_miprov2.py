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
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.miprov2 import (
    C19_DEMO_MODES,
    MIPROV2_COMPONENT_ID,
    Miprov2DemoMode,
    build_c19_miprov2_adapter,
    build_c19_miprov2_control,
    build_c19_miprov2_state,
    c19_miprov2_proposal_bodies,
    c19_miprov2_run_ref,
)
from whetstone_envs.optim.provider import (
    c19_fake_gold_by_prompt,
    c19_fake_transport_factory,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore


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
            transport_factory=c19_fake_transport_factory(
                gold_by_prompt=c19_fake_gold_by_prompt(prepared.experiment)
            ),
        )
        yield engine, store


def test_demo_modes_cover_the_whetstone_enumeration() -> None:
    assert set(C19_DEMO_MODES) == {mode.value for mode in Miprov2DemoMode}
    assert C19_DEMO_MODES == ("fewshot", "zeroshot", "ground_only")


@pytest.mark.parametrize("demo_mode", list(Miprov2DemoMode))
def test_control_binds_the_c19_mutation_surface(
    engine_and_store, prepared, demo_mode: Miprov2DemoMode
) -> None:
    engine, _store = engine_and_store
    control = build_c19_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        demo_mode=demo_mode,
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


def test_zeroshot_carries_zero_demo_maxima(engine_and_store, prepared) -> None:
    engine, _store = engine_and_store
    control = build_c19_miprov2_control(
        engine=engine,
        experiment=prepared.experiment,
        demo_mode=Miprov2DemoMode.ZEROSHOT,
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
    for body in c19_miprov2_proposal_bodies():
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
        "instruction": c19_miprov2_proposal_bodies()[0],
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
    control = build_c19_miprov2_control(
        engine=engine, experiment=prepared.experiment
    )
    adapter = build_c19_miprov2_adapter(
        store=store, engine=engine, control=control
    )
    assert adapter.key == MIPROV2_ADAPTER_KEY
    run_ref = c19_miprov2_run_ref(
        run_id="c19-miprov2-unit",
        control=control,
        experiment=prepared.experiment,
    )
    state = build_c19_miprov2_state(
        run=run_ref, control=control, engine=engine, adapter=adapter
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
            transport_factory=c19_fake_transport_factory(
                gold_by_prompt=c19_fake_gold_by_prompt(single.experiment)
            ),
        )
        assert len(engine.sampling.task_hashes) == 1
        with pytest.raises(ValueError, match="at least two tasks"):
            build_c19_miprov2_control(
                engine=engine, experiment=single.experiment
            )
