from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("whetstone.experiment.env")

if TYPE_CHECKING:
    from dr_store import ObjectStore

from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    build_toy_copro_control,
    copro_run_request,
    prepare_gepa_run,
    register_runtime,
)
from whetstone.core.identity import TypedRef
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.optim.gepa.adapter import GepaDetailedResult, GepaTerminalResult
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
)
from whetstone.optim.gepa.step_engine import GepaStepCheckpoint
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    build_c19_experiment,
    c19_render_contract,
)
from whetstone_envs.optim.run import C19RunSpec, run_c19_optimizer
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner


def test_copro_fake_transport_completes(tmp_path) -> None:
    output = run_c19_optimizer(
        C19RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            output_dir=tmp_path / "copro-run",
            run_id="c19-copro-e2e",
        )
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result.step_results
    assert result.step_results[-1].record.status.value in {
        "complete",
        "failed",
    }


def test_gepa_fake_transport_completes(tmp_path) -> None:
    sqlite_path = tmp_path / "gepa.sqlite"
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    experiment = build_c19_experiment(pool, split_sizes=(2, 2, 0), num_seeds=1)
    with open_sqlite(str(sqlite_path)) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
        )
        prompt_adapter = PlainPromptAdapter()
        task_hashes = engine.sampling.task_hashes[:2]
        control = configure_gepa(
            reflection_model=ProposerConfig(
                provider_call_config=engine.provider_execution_policy_ref,
            ),
            metric=engine.eval_config_ref,
            reward_policy_hash=experiment.reward_policy.identity_hash(),
            evaluation_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            proposal_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            proposal_prompt_adapter_identity_hash=(
                prompt_adapter_identity_hash(prompt_adapter)
            ),
            proposal_durability_policy_identity_hash="c" * 64,
            task_model_identity_hash=engine.task_model_identity_hash(),
            prompt_format_identity_hash="d" * 64,
            prompt_binding_identity_hash="e" * 64,
            trainset_task_hashes=task_hashes,
            valset_task_hashes=None,
            component_names=("generate",),
            num_predictors=1,
            max_metric_calls=2,
        )
        factory = MagicMock()
        factory.create.return_value = MagicMock()
        factory.persist_result.return_value = TypedRef(
            schema_name="whetstone.gepa.result",
            content_hash="a" * 64,
        )
        adapter = GepaHarnessAdapter(
            control=control,
            seed_candidate={"generate": PROBES.naive_template},
            trainset=(),
            valset=None,
            adapter_factory=GepaHarnessAdapterFactory(factory=factory),
        )
        runtime = register_runtime(
            store=store,
            engine=engine,
            copro_control=build_toy_copro_control(engine=engine),
            extra_adapters={GEPA_ADAPTER_KEY: adapter},
        )
        launch = prepare_gepa_run(
            runtime,
            run_id="c19-gepa-e2e",
            control=control,
            experiment=experiment,
            render_contract=c19_render_contract(),
            mutation_field=C19_MUTATION_FIELD,
        )
        request = copro_run_request(
            launch,
            controller_identity_hash=runtime.controller.runtime_hash,
        )
        detailed = GepaDetailedResult(
            candidates=({"generate": PROBES.naive_template},),
            parents=((None,),),
            val_aggregate_scores=(1.0,),
            val_subscores=({"task": 1.0},),
            per_val_instance_best_candidates={"task": (0,)},
            discovery_eval_counts=(1,),
            seed=0,
            best_idx=0,
            control_identity_hash=control.identity_hash(),
        )
        with (
            patch(
                "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
                side_effect=[
                    (
                        detailed,
                        GepaStepCheckpoint(
                            metric_calls_consumed=1,
                            terminal=False,
                        ),
                    ),
                    (
                        detailed,
                        GepaStepCheckpoint(
                            metric_calls_consumed=2,
                            terminal=True,
                        ),
                    ),
                ],
            ),
            patch(
                "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
                return_value=GepaTerminalResult(
                    best_candidate={"generate": PROBES.ceiling_template},
                    control_identity_hash=control.identity_hash(),
                    artifact_ref=TypedRef(
                        schema_name="whetstone.gepa.result",
                        content_hash="a" * 64,
                    ),
                ),
            ),
        ):
            result_ref = runtime.controller.drive(request)
        assert result_ref.schema_name == OPTIM_RESULT_SCHEMA
        result = OptimResult.model_validate(
            runtime.store.get(result_ref.reference)
        )
        assert result.step_results[-1].record.status.value == "complete"


def test_run_refuses_in_repo_output() -> None:
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=Path("artifacts") / "c19-run",
            )
        )


def test_run_refuses_in_repo_output_when_cwd_is_elsewhere(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="must not be written inside the repo"
    ):
        run_c19_optimizer(
            C19RunSpec(
                optimizer="copro",
                transport="fake",
                output_dir=repo_root / "artifacts" / "c19-run",
            )
        )
