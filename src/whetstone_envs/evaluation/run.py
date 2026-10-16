from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import ProviderKind
from dr_store.sync import open_sqlite
from whetstone.eval import EvalRequest
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine

from whetstone_envs.artifacts import (
    default_output_dir,
    durable_run_boundary,
    prepare_output_root,
)
from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.evaluation.candidates import (
    default_candidates,
    prepare_candidates,
)
from whetstone_envs.evidence import (
    SPLIT_ROLE_BY_REPORT_ROLE,
    EvalReport,
    publish_eval_report,
)
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.extraction.evaluation import project_eval_report
from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    widened_execution_policy,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone_envs.evaluation.spec import C19EvalSpec
    from whetstone_envs.optim.experiment import (
        PreparedSplitExperiment,
    )


_C19_CATEGORY_COUNT = 22


@dataclass(frozen=True, slots=True)
class EvalRunOutput:
    directory: Path
    report: EvalReport


def _prepare_experiment(spec: C19EvalSpec) -> PreparedSplitExperiment:
    provider = None
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(
            model=spec.model, reasoning_effort=spec.task_reasoning_effort
        )
    requested_tasks = sum(spec.split_sizes)
    n_per_category = max(
        1,
        (requested_tasks + _C19_CATEGORY_COUNT - 1) // _C19_CATEGORY_COUNT,
    )
    pool = generate_pool(
        n_per_category=n_per_category,
        seed_start=765_432,
    )
    return prepare_c19_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=spec.repeats,
        provider_call_config=provider,
    )


def _build_engine(
    spec: C19EvalSpec, prepared: PreparedSplitExperiment, store: ObjectStore
) -> RuntimeEvalEngine:
    runtime_config = ReferenceEvalRuntimeConfig(
        split_role=SPLIT_ROLE_BY_REPORT_ROLE[spec.role],
        transport_api_key_env=(
            "OPENROUTER_API_KEY"
            if spec.transport == "openrouter"
            else "WHETSTONE_TOY_API_KEY"
        ),
        provider_kind=(
            ProviderKind.OPENROUTER
            if spec.transport == "openrouter"
            else ProviderKind.OPENAI
        ),
    )
    execution_policy = widened_execution_policy(
        runtime_config.execution_policy,
        concurrency=spec.provider_concurrency,
    )
    if spec.transport == "openrouter":
        execution_policy = hardened_execution_policy(execution_policy)
        _, transport_factory = bind_openrouter_transport(execution_policy)
    else:
        transport_factory = fake_transport_factory(
            gold_by_prompt=fake_gold_by_prompt(
                prepared.experiment,
                render_contract=c19_render_contract(),
                ceiling_template=PROBES.ceiling_template,
            )
        )
    # build_engine cannot apply the widened policy or requested concurrency.
    driver = GraphRolloutEvalDriver(
        eval_runner=ExactMatchEvalProcedureRunner(),
        mutation_field=C19_MUTATION_FIELD,
        render_contract=c19_render_contract(),
        transport_factory=transport_factory,
    )
    return RuntimeEvalEngine(
        store=store,
        experiment=prepared.experiment,
        sampling=prepared.experiment.eval_configs.split_for(
            runtime_config.split_role
        ),
        execution_policy=execution_policy,
        driver=driver,
        concurrency=spec.provider_concurrency,
    )


def run_c19_evaluation(spec: C19EvalSpec) -> EvalRunOutput:
    candidate_refs = prepare_candidates(
        spec.candidates or default_candidates()
    )
    prepared = _prepare_experiment(spec)
    run_id = spec.run_id or f"c19-eval-{uuid4().hex[:8]}"
    output = prepare_output_root(spec.output_dir or default_output_dir(run_id))
    with (
        durable_run_boundary(output),
        open_sqlite(str(output / "runtime.sqlite")) as store,
    ):
        object_store = cast("ObjectStore", store)
        engine = _build_engine(spec, prepared, object_store)
        results = tuple(
            engine.evaluate(
                EvalRequest(
                    request_id=f"{run_id}:{index}:{item.name}",
                    candidate=item.candidate.record,
                    metadata={},
                )
            )
            for index, item in enumerate(candidate_refs)
        )
        report = project_eval_report(
            store=object_store,
            prepared=prepared,
            run_id=run_id,
            transport=spec.transport,
            model=spec.model,
            role=spec.role,
            split_sizes=spec.split_sizes,
            candidates=candidate_refs,
            results=results,
        )
        publish_eval_report(output, report)
    return EvalRunOutput(directory=output, report=report)


__all__ = ["EvalRunOutput", "run_c19_evaluation"]
