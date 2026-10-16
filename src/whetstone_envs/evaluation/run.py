from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_store.sync import open_sqlite
from whetstone.eval import EvalRequest

from whetstone_envs.artifacts import (
    default_output_dir,
    durable_run_boundary,
    prepare_output_root,
)
from whetstone_envs.c19 import generate_pool
from whetstone_envs.evaluation.candidates import (
    default_candidates,
    prepare_candidates,
)
from whetstone_envs.evaluation.runtime import (
    TransportKind,
    evaluation_reference_config,
    prepare_evaluation_runtime,
)
from whetstone_envs.evidence import (
    SPLIT_ROLE_BY_REPORT_ROLE,
    EvalReport,
    publish_eval_report,
)
from whetstone_envs.optim.experiment import (
    prepare_c19_experiment,
)
from whetstone_envs.optim.extraction.evaluation import project_eval_report
from whetstone_envs.optim.families import FamilyId, family_spec
from whetstone_envs.optim.provider import (
    openrouter_seeded_call_config,
)

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
        transport = TransportKind(spec.transport)
        runtime = prepare_evaluation_runtime(
            experiment=prepared.experiment,
            family=family_spec(FamilyId.C19),
            reference=evaluation_reference_config(
                transport, split_role=SPLIT_ROLE_BY_REPORT_ROLE[spec.role]
            ),
            transport=transport,
            concurrency=spec.provider_concurrency,
        )
        engine = runtime.build_engine(object_store)
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
