from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from dr_providers import ProviderKind, ReasoningEffort
from dr_store.sync import open_sqlite
from whetstone.eval import EvalRequest
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.experiment.candidate import CandidateRef, candidate_reference

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.evidence import (
    SPLIT_ROLE_BY_REPORT_ROLE,
    CandidateSource,
    EvalReport,
    EvalRoleName,
    publish_eval_report,
)
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    c19_candidate,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    bind_openrouter_transport,
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    widened_execution_policy,
)
from whetstone_envs.optim.run import DEFAULT_OUTPUT_ROOT
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner
from whetstone_envs.reporting.projection import project_eval_report
from whetstone_envs.reporting.publication import (
    durable_run_boundary,
    prepare_output_root,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone_envs.optim.experiment import (
        PreparedSplitExperiment,
    )

_C19_CATEGORY_COUNT = 22


@dataclass(frozen=True, slots=True)
class CandidateInput:
    name: str
    source: Literal["naive", "ceiling", "custom"]
    template: str


@dataclass(frozen=True, slots=True)
class C19EvalSpec:
    transport: Literal["fake", "openrouter"]
    role: EvalRoleName
    candidates: tuple[CandidateInput, ...] = ()
    repeats: int = 1
    split_sizes: tuple[int, int, int] = (20, 20, 0)
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"
    task_reasoning_effort: ReasoningEffort | None = None
    provider_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY


@dataclass(frozen=True, slots=True)
class EvalRunOutput:
    directory: Path
    report: EvalReport


def default_eval_output_dir(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / run_id


def default_candidates() -> tuple[CandidateInput, ...]:
    return (
        CandidateInput("naive", "naive", PROBES.naive_template),
        CandidateInput("ceiling", "ceiling", PROBES.ceiling_template),
    )


def _validate_candidates(
    candidates: tuple[CandidateInput, ...],
) -> tuple[tuple[str, CandidateSource, CandidateRef], ...]:
    names: set[str] = set()
    prepared: list[tuple[str, CandidateSource, CandidateRef]] = []
    for item in candidates:
        if not item.name.strip():
            raise ValueError("candidate names must be nonblank")
        if item.name in names:
            raise ValueError(f"duplicate candidate name {item.name!r}")
        names.add(item.name)
        candidate = c19_candidate(
            candidate_id=f"c19-{item.name}", template=item.template
        )
        prepared.append(
            (item.name, item.source, candidate_reference(candidate))
        )
    return tuple(prepared)


def _require_split_for_role(
    prepared: PreparedSplitExperiment, role: EvalRoleName
) -> None:
    """Refuse a role whose split this experiment does not carry.

    Held-out is the only optional split, so a zero held-out size must fail by
    name here rather than degrade into evaluating some other role's tasks.
    """
    split_role = SPLIT_ROLE_BY_REPORT_ROLE[role]
    if split_role not in prepared.experiment.eval_configs.splits():
        raise ValueError(
            f"this experiment has no {role} split: "
            f"split sizes must give the {role} role a positive size"
        )


def run_c19_evaluation(spec: C19EvalSpec) -> EvalRunOutput:
    """Evaluate selected C19 candidates and publish one strict local report."""
    if spec.transport not in {"fake", "openrouter"}:
        raise ValueError(f"unsupported transport {spec.transport!r}")
    if spec.role not in SPLIT_ROLE_BY_REPORT_ROLE:
        raise ValueError(f"unsupported role {spec.role!r}")
    if spec.repeats < 1:
        raise ValueError("repeats must be at least 1")
    candidates = spec.candidates or default_candidates()
    candidate_refs = _validate_candidates(candidates)
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(
            model=spec.model, reasoning_effort=spec.task_reasoning_effort
        )
        api_key_env = "OPENROUTER_API_KEY"
    requested_tasks = sum(spec.split_sizes)
    n_per_category = max(
        1,
        (requested_tasks + _C19_CATEGORY_COUNT - 1) // _C19_CATEGORY_COUNT,
    )
    pool = generate_pool(
        n_per_category=n_per_category,
        seed_start=765_432,
    )
    prepared = prepare_c19_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=spec.repeats,
        provider_call_config=provider,
    )
    _require_split_for_role(prepared, spec.role)
    run_id = spec.run_id or f"c19-eval-{uuid4().hex[:8]}"
    output = prepare_output_root(
        spec.output_dir or default_eval_output_dir(run_id)
    )
    with durable_run_boundary(output):
        runtime_config = ReferenceEvalRuntimeConfig(
            split_role=SPLIT_ROLE_BY_REPORT_ROLE[spec.role],
            transport_api_key_env=api_key_env,
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
        with open_sqlite(str(output / "runtime.sqlite")) as store:
            # Assembled here rather than through ``build_engine``, which
            # takes neither a concurrency nor a policy and so would yield
            # whetstone's default width over the *unwidened, unhardened*
            # policy -- silently discarding both transforms above. This is
            # the same in-process pair that helper builds for
            # ``driver_mode="in_process"``, which is this config's mode.
            driver = GraphRolloutEvalDriver(
                eval_runner=ExactMatchEvalProcedureRunner(),
                mutation_field=C19_MUTATION_FIELD,
                render_contract=c19_render_contract(),
                transport_factory=transport_factory,
            )
            engine = RuntimeEvalEngine(
                store=cast("ObjectStore", store),
                experiment=prepared.experiment,
                sampling=prepared.experiment.eval_configs.split_for(
                    runtime_config.split_role
                ),
                execution_policy=execution_policy,
                driver=driver,
                concurrency=spec.provider_concurrency,
            )
            results = tuple(
                engine.evaluate(
                    EvalRequest(
                        request_id=f"{run_id}:{index}:{name}",
                        candidate=candidate.record,
                        metadata={},
                    )
                )
                for index, (name, _source, candidate) in enumerate(
                    candidate_refs
                )
            )
            report = project_eval_report(
                store=cast("ObjectStore", store),
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


__all__ = [
    "C19EvalSpec",
    "CandidateInput",
    "EvalRunOutput",
    "default_candidates",
    "default_eval_output_dir",
    "run_c19_evaluation",
]
