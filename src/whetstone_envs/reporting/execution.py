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
from whetstone.eval.schema import EvalEvidence
from whetstone.experiment.candidate import CandidateRef, candidate_reference

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.completeness import require_task_completeness
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
from whetstone_envs.optim.run_cost import RunCostDocument, write_run_cost
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner
from whetstone_envs.optim.study.manifest import RunSpendRecord
from whetstone_envs.reporting.projection import project_eval_report
from whetstone_envs.reporting.publication import (
    durable_run_boundary,
    prepare_output_root,
    publish_eval_report,
)
from whetstone_envs.reporting.schema import (
    SPLIT_ROLE_BY_REPORT_ROLE,
    CandidateSource,
    EvalReport,
    EvalRoleName,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from dr_store.sync import BlockingObjectStore

    from whetstone_envs.optim.experiment import (
        PreparedSplitExperiment,
    )

_C19_STRATUM_COUNT = 22


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
    #: The task route's reasoning effort. ``None`` sends no reasoning key,
    #: which is what this CLI did before the field existed.
    #:
    #: Threaded so a standalone report can reproduce a study arm's route
    #: exactly: an evaluation of the study's own candidates at a different
    #: effort is an evaluation of a different task model.
    task_reasoning_effort: ReasoningEffort | None = None
    #: How many task evaluations run against the provider at once.
    #:
    #: The same operator setting the study path takes, for the same
    #: reason: this CLI reaches the same provider with the same
    #: reasoning-model latency, and a standalone evaluation of a few
    #: hundred rows is exactly as bound by width as a stage is.
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
    n_per_stratum = max(
        1,
        (requested_tasks + _C19_STRATUM_COUNT - 1) // _C19_STRATUM_COUNT,
    )
    pool = generate_pool(
        n_per_stratum=n_per_stratum,
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
        # The same two transforms, in the same order, the study path
        # applies -- widen the connection pool to the requested width,
        # then harden the paid route's timeout and retry ownership. This
        # CLI talks to the same provider at the same token sizes, so a
        # policy that differed here would mean the standalone tool ran at
        # whetstone's 30 s chat-completion bound with retries that never
        # waited, which is the failure that aborted the live Stage 0.
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
            _require_reported_completeness(results, candidate_refs)
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
            _publish_eval_cost(output, report, store=store)
    return EvalRunOutput(directory=output, report=report)


def _require_reported_completeness(
    results: tuple[object, ...],
    candidate_refs: tuple[tuple[str, object, object], ...],
) -> None:
    """Apply the study's per-task floor to the standalone report too.

    ``whetstone-eval`` publishes a held-out number that a claim is made
    from, exactly as a study stage does, and it is subject to exactly the
    same loss: a task whose every repeat was dropped leaves the task
    mean averaging over a smaller population than it reports, biased
    upward by the slow tasks whose absence caused it. The floor lived
    only in :class:`~whetstone_envs.optim.study.arms.RoleScorer`, so this
    path published the very number the floor exists to refuse.

    Checked after the evaluations rather than during, so the report path
    matches the stage's "priced first, then judged" ordering: the calls
    were billed whether or not the evidence is fit to report, and the
    cost artifact is not written for a run that refuses -- but the rows
    persist, so a refused run's spend is still recoverable from the
    store.

    An evaluation that produced no evidence at all is left alone here;
    the projection already refuses a result without evidence, and
    reporting a completeness failure for it would name the wrong cause.
    """
    for (name, _source, _candidate), result in zip(
        candidate_refs, results, strict=True
    ):
        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, EvalEvidence):
            continue
        require_task_completeness(evidence, purpose=f"eval:{name}")


def _publish_eval_cost(
    output: Path,
    report: EvalReport,
    *,
    store: BlockingObjectStore,
) -> None:
    """Write ``cost.json`` beside the report, when there was a bill.

    Optimizer runs publish one of these and the standalone eval path did
    not, so held-out evaluation -- the role every efficacy claim is finally
    made against -- spent real money that no artifact recorded. A study's
    reported cost understated its true spend by the whole reporting pass.

    Projected from the report rather than re-read from the rows: the report
    already re-derived the bill from the persisted evidence, and a second
    reading could disagree with the number the report publishes.

    A run that evidenced no provider call writes nothing. An all-zero
    document would claim the evaluation was measured and found free, which
    is the same untrue claim ``project_run_cost`` declines to make.
    """
    if report.spend is None:
        return
    task_model = report.spend.task_model
    write_run_cost(
        output,
        RunCostDocument(
            run_id=report.run.run_id,
            cost_report_schema_version=report.spend.schema_version,
            spend=(
                RunSpendRecord(
                    role=task_model.role,
                    calls=task_model.calls,
                    cached_calls=task_model.cached_calls,
                    input_tokens=task_model.input_tokens,
                    output_tokens=task_model.output_tokens,
                    priced_calls=task_model.priced_calls,
                    unpriced_calls=task_model.unpriced_calls,
                    rows_missing_token_breakdown=(
                        task_model.rows_missing_token_breakdown
                    ),
                    usd=task_model.usd,
                ),
            ),
        ),
        store=store,
    )


__all__ = [
    "C19EvalSpec",
    "CandidateInput",
    "EvalRunOutput",
    "default_candidates",
    "default_eval_output_dir",
    "run_c19_evaluation",
]
