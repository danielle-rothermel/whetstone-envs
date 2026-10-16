from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    StrictInt,
    StrictStr,
)

from whetstone_envs.evaluation.concurrency import DEFAULT_PROVIDER_CONCURRENCY
from whetstone_envs.evaluation.runtime import (
    TransportKind,
    evaluation_execution_policy,
    evaluation_reference_config,
    prepare_evaluation_runtime,
)
from whetstone_envs.optim.problem import OptimizationProblem, load_problem

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.provider.policy import ProviderExecutionPolicy

    from whetstone_envs.optim.spec import RunSpec


CodexRuntimeTransport = Literal["fake", "openrouter"]


class EnvsCodexRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    problem_file: StrictStr
    transport: CodexRuntimeTransport
    provider_concurrency: StrictInt = Field(
        default=DEFAULT_PROVIDER_CONCURRENCY, ge=1
    )
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None
    row_job_entrypoint: StrictStr = (
        "whetstone.eval.drivers.graph_worker:run_row"
    )
    unit_deadline_seconds: PositiveFloat = 86_400.0

    @cached_property
    def _problem(self) -> OptimizationProblem:
        return load_problem(Path(self.problem_file))

    @property
    def _reference(self) -> ReferenceEvalRuntimeConfig:
        return evaluation_reference_config(
            TransportKind(self.transport)
        ).model_copy(
            update={
                "partial_log_path": self.partial_log_path,
                "prompt_cache_path": self.prompt_cache_path,
                "row_job_entrypoint": self.row_job_entrypoint,
                "unit_deadline_seconds": self.unit_deadline_seconds,
                "mutation_field": self.mutation_field,
                "render_contract": self.render_contract,
            }
        )

    @property
    def execution_policy(self) -> ProviderExecutionPolicy:
        return evaluation_execution_policy(
            self._reference,
            transport=TransportKind(self.transport),
            concurrency=self.provider_concurrency,
        )

    @property
    def mutation_field(self) -> str:
        return self._problem.contract.mutation_field

    @property
    def render_contract(self) -> TemplateRenderContract:
        return self._problem.contract.render_contract()

    def build_engine(self, store: ObjectStore) -> EvalEngine:
        problem = self._problem
        runtime = prepare_evaluation_runtime(
            experiment=problem.prepared.experiment,
            eval_runner=problem.eval_runner,
            mutation_field=self.mutation_field,
            render_contract=self.render_contract,
            fake_replies=problem.fake_replies,
            reference=self._reference,
            transport=TransportKind(self.transport),
            concurrency=self.provider_concurrency,
        )
        return runtime.build_engine(store)


# Persisted subprocess loader path.
ENVS_CODEX_RUNTIME_CONFIG_CLASS = (
    "whetstone_envs.optim.codex_runtime:EnvsCodexRuntimeConfig"
)


def runtime_config_for_run(
    *, spec: RunSpec, problem_file: Path | None
) -> EnvsCodexRuntimeConfig:
    if problem_file is None:
        raise ValueError("Codex requires a problem_file")
    return EnvsCodexRuntimeConfig(
        problem_file=str(problem_file.resolve()),
        transport=spec.transport.value,
        provider_concurrency=spec.provider_concurrency,
    )


__all__ = [
    "ENVS_CODEX_RUNTIME_CONFIG_CLASS",
    "CodexRuntimeTransport",
    "EnvsCodexRuntimeConfig",
]
