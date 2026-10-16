from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from dr_providers import ProviderKind
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.eval_procedure import EvalProcedureRunner
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import INTERNAL_EVAL
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy

from whetstone_envs.evaluation.provider import (
    RetryingTransport,
    bind_openrouter_transport,
    fake_transport_factory,
    hardened_execution_policy,
    widened_execution_policy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dr_store import ObjectStore


@verify(UNIQUE)
class TransportKind(StrEnum):
    FAKE = auto()
    OPENROUTER = auto()


def evaluation_reference_config(
    transport: TransportKind,
    *,
    split_role: str = INTERNAL_EVAL,
) -> ReferenceEvalRuntimeConfig:
    return ReferenceEvalRuntimeConfig(
        split_role=split_role,
        transport_api_key_env=(
            "OPENROUTER_API_KEY"
            if transport is TransportKind.OPENROUTER
            else "WHETSTONE_TOY_API_KEY"
        ),
        provider_kind=(
            ProviderKind.OPENROUTER
            if transport is TransportKind.OPENROUTER
            else ProviderKind.OPENAI
        ),
    )


def evaluation_execution_policy(
    reference: ReferenceEvalRuntimeConfig,
    *,
    transport: TransportKind,
    concurrency: int,
) -> ProviderExecutionPolicy:
    policy = widened_execution_policy(
        reference.execution_policy, concurrency=concurrency
    )
    return (
        hardened_execution_policy(policy)
        if transport is TransportKind.OPENROUTER
        else policy
    )


@dataclass(frozen=True, slots=True)
class PreparedEvaluationRuntime:
    experiment: Experiment
    eval_runner: EvalProcedureRunner
    mutation_field: str
    render_contract: TemplateRenderContract
    reference: ReferenceEvalRuntimeConfig
    execution_policy: ProviderExecutionPolicy
    concurrency: int
    transport_factory: Callable[[ProviderExecutionPolicy], TransportCall]
    live_transport: RetryingTransport | None

    def build_engine(self, store: ObjectStore) -> RuntimeEvalEngine:
        # Whetstone's reference builder cannot apply our policy or concurrency.
        driver = GraphRolloutEvalDriver(
            eval_runner=self.eval_runner,
            mutation_field=self.mutation_field,
            render_contract=self.render_contract,
            transport_factory=self.transport_factory,
        )
        return RuntimeEvalEngine(
            store=store,
            experiment=self.experiment,
            sampling=self.experiment.eval_configs.split_for(
                self.reference.split_role
            ),
            execution_policy=self.execution_policy,
            driver=driver,
            concurrency=self.concurrency,
        )


def prepare_evaluation_runtime(  # noqa: PLR0913
    *,
    experiment: Experiment,
    eval_runner: EvalProcedureRunner,
    mutation_field: str,
    render_contract: TemplateRenderContract,
    fake_replies: Mapping[str, str],
    reference: ReferenceEvalRuntimeConfig,
    transport: TransportKind,
    concurrency: int,
) -> PreparedEvaluationRuntime:
    policy = evaluation_execution_policy(
        reference, transport=transport, concurrency=concurrency
    )
    live_transport = None
    if transport is TransportKind.OPENROUTER:
        live_transport, transport_factory = bind_openrouter_transport(policy)
    else:
        transport_factory = fake_transport_factory(replies=fake_replies)
    return PreparedEvaluationRuntime(
        experiment=experiment,
        eval_runner=eval_runner,
        mutation_field=mutation_field,
        render_contract=render_contract,
        reference=reference,
        execution_policy=policy,
        concurrency=concurrency,
        transport_factory=transport_factory,
        live_transport=live_transport,
    )


__all__ = [
    "PreparedEvaluationRuntime",
    "TransportKind",
    "evaluation_execution_policy",
    "evaluation_reference_config",
    "prepare_evaluation_runtime",
]
