from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_graph import GraphConfig, graph_hash
from dr_providers import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    ControlConstraints,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderKind,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers import (
    Protocol as WireProtocol,
)
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.eval import (
    SCHEMA_EVAL_PROCEDURE_CONFIG,
    EvalProcedureConfig,
    EvalProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
)
from whetstone.eval.aggregate import CompletenessPolicy, aggregation_definition
from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
    candidate_reference,
)
from whetstone.experiment.env import Experiment
from whetstone.experiment.graph.rollout_template import (
    build_single_llm_eval_graph,
)
from whetstone.experiment.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
)
from whetstone.experiment.sampling import (
    HELD_OUT,
    INTERNAL_EVAL,
    OFFICIAL,
    EvalConfigs,
    derive_eval_split,
)

from whetstone_envs.evaluation.rows import TaskRow, task_rows_from_instances
from whetstone_envs.instances import InstancePoolSplit

if TYPE_CHECKING:
    from whetstone_envs.instances import InstancePool


@dataclass(frozen=True, slots=True)
class _RolloutGraph:
    graph_config: GraphConfig
    provider_call_config: ProviderCallConfig
    procedure_hash: str
    graph_hash_value: str

    @property
    def graph_hash(self) -> str:
        return self.graph_hash_value

    @property
    def procedure_config_hash(self) -> str:
        return self.procedure_hash


@dataclass(frozen=True, slots=True)
class PreparedSplitExperiment:
    experiment: Experiment
    split: InstancePoolSplit


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    namespace: str
    dataset_revision: str
    mutation_field: str
    prompt_fields: tuple[str, ...]
    response_field: str
    root_base_schema: str
    reward_policy_name: str
    candidate_id_prefix: str
    naive_template: str
    ceiling_template: str

    def render_contract(self) -> TemplateRenderContract:
        return TemplateRenderContract(
            kind=TemplateRenderKind.PYTHON_FORMAT_V1,
            available_fields=self.prompt_fields,
            required_fields=self.prompt_fields,
        )

    def candidate(self, *, candidate_id: str, template: str) -> Candidate:
        self.render_contract().validate_template(template)
        root_ref = typed_ref_for_record(
            self.root_base_schema, {"kind": "root"}
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            base_ref=root_ref,
            payload={self.mutation_field: template},
        )
        return candidate_reference(candidate).record

    def reward_policy(self) -> RewardPolicy:
        return RewardPolicy(
            policy_name=self.reward_policy_name,
            terms=(RewardTerm(name="score", weight=1.0),),
            missing_data=MissingDataPolicy.SKIP,
        )


def _task_hash(task: object) -> str:
    hashed = getattr(task, "task_hash", None)
    if not isinstance(hashed, str) or not hashed:
        raise TypeError("task must expose a nonempty task_hash")
    return hashed


def _reference_procedure(
    namespace: str,
) -> tuple[EvalProcedureConfig, str]:
    preprocessing = PreprocessingDefinition(
        definition_id=f"{namespace}.preprocessing",
        version="1",
        steps=(),
    ).materialize()
    metric_extraction = MetricExtractionDefinition(
        definition_id=f"{namespace}.metric_extraction",
        version="1",
        questions=(MetricQuestionBinding(metric="score", on="submission"),),
    ).materialize(resolved_operators=(("score", "1"),))
    procedure = EvalProcedureDefinition(
        definition_id=f"{namespace}.evaluation_procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": "not_applicable"},
    )
    return procedure, procedure.config_hash


def _seeded_provider_call_config(
    provider_call_config: ProviderCallConfig | None,
    *,
    namespace: str,
) -> ProviderCallConfig:
    if provider_call_config is not None:
        return provider_call_config
    definition = ProviderCallDefinition(
        definition_id=f"{namespace}.provider/v1",
        route={
            "provider": ProviderKind.OPENAI,
            "protocol": WireProtocol.CHAT_COMPLETIONS,
            "model": "fake-model",
        },
        supported_kinds=frozenset({ProviderCallKind.GENERATE}),
        constraints=ControlConstraints(
            supported_controls=frozenset(
                {
                    RequestControl.TEMPERATURE,
                    RequestControl.TOP_P,
                    RequestControl.TOKEN_LIMIT,
                    RequestControl.SEED,
                }
            ),
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
        ),
    )
    return ProviderCallConfig(
        definition=definition, controls={}, extensions={}
    )


def _reference_rollout_graph(
    provider_call_config: ProviderCallConfig,
    procedure_hash: str,
) -> _RolloutGraph:
    provider_ref_hash = str(
        typed_ref_for_record(
            PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config.model_dump(mode="json"),
        ).content_hash
    )
    graph_config = build_single_llm_eval_graph(
        provider_call_config_hash=provider_ref_hash,
        evaluation_procedure_config_schema=SCHEMA_EVAL_PROCEDURE_CONFIG,
        evaluation_procedure_config_hash=procedure_hash,
    )
    return _RolloutGraph(
        graph_config=graph_config,
        provider_call_config=provider_call_config,
        procedure_hash=procedure_hash,
        graph_hash_value=graph_hash(graph_config),
    )


MAX_SKIP_FRACTION = 0.10


def _reference_aggregation(namespace: str):
    return aggregation_definition(f"{namespace}.aggregation").materialize(
        {
            "reduction": "mean",
            "missing_data": "skip",
            "max_skip_fraction": repr(MAX_SKIP_FRACTION),
        }
    )


def _required_evaluation_rows(
    split: InstancePoolSplit,
    *,
    namespace: str,
) -> tuple[tuple[TaskRow, ...], tuple[TaskRow, ...]]:
    internal = task_rows_from_instances(split.splits[INTERNAL_EVAL].instances)
    official = task_rows_from_instances(split.splits[OFFICIAL].instances)
    if not internal or not official:
        msg = (
            f"{namespace} experiment requires non-empty internal and "
            "official splits"
        )
        raise ValueError(msg)
    return internal, official


def prepare_experiment(
    pool: InstancePool,
    *,
    contract: ExperimentContract,
    split_sizes: tuple[int, int, int],
    num_seeds: int = 1,
    provider_call_config: ProviderCallConfig | None = None,
) -> PreparedSplitExperiment:
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    split = pool.split(
        dict(
            zip((INTERNAL_EVAL, OFFICIAL, HELD_OUT), split_sizes, strict=True)
        ),
    )
    internal_rows, official_rows = _required_evaluation_rows(
        split, namespace=contract.namespace
    )
    procedure, procedure_hash = _reference_procedure(contract.namespace)
    aggregation = _reference_aggregation(contract.namespace)
    resolved_provider = _seeded_provider_call_config(
        provider_call_config, namespace=contract.namespace
    )
    rollout_graph = _reference_rollout_graph(resolved_provider, procedure_hash)

    def derive(split_role, rows):
        return derive_eval_split(
            namespace=contract.namespace,
            dataset_revision=contract.dataset_revision,
            split_role=split_role,
            tasks=rows,
            task_hash_of=_task_hash,
            procedure=procedure,
            aggregation=aggregation,
            num_seeds=num_seeds,
        )

    held_out_rows = task_rows_from_instances(split.splits[HELD_OUT].instances)
    eval_configs = EvalConfigs(
        env_name=contract.namespace,
        procedure_config_hash=procedure_hash,
        internal=derive(INTERNAL_EVAL, internal_rows),
        official=derive(OFFICIAL, official_rows),
        held_out=derive(HELD_OUT, held_out_rows) if held_out_rows else None,
    )
    experiment = Experiment(
        env_name=contract.namespace,
        rollout_graph=rollout_graph,
        initial_candidate=contract.candidate(
            candidate_id=f"{contract.candidate_id_prefix}-initial",
            template=contract.naive_template,
        ),
        ceiling_candidate=contract.candidate(
            candidate_id=f"{contract.candidate_id_prefix}-ceiling",
            template=contract.ceiling_template,
        ),
        eval_configs=eval_configs,
        reward_policy=contract.reward_policy(),
        completeness_policy=CompletenessPolicy(),
    )
    return PreparedSplitExperiment(experiment=experiment, split=split)


def provider_config_ref(config: ProviderCallConfig) -> IdentityRef:
    payload = config.model_dump(mode="json")
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    return provider_config_ref(experiment.rollout_graph.provider_call_config)


def gold_by_task_hash(experiment: Experiment) -> dict[str, str]:
    gold_by_hash: dict[str, str] = {}
    for split in (
        experiment.eval_configs.internal,
        experiment.eval_configs.official,
        experiment.eval_configs.held_out,
    ):
        if split is None:
            continue
        for task in split.tasks:
            gold = getattr(task, "gold", None)
            if not isinstance(gold, str):
                raise TypeError("an eval task must expose a strict gold")
            gold_by_hash[_task_hash(task)] = gold
    return gold_by_hash


__all__ = [
    "ExperimentContract",
    "PreparedSplitExperiment",
    "gold_by_task_hash",
    "prepare_experiment",
    "provider_call_config_ref",
    "provider_config_ref",
]
