from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_graph import GraphConfig, graph_hash
from dr_providers import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    ControlConstraints,
    Protocol,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderKind,
    RequestControl,
    TokenLimitParameter,
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
from whetstone.experiment.reward import RewardPolicy, RewardTerm
from whetstone.experiment.sampling import (
    HELD_OUT,
    INTERNAL_EVAL,
    OFFICIAL,
    EvalConfigs,
    derive_eval_split,
)

from whetstone_envs.c19 import PROBES
from whetstone_envs.optim.rows import TaskRow, task_rows_from_instances
from whetstone_envs.pools import PoolSplit

if TYPE_CHECKING:
    from whetstone_envs.pools import TaskPool

C19_NAMESPACE = "whetstone_envs.c19"
C19_DATASET_REVISION = "c19/v1"
C19_MUTATION_FIELD = "prompt_template"
C19_ROOT_BASE_SCHEMA = "whetstone_envs.c19.root_candidate"
C19_PROMPT_FIELDS = ("grid", "command", "question")
#: The C19 family names its single generated component output "response";
#: demonstrations and traces carry the task gold under this exact key.
C19_RESPONSE_FIELD = "response"


@dataclass(frozen=True, slots=True)
class _C19RolloutGraph:
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
class PreparedC19Experiment:
    experiment: Experiment
    split: PoolSplit


def c19_render_contract() -> TemplateRenderContract:
    return TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=C19_PROMPT_FIELDS,
        required_fields=C19_PROMPT_FIELDS,
    )


def _task_hash(task: object) -> str:
    hashed = getattr(task, "task_hash", None)
    if not isinstance(hashed, str) or not hashed:
        raise TypeError("task must expose a nonempty task_hash")
    return hashed


def _reference_procedure() -> tuple[EvalProcedureConfig, str]:
    preprocessing = PreprocessingDefinition(
        definition_id=f"{C19_NAMESPACE}.preprocessing",
        version="1",
        steps=(),
    ).materialize()
    metric_extraction = MetricExtractionDefinition(
        definition_id=f"{C19_NAMESPACE}.metric_extraction",
        version="1",
        questions=(MetricQuestionBinding(metric="score", on="submission"),),
    ).materialize(resolved_operators=(("score", "1"),))
    procedure = EvalProcedureDefinition(
        definition_id=f"{C19_NAMESPACE}.evaluation_procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": "not_applicable"},
    )
    return procedure, procedure.config_hash


def _seeded_provider_call_config(
    provider_call_config: ProviderCallConfig | None,
) -> ProviderCallConfig:
    if provider_call_config is not None:
        return provider_call_config
    definition = ProviderCallDefinition(
        definition_id=f"{C19_NAMESPACE}.provider/v1",
        route={
            "provider": ProviderKind.OPENAI,
            "protocol": Protocol.CHAT_COMPLETIONS,
            "model": "fake-model",
        },
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
) -> _C19RolloutGraph:
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
    return _C19RolloutGraph(
        graph_config=graph_config,
        provider_call_config=provider_call_config,
        procedure_hash=procedure_hash,
        graph_hash_value=graph_hash(graph_config),
    )


def _reference_aggregation():
    return aggregation_definition(f"{C19_NAMESPACE}.aggregation").materialize(
        {
            "reduction": "mean",
            "missing_data": "propagate",
        }
    )


def c19_candidate(*, candidate_id: str, template: str) -> Candidate:
    """Build one validated C19 prompt candidate."""
    c19_render_contract().validate_template(template)
    root_ref = typed_ref_for_record(C19_ROOT_BASE_SCHEMA, {"kind": "root"})
    candidate = Candidate(
        candidate_id=candidate_id,
        base_ref=root_ref,
        payload={C19_MUTATION_FIELD: template},
    )
    return candidate_reference(candidate).record


def _rows_from_split(
    split: PoolSplit,
) -> tuple[tuple[TaskRow, ...], tuple[TaskRow, ...]]:
    internal = task_rows_from_instances(split.internal_eval)
    official = task_rows_from_instances(split.official)
    if not internal or not official:
        msg = "c19 experiment requires non-empty internal and official splits"
        raise ValueError(msg)
    return internal, official


def prepare_c19_experiment(
    pool: TaskPool,
    *,
    split_sizes: tuple[int, int, int],
    num_seeds: int = 1,
    provider_call_config: ProviderCallConfig | None = None,
) -> PreparedC19Experiment:
    """Prepare an Experiment and retain its authoritative source split."""
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    split = pool.split(*split_sizes)
    internal_rows, official_rows = _rows_from_split(split)
    procedure, procedure_hash = _reference_procedure()
    aggregation = _reference_aggregation()
    resolved_provider = _seeded_provider_call_config(provider_call_config)
    rollout_graph = _reference_rollout_graph(resolved_provider, procedure_hash)
    internal = derive_eval_split(
        namespace=C19_NAMESPACE,
        dataset_revision=C19_DATASET_REVISION,
        split_role=INTERNAL_EVAL,
        tasks=internal_rows,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=num_seeds,
    )
    official = derive_eval_split(
        namespace=C19_NAMESPACE,
        dataset_revision=C19_DATASET_REVISION,
        split_role=OFFICIAL,
        tasks=official_rows,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=num_seeds,
    )
    held_out_rows = task_rows_from_instances(split.held_out)
    held_out = (
        derive_eval_split(
            namespace=C19_NAMESPACE,
            dataset_revision=C19_DATASET_REVISION,
            split_role=HELD_OUT,
            tasks=held_out_rows,
            task_hash_of=_task_hash,
            procedure=procedure,
            aggregation=aggregation,
            num_seeds=num_seeds,
        )
        if held_out_rows
        else None
    )
    eval_configs = EvalConfigs(
        env_name=C19_NAMESPACE,
        procedure_config_hash=procedure_hash,
        internal=internal,
        official=official,
        held_out=held_out,
    )
    resolved_initial = PROBES.naive_template
    resolved_ceiling = PROBES.ceiling_template
    render_contract = c19_render_contract()
    render_contract.validate_template(resolved_initial)
    render_contract.validate_template(resolved_ceiling)
    reward_policy = RewardPolicy(
        policy_name="c19-exact-match",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    experiment = Experiment(
        env_name=C19_NAMESPACE,
        rollout_graph=rollout_graph,
        initial_candidate=c19_candidate(
            candidate_id="c19-initial",
            template=resolved_initial,
        ),
        ceiling_candidate=c19_candidate(
            candidate_id="c19-ceiling",
            template=resolved_ceiling,
        ),
        eval_configs=eval_configs,
        reward_policy=reward_policy,
        completeness_policy=CompletenessPolicy(),
    )
    return PreparedC19Experiment(experiment=experiment, split=split)


def c19_provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    """The typed reference to this experiment's provider call config.

    Optimizer proposer transports resolve the prompt model through this
    reference, so it must carry the ``PROVIDER_CALL_CONFIG_SCHEMA`` identity
    of the experiment's rollout config -- not an execution-policy reference.
    """
    payload = experiment.rollout_graph.provider_call_config.model_dump(
        mode="json"
    )
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def c19_gold_by_task_hash(experiment: Experiment) -> dict[str, str]:
    """Map every prepared C19 task hash to its exact oracle gold.

    The eval engine deliberately withholds gold from its sampling view, so
    labeled demonstrations read it from the family's own experiment splits.
    """
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
                raise TypeError("C19 task must expose a strict gold")
            gold_by_hash[_task_hash(task)] = gold
    return gold_by_hash


def probe_candidates_from_templates(
    *,
    naive_template: str,
    ceiling_template: str,
) -> tuple[Candidate, Candidate]:
    """Map a probe pair's templates onto initial and ceiling candidates."""
    render_contract = c19_render_contract()
    render_contract.validate_template(naive_template)
    render_contract.validate_template(ceiling_template)
    return (
        c19_candidate(candidate_id="c19-initial", template=naive_template),
        c19_candidate(candidate_id="c19-ceiling", template=ceiling_template),
    )


def reward_policy_for_exact_match() -> RewardPolicy:
    return RewardPolicy(
        policy_name="c19-exact-match",
        terms=(RewardTerm(name="score", weight=1.0),),
    )


__all__ = [
    "C19_MUTATION_FIELD",
    "C19_NAMESPACE",
    "C19_PROMPT_FIELDS",
    "C19_RESPONSE_FIELD",
    "PreparedC19Experiment",
    "c19_candidate",
    "c19_gold_by_task_hash",
    "c19_provider_call_config_ref",
    "c19_render_contract",
    "prepare_c19_experiment",
    "probe_candidates_from_templates",
    "reward_policy_for_exact_match",
]
