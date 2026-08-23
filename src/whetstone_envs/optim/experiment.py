"""The prepared-experiment shape every task family hands the runner.

One family-generic builder, :func:`prepare_experiment`, turns a pool and a
:class:`ExperimentContract` into the ``Experiment`` the optimizers drive.
``prepare_c19_experiment`` is that builder bound to C19's contract, and a
second family binds its own; nothing here is C19-shaped except the
``C19_*`` constants and the binding at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dr_graph import GraphConfig, graph_hash
from dr_providers import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    ControlConstraints,
    ProviderCallConfig,
    ProviderCallDefinition,
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
#: demonstrations and traces carry the task gold under this exact key. It
#: reaches the optimizers through ``C19_CONTRACT.response_field``, never as
#: a literal in the runner.
C19_RESPONSE_FIELD = "response"


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


class PreparedExperiment(Protocol):
    """What a family's experiment builder hands the runner and the report.

    ``run_optimizer`` and ``project_trajectory_report`` read exactly these
    two attributes, so a second family satisfies the contract by returning
    its own prepared pair rather than by subclassing anything.
    """

    @property
    def experiment(self) -> Experiment: ...

    @property
    def split(self) -> PoolSplit: ...


@dataclass(frozen=True, slots=True)
class PreparedSplitExperiment:
    """One family's Experiment beside the pool split that authored it.

    The split is retained because the eval configs carry task hashes, not
    the source instances, and the report projects per-stratum results from
    the instances.
    """

    experiment: Experiment
    split: PoolSplit


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Everything :func:`prepare_experiment` needs that is family-specific.

    A family names its persisted identity (``namespace`` and
    ``dataset_revision``, which reach the derived eval splits), the payload
    field its optimizers mutate, the placeholders its templates must keep,
    the key its generated component answers under, and the two probe
    templates that anchor its initial and ceiling candidates. Nothing else
    about the family reaches the builder.
    """

    namespace: str
    dataset_revision: str
    mutation_field: str
    prompt_fields: tuple[str, ...]
    #: The key a generated component writes its answer under. A labeled
    #: demonstration files the task's gold here, so a family that names its
    #: output differently says so once, in its contract.
    response_field: str
    root_base_schema: str
    reward_policy_name: str
    candidate_id_prefix: str
    naive_template: str
    ceiling_template: str

    def render_contract(self) -> TemplateRenderContract:
        """The contract every template for this family must satisfy."""
        return TemplateRenderContract(
            kind=TemplateRenderKind.PYTHON_FORMAT_V1,
            available_fields=self.prompt_fields,
            required_fields=self.prompt_fields,
        )

    def candidate(self, *, candidate_id: str, template: str) -> Candidate:
        """Build one validated prompt candidate for this family."""
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
        """This family's single exact-match reward term."""
        return RewardPolicy(
            policy_name=self.reward_policy_name,
            terms=(RewardTerm(name="score", weight=1.0),),
        )


C19_CONTRACT = ExperimentContract(
    namespace=C19_NAMESPACE,
    dataset_revision=C19_DATASET_REVISION,
    mutation_field=C19_MUTATION_FIELD,
    prompt_fields=C19_PROMPT_FIELDS,
    response_field=C19_RESPONSE_FIELD,
    root_base_schema=C19_ROOT_BASE_SCHEMA,
    reward_policy_name="c19-exact-match",
    candidate_id_prefix="c19",
    naive_template=PROBES.naive_template,
    ceiling_template=PROBES.ceiling_template,
)


def c19_render_contract() -> TemplateRenderContract:
    return C19_CONTRACT.render_contract()


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


#: The fraction of an evaluation's planned rows that may be lost before
#: the evaluation itself is void.
#:
#: **Why this is not zero.** A task-model call can fail for reasons that
#: are nothing to do with the candidate being measured -- a rate limit, a
#: transient 5xx, a timeout. Under ``missing_data="propagate"`` a single
#: such row sets the whole aggregate to ``None``: the live Stage 0 lost
#: exactly one row out of 352 and the entire naive-anchor evaluation, and
#: with it the stage, was voided. Discarding 351 good rows -- and their
#: cost -- over one 429 measures nothing and reports nothing.
#:
#: ``skip`` instead aggregates over the rows that are present and lets the
#: shortfall show up as reduced completeness, which is the handling the
#: protocol already pre-registered for this problem: §8's O7 recommends
#: per-task weighting by achieved sample count with a hard backstop at
#: 90%, the analysis already weights by ``per_task_counts``, and
#: :data:`~whetstone_envs.optim.study.manifest.COMPLETENESS_BACKSTOP` is
#: already 0.90. Propagating was the one piece inconsistent with that
#: rule.
#:
#: The tolerance is set to match the backstop rather than to something
#: laxer: beyond it the aggregate goes back to ``None``, so an evaluation
#: that lost more than a tenth of its rows still refuses to report a
#: number rather than quietly averaging a biased subset. Retries -- see
#: :class:`~whetstone_envs.optim.provider.RetryingTransport` -- are what
#: keep normal operation far away from this bound; this is the floor, not
#: the plan.
MAX_SKIP_FRACTION = 0.10


def _reference_aggregation(namespace: str):
    return aggregation_definition(f"{namespace}.aggregation").materialize(
        {
            "reduction": "mean",
            "missing_data": "skip",
            "max_skip_fraction": repr(MAX_SKIP_FRACTION),
        }
    )


def c19_candidate(*, candidate_id: str, template: str) -> Candidate:
    """Build one validated C19 prompt candidate."""
    return C19_CONTRACT.candidate(candidate_id=candidate_id, template=template)


def _rows_from_split(
    split: PoolSplit,
    *,
    namespace: str,
) -> tuple[tuple[TaskRow, ...], tuple[TaskRow, ...]]:
    internal = task_rows_from_instances(split.internal_eval)
    official = task_rows_from_instances(split.official)
    if not internal or not official:
        msg = (
            f"{namespace} experiment requires non-empty internal and "
            "official splits"
        )
        raise ValueError(msg)
    return internal, official


def prepare_experiment(
    pool: TaskPool,
    *,
    contract: ExperimentContract,
    split_sizes: tuple[int, int, int],
    num_seeds: int = 1,
    provider_call_config: ProviderCallConfig | None = None,
) -> PreparedSplitExperiment:
    """Prepare one family's Experiment and retain its source split.

    Every family reaches the optimizers through this one builder; the only
    thing that varies between two families is the ``contract`` handed in.
    Held-out is derived under its own role when the split requests one, and
    omitted otherwise, because upstream treats it as optional.
    """
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    split = pool.split(*split_sizes)
    internal_rows, official_rows = _rows_from_split(
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

    held_out_rows = task_rows_from_instances(split.held_out)
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


def prepare_c19_experiment(
    pool: TaskPool,
    *,
    split_sizes: tuple[int, int, int],
    num_seeds: int = 1,
    provider_call_config: ProviderCallConfig | None = None,
) -> PreparedSplitExperiment:
    """Prepare the C19 Experiment: :func:`prepare_experiment` under C19."""
    return prepare_experiment(
        pool,
        contract=C19_CONTRACT,
        split_sizes=split_sizes,
        num_seeds=num_seeds,
        provider_call_config=provider_call_config,
    )


def provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    """The typed reference to this experiment's provider call config.

    Optimizer proposer transports resolve the prompt model through this
    reference, so it must carry the ``PROVIDER_CALL_CONFIG_SCHEMA`` identity
    of the experiment's rollout config -- not an execution-policy reference.

    Every family's prepared experiment carries one rollout provider call
    config, so this reads the experiment alone: COPRO, GEPA, and MIPROv2
    bind one derivation rather than three copies.
    """
    payload = experiment.rollout_graph.provider_call_config.model_dump(
        mode="json"
    )
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def gold_by_task_hash(experiment: Experiment) -> dict[str, str]:
    """Map every prepared task hash to its exact oracle gold.

    The eval engine deliberately withholds gold from its sampling view, so
    labeled demonstrations read it from the family's own experiment splits.
    Every family's task rows carry a strict ``gold``, so this names no
    family.
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
                raise TypeError("an eval task must expose a strict gold")
            gold_by_hash[_task_hash(task)] = gold
    return gold_by_hash


def probe_candidates_from_templates(
    *,
    naive_template: str,
    ceiling_template: str,
) -> tuple[Candidate, Candidate]:
    """Map a C19 probe pair's templates onto initial and ceiling candidates."""
    return (
        C19_CONTRACT.candidate(
            candidate_id="c19-initial", template=naive_template
        ),
        C19_CONTRACT.candidate(
            candidate_id="c19-ceiling", template=ceiling_template
        ),
    )


def reward_policy_for_exact_match() -> RewardPolicy:
    return C19_CONTRACT.reward_policy()


__all__ = [
    "C19_CONTRACT",
    "C19_MUTATION_FIELD",
    "C19_NAMESPACE",
    "C19_PROMPT_FIELDS",
    "C19_RESPONSE_FIELD",
    "ExperimentContract",
    "PreparedExperiment",
    "PreparedSplitExperiment",
    "c19_candidate",
    "c19_render_contract",
    "gold_by_task_hash",
    "prepare_c19_experiment",
    "prepare_experiment",
    "probe_candidates_from_templates",
    "provider_call_config_ref",
    "reward_policy_for_exact_match",
]
