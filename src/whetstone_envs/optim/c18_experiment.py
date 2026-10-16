from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.c18 import DEFAULT_CONFIG, PROBES
from whetstone_envs.c18 import generate_pool as _c18_generate_pool
from whetstone_envs.c18.config import GenerationConfig
from whetstone_envs.optim.experiment import (
    ExperimentContract,
    prepare_experiment,
)
from whetstone_envs.scoring.families import family_score

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dr_providers import ProviderCallConfig

    from whetstone_envs.instances import InstancePool
    from whetstone_envs.optim.experiment import PreparedSplitExperiment

__all__ = [
    "C18_CONTRACT",
    "C18_DATASET_REVISION",
    "C18_DEFAULT_N_PER_CATEGORY",
    "C18_DEFAULT_POOL_SEED_START",
    "C18_MUTATION_FIELD",
    "C18_NAMESPACE",
    "C18_PROMPT_FIELDS",
    "C18_RESPONSE_FIELD",
    "C18_TASK_CONTEXT",
    "C18VerdictEvalProcedureRunner",
    "c18_candidate",
    "c18_generate_pool",
    "c18_render_contract",
    "prepare_c18_experiment",
]

C18_NAMESPACE = "whetstone_envs.c18"
C18_DATASET_REVISION = "c18/v1"
#: C18 shares C19's payload field name because both families carry exactly
#: one optimizable prompt template; the field is a whetstone-side payload
#: key, not a C19 concept.
C18_MUTATION_FIELD = "prompt_template"
C18_ROOT_BASE_SCHEMA = "whetstone_envs.c18.root_candidate"
#: Both C18 probe templates use exactly these two placeholders.
C18_PROMPT_FIELDS = ("question", "query")
#: C18 shares C19's generated-output key: both families have exactly one
#: generated component whose reply is the answer under evaluation.
C18_RESPONSE_FIELD = "response"
C18_TASK_CONTEXT = (
    "Decide whether the query statement is entailed by the given fictional "
    "facts and rules."
)

#: C18's own generation defaults, so an unparameterised C18 run generates
#: the pool the family's ``DEFAULT_CONFIG`` describes.
C18_DEFAULT_N_PER_CATEGORY = DEFAULT_CONFIG.n_per_category
C18_DEFAULT_POOL_SEED_START = DEFAULT_CONFIG.seed_start


C18_CONTRACT = ExperimentContract(
    namespace=C18_NAMESPACE,
    dataset_revision=C18_DATASET_REVISION,
    mutation_field=C18_MUTATION_FIELD,
    prompt_fields=C18_PROMPT_FIELDS,
    response_field=C18_RESPONSE_FIELD,
    root_base_schema=C18_ROOT_BASE_SCHEMA,
    reward_policy_name="c18-exact-match",
    candidate_id_prefix="c18",
    naive_template=PROBES.naive_template,
    ceiling_template=PROBES.ceiling_template,
)


def c18_render_contract():
    """The contract every C18 template must satisfy."""
    return C18_CONTRACT.render_contract()


def c18_candidate(*, candidate_id: str, template: str):
    """Build one validated C18 prompt candidate."""
    return C18_CONTRACT.candidate(candidate_id=candidate_id, template=template)


def c18_generate_pool(*, n_per_category: int, seed_start: int) -> InstancePool:
    """Generate a C18 pool under the registry's calling convention.

    ``whetstone_envs.c18.generate_pool`` takes its seed start from a
    :class:`GenerationConfig` rather than as an argument, so this adapter
    rebuilds the default config at the requested seed. The config validates
    the seed itself, which keeps a seed outside PrOntoQA's admissible range
    a refusal here rather than a silent generator wraparound.
    """
    config = (
        DEFAULT_CONFIG
        if seed_start == C18_DEFAULT_POOL_SEED_START
        else GenerationConfig(
            generator_version=DEFAULT_CONFIG.generator_version,
            seed_start=seed_start,
            n_per_category=DEFAULT_CONFIG.n_per_category,
            categories=DEFAULT_CONFIG.categories,
            split=DEFAULT_CONFIG.split,
        )
    )
    return _c18_generate_pool(config, n_per_category=n_per_category)


def prepare_c18_experiment(
    pool: InstancePool,
    *,
    split_sizes: tuple[int, int, int],
    num_seeds: int = 1,
    provider_call_config: ProviderCallConfig | None = None,
) -> PreparedSplitExperiment:
    """Prepare the C18 Experiment: :func:`prepare_experiment` under C18.

    This is the whole second-family adapter for experiment preparation. It
    is the same call C19 makes with a different contract, which is what
    makes the two runs structurally identical downstream.
    """
    return prepare_experiment(
        pool,
        contract=C18_CONTRACT,
        split_sizes=split_sizes,
        num_seeds=num_seeds,
        provider_call_config=provider_call_config,
    )


class C18VerdictEvalProcedureRunner:
    """Score a C18 generation by its terminal True/False verdict.

    Workers reconstruct this type via ``runner_type()`` and transfer no
    constructor state, so it takes no arguments -- the same contract
    ``ExactMatchEvalProcedureRunner`` satisfies.

    C18's ceiling probe asks for reasoning ending in a lone verdict line, so
    an exact-match runner over the whole reply would score every reasoned
    answer zero and make the ceiling anchor indistinguishable from noise.
    The rule itself is not restated here: it comes from
    :func:`whetstone_envs.scoring.families.family_score`, whose ``c18``
    entry performs the extraction and the comparison via
    :func:`whetstone_envs.c18.score_gold`. A report's own score check
    calls that same function, which keeps one owner for how a C18 reply is
    scored and stops a recorded score drifting from its re-derivation.
    """

    #: The family whose rule this runner applies, resolved through the
    #: shared scorer registry rather than reimplemented.
    family_id = "c18"

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: object,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        generation = node_inputs.get("provider_generation")
        text = (
            generation
            if isinstance(generation, str)
            else str(generation or "")
        )
        raw_gold = getattr(task, "gold", "")
        gold = raw_gold if isinstance(raw_gold, str) else ""
        score = family_score(
            family=self.family_id, output_text=text, gold=gold
        )
        return score, {"text": text}, {}
