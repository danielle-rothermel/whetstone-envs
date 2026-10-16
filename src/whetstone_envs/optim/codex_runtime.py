from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dr_providers import ProviderKind, ReasoningEffort
from pydantic import BaseModel, ConfigDict, PositiveFloat, StrictInt, StrictStr
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.evaluation.runtime import (
    TransportKind,
    evaluation_execution_policy,
    prepare_evaluation_runtime,
)
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    openrouter_seeded_call_config,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.provider.policy import ProviderExecutionPolicy

    from whetstone_envs.optim.spec import RunSpec

#: The two transports a run may name, mirroring ``RunSpec.transport``.
#: Spelled here as a persisted enumeration because this config crosses a
#: process boundary.
CodexRuntimeTransport = Literal["fake", "openrouter"]


class EnvsCodexRuntimeConfig(BaseModel):
    """One family's evaluation runtime, rebuildable in another process.

    Every field is a *generation parameter*, not a derived artifact: the
    experiment is rebuilt from them rather than serialized, because an
    ``Experiment`` owns a live rollout graph that does not serialize.

    This is a persisted cross-process contract -- the MCP server validates
    it out of a subprocess environment variable -- so it is a frozen,
    ``extra="forbid"`` model and its field spellings are pinned by a
    golden test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_id: StrictStr
    split_sizes: tuple[StrictInt, StrictInt, StrictInt]
    n_per_category: StrictInt
    pool_seed_start: StrictInt
    num_seeds: StrictInt
    transport: CodexRuntimeTransport
    model: StrictStr

    #: The task route's reasoning effort, or ``None`` for the provider's
    #: default.
    #:
    #: Carried across the process boundary because the effort is part of
    #: the bound provider call config, and therefore part of the Eval
    #: Config the server must land on: ``build_codex_adapter`` compares
    #: this process's ``task_model_identity_hash`` against the harness's
    #: and refuses a rebuild that disagrees. A runtime that dropped the
    #: field would not evaluate at an unpinned effort -- it would fail to
    #: start, loudly, which is the guard working.
    #:
    #: Defaulted to ``None`` rather than required so a config written
    #: before this field existed still validates under ``extra="forbid"``,
    #: and so the fake transport keeps the identity it always had.
    reasoning_effort: ReasoningEffort | None = None

    #: How many task evaluations the server runs against the provider at
    #: once.
    #:
    #: Carried across the process boundary rather than defaulted on the
    #: far side, because the server has no other way to learn it: it
    #: rebuilds its engine from this config and nothing else, so an
    #: unset field means the Codex arm silently evaluates at whetstone's
    #: default width while every other arm runs at the operator's. Two
    #: arms measured at different widths is not a correctness problem,
    #: but it is a wall-clock one, and it is invisible without this
    #: field.
    provider_concurrency: StrictInt = DEFAULT_PROVIDER_CONCURRENCY

    # The three fields ``EvalRuntimeConfig`` requires structurally. They
    # are delegated straight to the reference config, so a Codex run gets
    # the same driver settings every other run does.
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None
    row_job_entrypoint: StrictStr = (
        "whetstone.eval.drivers.graph_worker:run_row"
    )
    unit_deadline_seconds: PositiveFloat = 86_400.0

    @property
    def _reference(self) -> ReferenceEvalRuntimeConfig:
        """The whetstone-ai config this one delegates its driver settings to.

        Only the experiment is envs' concern; everything about *how* a row
        is executed stays whetstone-ai's, so it is not restated here.
        """
        family = family_spec(self.family_id)
        return ReferenceEvalRuntimeConfig(
            partial_log_path=self.partial_log_path,
            prompt_cache_path=self.prompt_cache_path,
            row_job_entrypoint=self.row_job_entrypoint,
            unit_deadline_seconds=self.unit_deadline_seconds,
            transport_api_key_env=(
                "OPENROUTER_API_KEY"
                if self.transport == "openrouter"
                else "WHETSTONE_TOY_API_KEY"
            ),
            provider_kind=(
                ProviderKind.OPENROUTER
                if self.transport == "openrouter"
                else ProviderKind.OPENAI
            ),
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
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
        """The launch's mutation field, which the MCP server cross-checks."""
        return family_spec(self.family_id).mutation_field

    @property
    def render_contract(self):
        """The launch's render contract; the MCP server requires one."""
        return family_spec(self.family_id).render_contract()

    def build_engine(self, store: ObjectStore) -> EvalEngine:
        """Rebuild this run's exact evaluation engine from its parameters.

        The engine this returns must address the same ``eval_config_ref``
        as the harness's, because the Tool Config is bound to that hash
        and the evaluator refuses any call whose config does not match.
        """
        family = family_spec(self.family_id)
        pool = family.generate_pool(
            n_per_category=self.n_per_category,
            seed_start=self.pool_seed_start,
        )
        prepared = family.build_experiment(
            pool,
            split_sizes=self.split_sizes,
            num_seeds=self.num_seeds,
            provider_call_config=(
                openrouter_seeded_call_config(
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                )
                if self.transport == "openrouter"
                else None
            ),
        )
        experiment = prepared.experiment
        runtime = prepare_evaluation_runtime(
            experiment=experiment,
            family=family,
            reference=self._reference,
            transport=TransportKind(self.transport),
            concurrency=self.provider_concurrency,
        )
        return runtime.build_engine(store)


#: The ``module:Class`` path the Codex runner records so the MCP server
#: can load this config. It is a persisted string crossing a process
#: boundary, so it is named here and pinned by a golden test rather than
#: derived from ``__name__`` at a call site.
ENVS_CODEX_RUNTIME_CONFIG_CLASS = (
    "whetstone_envs.optim.codex_runtime:EnvsCodexRuntimeConfig"
)


__all__ = [
    "ENVS_CODEX_RUNTIME_CONFIG_CLASS",
    "CodexRuntimeTransport",
    "EnvsCodexRuntimeConfig",
]


def runtime_config_for_run(
    *, spec: RunSpec, n_per_category: int, pool_seed_start: int
) -> EnvsCodexRuntimeConfig:
    return EnvsCodexRuntimeConfig(
        family_id=spec.family.value,
        split_sizes=spec.split_sizes,
        n_per_category=n_per_category,
        pool_seed_start=pool_seed_start,
        num_seeds=spec.num_seeds,
        transport=spec.transport.value,
        model=spec.model,
        reasoning_effort=spec.task_reasoning_effort,
        provider_concurrency=spec.provider_concurrency,
    )
