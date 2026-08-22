"""The eval runtime config the out-of-process Codex MCP server rebuilds.

Codex evaluates through an MCP server whetstone starts in a *separate
process*, and that process rebuilds its evaluation engine from one
serialized config and nothing else
(``whetstone.optim.codex.mcp_server.build_server_from_env``). whetstone-ai
ships :class:`~whetstone.eval.reference_runtime.ReferenceEvalRuntimeConfig`
for this, but its experiment is always the in-memory *toy* experiment --
it can carry a launch's ``mutation_field`` and ``render_contract``, but it
has no way to carry the launch's ``Experiment``. whetstone-ai names that
limitation itself, in ``platform/cli.py``'s ``ToyExperimentOnlyError``: a
launch's persisted record cannot supply an experiment, because an
``Experiment`` owns a live ``rollout_graph`` and the launch persists only
identity hashes over the eval config.

The consequence for a study run is not a crash: the server comes up
happily on the toy experiment, its Eval Config hash does not match the
run's Tool Config, and **every** tool call is refused as
``tool config is not bound to the engine's exact Eval Config``. The agent
then has nothing to select and the Step terminalizes. That is what this
module exists to prevent.

**Why an envs-owned config is the right shape.** ``EvalRuntimeConfig`` is
an open ``typing.Protocol``, and the server loads whichever class the
runner names by ``module:Class`` path, so supplying one is the extension
point whetstone-ai already provides rather than a workaround. And envs
*can* do what a persisted launch cannot: a family's experiment is a pure
function of its generation parameters, so this config carries those
parameters -- the family id, the split sizes, the pool shape, the repeat
count, and the task route -- and rebuilds the identical experiment on the
other side.

**Determinism is the whole contract.** The rebuilt engine must produce
the same ``eval_config_ref`` as the harness's, or every call is refused
exactly as above. That holds because ``FamilySpec.generate_pool`` is
seeded and ``build_experiment`` is a pure function of the pool and the
split sizes. :func:`build_codex_runtime_config` derives every field from
the same :class:`~whetstone_envs.optim.run.RunSpec` the in-process engine
was built from, so the two cannot drift apart silently; the run asserts
the two hashes agree before it spawns anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dr_providers import ProviderKind
from pydantic import BaseModel, ConfigDict, PositiveFloat, StrictInt, StrictStr
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.provider import (
    fake_gold_by_prompt,
    fake_transport_factory,
    openrouter_seeded_call_config,
    openrouter_transport_factory,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.provider.policy import ProviderExecutionPolicy

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
    n_per_stratum: StrictInt
    pool_seed_start: StrictInt
    num_seeds: StrictInt
    transport: CodexRuntimeTransport
    model: StrictStr

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
        return self._reference.execution_policy

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
            n_per_stratum=self.n_per_stratum,
            seed_start=self.pool_seed_start,
        )
        prepared = family.build_experiment(
            pool,
            split_sizes=self.split_sizes,
            num_seeds=self.num_seeds,
            provider_call_config=(
                openrouter_seeded_call_config(model=self.model)
                if self.transport == "openrouter"
                else None
            ),
        )
        experiment = prepared.experiment
        if self.transport == "openrouter":
            transport_factory = openrouter_transport_factory
        else:
            transport_factory = fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    experiment,
                    render_contract=family.render_contract(),
                    ceiling_template=family.probes.ceiling_template,
                )
            )
        return self._reference.build_engine(
            store,
            experiment=experiment,
            eval_runner=family.eval_runner(),
            transport_factory=transport_factory,
        )


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
