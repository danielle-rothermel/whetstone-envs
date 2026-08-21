from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA, ProviderKind
from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    copro_run_request,
    prepare_copro_run,
    register_runtime,
)
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.optim.copro.control import (
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optim.copro.proposal_contract import CoproProposalContractRecord
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    ProviderProposerTransport,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.c19 import PROBES, generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    build_c19_experiment,
    c19_render_contract,
)
from whetstone_envs.optim.provider import (
    openrouter_seeded_call_config,
    openrouter_transport_factory,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.experiment.env import Experiment

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm" / "data" / "runs" / ("whetstone-envs")
)
C19_PROPOSAL_BODIES = (
    PROBES.naive_template,
    PROBES.ceiling_template,
)


@dataclass(frozen=True, slots=True)
class C19RunSpec:
    optimizer: str
    transport: str
    split_sizes: tuple[int, int, int] = (2, 2, 0)
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"


def default_output_dir(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / run_id


def _provider_config_resolver(experiment: Experiment):
    provider_config = experiment.rollout_graph.provider_call_config

    def resolve(_ref: object):
        return provider_config

    return resolve


def _provider_call_config_ref(experiment: Experiment) -> IdentityRef:
    payload = experiment.rollout_graph.provider_call_config.model_dump(
        mode="json"
    )
    record_ref = typed_ref_for_record(PROVIDER_CALL_CONFIG_SCHEMA, payload)
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def _c19_proposal_contract() -> CoproProposalContractRecord:
    return CoproProposalContractRecord(
        target_name="c19_prompt_template",
        task_context="Predict the MiniGrid fact asked by the question.",
        output_rule=(
            "Return one non-empty prompt template that uses {grid}, "
            "{command}, and {question}."
        ),
    )


def run_c19_optimizer(spec: C19RunSpec) -> Path:
    """Run COPRO on a small C19 split and write artifacts off-repo."""
    if spec.optimizer != "copro":
        raise ValueError(f"unsupported optimizer {spec.optimizer!r}")
    if spec.transport not in {"fake", "openrouter"}:
        raise ValueError(f"unsupported transport {spec.transport!r}")
    resolved_run_id = spec.run_id or (
        f"c19-{spec.optimizer}-{uuid4().hex[:8]}"
    )
    resolved_output = (
        spec.output_dir or default_output_dir(resolved_run_id)
    ).resolve()
    if resolved_output.is_relative_to(Path.cwd().resolve()):
        raise ValueError("run artifacts must not be written inside the repo")
    resolved_output.mkdir(parents=True, exist_ok=True)
    sqlite_path = resolved_output / "runtime.sqlite"
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    live_factory = None
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(model=spec.model)
        api_key_env = "OPENROUTER_API_KEY"
        live_factory = openrouter_transport_factory
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    experiment = build_c19_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=1,
        provider_call_config=provider,
    )
    runtime_kwargs = {"transport_api_key_env": api_key_env}
    if spec.transport == "openrouter":
        runtime_kwargs["provider_kind"] = ProviderKind.OPENROUTER
    runtime_config = ReferenceEvalRuntimeConfig(**runtime_kwargs)
    with open_sqlite(str(sqlite_path)) as store:
        engine_kwargs = {}
        if live_factory is not None:
            engine_kwargs["transport_factory"] = live_factory
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            **engine_kwargs,
        )
        prompt_adapter = PlainPromptAdapter()
        proposer_transport = None
        if live_factory is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=_provider_config_resolver(
                    experiment
                ),
                transport=live_factory(runtime_config.execution_policy),
                execution_policy=runtime_config.execution_policy,
                prompt_adapter=prompt_adapter,
            )
        defaults = CoproInjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=_provider_call_config_ref(experiment),
                temperature=None,
            ),
            proposal_contract=_c19_proposal_contract(),
            eval_config_ref=engine.eval_config_ref,
            eval_role=EvalRole.INTERNAL,
            provider_execution_policy_ref=(
                engine.provider_execution_policy_ref
            ),
            expected_reward_policy_hash=(
                experiment.reward_policy.identity_hash()
            ),
            provider_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            prompt_adapter=prompt_adapter,
        )
        copro_control = configure_copro(
            breadth=2,
            depth=1,
            track_stats=False,
            defaults=defaults,
        )
        runtime = register_runtime(
            store=store,
            engine=engine,
            copro_control=copro_control,
            proposal_bodies=C19_PROPOSAL_BODIES,
            proposer_transport=proposer_transport,
        )
        launch = prepare_copro_run(
            runtime,
            run_id=resolved_run_id,
            control=copro_control,
            experiment=experiment,
            render_contract=c19_render_contract(),
            mutation_field=C19_MUTATION_FIELD,
        )
        request = copro_run_request(
            launch,
            controller_identity_hash=runtime.controller.runtime_hash,
        )
        result_ref = runtime.controller.drive(request)
        if result_ref.schema_name != OPTIM_RESULT_SCHEMA:
            raise ValueError("optimizer run did not produce an OptimResult")
        result = OptimResult.model_validate(
            runtime.store.get(result_ref.reference)
        )
        (resolved_output / "result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return resolved_output


__all__ = [
    "C19_PROPOSAL_BODIES",
    "DEFAULT_OUTPUT_ROOT",
    "C19RunSpec",
    "default_output_dir",
    "run_c19_optimizer",
]
