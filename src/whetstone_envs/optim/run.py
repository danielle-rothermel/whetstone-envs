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
    prepare_gepa_run,
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
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
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
from whetstone_envs.optim.gepa import build_c19_gepa_adapter
from whetstone_envs.optim.provider import (
    bind_openrouter_transport,
    c19_fake_transport_factory,
    openrouter_seeded_call_config,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore
    from whetstone.experiment.env import Experiment

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm" / "data" / "runs" / ("whetstone-envs")
)
# Seed COPRO asks for one draft and keeps the naive initial candidate. The
# first body must differ from that seed or COPRO rejects a no-op mutation.
C19_PROPOSAL_BODIES = (
    PROBES.ceiling_template,
    PROBES.naive_template,
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


def _git_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _repository_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        root = _git_root(start)
        if root is not None and root not in seen:
            seen.add(root)
            roots.append(root)
    return tuple(roots)


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


def _resolve_run_layout(spec: C19RunSpec) -> tuple[str, Path]:
    resolved_run_id = spec.run_id or (
        f"c19-{spec.optimizer}-{uuid4().hex[:8]}"
    )
    resolved_output = (
        spec.output_dir or default_output_dir(resolved_run_id)
    ).resolve()
    repo_roots = _repository_roots()
    if any(resolved_output.is_relative_to(root) for root in repo_roots):
        raise ValueError("run artifacts must not be written inside the repo")
    resolved_output.mkdir(parents=True, exist_ok=True)
    return resolved_run_id, resolved_output


def _ceiling_gold_by_prompt(experiment: Experiment) -> dict[str, str]:
    contract = c19_render_contract()
    gold_by_prompt: dict[str, str] = {}
    for task in experiment.eval_configs.internal.tasks:
        gold = getattr(task, "gold", None)
        inputs = getattr(task, "prompt_inputs", None)
        if not isinstance(gold, str) or not isinstance(inputs, dict):
            raise TypeError("internal task must expose prompt_inputs and gold")
        gold_by_prompt[contract.render(PROBES.ceiling_template, inputs)] = gold
    return gold_by_prompt


def run_c19_optimizer(spec: C19RunSpec) -> Path:
    """Run COPRO or GEPA on a small C19 split and write artifacts off-repo."""
    if spec.optimizer not in {"copro", "gepa"}:
        raise ValueError(f"unsupported optimizer {spec.optimizer!r}")
    if spec.transport not in {"fake", "openrouter"}:
        raise ValueError(f"unsupported transport {spec.transport!r}")
    resolved_run_id, resolved_output = _resolve_run_layout(spec)
    sqlite_path = resolved_output / "runtime.sqlite"
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(model=spec.model)
        api_key_env = "OPENROUTER_API_KEY"
    pool = generate_pool(n_per_stratum=2, seed_start=765_432)
    experiment = build_c19_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=1,
        provider_call_config=provider,
    )
    if spec.transport == "openrouter":
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
            provider_kind=ProviderKind.OPENROUTER,
        )
    else:
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
        )
    live_transport = None
    if spec.transport == "openrouter":
        live_transport, transport_factory = bind_openrouter_transport(
            runtime_config.execution_policy
        )
    else:
        transport_factory = c19_fake_transport_factory(
            gold_by_prompt=_ceiling_gold_by_prompt(experiment)
        )
    with open_sqlite(str(sqlite_path)) as store:
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=transport_factory,
        )
        prompt_adapter = PlainPromptAdapter()
        proposer_transport = None
        if live_transport is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=_provider_config_resolver(
                    experiment
                ),
                transport=live_transport,
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
        extra_adapters = None
        if spec.optimizer == "gepa":
            gepa_adapter = build_c19_gepa_adapter(
                store=cast("ObjectStore", store),
                engine=engine,
                experiment=experiment,
                run_id=resolved_run_id,
                proposer_transport=proposer_transport,
            )
            extra_adapters = {GEPA_ADAPTER_KEY: gepa_adapter}
        runtime = register_runtime(
            store=store,
            engine=engine,
            copro_control=copro_control,
            extra_adapters=extra_adapters,
            proposal_bodies=C19_PROPOSAL_BODIES,
            proposer_transport=proposer_transport,
        )
        if spec.optimizer == "gepa":
            launch = prepare_gepa_run(
                runtime,
                run_id=resolved_run_id,
                control=gepa_adapter.control,
                experiment=experiment,
                render_contract=c19_render_contract(),
                mutation_field=C19_MUTATION_FIELD,
            )
        else:
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
