from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import (
    compute_identity_hash,
)
from whetstone.core.roles import EvalRole
from whetstone.optim.copro.adapter import CoproAdapter
from whetstone.optim.copro.control import (
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optim.copro.proposal_contract import CoproProposalContractRecord
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)

if TYPE_CHECKING:
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.proposal.proposer import ProposerTransport
    from whetstone.provider.language_model import PlainPromptAdapter

    from whetstone_envs.optim.families import (
        FamilySpec,
    )
    from whetstone_envs.optim.proposals import ProposalRuntime
    from whetstone_envs.optim.spec import (
        CoproSettings,
    )


from whetstone_envs.optim._bindings import CoproBinding

COPRO_EXECUTOR_SCHEMA_SUFFIX = "copro_proposal_executor"


def _proposal_contract(family: FamilySpec) -> CoproProposalContractRecord:
    placeholders = ", ".join(f"{{{field}}}" for field in family.prompt_fields)
    return CoproProposalContractRecord(
        target_name=f"{family.family_id}_prompt_template",
        task_context=family.task_context,
        output_rule=(
            f"Return one non-empty prompt template that uses {placeholders}."
        ),
    )


def _copro_adapter(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    family: FamilySpec,
    extra_proposal_bodies: tuple[str, ...] = (),
) -> CoproAdapter:
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=family.proposal_bodies(extra_proposal_bodies),
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
    )
    return CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema=(f"{family.namespace}.{COPRO_EXECUTOR_SCHEMA_SUFFIX}"),
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )


def bind_copro(
    *,
    settings: CoproSettings,
    engine: EvalEngine,
    experiment: Experiment,
    family: FamilySpec,
    proposer: ProposalRuntime,
) -> CoproBinding:
    defaults = CoproInjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=proposer.route.ref, temperature=None
        ),
        proposal_contract=_proposal_contract(family),
        eval_config_ref=engine.eval_config_ref,
        eval_role=EvalRole.INTERNAL,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
        provider_execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter=proposer.prompt_adapter,
    )
    control = configure_copro(
        breadth=settings.breadth,
        depth=settings.depth,
        track_stats=False,
        defaults=defaults,
    )
    return CoproBinding(
        control=control,
        adapter=_copro_adapter(
            engine=engine,
            control=control,
            prompt_adapter=proposer.prompt_adapter,
            proposer_transport=proposer.transport,
            family=family,
            extra_proposal_bodies=settings.extra_proposal_bodies,
        ),
    )
