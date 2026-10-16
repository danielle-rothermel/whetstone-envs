from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_providers import ProviderCallConfig
from whetstone.core.identity import (
    IdentityRef,
)
from whetstone.optim.proposal.proposer import (
    ProposerTransport,
    ProviderProposerTransport,
)
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.optim.experiment import (
    provider_call_config_ref,
    provider_config_ref,
)
from whetstone_envs.optim.provider import (
    openrouter_seeded_call_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone.experiment.env import Experiment
    from whetstone.provider.policy import ProviderExecutionPolicy

    from whetstone_envs.optim.provider import RetryingTransport


@dataclass(frozen=True, slots=True)
class ProposerRoute:
    config: ProviderCallConfig
    ref: IdentityRef

    def resolver(self) -> Callable[[IdentityRef], ProviderCallConfig]:
        def resolve(_ref: IdentityRef) -> ProviderCallConfig:
            return self.config

        return resolve


def proposer_route(
    *,
    experiment: Experiment,
    proposer_model: str | None,
) -> ProposerRoute:
    if proposer_model is None:
        config = experiment.rollout_graph.provider_call_config
        return ProposerRoute(
            config=config, ref=provider_call_config_ref(experiment)
        )
    config = openrouter_seeded_call_config(model=proposer_model)
    return ProposerRoute(config=config, ref=provider_config_ref(config))


@dataclass(frozen=True, slots=True)
class ProposalRuntime:
    route: ProposerRoute
    prompt_adapter: PlainPromptAdapter
    transport: ProposerTransport | None


def prepare_proposer(
    *,
    experiment: Experiment,
    model: str | None,
    live_transport: RetryingTransport | None,
    execution_policy: ProviderExecutionPolicy,
) -> ProposalRuntime:
    prompt_adapter = PlainPromptAdapter()
    route = proposer_route(experiment=experiment, proposer_model=model)
    transport = (
        None
        if live_transport is None
        else ProviderProposerTransport(
            resolve_provider_call_config=route.resolver(),
            transport=live_transport,
            execution_policy=execution_policy,
            prompt_adapter=prompt_adapter,
        )
    )
    return ProposalRuntime(
        route=route, prompt_adapter=prompt_adapter, transport=transport
    )
