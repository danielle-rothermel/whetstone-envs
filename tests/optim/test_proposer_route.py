"""The proposal route's reference and record must name one config.

Every optimizer hands :class:`ProviderProposerTransport` a
``ProposerConfig`` carrying an ``IdentityRef``, and the transport resolves
that reference and asserts the resolved record matches it. When a run names
a ``proposer_model`` distinct from its task model, the two sides are minted
from different configs unless the reference is derived from the proposer's
own config -- and the mismatch surfaces mid-run, inside the durable
boundary, as a ``DurableRunError``.

These tests drive the *real* transport's assertion. The fake proposer
transport never calls a resolver at all, so no fake-path test can reach it.
No network: the assertion fires at config-resolution time, before the
transport callable is ever invoked, and the stub here raises if it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_providers import ProviderKind, ReasoningEffort
from dr_store.sync import open_sqlite
from whetstone.core.identity import ImmutableJsonObject
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.proposal.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProviderProposerTransport,
)

from whetstone_envs.c19 import generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    c19_render_contract,
    prepare_c19_experiment,
    provider_call_config_ref,
)
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.gepa import build_gepa_control, gepa_prompt_services
from whetstone_envs.optim.miprov2 import build_miprov2_control
from whetstone_envs.optim.provider import (
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    widened_execution_policy,
)
from whetstone_envs.optim.run import _proposer_route
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner

if TYPE_CHECKING:
    from dr_store import ObjectStore

C19 = family_spec("c19")

#: A proposer distinct from the task model, which is the whole trigger
#: condition: the task route is pinned to a reasoning effort and the
#: proposer route is deliberately unpinned, so the two configs -- and
#: therefore their content hashes -- differ.
PROPOSER_MODEL = "openai/gpt-5.4-nano"
TASK_MODEL = "openai/gpt-5-nano"


@pytest.fixture
def prepared():
    return prepare_c19_experiment(
        generate_pool(n_per_stratum=2, seed_start=765_432),
        split_sizes=(2, 2, 0),
        num_seeds=1,
        provider_call_config=openrouter_seeded_call_config(
            model=TASK_MODEL, reasoning_effort=ReasoningEffort.LOW
        ),
    )


@pytest.fixture
def engine_and_store(tmp_path, prepared):
    runtime_config = ReferenceEvalRuntimeConfig(
        transport_api_key_env="WHETSTONE_TOY_API_KEY",
    )
    with open_sqlite(str(tmp_path / "runtime.sqlite")) as store:
        engine = runtime_config.build_engine(
            cast("ObjectStore", store),
            experiment=prepared.experiment,
            eval_runner=ExactMatchEvalProcedureRunner(),
            mutation_field=C19_MUTATION_FIELD,
            render_contract=c19_render_contract(),
            transport_factory=fake_transport_factory(
                gold_by_prompt=fake_gold_by_prompt(
                    prepared.experiment,
                    render_contract=C19.render_contract(),
                    ceiling_template=C19.probes.ceiling_template,
                )
            ),
        )
        yield engine, store


class _RouteAcceptedError(Exception):
    """Raised in place of the live call once the assertion has passed.

    The mint/resolve assertion is the last thing between ``draft`` and the
    first provider request, so reaching the transport at all *is* the
    evidence that the reference and the resolved record agreed. Raising
    here rather than returning a fake response keeps the test at the
    boundary it is about and guarantees no request is ever issued.
    """


def _refuse_call(*_args: object, **_kwargs: object):
    raise _RouteAcceptedError


def _openrouter_execution_policy():
    """The policy a live ``--transport openrouter`` run hands the proposer.

    The assertion under test does not read the policy, but building the
    transport the way the runner builds it keeps the slice honest rather
    than substituting a policy no run would ever use.
    """
    return hardened_execution_policy(
        widened_execution_policy(
            ReferenceEvalRuntimeConfig(
                transport_api_key_env="OPENROUTER_API_KEY",
                provider_kind=ProviderKind.OPENROUTER,
            ).execution_policy,
            concurrency=1,
        )
    )


def _drive_assertion(*, prompt_model: ProposerConfig, experiment) -> None:
    """Resolve ``prompt_model``'s route through the production transport.

    This is the smallest slice of ``run_optimizer`` that reaches the real
    mint/resolve assertion: the same ``ProviderProposerTransport`` the
    runner builds, resolving through the same route object, against the
    reference the optimizer's injected defaults actually carry.
    """
    route = _proposer_route(
        experiment=experiment, proposer_model=PROPOSER_MODEL
    )
    transport = ProviderProposerTransport(
        resolve_provider_call_config=route.resolver(),
        transport=_refuse_call,
        execution_policy=_openrouter_execution_policy(),
    )
    request = ProposalRequest(
        proposal_mode="instruction",
        request_ordinal=0,
        proposal_authority_identity_hash="0" * 64,
        mutation_field=C19_MUTATION_FIELD,
        base_candidate=candidate_reference(experiment.initial_candidate),
        context=ImmutableJsonObject({"proposal_prompt": "write a template"}),
    )
    # Reaching the refusing transport means the reference and the resolved
    # record agreed; a mismatch raises ``ValueError`` strictly earlier.
    with pytest.raises(_RouteAcceptedError):
        transport.draft(prompt_model, request, 1)


def _halves(engine) -> tuple[tuple[str, ...], tuple[str, ...]]:
    task_hashes = tuple(engine.sampling.task_hashes)
    midpoint = len(task_hashes) // 2
    return task_hashes[:midpoint], task_hashes[midpoint:]


def test_copro_proposer_route_reference_matches_the_resolved_record(
    prepared,
) -> None:
    """FAILS-BEFORE: COPRO minted the ref from the task config."""
    experiment = prepared.experiment
    route = _proposer_route(
        experiment=experiment, proposer_model=PROPOSER_MODEL
    )
    # Exactly the ``CoproInjectedDefaults.prompt_model`` the runner builds.
    prompt_model = ProposerConfig(
        provider_call_config=route.ref, temperature=None
    )
    _drive_assertion(prompt_model=prompt_model, experiment=experiment)


def test_gepa_reflection_route_reference_matches_the_resolved_record(
    engine_and_store, prepared
) -> None:
    """FAILS-BEFORE: GEPA minted the ref from the task config."""
    engine, _store = engine_and_store
    experiment = prepared.experiment
    route = _proposer_route(
        experiment=experiment, proposer_model=PROPOSER_MODEL
    )
    trainset, valset = _halves(engine)
    control = build_gepa_control(
        engine=engine,
        experiment=experiment,
        family=C19,
        prompt_services=gepa_prompt_services(C19),
        policy_identity_hash="0" * 64,
        proposer_config_ref=route.ref,
        trainset_task_hashes=trainset,
        valset_task_hashes=valset,
    )
    _drive_assertion(
        prompt_model=control.reflection_model, experiment=experiment
    )


def test_miprov2_prompt_route_reference_matches_the_resolved_record(
    engine_and_store, prepared
) -> None:
    """FAILS-BEFORE: MIPROv2 minted the ref from the task config."""
    engine, _store = engine_and_store
    experiment = prepared.experiment
    route = _proposer_route(
        experiment=experiment, proposer_model=PROPOSER_MODEL
    )
    trainset, valset = _halves(engine)
    control = build_miprov2_control(
        engine=engine,
        experiment=experiment,
        family=C19,
        proposer_config_ref=route.ref,
        trainset_task_hashes=trainset,
        valset_task_hashes=valset,
    )
    _drive_assertion(prompt_model=control.prompt_model, experiment=experiment)


def test_the_task_route_reference_would_not_have_matched(prepared) -> None:
    """The bug this fixes, stated as the mismatch it actually was.

    Minting from the experiment while resolving to a distinct proposer is
    precisely what the three optimizers did, and it is what the transport
    refuses. Pinning the refusal keeps the fix from being undone by a
    later edit that reaches for ``provider_call_config_ref`` again.
    """
    experiment = prepared.experiment
    mismatched = ProposerConfig(
        provider_call_config=provider_call_config_ref(experiment),
        temperature=None,
    )
    with pytest.raises(
        ValueError, match="record reference does not match IdentityRef"
    ):
        _drive_assertion(prompt_model=mismatched, experiment=experiment)


def test_an_unnamed_proposer_keeps_the_experiment_route_unchanged(
    prepared,
) -> None:
    """``proposer_model=None`` must stay byte-identical to the old path.

    A single-model run records the experiment's own reference. Changing
    that hash would rewrite the recorded identity of every run that never
    named a proposer, so it is pinned rather than merely derived.
    """
    experiment = prepared.experiment
    route = _proposer_route(experiment=experiment, proposer_model=None)
    expected = provider_call_config_ref(experiment)
    assert route.ref == expected
    assert route.resolver()(None) is (
        experiment.rollout_graph.provider_call_config
    )


def test_the_proposer_route_carries_no_reasoning_controls(prepared) -> None:
    """The proposer stays unpinned even when the task route is pinned.

    Reasoning effort is a property of the model a study *measures*, so the
    proposer must not inherit the task route's pin. Deriving the reference
    from the proposer config is what makes that difference representable
    at all, which is why it is asserted alongside the reference.
    """
    experiment = prepared.experiment
    route = _proposer_route(
        experiment=experiment, proposer_model=PROPOSER_MODEL
    )
    assert route.config.controls.reasoning is None
    task_config = experiment.rollout_graph.provider_call_config
    assert task_config.controls.reasoning is ReasoningEffort.LOW
    assert route.ref != provider_call_config_ref(experiment)
