from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
    candidate_reference,
)
from whetstone.optim.contracts import OptimRun, OutputContract, StepMode
from whetstone.optim.proposal.mutation import DiffCheckError, diff_check
from whetstone.optim.proposal.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)

from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_ROOT_BASE_SCHEMA,
    c19_render_contract,
)
from whetstone_envs.optim.nulls import (
    NULL_PERTURBATION_RATE,
    NullIdentityTransport,
    NullRandomTransport,
    perturb_template,
)

_HASH = "a" * 64

#: A seed long enough that the perturber has real choices to make. The c19
#: naive template is short, so a deliberately wordier seed exercises the
#: swap/delete/duplicate mix rather than the one-operation corner.
SEED_TEMPLATE = (
    "You are given the MiniGrid observation {grid} together with the agent "
    "command {command}. Read both carefully and then answer the following "
    "question {question} with the exact fact and nothing else."
)
SHORT_TEMPLATE = "Given {grid} and {command}, answer {question}."


def _root_ref():
    return typed_ref_for_record(C19_ROOT_BASE_SCHEMA, {"kind": "root"})


def _seed_candidate(template: str = SEED_TEMPLATE) -> Candidate:
    return Candidate(
        candidate_id="null-seed",
        base_ref=_root_ref(),
        payload={C19_MUTATION_FIELD: template},
    )


def _request(template: str = SEED_TEMPLATE) -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        proposal_authority_identity_hash=_HASH,
        mutation_field=C19_MUTATION_FIELD,
        base_candidate=candidate_reference(_seed_candidate(template)),
    )


def _config() -> ProposerConfig:
    root = _root_ref()
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=root,
            record_hash=root.content_hash,
        ),
        temperature=None,
    )


def _null_random(seed: int = 5000) -> NullRandomTransport:
    return NullRandomTransport(
        seed=seed,
        render_contract=c19_render_contract(),
        execution_policy_hash=_HASH,
        prompt_adapter_identity_hash=_HASH,
    )


def _null_identity() -> NullIdentityTransport:
    return NullIdentityTransport(
        execution_policy_hash=_HASH,
        prompt_adapter_identity_hash=_HASH,
    )


def test_both_nulls_satisfy_the_proposer_transport_protocol() -> None:
    """Each null is usable wherever the runner binds a proposer."""
    random_transport: ProposerTransport = _null_random()
    identity_transport: ProposerTransport = _null_identity()
    for transport in (random_transport, identity_transport):
        assert len(transport.execution_policy_hash) == 64
        assert len(transport.prompt_adapter_identity_hash) == 64
        assert len(transport.durability_identity_hash) == 64


def test_nulls_have_distinct_durability_identities() -> None:
    """A null is never mistaken for the other null or for a real proposer."""
    assert (
        _null_random().durability_identity_hash
        != _null_identity().durability_identity_hash
    )


# --- null-A: determinism -------------------------------------------------


def test_same_seed_yields_identical_perturbation() -> None:
    contract = c19_render_contract()
    first = perturb_template(SEED_TEMPLATE, seed=17, render_contract=contract)
    second = perturb_template(SEED_TEMPLATE, seed=17, render_contract=contract)
    assert first == second


def test_different_seeds_yield_different_perturbations() -> None:
    """The perturber is seeded, not constant."""
    contract = c19_render_contract()
    drawn = {
        perturb_template(SEED_TEMPLATE, seed=seed, render_contract=contract)
        for seed in range(12)
    }
    assert len(drawn) > 1


def test_null_random_run_replays_exactly() -> None:
    """Two transports on one seed draft byte-identical candidates."""
    request, config = _request(), _config()
    first = _null_random().draft(config, request, 4)
    second = _null_random().draft(config, request, 4)
    assert [d.template for d in first] == [d.template for d in second]


def test_null_random_seeds_are_independent() -> None:
    """Two run seeds are two different noise draws, not one repeated."""
    request, config = _request(), _config()
    first = _null_random(seed=5000).draft(config, request, 4)
    second = _null_random(seed=5001).draft(config, request, 4)
    assert [d.template for d in first] != [d.template for d in second]


# --- null-A: placeholder preservation ------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_perturbation_preserves_every_render_contract_placeholder(
    seed: int,
) -> None:
    """The study's load-bearing assertion: no draw drops a placeholder.

    A perturbation that lost ``{grid}`` would not be a weaker candidate; it
    would be a rejected one, and null-A would stop being a control for
    selection.
    """
    contract = c19_render_contract()
    result = perturb_template(
        SEED_TEMPLATE, seed=seed, render_contract=contract
    )
    # Raises if a required field is missing or an unavailable one appeared.
    contract.validate_template(result)
    for field in ("{grid}", "{command}", "{question}"):
        assert field in result


@pytest.mark.parametrize("seed", range(20))
def test_short_template_perturbation_preserves_placeholders(
    seed: int,
) -> None:
    """A template that is almost entirely placeholders still survives."""
    contract = c19_render_contract()
    result = perturb_template(
        SHORT_TEMPLATE, seed=seed, render_contract=contract
    )
    contract.validate_template(result)


def test_perturbation_never_duplicates_a_placeholder_token() -> None:
    """Duplication draws only from non-placeholder tokens.

    A duplicated ``{grid}`` still validates, so only a count check catches
    it; the contract requires each field at least once, not exactly once.
    """
    contract = c19_render_contract()
    for seed in range(40):
        result = perturb_template(
            SEED_TEMPLATE, seed=seed, render_contract=contract
        )
        for field in ("{grid}", "{command}", "{question}"):
            assert result.count(field) == 1


def test_drafts_are_valid_candidates_under_the_render_contract() -> None:
    """Every null-A draft survives the mutation path COPRO drives."""
    contract = c19_render_contract()
    drafts = _null_random().draft(_config(), _request(), 4)
    for draft in drafts:
        assert not draft.failed
        contract.validate_template(draft.template)


def test_all_placeholders_required_by_the_contract_are_retained() -> None:
    """Guards the contract itself, not just this seed's spelling."""
    contract = c19_render_contract()
    assert set(contract.required_fields) == {"grid", "command", "question"}
    result = perturb_template(SEED_TEMPLATE, seed=3, render_contract=contract)
    assert set(contract.placeholder_fields(result)) == set(
        contract.required_fields
    )


def test_perturber_falls_back_to_identity_when_no_draw_validates() -> None:
    """A bounded, recorded no-op rather than an unbounded retry loop.

    With a template of nothing but required placeholders, every eligible
    token set is empty, so no perturbation exists and the seed comes back.
    """
    contract = c19_render_contract()
    template = "{grid}{command}{question}"
    result = perturb_template(template, seed=1, render_contract=contract)
    assert result == template


def test_identity_fallback_is_recorded_on_the_draft() -> None:
    """A null-A run's evidence shows when it degenerated toward null-B."""
    transport = NullRandomTransport(
        seed=5000,
        render_contract=c19_render_contract(),
        execution_policy_hash=_HASH,
        prompt_adapter_identity_hash=_HASH,
    )
    request = _request("{grid}{command}{question}")
    drafts = transport.draft(_config(), request, 2)
    assert all(
        draft.response_evidence.to_json()["identity_fallback"]
        for draft in drafts
    )


def test_perturbation_respects_a_foreign_render_contract() -> None:
    """Placeholder handling reads the contract; it hardcodes no c19 field."""
    contract = TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=("question", "query"),
        required_fields=("question", "query"),
    )
    template = (
        "Answer the question {question} using only the retrieved "
        "passage {query} and nothing else at all."
    )
    for seed in range(20):
        result = perturb_template(
            template, seed=seed, render_contract=contract
        )
        contract.validate_template(result)


# --- null-A vs null-B: the differing/identical contrast -------------------


def test_null_random_candidates_differ_from_the_seed() -> None:
    """null-A perturbs: its candidates are not the seed."""
    request = _request()
    drafts = _null_random().draft(_config(), request, 4)
    for draft in drafts:
        assert draft.template != SEED_TEMPLATE


def test_null_random_candidates_are_pairwise_distinct() -> None:
    """COPRO requires distinct proposals within one round.

    A short template has a small perturbation space, so two slots can draw
    the same result unless distinctness is enforced.
    """
    for template in (SEED_TEMPLATE, SHORT_TEMPLATE):
        drafts = _null_random().draft(_config(), _request(template), 4)
        templates = [draft.template for draft in drafts]
        assert len(set(templates)) == len(templates)


def test_null_identity_proposes_no_mutation() -> None:
    """null-B returns the seed: it proposes nothing, and says so."""
    drafts = _null_identity().draft(_config(), _request(), 3)
    assert len(drafts) == 3
    for draft in drafts:
        assert draft.failed
        assert draft.template == ""
        assert (
            draft.request_evidence.to_json()["seed_template"] == SEED_TEMPLATE
        )


def test_null_identity_cannot_be_a_successful_no_op_draft() -> None:
    """Why null-B reports an unfilled slot rather than repeating the seed.

    ``diff_check`` rejects a proposal whose mutation field equals its base,
    so a "successful" identity draft is not representable: a no-op is not a
    proposal, and the honest transport-level statement is that none was made.
    This pins the upstream rule the transport's shape depends on.
    """
    base = _seed_candidate()
    identical = Candidate(
        candidate_id="identity",
        base_ref=candidate_reference(base).record_ref,
        payload={C19_MUTATION_FIELD: SEED_TEMPLATE},
    )
    with pytest.raises(DiffCheckError, match="must differ from its base"):
        diff_check(base=base, proposed=identical, run=_null_b_run())


def _null_b_run():
    """The minimal run identity ``diff_check`` reads."""
    root = _root_ref()
    return OptimRun(
        run_id="null-b-check",
        optimizer_config=IdentityRef(
            record_ref=root, record_hash=root.content_hash
        ),
        adapter_key="copro",
        mode=StepMode.PURE,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=c19_render_contract(),
        initial_candidate_ref=candidate_reference(_seed_candidate()),
        mutation_field=C19_MUTATION_FIELD,
    )


def test_null_a_differs_and_null_b_does_not() -> None:
    """The two controls differ in exactly the intended way."""
    request, config = _request(), _config()
    perturbed = _null_random().draft(config, request, 3)
    identical = _null_identity().draft(config, request, 3)
    assert all(draft.template != SEED_TEMPLATE for draft in perturbed)
    assert all(draft.failed for draft in identical)
    assert all(
        draft.request_evidence.to_json()["seed_template"] == SEED_TEMPLATE
        for draft in identical
    )


# --- spend and evidence ---------------------------------------------------


def test_nulls_report_no_proposer_spend() -> None:
    """A null makes no provider call, so run cost records nothing."""
    request, config = _request(), _config()
    for drafts in (
        _null_random().draft(config, request, 3),
        _null_identity().draft(config, request, 3),
    ):
        for draft in drafts:
            assert draft.call_usage() is None
            assert draft.cost is None
            assert draft.usage.to_json()["proposer_calls"] == 0


def test_nulls_fill_every_requested_slot() -> None:
    """Same budget and the same slot cardinality as a real proposer."""
    request, config = _request(), _config()
    for count in (0, 1, 2, 5):
        assert len(_null_random().draft(config, request, count)) == count
        assert len(_null_identity().draft(config, request, count)) == count


def test_nulls_record_the_calls_they_were_asked_for() -> None:
    request, config = _request(), _config()
    transport = _null_random()
    transport.draft(config, request, 2)
    transport.draft(config, request, 3)
    assert [count for _, _, count in transport.calls] == [2, 3]


@pytest.mark.parametrize("count", [-1, -5])
def test_negative_draft_count_is_rejected(count: int) -> None:
    request, config = _request(), _config()
    with pytest.raises(ValueError, match="nonnegative"):
        _null_random().draft(config, request, count)
    with pytest.raises(ValueError, match="nonnegative"):
        _null_identity().draft(config, request, count)


def test_null_random_perturbs_the_requested_base_not_the_run_seed() -> None:
    """null-A tracks the optimizer's search rather than always resetting.

    COPRO's history rounds mutate the incumbent, so a null that always
    perturbed the run seed would be a control for a different search shape.
    """
    other = (
        "A completely different incumbent template mentioning {grid} and "
        "{command} before it finally asks {question} at the very end."
    )
    drafts = _null_random().draft(_config(), _request(other), 2)
    # A perturbation reorders, drops, or repeats whole tokens, so compare
    # token multisets rather than substrings: the draft must be a near-copy
    # of the requested base, not of the module-level seed.
    incumbent_tokens = set(other.split())
    seed_only_tokens = set(SEED_TEMPLATE.split()) - incumbent_tokens
    for draft in drafts:
        drafted = set(draft.template.split())
        assert len(drafted & incumbent_tokens) > len(incumbent_tokens) // 2
        assert not (drafted & seed_only_tokens)


def test_perturbation_rate_is_the_protocol_value() -> None:
    assert NULL_PERTURBATION_RATE == 0.05
