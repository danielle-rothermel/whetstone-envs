"""The family registry the shared optimizer runner reads."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("whetstone.experiment.env")

from dataclasses import replace

from whetstone_envs.c18 import PROBES as C18_PROBES
from whetstone_envs.c19 import PROBES as C19_PROBES
from whetstone_envs.c19 import generate_pool as c19_generate_pool
from whetstone_envs.optim.c18_experiment import (
    C18_CONTRACT,
    C18_PROMPT_FIELDS,
    C18_PROTOCOL_SPLIT_SIZES,
    c18_generate_pool,
    c18_protocol_split_sizes,
)
from whetstone_envs.optim.experiment import (
    C19_CONTRACT,
    C19_MUTATION_FIELD,
    C19_NAMESPACE,
    C19_PROMPT_FIELDS,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.families import (
    KNOWN_FAMILY_IDS,
    FamilyId,
    FamilySpec,
    family_spec,
    register_family,
    registered_family_ids,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner


def _c19_like_spec(**overrides: object) -> FamilySpec:
    """A spec shaped like c19's, for testing the registry's own rules."""
    fields: dict[str, Any] = {
        "family_id": FamilyId.C18.value,
        "contract": C19_CONTRACT,
        "task_context": "A test family's task.",
        "rendering_rules": "Render it.",
        "example_execution": "Score it.",
        "probes": C19_PROBES,
        "generate_pool": c19_generate_pool,
        "build_experiment": prepare_c19_experiment,
        "eval_runner": ExactMatchEvalProcedureRunner,
        "default_n_per_stratum": 2,
        "default_pool_seed_start": 1,
        "run_id_prefix": "test",
    }
    fields.update(overrides)
    return FamilySpec(**fields)


def test_c19_is_registered_and_carries_its_own_contract() -> None:
    spec = family_spec("c19")
    assert spec.family_id == "c19"
    assert spec.namespace == C19_NAMESPACE
    assert spec.mutation_field == C19_MUTATION_FIELD
    assert spec.prompt_fields == C19_PROMPT_FIELDS
    assert "MiniGrid" in spec.task_context
    assert spec.probes is C19_PROBES
    assert spec.render_contract() == c19_render_contract()
    assert "c19" in registered_family_ids()


def test_c18_is_registered_and_carries_its_own_contract() -> None:
    spec = family_spec("c18")
    assert spec.family_id == "c18"
    assert spec.namespace == "whetstone_envs.c18"
    assert spec.contract is C18_CONTRACT
    assert spec.prompt_fields == C18_PROMPT_FIELDS
    assert spec.probes is C18_PROBES
    assert spec.generate_pool is c18_generate_pool
    assert "c18" in KNOWN_FAMILY_IDS
    assert "c18" in registered_family_ids()


def test_the_two_families_differ_in_every_identity_that_matters() -> None:
    """Nothing about c18 is c19's, and nothing is accidentally shared.

    A second family that silently reused the first's namespace, dataset
    revision, or placeholders would make the C3 evidence meaningless: two
    runs would address the same persisted identities.
    """
    c19 = family_spec("c19")
    c18 = family_spec("c18")
    assert c19.namespace != c18.namespace
    assert c19.contract.dataset_revision != c18.contract.dataset_revision
    assert c19.contract.root_base_schema != c18.contract.root_base_schema
    assert c19.contract.reward_policy_name != c18.contract.reward_policy_name
    assert set(c19.prompt_fields).isdisjoint(set(c18.prompt_fields)) or (
        c19.prompt_fields != c18.prompt_fields
    )
    assert c19.probes.naive_template != c18.probes.naive_template
    assert c19.probes.ceiling_template != c18.probes.ceiling_template
    # Both families carry exactly one optimizable template, so they share
    # the payload key. That is a whetstone-side field name, not a domain
    # concept, and the differing root schema keeps their candidates apart.
    assert c19.mutation_field == c18.mutation_field


def test_c18_protocol_split_sizes_match_the_generators_own_plan() -> None:
    """The pinned (24, 48, 48) is what C18's config actually derives.

    The literal is what the study manifest records, so it must not drift
    away from ``default_split_sizes`` unnoticed.
    """
    pool = c18_generate_pool(
        n_per_stratum=30,
        seed_start=family_spec("c18").default_pool_seed_start,
    )
    assert c18_protocol_split_sizes(pool) == C18_PROTOCOL_SPLIT_SIZES
    assert sum(C18_PROTOCOL_SPLIT_SIZES) == len(pool)


def test_unrecognised_family_reports_a_typo_not_a_wiring_gap() -> None:
    with pytest.raises(ValueError, match="unsupported family 'c17'"):
        family_spec("c17")


def test_registering_an_unknown_identifier_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown family 'c17'"):
        register_family(_c19_like_spec(family_id="c17"))


@pytest.mark.parametrize("family_id", list(KNOWN_FAMILY_IDS))
def test_registering_a_duplicate_family_is_refused(family_id: str) -> None:
    """A second registration must not silently shadow the first."""
    with pytest.raises(ValueError, match="already registered"):
        register_family(_c19_like_spec(family_id=family_id))


def test_a_family_must_declare_prompt_fields() -> None:
    empty = replace(C19_CONTRACT, prompt_fields=())
    with pytest.raises(ValueError, match="declares no prompt fields"):
        _c19_like_spec(contract=empty)


def test_a_family_default_pool_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="default_n_per_stratum"):
        _c19_like_spec(default_n_per_stratum=0)


@pytest.mark.parametrize("family_id", list(KNOWN_FAMILY_IDS))
def test_proposal_bodies_lead_with_a_body_differing_from_the_seed(
    family_id: str,
) -> None:
    """A scripted first draft must differ from the naive seed.

    A seed optimizer keeps the naive initial candidate, so a first body
    equal to it is a no-op mutation the optimizer rejects.
    """
    spec = family_spec(family_id)
    bodies = spec.proposal_bodies()
    assert bodies[0] == spec.probes.ceiling_template
    assert bodies[0] != spec.probes.naive_template
    assert spec.probes.naive_template in bodies


@pytest.mark.parametrize("family_id", list(KNOWN_FAMILY_IDS))
def test_every_proposal_body_satisfies_the_family_render_contract(
    family_id: str,
) -> None:
    spec = family_spec(family_id)
    contract = spec.render_contract()
    for body in spec.proposal_bodies():
        contract.validate_template(body)


def test_the_registry_generates_the_family_pool_it_names() -> None:
    """The registered generator is the family's own, not a copy."""
    spec = family_spec("c19")
    through_registry = spec.generate_pool(n_per_stratum=1, seed_start=11)
    direct = c19_generate_pool(n_per_stratum=1, seed_start=11)
    assert [instance.id for instance in through_registry.as_sequence()] == [
        instance.id for instance in direct.as_sequence()
    ]


def test_a_family_mints_its_own_candidates() -> None:
    """The study's anchors and representatives reach the engine through the
    registry, so how a candidate is built is family knowledge like any
    other -- not something a caller reaches around the registry for."""
    spec = family_spec("c19")
    candidate = spec.build_candidate(
        candidate_id="c19-naive", template=spec.probes.naive_template
    )
    assert candidate.candidate_id == "c19-naive"
    assert candidate.payload[spec.mutation_field] == (
        spec.probes.naive_template
    )


def test_a_candidate_that_breaks_the_render_contract_is_refused() -> None:
    spec = family_spec("c19")
    with pytest.raises(ValueError, match="field"):
        spec.build_candidate(
            candidate_id="c19-broken", template="no placeholders here"
        )
