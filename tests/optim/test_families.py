"""The family registry the shared optimizer runner reads."""

from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.c19 import PROBES as C19_PROBES
from whetstone_envs.c19 import generate_pool as c19_generate_pool
from whetstone_envs.optim.experiment import (
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


def _c19_like_spec(**overrides) -> FamilySpec:
    """A spec shaped like c19's, for testing the registry's own rules."""
    fields = {
        "family_id": FamilyId.C18.value,
        "namespace": "whetstone_envs.test",
        "mutation_field": C19_MUTATION_FIELD,
        "prompt_fields": C19_PROMPT_FIELDS,
        "task_context": "A test family's task.",
        "probes": C19_PROBES,
        "generate_pool": c19_generate_pool,
        "build_experiment": prepare_c19_experiment,
        "render_contract": c19_render_contract,
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


def test_c18_is_a_known_identifier_the_registry_admits() -> None:
    """The registry admits c18 before its experiment builder exists.

    Wave 5 registers the real spec; what this pins is that the identifier is
    already known, so registering it is a registration and not a change to
    the registry's own vocabulary.
    """
    assert "c18" in KNOWN_FAMILY_IDS
    assert FamilyId.C18.value == "c18"


def test_known_but_unregistered_family_reports_a_wiring_gap() -> None:
    with pytest.raises(ValueError, match="known but not registered"):
        family_spec("c18")


def test_unrecognised_family_reports_a_typo_not_a_wiring_gap() -> None:
    with pytest.raises(ValueError, match="unsupported family 'c17'"):
        family_spec("c17")


def test_registering_an_unknown_identifier_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown family 'c17'"):
        register_family(_c19_like_spec(family_id="c17"))


def test_registering_a_duplicate_family_is_refused() -> None:
    """A second family must not silently shadow the first."""
    with pytest.raises(ValueError, match="already registered"):
        register_family(_c19_like_spec(family_id=FamilyId.C19.value))


def test_a_family_must_declare_prompt_fields() -> None:
    with pytest.raises(ValueError, match="declares no prompt fields"):
        _c19_like_spec(prompt_fields=())


def test_a_family_default_pool_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="default_n_per_stratum"):
        _c19_like_spec(default_n_per_stratum=0)


def test_proposal_bodies_lead_with_a_body_differing_from_the_seed() -> None:
    """A scripted first draft must differ from the naive seed.

    A seed optimizer keeps the naive initial candidate, so a first body
    equal to it is a no-op mutation the optimizer rejects.
    """
    spec = family_spec("c19")
    bodies = spec.proposal_bodies()
    assert bodies[0] == spec.probes.ceiling_template
    assert bodies[0] != spec.probes.naive_template
    assert spec.probes.naive_template in bodies


def test_every_proposal_body_satisfies_the_family_render_contract() -> None:
    spec = family_spec("c19")
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
