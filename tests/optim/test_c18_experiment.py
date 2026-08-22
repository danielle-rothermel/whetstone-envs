"""C18's optimizer adapter: contract, pool, splits, and scoring."""

from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.experiment.sampling import HELD_OUT, INTERNAL_EVAL, OFFICIAL

from whetstone_envs.c18 import PROBES
from whetstone_envs.c18.config import DEFAULT_CONFIG
from whetstone_envs.optim.c18_experiment import (
    C18_CONTRACT,
    C18_DEFAULT_N_PER_STRATUM,
    C18_DEFAULT_POOL_SEED_START,
    C18_MUTATION_FIELD,
    C18_NAMESPACE,
    C18_PROMPT_FIELDS,
    C18VerdictEvalProcedureRunner,
    c18_candidate,
    c18_generate_pool,
    c18_render_contract,
    prepare_c18_experiment,
)
from whetstone_envs.optim.rows import task_rows_from_instances

#: One instance per stratum: four instances, enough to fill a 2/1/1 split
#: while keeping PrOntoQA generation to about a second.
SMOKE_N_PER_STRATUM = 1


def _smoke_pool():
    return c18_generate_pool(
        n_per_stratum=SMOKE_N_PER_STRATUM,
        seed_start=C18_DEFAULT_POOL_SEED_START,
    )


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_render_contract_requires_exactly_question_and_query() -> None:
    contract = c18_render_contract()
    assert contract.available_fields == C18_PROMPT_FIELDS
    assert contract.required_fields == C18_PROMPT_FIELDS


@pytest.mark.parametrize(
    "template", [PROBES.naive_template, PROBES.ceiling_template]
)
def test_both_probes_satisfy_the_contract(template: str) -> None:
    """The contract is read off the probes, so it must admit both."""
    observed = c18_render_contract().validate_template(template)
    assert set(observed) == set(C18_PROMPT_FIELDS)


def test_a_candidate_dropping_a_placeholder_is_refused() -> None:
    with pytest.raises(ValueError, match="query"):
        c18_candidate(candidate_id="c18-bad", template="{question} only")


def test_candidate_payload_carries_the_template_under_the_field() -> None:
    candidate = c18_candidate(
        candidate_id="c18-probe", template=PROBES.naive_template
    )
    assert candidate.payload[C18_MUTATION_FIELD] == PROBES.naive_template


def test_the_contract_names_c18s_own_persisted_identities() -> None:
    """C18 must not address C19's records, or the two runs collide."""
    assert C18_CONTRACT.namespace == C18_NAMESPACE
    assert C18_CONTRACT.dataset_revision == "c18/v1"
    assert C18_CONTRACT.root_base_schema.startswith(C18_NAMESPACE)
    assert C18_CONTRACT.reward_policy_name == "c18-exact-match"


# --------------------------------------------------------------------------
# Pool generation
# --------------------------------------------------------------------------


def test_the_pool_adapter_matches_the_registry_calling_convention() -> None:
    """The registry calls with keyword ``n_per_stratum`` and ``seed_start``.

    C18's own generator takes its seed start from a config instead, so the
    adapter exists to bridge that; this pins that it bridges rather than
    silently ignoring the requested seed.
    """
    pool = _smoke_pool()
    assert len(pool) == SMOKE_N_PER_STRATUM * len(DEFAULT_CONFIG.strata)
    seeds = {instance.seed for instance in pool.as_sequence()}
    assert min(seeds) == C18_DEFAULT_POOL_SEED_START


def test_a_different_seed_start_generates_a_different_pool() -> None:
    """A requested seed must actually reach the generator."""
    default = _smoke_pool()
    shifted = c18_generate_pool(
        n_per_stratum=SMOKE_N_PER_STRATUM,
        seed_start=C18_DEFAULT_POOL_SEED_START + 100,
    )
    assert {i.id for i in default.as_sequence()} != {
        i.id for i in shifted.as_sequence()
    }


def test_pool_generation_is_deterministic() -> None:
    first = _smoke_pool()
    second = _smoke_pool()
    assert [i.id for i in first.as_sequence()] == [
        i.id for i in second.as_sequence()
    ]
    assert [i.gold for i in first.as_sequence()] == [
        i.gold for i in second.as_sequence()
    ]


def test_an_inadmissible_seed_start_is_refused_by_the_config() -> None:
    """PrOntoQA reserves low seeds; the config, not the adapter, rules."""
    with pytest.raises(ValueError, match="seed_start must be above"):
        c18_generate_pool(n_per_stratum=1, seed_start=1)


def test_family_defaults_come_from_c18s_own_configuration() -> None:
    assert DEFAULT_CONFIG.n_per_stratum == C18_DEFAULT_N_PER_STRATUM
    assert DEFAULT_CONFIG.seed_start == C18_DEFAULT_POOL_SEED_START


# --------------------------------------------------------------------------
# Experiment preparation
# --------------------------------------------------------------------------


def test_prepare_maps_the_split_to_eval_rows_under_their_roles() -> None:
    pool = _smoke_pool()
    prepared = prepare_c18_experiment(pool, split_sizes=(2, 1, 1))
    split = pool.split(2, 1, 1)
    assert prepared.split == split
    configs = prepared.experiment.eval_configs
    assert {row.task_id for row in configs.internal.tasks} == {
        instance.id for instance in split.internal_eval
    }
    assert {row.task_id for row in configs.official.tasks} == {
        instance.id for instance in split.official
    }
    assert configs.internal.task_set.dataset_revision == "c18/v1"
    for eval_split, role in (
        (configs.internal, INTERNAL_EVAL),
        (configs.official, OFFICIAL),
        (configs.held_out, HELD_OUT),
    ):
        assert eval_split is not None
        assert eval_split.split_role == role


def test_held_out_is_omitted_when_the_split_requests_none() -> None:
    prepared = prepare_c18_experiment(_smoke_pool(), split_sizes=(2, 2, 0))
    assert prepared.experiment.eval_configs.held_out is None
    assert prepared.experiment.eval_configs.held_out_task_hashes == ()


def test_held_out_rows_match_the_source_split() -> None:
    pool = _smoke_pool()
    prepared = prepare_c18_experiment(pool, split_sizes=(1, 1, 1))
    expected = tuple(
        row.task_hash
        for row in task_rows_from_instances(pool.split(1, 1, 1).held_out)
    )
    assert prepared.experiment.eval_configs.held_out_task_hashes == expected


def test_the_experiment_anchors_on_c18s_own_probes() -> None:
    experiment = prepare_c18_experiment(
        _smoke_pool(), split_sizes=(2, 2, 0)
    ).experiment
    initial = experiment.initial_candidate.payload[C18_MUTATION_FIELD]
    ceiling = experiment.ceiling_candidate.payload[C18_MUTATION_FIELD]
    assert initial == PROBES.naive_template
    assert ceiling == PROBES.ceiling_template
    assert experiment.env_name == C18_NAMESPACE
    assert experiment.reward_policy.policy_name == "c18-exact-match"


def test_prepare_refuses_a_non_positive_repeat_count() -> None:
    with pytest.raises(ValueError, match="num_seeds must be at least 1"):
        prepare_c18_experiment(
            _smoke_pool(), split_sizes=(2, 2, 0), num_seeds=0
        )


def test_prepare_refuses_an_empty_required_split() -> None:
    with pytest.raises(ValueError, match="non-empty internal and official"):
        prepare_c18_experiment(_smoke_pool(), split_sizes=(0, 2, 0))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


class _Task:
    def __init__(self, gold: str) -> None:
        self.gold = gold


def _score(generation: object, gold: str) -> float | None:
    score, _output, _extra = C18VerdictEvalProcedureRunner().run_eval_node(
        node_id="generate",
        node_inputs={"provider_generation": generation},
        evaluation_procedure_config_hash="unused",
        task=_Task(gold),
    )
    return score


@pytest.mark.parametrize(
    ("generation", "gold", "expected"),
    [
        ("True", "True", 1.0),
        ("False", "False", 1.0),
        ("True", "False", 0.0),
        # The ceiling probe asks for reasoning ending in a lone verdict, so
        # the terminal line is what counts. A plain exact-match runner would
        # score this zero and flatten the ceiling anchor into the floor.
        ("Sam is a rompus, so it follows.\nTrue", "True", 1.0),
        ("Chain of rules leads here.\nFalse", "True", 0.0),
        # Ambiguity is not a verdict.
        ("Probably true, probably false", "True", 0.0),
        ("", "True", 0.0),
    ],
)
def test_the_terminal_verdict_is_what_is_scored(
    generation: str, gold: str, expected: float
) -> None:
    assert _score(generation, gold) == expected


def test_a_non_string_generation_scores_zero_rather_than_raising() -> None:
    """A malformed provider reply is a zero, not a run-ending crash."""
    assert _score(None, "True") == 0.0


def test_the_runner_takes_no_constructor_state() -> None:
    """Workers reconstruct the runner by type, transferring no state."""
    assert C18VerdictEvalProcedureRunner() is not None
    assert type(C18VerdictEvalProcedureRunner()).__init__ is object.__init__
