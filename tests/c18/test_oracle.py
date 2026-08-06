from __future__ import annotations

import pytest

from whetstone_envs.c18 import score_gold
from whetstone_envs.c18.oracle import (
    OracleError,
    entailment_label,
    extract_verdict,
    score,
)

# --- D1: a one-hop chain (fact -> one rule -> queried property) -----------
# Sally is a brimpus; every brimpus is sour  =>  Sally is sour.
_D1 = "Sally is a brimpus. Every brimpus is sour. Each wumpus is not sour."

# --- D2: a two-hop chain (fact -> membership rule -> property rule) --------
# Stella is a lempus; lempuses are zumpuses  => Stella is a zumpus;
# every zumpus is not floral  =>  Stella is not floral.
_D2 = (
    "Stella is a lempus. Lempuses are zumpuses. Every zumpus is not floral. "
    "Each tumpus is small."
)

# --- D3: a three-hop chain (two membership hops, then a property) ----------
# Max is a grimpus; each grimpus is a brimpus => Max is a brimpus;
# brimpuses are impuses => Max is an impus; every impus is not rainy
#   =>  Max is not rainy.
_D3 = (
    "Max is a grimpus. Each grimpus is a brimpus. Brimpuses are impuses. "
    "Every impus is not rainy. Gorpuses are loud."
)

# --- D5: a five-hop chain (four membership hops, then a property) ----------
# Wren is an impus; impuses are vumpuses => Wren is a vumpus;
# vumpuses are gorpuses => Wren is a gorpus; gorpuses are zumpuses
#   => Wren is a zumpus; zumpuses are shumpuses => Wren is a shumpus;
# every shumpus is amenable  =>  Wren is amenable.
_D5 = (
    "Wren is an impus. Impuses are vumpuses. Vumpuses are gorpuses. "
    "Gorpuses are zumpuses. Zumpuses are shumpuses. "
    "Every shumpus is amenable. Each lorpus is not amenable."
)


def test_d1_one_hop_chain_both_polarities() -> None:
    assert entailment_label(_D1, "True or false: Sally is sour.") == "True"
    assert (
        entailment_label(_D1, "True or false: Sally is not sour.") == "False"
    )


def test_d2_two_hop_chain_both_polarities() -> None:
    assert (
        entailment_label(_D2, "True or false: Stella is not floral.") == "True"
    )
    assert entailment_label(_D2, "True or false: Stella is floral.") == "False"


def test_d3_three_hop_chain_both_polarities() -> None:
    assert entailment_label(_D3, "True or false: Max is not rainy.") == "True"
    assert entailment_label(_D3, "True or false: Max is rainy.") == "False"


def test_d5_five_hop_chain_both_polarities() -> None:
    assert entailment_label(_D5, "True or false: Wren is amenable.") == "True"
    not_amenable = "True or false: Wren is not amenable."
    assert entailment_label(_D5, not_amenable) == "False"


def test_unrelated_property_is_false() -> None:
    assert entailment_label(_D1, "True or false: Sally is loud.") == "False"


def test_fixpoint_requires_iterating_to_convergence() -> None:
    shuffled = (
        "Every impus is not rainy. Brimpuses are impuses. "
        "Each grimpus is a brimpus. Max is a grimpus."
    )
    assert entailment_label(shuffled, "True or false: Max is not rainy.") == (
        "True"
    )


def test_singular_and_plural_kinds_unify() -> None:
    theory = "Fae is a wumpus. Every wumpus is a jompus. Jompuses are sunny."
    assert entailment_label(theory, "True or false: Fae is sunny.") == "True"


def test_plural_membership_without_article_is_a_kind() -> None:
    theory = (
        "Stella is a tumpus. Tumpuses are wumpuses. "
        "Every wumpus is not bitter."
    )
    not_bitter = "True or false: Stella is not bitter."
    assert entailment_label(theory, not_bitter) == "True"
    theory2 = "Stella is a tumpus. Tumpuses are small."
    assert entailment_label(theory2, "True or false: Stella is small.") == (
        "True"
    )


def test_negation_polarity_is_exact() -> None:
    theory = "Rex is a numpus. Numpuses are not luminous."
    assert entailment_label(theory, "True or false: Rex is not luminous.") == (
        "True"
    )
    assert entailment_label(theory, "True or false: Rex is luminous.") == (
        "False"
    )


def test_facts_for_a_distractor_entity_do_not_apply_to_query_entity() -> None:
    theory = (
        "Rex is a numpus. Sam is a wumpus. Every wumpus is luminous. "
        "Every numpus is wooden."
    )
    assert (
        entailment_label(theory, "True or false: Rex is luminous.") == "False"
    )
    assert entailment_label(theory, "True or false: Rex is wooden.") == "True"


_CEILING_REASONING = (
    "Wren is an impus. Impuses are vumpuses, so Wren is a vumpus. "
    "Vumpuses are gorpuses, so Wren is a gorpus. Following the rules step "
    "by step, no rule makes Wren amenable; the query property is not "
    "entailed.\n\nFalse"
)


def test_extract_verdict_bare_tokens() -> None:
    assert extract_verdict("True") == "True"
    assert extract_verdict("False") == "False"
    assert extract_verdict("true") == "True"
    assert extract_verdict("FALSE") == "False"


def test_extract_verdict_cot_takes_final_line_token() -> None:
    assert extract_verdict(_CEILING_REASONING) == "False"


def test_extract_verdict_accepts_period_on_verdict_only_final_line() -> None:
    prediction = (
        "The premises mention True, but do not entail the query.\nFalse."
    )
    assert extract_verdict(prediction) == "False"
    assert score_gold(prediction, "False") == 1


@pytest.mark.parametrize(
    "prediction",
    [
        "True or False",
        "True/False",
        "True but I cannot determine which",
        "True or false: the statement is not entailed; therefore False.",
        "True, or False",
        "False. Or True.",
        "True, but either answer could be correct",
        "True, because either answer may be correct.",
        "False, since the query may actually be true.",
        "True. It may actually be false.",
        "False. The answer could also be true.",
        "True. This may be false.",
        "False. That may be true.",
    ],
)
def test_ambiguous_prefix_is_not_a_verdict(prediction: str) -> None:
    assert extract_verdict(prediction) == prediction
    assert score_gold(prediction, "True") == 0
    assert score_gold(prediction, "False") == 0


def test_extract_verdict_no_token_returned_unchanged() -> None:
    assert extract_verdict("I cannot determine the answer.") == (
        "I cannot determine the answer."
    )


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        (_CEILING_REASONING, "False", 1),
        ("True", "True", 1),
        ("False", "True", 0),
    ],
)
def test_score_gold_handles_protocol_responses(
    prediction: str,
    gold: str,
    expected: int,
) -> None:
    assert score_gold(prediction, gold) == expected


def test_score_cot_response_via_rederive_path() -> None:
    q = (
        "Wren is an impus. Impuses are vumpuses. Vumpuses are gorpuses. "
        "Gorpuses are zumpuses. Zumpuses are shumpuses. "
        "Every shumpus is amenable. Each lorpus is not amenable."
    )
    query = "True or false: Wren is not amenable."  # gold False
    assert score(_CEILING_REASONING, q, query) == 1


def test_score_unparsable_input_is_zero_not_raise() -> None:
    assert score("True", _D1, "this is not a query") == 0


def test_query_rejecting_a_kind_target() -> None:
    with pytest.raises(OracleError):
        entailment_label(_D1, "True or false: Sally is a brimpus.")


def test_unparsable_query_raises() -> None:
    with pytest.raises(OracleError):
        entailment_label(_D1, "Is Sally sour?")


def test_unparsable_sentence_raises() -> None:
    with pytest.raises(OracleError):
        entailment_label(
            "Sally frobnicates the brimpus widget",
            "True or false: Sally is sour.",
        )


def test_question_requires_a_fact_for_the_query_entity() -> None:
    with pytest.raises(OracleError):
        entailment_label(
            "Every brimpus is sour.",
            "True or false: Sally is sour.",
        )
