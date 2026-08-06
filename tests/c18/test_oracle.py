"""Forward-chaining oracle correctness on hand-traced fixtures.

Every theory below is hand-constructed and hand-traced -- none is
generator-produced -- so this file catches an oracle that is silently a
re-derivation of generator internals rather than a true independent
forward-chaining fixpoint over the public text.

The four multi-hop chains cover the default depth strata D1, D2,
D3, D5. Each chain is worked out step by step in the comment beside it,
and each is exercised in both polarities (a derivable query -> True, and
the opposite/undrivable query -> False), so the fixtures pin both label
outcomes at every depth.
"""

from __future__ import annotations

import pytest

from whetstone_envs.c18 import oracle
from whetstone_envs.c18.oracle import (
    OracleError,
    entailment_label,
    extract_verdict,
    score,
    score_gold,
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
    # Derivable: Sally is sour (True). Opposite polarity not derivable.
    assert entailment_label(_D1, "True or false: Sally is sour.") == "True"
    assert (
        entailment_label(_D1, "True or false: Sally is not sour.") == "False"
    )


def test_d2_two_hop_chain_both_polarities() -> None:
    # Derivable negated property: Stella is not floral (True).
    assert (
        entailment_label(_D2, "True or false: Stella is not floral.") == "True"
    )
    # The un-negated form is not derivable -> False.
    assert entailment_label(_D2, "True or false: Stella is floral.") == "False"


def test_d3_three_hop_chain_both_polarities() -> None:
    assert entailment_label(_D3, "True or false: Max is not rainy.") == "True"
    assert entailment_label(_D3, "True or false: Max is rainy.") == "False"


def test_d5_five_hop_chain_both_polarities() -> None:
    assert entailment_label(_D5, "True or false: Wren is amenable.") == "True"
    not_amenable = "True or false: Wren is not amenable."
    assert entailment_label(_D5, not_amenable) == "False"


def test_unrelated_property_is_false() -> None:
    # A property mentioned in a rule the individual never reaches is not
    # derivable: "Each wumpus is not sour" never fires (Sally is no wumpus).
    assert entailment_label(_D1, "True or false: Sally is loud.") == "False"


def test_fixpoint_requires_iterating_to_convergence() -> None:
    # Rules are given out of chain order; the fixpoint loop must still
    # propagate through all three membership hops regardless of order.
    shuffled = (
        "Every impus is not rainy. Brimpuses are impuses. "
        "Each grimpus is a brimpus. Max is a grimpus."
    )
    assert entailment_label(shuffled, "True or false: Max is not rainy.") == (
        "True"
    )


def test_singular_and_plural_kinds_unify() -> None:
    # "Every jompus ..." (singular antecedent) and "Jompuses are ..."
    # (plural antecedent) must canonicalize to the same kind key, or a
    # chain crossing the two forms would silently break.
    theory = "Fae is a wumpus. Every wumpus is a jompus. Jompuses are sunny."
    assert entailment_label(theory, "True or false: Fae is sunny.") == "True"


def test_plural_membership_without_article_is_a_kind() -> None:
    # "Tumpuses are wumpuses" is a kind membership (no article), not a
    # property assignment; the chain must treat "wumpuses" as a kind.
    theory = (
        "Stella is a tumpus. Tumpuses are wumpuses. "
        "Every wumpus is not bitter."
    )
    not_bitter = "True or false: Stella is not bitter."
    assert entailment_label(theory, not_bitter) == "True"
    # Sanity: an article-less adjective in the same shape is a property.
    theory2 = "Stella is a tumpus. Tumpuses are small."
    assert entailment_label(theory2, "True or false: Stella is small.") == (
        "True"
    )


def test_negation_polarity_is_exact() -> None:
    # Deriving "not X" must not satisfy a query for "X" and vice versa.
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


def test_score_exact_match_and_normalization() -> None:
    q, query = _D1, "True or false: Sally is sour."
    assert score("True", q, query) == 1
    # Case-insensitive, whitespace/fence tolerant.
    assert score("true", q, query) == 1
    assert score("  True\n", q, query) == 1
    assert score("```\nTrue\n```", q, query) == 1
    assert score("False", q, query) == 0


# A reasoned ceiling reply ends with the verdict on its own final line; a
# naive reply may append a trailing rationale.
# Scoring the whole reply against the gold token scores every such reply 0.
# These fixtures pin extraction of the final True/False verdict for each
# response shape.
_CEILING_REASONING = (
    "Wren is an impus. Impuses are vumpuses, so Wren is a vumpus. "
    "Vumpuses are gorpuses, so Wren is a gorpus. Following the rules step "
    "by step, no rule makes Wren amenable; the query property is not "
    "entailed.\n\nFalse"
)
_NAIVE_RATIONALE = "True, since Sally is a brimpus and every brimpus is sour."


def test_extract_verdict_bare_tokens() -> None:
    assert extract_verdict("True") == "True"
    assert extract_verdict("False") == "False"
    assert extract_verdict("true") == "True"
    assert extract_verdict("FALSE") == "False"


def test_extract_verdict_cot_takes_final_line_token() -> None:
    # The ceiling protocol makes the verdict the final standalone token.
    assert extract_verdict(_CEILING_REASONING) == "False"


def test_extract_verdict_accepts_period_on_verdict_only_final_line() -> None:
    prediction = (
        "The premises mention True, but do not entail the query.\nFalse."
    )
    assert extract_verdict(prediction) == "False"
    assert score_gold(prediction, "False") == 1


def test_extract_verdict_naive_trailing_rationale() -> None:
    # A verdict followed by an appended rationale still resolves to the
    # verdict (the rationale here restates no verdict token).
    assert extract_verdict(_NAIVE_RATIONALE) == "True"
    assert extract_verdict("False. It is not derivable from the rules.") == (
        "False"
    )


def test_rationale_boolean_does_not_override_leading_verdict() -> None:
    prediction = "True, because the statement is not false."
    assert extract_verdict(prediction) == "True"
    assert score_gold(prediction, "True") == 1


def test_extract_verdict_no_token_returned_unchanged() -> None:
    # No verdict token -> unchanged, so it still fails exact match (0).
    assert extract_verdict("I cannot determine the answer.") == (
        "I cannot determine the answer."
    )


def test_score_ceiling_reasoning_response_scores_correctly() -> None:
    assert score_gold(_CEILING_REASONING, "False") == 1
    assert score_gold(_CEILING_REASONING, "True") == 0


def test_score_naive_trailing_rationale_scores_correctly() -> None:
    assert score_gold(_NAIVE_RATIONALE, "True") == 1
    assert score_gold(_NAIVE_RATIONALE, "False") == 0


def test_score_cot_response_via_reredive_path() -> None:
    # The same extraction applies on the re-derive-from-text scoring path.
    q = (
        "Wren is an impus. Impuses are vumpuses. Vumpuses are gorpuses. "
        "Gorpuses are zumpuses. Zumpuses are shumpuses. "
        "Every shumpus is amenable. Each lorpus is not amenable."
    )
    query = "True or false: Wren is not amenable."  # gold False
    assert score(_CEILING_REASONING, q, query) == 1


def test_score_unparsable_input_is_zero_not_raise() -> None:
    # A malformed query scores 0 rather than raising (a response is graded).
    assert score("True", _D1, "this is not a query") == 0


def test_score_gold_matches_frozen_gold() -> None:
    assert score_gold("True", "True") == 1
    assert score_gold(" true ", "True") == 1
    assert score_gold("False", "True") == 0


def test_query_rejecting_a_kind_target() -> None:
    # The ontology only ever queries a property; a kind-valued query is
    # rejected as malformed rather than silently answered.
    with pytest.raises(OracleError, match="targets a kind"):
        entailment_label(_D1, "True or false: Sally is a brimpus.")


def test_unparsable_query_raises() -> None:
    with pytest.raises(OracleError, match="unparsable query"):
        entailment_label(_D1, "Is Sally sour?")


def test_unparsable_sentence_raises() -> None:
    with pytest.raises(OracleError, match="unparsable sentence"):
        entailment_label(
            "Sally frobnicates the brimpus widget",
            "True or false: Sally is sour.",
        )


def test_entity_names_pin_the_fact_vs_rule_split() -> None:
    # The nine upstream proper nouns are what separate a ground fact from a
    # universally-quantified rule; the set is pinned for that classification.
    expected = {
        "Fae",
        "Rex",
        "Sally",
        "Max",
        "Alex",
        "Sam",
        "Polly",
        "Stella",
        "Wren",
    }
    assert set(oracle.ENTITY_NAMES) == expected
