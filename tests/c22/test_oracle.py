"""Oracle correctness on HAND-BUILT fixtures (not generator-produced).

Checklist A requires at least one hand-constructed instance per atom in
the easy and hard pools, with an independently-verified expected verdict,
asserting the oracle's pass/fail matches manual inspection of the
constraint text. The fixtures below are written by hand -- their
``instruction_id_list`` / ``kwargs`` / responses are stated directly, so
this file exercises the oracle as an independent check and would catch an
oracle that had silently become a re-derivation of generator internals.
"""

from __future__ import annotations

import pytest

from whetstone_envs.c22 import oracle
from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.atoms import EASY_POOL, HARD_POOL
from whetstone_envs.c22.spec import ConstraintSpec


def _single_atom_spec(
    instruction_id: str,
    kwargs: dict[str, object],
) -> ConstraintSpec:
    """A hand-built one-atom spec with its checker-canonical description."""
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](
        instruction_id,
    )
    description = instruction.build_description(**kwargs)
    return ConstraintSpec(
        base_task="Answer the question.",
        constraint_descriptions=(description,),
        instruction_id_list=(instruction_id,),
        kwargs_list=(kwargs,),
    )


# (instruction_id, kwargs, passing_response, failing_response)
# Each pair is hand-verified against the constraint's plain meaning.
_EASY_FIXTURES: list[tuple[str, dict[str, object], str, str]] = [
    (
        "keywords:existence",
        {"keywords": ["zylthorn"]},
        "the answer mentions zylthorn plainly",  # contains the keyword
        "the answer omits it entirely",  # keyword absent
    ),
    (
        "startend:end_checker",
        {"end_phrase": "END OF ANSWER"},
        "some words END OF ANSWER",  # ends with the exact phrase
        "some words and nothing else",  # wrong ending
    ),
    (
        "detectable_format:title",
        {},
        "here is <<A Real Title>> and text",  # has <<...>> title
        "here is a plain answer",  # no title wrapper
    ),
    (
        "startend:quotation",
        {},
        '"the whole thing is quoted"',  # wrapped in double quotes
        "not wrapped in quotes",  # unwrapped
    ),
    (
        "punctuation:no_comma",
        {},
        "no commas anywhere here",  # comma-free
        "this one, unfortunately, has commas",  # contains commas
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "fill in [name] and [date] please",  # two [placeholders]
        "fill in [name] only",  # only one placeholder
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "main text P.S. an afterthought",  # has the P.S. marker
        "main text with no afterthought",  # no postscript
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 1},
        "an *emphasized* fragment appears",  # one *highlight*
        "nothing is emphasized here",  # no highlights
    ),
]

_HARD_FIXTURES: list[tuple[str, dict[str, object], str, str]] = [
    (
        "length_constraints:number_words",
        {"num_words": 4, "relation": "exactly"},
        "exactly four words here",  # exactly 4 words
        "only three words",  # 3 words
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_frequency": 1, "let_relation": "less than"},
        "the letter is absent entirely",  # zero 'z' -> < 1
        "a puzzling buzzing haze",  # several 'z' -> not < 1
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["quarnex"]},
        "a perfectly clean response",  # forbidden word absent
        "this mentions quarnex directly",  # forbidden word present
    ),
]


def _all_atom_ids_covered() -> None:
    covered = {f[0] for f in (*_EASY_FIXTURES, *_HARD_FIXTURES)}
    easy_ids = {a.instruction_id for a in EASY_POOL}
    hard_ids = {a.instruction_id for a in HARD_POOL}
    assert easy_ids <= covered, easy_ids - covered
    assert hard_ids <= covered, hard_ids - covered


def test_every_pool_atom_has_a_hand_fixture() -> None:
    # Guards checklist A's "at least one instance per atom" requirement:
    # if a new atom is added to a pool without a fixture, this fails.
    _all_atom_ids_covered()


@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "passing", "failing"),
    [*_EASY_FIXTURES, *_HARD_FIXTURES],
    ids=[f[0] for f in (*_EASY_FIXTURES, *_HARD_FIXTURES)],
)
def test_single_atom_pass_and_fail(
    instruction_id: str,
    kwargs: dict[str, object],
    passing: str,
    failing: str,
) -> None:
    spec = _single_atom_spec(instruction_id, kwargs)
    assert oracle.check(spec, passing).score == 1
    assert oracle.check(spec, failing).score == 0


def test_exact_word_count_rejects_both_adjacent_counts() -> None:
    spec = _single_atom_spec(
        "length_constraints:number_words",
        {"num_words": 4, "relation": "exactly"},
    )
    assert oracle.check(spec, "one two three four").score == 1
    assert oracle.check(spec, "one two three").score == 0
    assert oracle.check(spec, "one two three four five").score == 0


def test_score_gold_round_trips_through_json() -> None:
    spec = _single_atom_spec("keywords:existence", {"keywords": ["zylthorn"]})
    gold = spec.to_gold()
    assert oracle.score_gold(gold, "here is zylthorn") == 1
    assert oracle.score_gold(gold, "not present") == 0


def test_strict_all_pass_requires_every_atom() -> None:
    # A hand-built 3-atom stack: keyword + no-comma + end phrase.
    spec = ConstraintSpec(
        base_task="Name a color.",
        constraint_descriptions=(
            "Include keywords ['zylthorn'] in the response.",
            "In your entire response, refrain from the use of any commas.",
            "Finish your response with this exact phrase DONE. "
            "No other words should follow this phrase.",
        ),
        instruction_id_list=(
            "keywords:existence",
            "punctuation:no_comma",
            "startend:end_checker",
        ),
        kwargs_list=(
            {"keywords": ["zylthorn"]},
            {},
            {"end_phrase": "DONE"},
        ),
    )
    # satisfies all three
    good = oracle.check(spec, "zylthorn blue DONE")
    assert good.score == 1
    assert good.follow_all
    assert [v for _, v in good.per_atom] == [True, True, True]

    # violates only the end phrase -> strict all-pass fails
    bad = oracle.check(spec, "zylthorn blue")
    assert bad.score == 0
    assert not bad.follow_all
    verdicts = dict(bad.per_atom)
    assert verdicts["keywords:existence"] is True
    assert verdicts["punctuation:no_comma"] is True
    assert verdicts["startend:end_checker"] is False


def test_per_atom_verdicts_align_with_stack_order() -> None:
    spec = _single_atom_spec("startend:quotation", {})
    result = oracle.check(spec, '"quoted"')
    assert result.per_atom == (("startend:quotation", True),)


def test_normalization_strips_wrapping_fence_before_scoring() -> None:
    # The shared normalize() removes a wrapping code fence; a fenced but
    # otherwise-correct answer must still pass.
    spec = _single_atom_spec("startend:end_checker", {"end_phrase": "END"})
    fenced = "```\nfinal words END\n```"
    assert oracle.check(spec, fenced).score == 1


def test_empty_response_scores_zero() -> None:
    # test_instruction_following_strict counts a blank response as a
    # failure of every constraint.
    spec = _single_atom_spec("detectable_format:title", {})
    assert oracle.check(spec, "   ").score == 0
