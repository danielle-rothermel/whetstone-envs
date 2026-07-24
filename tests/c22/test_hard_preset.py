"""Checks for the c22 'hard' mix and the HARD_PRESET pool variant.

The hard preset is the hardest configuration of the original IFEval suite
(no hidden-information design change): counts ``(3, 6, 8)`` crossed with a
single ``hard`` mix that is hard-first and conflict-aware -- every 'hard'
stack contains ALL three hard-pool atoms, filling any remaining count from
the easy pool. These are the no-LLM-call blocking checks (determinism,
strata composition, conflict feasibility, prompts-unchanged) plus
hand-built oracle fixtures exercising the word-count x forbidden-word
interaction on a pure-hard and an n8 instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone_envs.c22 import generate, oracle
from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.generate import (
    DEFAULT_SEED_START,
    HARD_PRESET,
    HARD_SEED_START,
    MIX_HARD,
    PUBLISHED_KEY_MAX,
    PUBLISHED_KEY_MIN,
)
from whetstone_envs.c22.prompts import PROBES
from whetstone_envs.c22.spec import ConstraintSpec, compatibility_error
from whetstone_envs.core.manifest import Manifest, content_hash

_HARD_MANIFEST_PATH = Path(generate.__file__).with_name("manifest_hard.json")

_HARD_IDS = {
    "length_constraints:number_words",
    "keywords:letter_frequency",
    "keywords:forbidden_words",
}
_EASY_IDS = {a.instruction_id for a in generate.EASY_POOL}


# --- Determinism ----------------------------------------------------------


def test_preset_regenerates_byte_identical() -> None:
    a = HARD_PRESET.generate(n_per_stratum=5)
    b = HARD_PRESET.generate(n_per_stratum=5)
    assert content_hash(a) == content_hash(b)
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]
    assert [i.id for i in a.instances] == [i.id for i in b.instances]


def test_committed_hard_manifest_matches_regenerated_pool() -> None:
    pool = HARD_PRESET.generate()
    frozen = Manifest.read(_HARD_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == "c22-generate-3+hard"


# --- Strata composition ---------------------------------------------------


def test_hard_preset_has_three_strata_with_declared_counts() -> None:
    pool = HARD_PRESET.generate()
    assert pool.stratum_counts() == {
        "n3_hard": 20,
        "n6_hard": 20,
        "n8_hard": 20,
    }
    assert len(pool) == 60


def test_every_hard_instance_contains_all_three_hard_atoms() -> None:
    # The defining property of the 'hard' mix: all three hard-pool atoms
    # are present in every stack (pure-hard at n=3, hard+easy-fill at n>3).
    pool = HARD_PRESET.generate()
    for inst in pool.instances:
        (label,) = inst.strata
        spec = ConstraintSpec.from_gold(inst.gold)
        ids = set(spec.instruction_id_list)
        assert ids >= _HARD_IDS, f"{inst.id} ({label}) missing a hard atom"
        n = int(label.split("_")[0][1:])  # "n6_hard" -> 6
        assert len(spec.instruction_id_list) == n
        assert len(spec.constraint_descriptions) == n
        assert len(spec.kwargs_list) == n


def test_pure_hard_stratum_is_exactly_the_three_hard_atoms() -> None:
    pool = HARD_PRESET.generate()
    for inst in pool.instances:
        (label,) = inst.strata
        if label != "n3_hard":
            continue
        ids = set(ConstraintSpec.from_gold(inst.gold).instruction_id_list)
        assert ids == _HARD_IDS  # no easy fill at n=3


def test_hard_fill_beyond_three_comes_from_the_easy_pool() -> None:
    pool = HARD_PRESET.generate()
    for inst in pool.instances:
        (label,) = inst.strata
        if label == "n3_hard":
            continue
        ids = set(ConstraintSpec.from_gold(inst.gold).instruction_id_list)
        assert (ids - _HARD_IDS) <= _EASY_IDS  # fill is easy-pool only


# --- Conflict feasibility across the whole pool ---------------------------


def test_no_stacked_pair_conflicts_across_the_whole_pool() -> None:
    conflicts = instructions_registry.INSTRUCTION_CONFLICTS
    pool = HARD_PRESET.generate()
    for inst in pool.instances:
        spec = ConstraintSpec.from_gold(inst.gold)
        ids = list(spec.instruction_id_list)
        assert len(ids) == len(set(ids)), f"{inst.id} has duplicate atoms"
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                assert b not in conflicts.get(a, set()), (
                    f"{inst.id}: {a} conflicts with {b}"
                )
        assert compatibility_error(ids, spec.kwargs_list) is None


# --- Contamination / seed disjointness ------------------------------------


def test_hard_seeds_are_disjoint_from_published_and_base_c22() -> None:
    pool = HARD_PRESET.generate()
    seeds = [i.seed for i in pool.instances]
    for seed in seeds:
        assert seed > PUBLISHED_KEY_MAX
        assert not (PUBLISHED_KEY_MIN <= seed <= PUBLISHED_KEY_MAX)
    # Disjoint from the base c22 pool's range too: HARD_SEED_START sits far
    # above the base default range so hard instances never collide with it.
    assert HARD_SEED_START > DEFAULT_SEED_START
    assert min(seeds) >= HARD_SEED_START
    # The base default pool is 120 instances from DEFAULT_SEED_START; the
    # hard range starts strictly above that ceiling.
    assert HARD_SEED_START > DEFAULT_SEED_START + 120


def test_hard_instance_ids_do_not_collide_with_base_c22_ids() -> None:
    base = generate.generate_pool(n_per_stratum=20)
    hard = HARD_PRESET.generate()
    base_ids = {i.id for i in base.instances}
    hard_ids = {i.id for i in hard.instances}
    assert base_ids.isdisjoint(hard_ids)


# --- Prompts unchanged (same two probe templates, no gold leak) -----------


def test_probes_render_for_hard_instances_without_gold_leak() -> None:
    pool = HARD_PRESET.generate(n_per_stratum=1)
    for inst in pool.instances:
        naive = PROBES.render_naive(inst)
        ceiling = PROBES.render_ceiling(inst)
        block = inst.prompt_inputs["constraints_block"]
        # The same two templates render, both embedding the public block.
        assert block in naive
        assert block in ceiling
        assert len(ceiling) > len(naive)
        # No gold/oracle-only field leaks into either rendered prompt.
        spec = ConstraintSpec.from_gold(inst.gold)
        for atom_id in spec.instruction_id_list:
            assert atom_id not in naive
            assert atom_id not in ceiling
        assert set(inst.prompt_inputs) == {"constraints_block"}


# --- Oracle fixtures: word-count x forbidden-word interaction --------------
# Hand-built (NOT generator-produced) instances with independently verified
# verdicts. Each exercises the hard interaction: hitting the exact word
# count while keeping a forbidden token out of the whole answer.

# A pure-hard 3-atom stack: exactly 5 words, no 'z', forbid 'quarnex'.
_PURE_HARD = ConstraintSpec(
    base_task="Name a fruit.",
    constraint_descriptions=(
        "Answer with exactly 5 words.",
        "In your response, the letter z should appear less than 1 times.",
        "Do not include keywords ['quarnex'] in the response.",
    ),
    instruction_id_list=(
        "length_constraints:number_words",
        "keywords:letter_frequency",
        "keywords:forbidden_words",
    ),
    kwargs_list=(
        {"num_words": 5, "relation": "exactly"},
        {"letter": "z", "let_frequency": 1, "let_relation": "less than"},
        {"forbidden_words": ["quarnex"]},
    ),
)

# An n8 stack: 3 hard (exactly 8 words, no 'q', forbid 'zylthorn') + 5 easy.
_N8 = ConstraintSpec(
    base_task="Describe a season in a few words.",
    constraint_descriptions=(
        "Answer with exactly 8 words.",
        "In your response, the letter q should appear less than 1 times.",
        "Do not include keywords ['zylthorn'] in the response.",
        "Include keywords ['vopflim'] in the response.",
        "Finish your response with this exact phrase END OF ANSWER. "
        "No other words should follow this phrase.",
        "In your entire response, refrain from the use of any commas.",
        "The response must contain at least 1 placeholders represented "
        "by square brackets, such as [address].",
        "At the end of your response, please explicitly add a postscript "
        "starting with P.S.",
    ),
    instruction_id_list=(
        "length_constraints:number_words",
        "keywords:letter_frequency",
        "keywords:forbidden_words",
        "keywords:existence",
        "startend:end_checker",
        "punctuation:no_comma",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
    ),
    kwargs_list=(
        {"num_words": 8, "relation": "exactly"},
        {"letter": "q", "let_frequency": 1, "let_relation": "less than"},
        {"forbidden_words": ["zylthorn"]},
        {"keywords": ["vopflim"]},
        {"end_phrase": "END OF ANSWER"},
        {},
        {"num_placeholders": 1},
        {"postscript_marker": "P.S."},
    ),
)


def test_pure_hard_oracle_pass_and_fail() -> None:
    # PASS: 5 words, no 'z', no 'quarnex'.
    good = oracle.check(_PURE_HARD, "apple banana cherry mango melon")
    assert good.score == 1
    assert dict(good.per_atom) == {
        "length_constraints:number_words": True,
        "keywords:letter_frequency": True,
        "keywords:forbidden_words": True,
    }
    # FAIL: still 5 words and 'z'-free, but padding the answer to length
    # with the forbidden token trips exactly the forbidden-word atom.
    bad = oracle.check(_PURE_HARD, "apple banana cherry mango quarnex")
    assert bad.score == 0
    verdicts = dict(bad.per_atom)
    assert verdicts["length_constraints:number_words"] is True
    assert verdicts["keywords:forbidden_words"] is False


def test_n8_oracle_pass_and_fail_word_count_x_forbidden() -> None:
    # PASS: exactly 8 tokenizer words plus every easy atom.
    passing = "cold vopflim [x] P.S. END OF ANSWER"
    good = oracle.check(_N8, passing)
    assert good.score == 1
    assert all(v for _, v in good.per_atom)
    # FAIL: replacing one word with the forbidden token preserves the
    # exact count, so only the forbidden-word atom fails.
    failing = "zylthorn vopflim [x] P.S. END OF ANSWER"
    bad = oracle.check(_N8, failing)
    assert bad.score == 0
    verdicts = dict(bad.per_atom)
    assert verdicts["length_constraints:number_words"] is True
    assert verdicts["keywords:forbidden_words"] is False


def test_hard_mix_generates_a_short_stack_deterministically() -> None:
    # The 'hard' mix is usable through the plain axes API too (config, not
    # a fork): a single-count 'hard' pool is expressible directly.
    pool = generate.generate_pool(
        n_per_stratum=2,
        constraint_counts=(3,),
        mixes=(MIX_HARD,),
        seed_start=HARD_SEED_START,
    )
    assert pool.stratum_counts() == {"n3_hard": 2}
    for inst in pool.instances:
        ids = set(ConstraintSpec.from_gold(inst.gold).instruction_id_list)
        assert ids == _HARD_IDS


@pytest.mark.parametrize(
    ("instance_id", "response"),
    [
        (
            "c22-2000037",
            "<<*red* *blue* [x] tree sun moon sky>>",
        ),
        (
            "c22-2000041",
            '"P.S.jaxbrynSIGNED OFF"',
        ),
        (
            "c22-2000048",
            "<<[P.P.S][a][b]>>FULLY DONE",
        ),
    ],
)
def test_reviewed_exact_word_seeds_have_full_satisfying_responses(
    instance_id: str,
    response: str,
) -> None:
    pool_by_id = {
        instance.id: instance for instance in HARD_PRESET.generate().instances
    }
    instance = pool_by_id[instance_id]
    assert oracle.score_gold(instance.gold, response) == 1
