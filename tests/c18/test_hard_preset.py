"""Checks for the c18 HARD_PRESET pool variant (deep chains + distractors).

The hard preset is the hardest configuration of the *upstream PrOntoQA*
suite along the two axes this env already exposes -- deduction depth and
distractors -- with NO hidden-information design change (no Unknown label,
no OOD rule type, no constraint-puzzle stratum). It pushes hop depth past
the base pool's D5 ceiling (depths ``(5, 8, 10)``), using relevant
distractors at D5 and none at D8/D10, where upstream cannot generate
deep relevant-distractor cases. It draws from a fresh seed range disjoint
from both the published PrOntoQA space and the base c18 pool.

These are the no-LLM-call blocking checks (preset determinism, strata
composition, contamination / seed disjointness, prompts-unchanged) plus the
load-bearing hand-traced ORACLE fixtures: a D8 chain, a D10-with-distractor
chain, and a distractor instance whose query is NOT entailed despite a
distractor rule mentioning the queried property. The oracle is a pure
function of the public text and is UNCHANGED for depth -- these fixtures
prove it closes to the correct fixpoint at maximal chain length.

Generation-path tests reseed the vendored PrOntoQA generator once per depth
(a subprocess); deep strata are slow, so those tests use ``n_per_stratum=1``
and (where the preset mechanism -- not the depth ceiling -- is what is under
test) the base pool's shallow depths. The depth ceiling itself is pinned as
a static assertion. The oracle fixtures call no subprocess and are fast.
"""

from __future__ import annotations

from pathlib import Path

from whetstone_envs.c18 import generate, oracle
from whetstone_envs.c18.generate import (
    DEFAULT_DISTRACTORS,
    DEFAULT_SEED_START,
    HARD_DEPTHS,
    HARD_DISTRACTORS_BY_DEPTH,
    HARD_HELD_OUT_PER_STRATUM,
    HARD_INTERNAL_EVAL_PER_STRATUM,
    HARD_N_PER_STRATUM,
    HARD_OFFICIAL_PER_STRATUM,
    HARD_PRESET,
    HARD_SEED_START,
    PUBLISHED_SEED,
    RESERVED_SEED_MAX,
    Preset,
    depth_label,
    generate_pool,
)
from whetstone_envs.c18.oracle import entailment_label
from whetstone_envs.c18.prompts import PROBES
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import Manifest, content_hash
from whetstone_envs.core.pool import TaskPool

_HARD_MANIFEST_PATH = Path(generate.__file__).with_name("manifest_hard.json")


# --- Preset identity: the hardest upstream configuration ------------------


def test_hard_preset_is_the_deep_distractor_configuration() -> None:
    # Depth pushed past the base D5 ceiling (D5, D8, D10), no hidden-info
    # change. Distractors ON is the base knob; the per-depth policy keeps
    # them ON at D5 and OFF at D8/D10 (upstream-infeasible there).
    assert HARD_DEPTHS == (5, 8, 10)
    assert HARD_PRESET.depths == (5, 8, 10)
    assert HARD_PRESET.distractors == "relevant"
    assert DEFAULT_DISTRACTORS == "relevant"
    assert HARD_PRESET.n_per_stratum == HARD_N_PER_STRATUM == 20
    assert HARD_PRESET.name == "hard"


def test_hard_distractor_policy_is_on_at_d5_off_at_deep_strata() -> None:
    # The one forced deviation from a uniform distractors-ON variant: the
    # pinned upstream generator can honor `relevant` distractors only up to
    # ~5 hops (deeper chains exhaust the fictional concept/property
    # vocabulary and never generate), so distractors are ON at D5 and `none`
    # at D8/D10, where the 8-/10-hop chain length is the hardness lever.
    assert HARD_PRESET.distractors_by_depth == HARD_DISTRACTORS_BY_DEPTH
    assert HARD_DISTRACTORS_BY_DEPTH == {5: "relevant", 8: "none", 10: "none"}
    # Every hard depth has an explicit policy entry (no silent fallback).
    assert set(HARD_DISTRACTORS_BY_DEPTH) == set(HARD_DEPTHS)
    # Distractors are ON at the deepest depth the generator can honor them.
    on_depths = [
        d for d, m in HARD_DISTRACTORS_BY_DEPTH.items() if m != "none"
    ]
    assert max(on_depths) == 5


# --- Determinism ----------------------------------------------------------
# Exercised through the preset mechanism at the base pool's shallow depths
# to avoid a second full deep-pool regeneration; the depth axis is only
# config, so the reproducibility property is identical at any depth.


def test_preset_regenerates_byte_identical() -> None:
    shallow = Preset(
        name="hard",
        depths=(1, 2),
        distractors=DEFAULT_DISTRACTORS,
        seed_start=HARD_SEED_START,
    )
    a = shallow.generate(n_per_stratum=2)
    b = shallow.generate(n_per_stratum=2)
    assert content_hash(a) == content_hash(b)
    assert [i.id for i in a.instances] == [i.id for i in b.instances]
    assert [i.gold for i in a.instances] == [i.gold for i in b.instances]


def test_committed_hard_manifest_matches_regenerated_pool() -> None:
    # The frozen manifest must still describe a freshly generated hard pool
    # (the regeneration diff check). This regenerates the FULL deep pool and
    # is the canonical guard that the committed hash is live. It is the ONE
    # full hard-pool regeneration; every other hard-preset property is
    # N-independent and runs at tiny N or off the committed manifest.
    pool = HARD_PRESET.generate()
    frozen = Manifest.read(_HARD_MANIFEST_PATH)
    assert frozen.matches_pool(pool)
    assert frozen.generator_version == "c18-generate-1+hard"


# --- Strata composition ---------------------------------------------------


def test_hard_manifest_declares_three_deep_strata() -> None:
    # Read from the frozen manifest so this check needs no generation.
    frozen = Manifest.read(_HARD_MANIFEST_PATH)
    assert frozen.stratum_counts == {
        depth_label(d): HARD_N_PER_STRATUM for d in HARD_DEPTHS
    }
    assert set(frozen.stratum_counts) == {"D5", "D8", "D10"}
    assert sum(frozen.stratum_counts.values()) == HARD_N_PER_STRATUM * 3


def test_preset_generate_labels_strata_by_depth() -> None:
    # Generate a tiny single-shallow-depth preset pool to confirm the preset
    # threads its depths through to the stratum labels (mechanism, not the
    # deep ceiling -- pinned statically above).
    shallow = Preset(
        name="hard",
        depths=(2,),
        distractors=DEFAULT_DISTRACTORS,
        seed_start=HARD_SEED_START,
    )
    pool = shallow.generate(n_per_stratum=1)
    assert set(pool.strata) == {"D2"}


def test_hard_split_sizes_partition_each_stratum_exactly() -> None:
    # internal 2 / official 6 / held_out 12 = 20 per stratum, no leftovers;
    # whole-pool totals (6, 18, 36) across the three depths.
    assert (
        HARD_INTERNAL_EVAL_PER_STRATUM
        + HARD_OFFICIAL_PER_STRATUM
        + HARD_HELD_OUT_PER_STRATUM
        == HARD_N_PER_STRATUM
    )

    pool = TaskPool(
        make_instance(
            id=f"hard-d{depth}",
            seed=HARD_SEED_START + index,
            strata=depth_label(depth),
        )
        for index, depth in enumerate(HARD_DEPTHS)
    )

    ie, off, ho = HARD_PRESET.default_split_sizes(pool)
    assert (ie, off, ho) == (6, 18, 36)
    assert ie + off + ho == HARD_N_PER_STRATUM * 3


# --- Contamination / seed disjointness ------------------------------------


def test_hard_seed_start_is_fresh_and_disjoint_from_base_and_published() -> (
    None
):
    # Strictly above the reserved published range and never the published
    # default seed, AND an order of magnitude above the base c18 fresh start
    # so a hard instance can never reuse a base-pool or published seed.
    assert HARD_SEED_START > RESERVED_SEED_MAX
    assert HARD_SEED_START != PUBLISHED_SEED
    assert HARD_SEED_START > DEFAULT_SEED_START
    # The base default pool consumes one seed per default depth from
    # DEFAULT_SEED_START; the hard window starts far above that block.
    assert HARD_SEED_START > DEFAULT_SEED_START + 100

    # Every seed the preset would consume (one per depth) is disjoint from
    # the base pool's consumed window.
    hard_seeds = {HARD_SEED_START + i for i in range(len(HARD_DEPTHS))}
    base_seeds = {
        DEFAULT_SEED_START + i for i in range(len(generate.DEFAULT_DEPTHS))
    }
    assert hard_seeds.isdisjoint(base_seeds)


def test_hard_preset_consumed_seeds_pass_the_freshness_guard() -> None:
    # The construction-time freshness assertion accepts the hard seed window
    # (it would raise if any consumed seed fell in the reserved range or hit
    # the published default). Uses a shallow single depth so no deep
    # subprocess runs; the freshness check is on the seed, not the depth.
    shallow = Preset(
        name="hard",
        depths=(1,),
        distractors=DEFAULT_DISTRACTORS,
        seed_start=HARD_SEED_START,
    )
    pool = shallow.generate(n_per_stratum=1)
    for inst in pool.instances:
        assert inst.seed >= HARD_SEED_START
        assert inst.seed > RESERVED_SEED_MAX
        assert inst.seed != PUBLISHED_SEED


# --- Prompts unchanged (same two probe templates, no gold leak) -----------


def test_probes_render_for_hard_instances_without_gold_leak() -> None:
    # A tiny shallow-depth preset pool renders through the SAME two probe
    # templates the base pool uses -- the hard variant changes only the pool,
    # never the prompt surface.
    shallow = Preset(
        name="hard",
        depths=(1,),
        distractors=DEFAULT_DISTRACTORS,
        seed_start=HARD_SEED_START,
    )
    pool = shallow.generate(n_per_stratum=1)
    for inst in pool.instances:
        naive = PROBES.render_naive(inst)
        ceiling = PROBES.render_ceiling(inst)
        assert inst.prompt_inputs["question"] in naive
        assert inst.prompt_inputs["query"] in naive
        assert inst.prompt_inputs["question"] in ceiling
        assert inst.prompt_inputs["query"] in ceiling
        assert len(ceiling) > len(naive)
        assert set(inst.prompt_inputs) == {"question", "query"}


# --- Oracle fixtures: DEEP chains + distractors, unchanged forward-chain ---
# Hand-constructed (NOT generator-produced) theories, each traced step by
# step in the comment beside it. These are the load-bearing checks that the
# forward-chaining fixpoint ORACLE handles deeper/distractor instances
# UNCHANGED -- its design property. Nothing in oracle.py special-cases depth
# or distractor count; these prove the fixpoint still converges correctly at
# hop depth 8 and 10 and is not fooled by distractor rules.

# D8: fact + 7 membership hops + 1 property rule, plus two distractor rules
# whose antecedent Sam never reaches (they never fire).
#   Sam is an aumpus;
#   aumpuses->bumpuses (1); bumpuses->cumpuses (2); cumpuses->dumpuses (3);
#   dumpuses->eumpuses (4); eumpuses->fumpuses (5); fumpuses->gumpuses (6);
#   gumpuses->humpuses (7); every humpus is metallic (8)  =>  Sam is metallic.
# The "zorpus is not metallic" / "lorpus is wooden" rules are inert (Sam is
# neither a zorpus nor a lorpus).
_D8 = (
    "Sam is an aumpus. "
    "Aumpuses are bumpuses. Bumpuses are cumpuses. Cumpuses are dumpuses. "
    "Dumpuses are eumpuses. Eumpuses are fumpuses. Fumpuses are gumpuses. "
    "Gumpuses are humpuses. Every humpus is metallic. "
    "Every zorpus is not metallic. Each lorpus is wooden."
)

# D10 WITH distractors: fact + 9 membership hops + 1 property rule, plus
# three distractor rules that never fire.
#   Polly is a jumpus;
#   jumpus->kumpus (1); kumpus->lumpus (2); lumpus->mumpus (3);
#   mumpus->numpus (4); numpus->oumpus (5); oumpus->pumpus (6);
#   pumpus->qumpus (7); qumpus->rumpus (8); rumpus->sumpus (9);
#   every sumpus is luminous (10)  =>  Polly is luminous.
# "tumpus is not luminous" mentions the queried property but never fires
# (Polly is no tumpus); "vumpus is quiet" / "wumpus is a xumpus" are inert.
_D10 = (
    "Polly is a jumpus. "
    "Jumpuses are kumpuses. Kumpuses are lumpuses. Lumpuses are mumpuses. "
    "Mumpuses are numpuses. Numpuses are oumpuses. Oumpuses are pumpuses. "
    "Pumpuses are qumpuses. Qumpuses are rumpuses. Rumpuses are sumpuses. "
    "Every sumpus is luminous. "
    "Every tumpus is not luminous. Vumpuses are quiet. "
    "Each wumpus is a xumpus."
)

# Distractor NOT-entailed: the chain derives 'metallic', but the QUERY asks
# about 'bitter'. A distractor rule "Every zorpus is bitter" MENTIONS the
# queried property -- a surface matcher might latch onto it -- yet Sam is not
# a zorpus, so the rule never fires and "Sam is bitter" is NOT entailed. The
# opposite polarity is undrivable too (no rule concludes "not bitter" for a
# kind Sam holds).
_DISTRACTOR_NOT_ENTAILED = (
    "Sam is an aumpus. Aumpuses are bumpuses. Bumpuses are cumpuses. "
    "Every cumpus is metallic. "
    "Every zorpus is bitter. Each dorpus is not bitter."
)


def test_d8_deep_chain_both_polarities() -> None:
    assert entailment_label(_D8, "True or false: Sam is metallic.") == "True"
    assert (
        entailment_label(_D8, "True or false: Sam is not metallic.") == "False"
    )


def test_d8_inert_distractor_rules_do_not_fire() -> None:
    # The distractor "every zorpus is not metallic" must not flip the verdict
    # and "each lorpus is wooden" must not make Sam wooden.
    assert entailment_label(_D8, "True or false: Sam is wooden.") == "False"


def test_d10_deep_chain_with_distractors_both_polarities() -> None:
    lum = "True or false: Polly is luminous."
    assert entailment_label(_D10, lum) == "True"
    not_lum = "True or false: Polly is not luminous."
    assert entailment_label(_D10, not_lum) == "False"


def test_d10_distractor_mentioning_property_does_not_fire() -> None:
    # "Every tumpus is not luminous" mentions 'luminous' but Polly is no
    # tumpus; the negated-property distractor must not be derivable.
    assert entailment_label(_D10, "True or false: Polly is quiet.") == "False"


def test_distractor_query_not_entailed_despite_suggestive_rule() -> None:
    # The chain entails 'metallic' (sanity), but the queried 'bitter' is NOT
    # entailed even though a distractor rule names it -- the exact failure a
    # distractor is meant to induce in a surface-matching solver.
    assert (
        entailment_label(
            _DISTRACTOR_NOT_ENTAILED, "True or false: Sam is metallic."
        )
        == "True"
    )
    assert (
        entailment_label(
            _DISTRACTOR_NOT_ENTAILED, "True or false: Sam is bitter."
        )
        == "False"
    )
    assert (
        entailment_label(
            _DISTRACTOR_NOT_ENTAILED, "True or false: Sam is not bitter."
        )
        == "False"
    )


def test_hard_pool_gold_agrees_with_oracle_on_a_deep_stratum() -> None:
    # End-to-end at real depth D8 (distractors `none`, as the hard policy
    # uses there): a single generated hard-depth instance's frozen gold must
    # equal the independent oracle's re-derivation from its public text --
    # the cross-check the generator asserts, re-verified here at a depth past
    # the base ceiling. One instance keeps the subprocess cost bounded.
    # (D8 WITH relevant distractors is upstream-infeasible and never
    # generates; that ceiling is why the hard policy drops distractors here.)
    pool = generate_pool(
        n_per_stratum=1,
        depths=(8,),
        distractors=HARD_DISTRACTORS_BY_DEPTH[8],
        seed_start=HARD_SEED_START + 1,
    )
    (inst,) = pool.instances
    derived = oracle.entailment_label(
        inst.prompt_inputs["question"],
        inst.prompt_inputs["query"],
    )
    assert derived == inst.gold
