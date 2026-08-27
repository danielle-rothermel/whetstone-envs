"""The committed protocol is the pre-registration, so it is golden-tested.

Every assertion here is about a value the study fixes before it spends. A
change to one of these literals is a change to the design, and it should
cost a failing test and a deliberate edit rather than passing silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest
from dr_providers import ReasoningEffort

pytest.importorskip("whetstone.experiment.env")

from pathlib import Path

from whetstone_envs.optim.c18_experiment import C18_PROTOCOL_SPLIT_SIZES
from whetstone_envs.optim.study.manifest import CODEX_AGENT_OMITTED
from whetstone_envs.optim.study.protocols import (
    C18_TASK_REASONING_EFFORT,
    CODEX_AGENT_MODEL,
    GEPA_MAX_METRIC_CALLS,
    MIPROV2_MINIBATCH_SIZE,
    MIPROV2_NUM_CANDIDATES,
    MIPROV2_NUM_TRIALS,
    PROPOSER_MODEL,
    PROTOCOL_DOC_PATH,
    PROTOCOL_DOC_SHA256,
    PROTOCOL_IDS,
    SIZED_FIELDS,
    STEP10_C18,
    STEP10_C18_TOY,
    STEP10_C19,
    STEP10_C19_TOY,
    TASK_MODEL,
    TASK_REASONING_EFFORT,
    StudyProtocol,
    protocol_doc_sha256,
    study_protocol,
    without_codex,
)
from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    FIDELITY_ARM_IDS,
    HOLM_FAMILY_SIZE,
    K_RUN_C18,
    PROTOCOL_SPLIT_SIZES,
    PROTOCOL_TRAIN_SIZE,
    PROTOCOL_VAL_SIZE,
    REAL_OPTIMIZER_ARM_IDS,
    ArmKind,
    StageId,
)

# --------------------------------------------------------------------------
# The pinned design
# --------------------------------------------------------------------------


def test_the_real_protocol_pins_the_protocol_documents_values() -> None:
    """Every design literal the protocol document fixes, in one place."""
    assert STEP10_C19.split_sizes == (88, 132, 440) == PROTOCOL_SPLIT_SIZES
    assert STEP10_C19.train_size == 44 == PROTOCOL_TRAIN_SIZE
    assert STEP10_C19.val_size == 44 == PROTOCOL_VAL_SIZE
    assert STEP10_C19.n_per_stratum == 32
    assert STEP10_C19.pool_seed_start == 1_000_000
    assert STEP10_C19.family == "c19"


def test_the_population_is_described_by_the_family_that_generated_it() -> None:
    """The generator version is the pool manifest's, not the protocol's.

    A protocol that restated it could name a version the generator no
    longer produces, and the manifest would record the claim rather than
    the fact.
    """
    from whetstone_envs.optim.families import family_spec

    manifest = family_spec(STEP10_C19.family).pool_manifest(
        n_per_stratum=STEP10_C19.n_per_stratum,
        seed_start=STEP10_C19.pool_seed_start,
    )
    assert manifest.generator_version == "c19-custom-v2"
    assert len(manifest.stratum_counts) == 22


def test_the_c18_protocol_pins_section_4_1s_values() -> None:
    """Every value section 4.1 fixes for the second family, as literals.

    Pinned here rather than merely read from the c18 adapter, because the
    adapter's ``DEFAULT_CONFIG`` is a *generation* default that a family
    change could legitimately move. These are what the study
    pre-registered, and a generator change that moved them should fail
    loudly rather than silently resize the study.
    """
    assert STEP10_C18.family == "c18"
    assert STEP10_C18.split_sizes == (24, 48, 48) == C18_PROTOCOL_SPLIT_SIZES
    assert STEP10_C18.n_per_stratum == 30
    assert STEP10_C18.pool_seed_start == 1_000_000_000
    # 12/12 is the even halving of the internal 24, which is what GEPA
    # requires train+val to cover exactly.
    assert (STEP10_C18.train_size, STEP10_C18.val_size) == (12, 12)
    assert STEP10_C18.train_size + STEP10_C18.val_size == 24


def test_the_c18_protocol_runs_every_arm_once() -> None:
    """``K_RUN = 1`` for every arm, at every stage that runs arms.

    Section 4.1 gives c18 no power analysis, so there is no pilot for a
    later stage to extend: one run per optimizer is the whole design. The
    single-run arms are single-run for their own reason and agree.
    """
    assert K_RUN_C18 == 1
    for stage in (StageId.STAGE1, StageId.STAGE2):
        counts = {
            arm.arm_id: arm.k_run for arm in STEP10_C18.arm_specs(stage=stage)
        }
        assert set(counts.values()) == {1}, stage
        assert len(counts) == 8, stage


def test_the_c18_protocol_does_not_minibatch() -> None:
    """C18's internal split is smaller than the pinned minibatch.

    24 < 35, so a batched c18 MIPROv2 arm would draw a larger batch than
    the validation split holds -- ``configure_miprov2`` refuses it. The
    design is pre-registered unbatched rather than at a resized batch,
    which would make the two families' MIPROv2 arms different searches.
    """
    assert STEP10_C18.miprov2_minibatch is False
    assert STEP10_C19.miprov2_minibatch is True
    assert STEP10_C18.internal_size < MIPROV2_MINIBATCH_SIZE
    for arm in STEP10_C18.arms:
        if arm.optimizer == "miprov2":
            assert arm.miprov2_minibatch is False
            # An unbatched arm carries no batch size: recording one would
            # pin a number no run reads into the design hash.
            assert arm.miprov2_minibatch_size is None


def test_the_two_protocols_share_every_unpopulation_pin() -> None:
    """C3's mechanical content: the same machine, a different family.

    Everything the generality claim says is *unchanged* between the two
    families is compared directly. A c18 study that ran a different task
    model, a different proposer, a different Codex agent or cap, or
    different COPRO/GEPA/MIPROv2 control shapes would not be evidence that
    the machinery carried -- it would be a second study.

    ``task_reasoning_effort`` is deliberately **not** in this list, since
    item 23. It is one of the two controls section 4.1 now pre-registers
    differently for the second family, and
    ``test_the_c18_protocol_pins_a_lower_reasoning_effort`` asserts the
    divergence directly rather than leaving it to a hole here.
    """
    shared = (
        "task_model",
        "proposer_model",
        "codex_agent_model",
        "temperature",
        "provider",
        "seed_control",
        "protocol_doc_path",
        "copro_breadth",
        "copro_depth",
        "gepa_max_metric_calls",
        "gepa_reflection_minibatch_size",
        "codex_evaluate_call_cap",
        "miprov2_num_trials",
        "miprov2_num_candidates",
    )
    for name in shared:
        assert getattr(STEP10_C18, name) == getattr(STEP10_C19, name), name
    # And the same eight arms, in the same order, on the same optimizers.
    assert tuple((a.arm_id, a.optimizer, a.kind) for a in STEP10_C18.arms) == (
        tuple((a.arm_id, a.optimizer, a.kind) for a in STEP10_C19.arms)
    )


def test_both_protocols_reference_the_same_frozen_document() -> None:
    """Section 4.1 is part of the c19 document, so c18 cites it.

    A second document would be a second pre-registration, and the digest
    that pins the text would no longer cover the section the c18 study
    runs under.
    """
    assert STEP10_C18.protocol_doc_path == STEP10_C19.protocol_doc_path
    assert (
        protocol_doc_sha256(Path(STEP10_C18.protocol_doc_path))
        == PROTOCOL_DOC_SHA256
    )


def test_the_real_protocol_pins_its_models() -> None:
    assert STEP10_C19.task_model == TASK_MODEL == "openai/gpt-5-nano"
    assert STEP10_C19.proposer_model == PROPOSER_MODEL == "openai/gpt-5.4-nano"
    assert STEP10_C19.codex_agent_model == CODEX_AGENT_MODEL == "gpt-5.6-sol"


def test_the_protocol_pins_the_task_models_reasoning_effort() -> None:
    """The pinned effort, as a literal.

    A reasoning effort is design rather than an invocation setting: it sets
    the task model's capability, so an effort chosen after Stage 0 measured
    the anchors would change the treatment under a pre-registration that
    had already named it. Pinned as a literal here so the value cannot
    drift silently -- ``low`` is what Danielle ratified, and a change to it
    is a change to the study.

    ``minimal`` is historical for c19 (item 19). Its Stage-0 probe failed
    the gate on capability rather than power -- ceiling 0.1977 against the
    0.30 floor -- so item 21 re-pinned the effort to ``low``.

    Since item 23 this constant is c19's pin rather than the study-wide
    one; c18 reads ``C18_TASK_REASONING_EFFORT``.
    """
    assert TASK_REASONING_EFFORT is ReasoningEffort.LOW
    assert TASK_REASONING_EFFORT.value == "low"
    assert STEP10_C19.task_reasoning_effort is TASK_REASONING_EFFORT


def test_the_c18_protocol_pins_a_lower_reasoning_effort() -> None:
    """C18's pin, as a literal, and the divergence it encodes (item 23).

    ``minimal`` is what Danielle ratified for this family after its Stage-0
    gate failed **by saturation** at c19's ``low``: the naive anchor scored
    0.9375 and the ceiling anchor scored 0.9375 on the same 48-task
    held-out split, leaving 0.0000 headroom against the 0.20 minimum. A
    naive prompt that already ties the ceiling bounds every arm's
    improvement at zero, so the design could not measure what it exists to
    measure, and escalating effort only pushes the naive anchor further in.

    Pinned as a literal here for the same reason c19's is: a change to it
    is a change to the study. ``low`` is historical for c18 -- it names the
    design attempt 1's gate refused.
    """
    assert C18_TASK_REASONING_EFFORT is ReasoningEffort.MINIMAL
    assert C18_TASK_REASONING_EFFORT.value == "minimal"
    assert STEP10_C18.task_reasoning_effort is C18_TASK_REASONING_EFFORT
    # The divergence itself, asserted rather than merely implied: the two
    # families measure the same machinery at different capability rungs.
    assert STEP10_C18.task_reasoning_effort != STEP10_C19.task_reasoning_effort


def test_each_toy_runs_at_its_own_familys_reasoning_effort() -> None:
    """The effort is not a sized field, and both toys prove it.

    A toy that ran at a different effort than the design it stands in for
    would be exercising a different task model -- which is precisely what
    ``SIZED_FIELDS`` exists to bound. The guard is *within* a pair: since
    item 23 the two families differ here by design, so comparing across
    them would assert the opposite of the pre-registration.
    """
    assert (
        STEP10_C19_TOY.task_reasoning_effort
        == STEP10_C19.task_reasoning_effort
        == ReasoningEffort.LOW
    )
    assert (
        STEP10_C18_TOY.task_reasoning_effort
        == STEP10_C18.task_reasoning_effort
        == ReasoningEffort.MINIMAL
    )
    assert "task_reasoning_effort" not in SIZED_FIELDS


def test_the_real_protocol_pins_its_control_shapes() -> None:
    assert STEP10_C19.gepa_max_metric_calls == GEPA_MAX_METRIC_CALLS == 200
    assert STEP10_C19.codex_evaluate_call_cap == CODEX_EVALUATE_CALL_CAP == 8
    assert STEP10_C19.miprov2_num_trials == MIPROV2_NUM_TRIALS == 10
    assert STEP10_C19.miprov2_num_candidates == MIPROV2_NUM_CANDIDATES == 3
    assert STEP10_C19.miprov2_minibatch_size == MIPROV2_MINIBATCH_SIZE == 35


def test_the_miprov2_candidate_count_is_at_the_runners_minibatch_floor() -> (
    None
):
    """The pin the import cycle stopped `protocols` from reading directly.

    Two candidates with minibatching exhausts MIPROv2's search space and
    raises inside the durable run boundary on releases before the fix
    (note 25d), so the runner refuses anything below its floor. The design
    states the number as a literal because the runner imports the study
    package for its spend record; this is the check that literal is for.
    """
    from whetstone_envs.optim.run import MIPROV2_MINIBATCH_MIN_CANDIDATES

    assert MIPROV2_NUM_CANDIDATES == MIPROV2_MINIBATCH_MIN_CANDIDATES


def test_the_protocol_declares_every_pre_registered_arm() -> None:
    """The arm list is design: six efficacy-or-control arms plus two
    MIPROv2 fidelity modes."""
    assert tuple(arm.arm_id for arm in STEP10_C19.arms) == (
        "copro",
        "miprov2",
        "miprov2-zeroshot",
        "miprov2-ground_only",
        "gepa",
        "codex",
        "null-random",
        "null-identity",
    )


def test_only_the_fewshot_miprov2_arm_is_an_efficacy_arm() -> None:
    """R2: which demo mode carries the claim is pre-registered.

    The two fidelity modes run once each; promoting one of them into the
    efficacy slot after seeing results is selection on held-out through the
    back door, so their run count is fixed here rather than later.
    """
    by_id = {arm.arm_id: arm for arm in STEP10_C19.arms}
    assert by_id["miprov2"].demo_mode == "fewshot"
    assert by_id["miprov2-zeroshot"].demo_mode == "zeroshot"
    assert by_id["miprov2-ground_only"].demo_mode == "ground_only"


def test_gepa_and_miprov2_carry_the_train_val_partition() -> None:
    """Note 18: the optimizers with the concept get an explicit split."""
    by_id = {arm.arm_id: arm for arm in STEP10_C19.arms}
    for arm_id in (
        "miprov2",
        "miprov2-zeroshot",
        "miprov2-ground_only",
        "gepa",
    ):
        assert by_id[arm_id].train_size == 44, arm_id
        assert by_id[arm_id].val_size == 44, arm_id
    for arm_id in ("copro", "codex", "null-random", "null-identity"):
        assert by_id[arm_id].train_size is None, arm_id
        assert by_id[arm_id].val_size is None, arm_id


# --------------------------------------------------------------------------
# The toy is the same protocol
# --------------------------------------------------------------------------


def _arm_shape(
    protocol: StudyProtocol,
) -> tuple[tuple[object, ...], ...]:
    """Each arm's design, with the sized partition fields removed.

    ``train_size``, ``val_size``, and the minibatch size are sized by
    construction; everything else about an arm is design the toy must
    share.
    """
    return tuple(
        (
            arm.arm_id,
            arm.optimizer,
            arm.kind,
            arm.demo_mode,
            arm.miprov2_num_trials,
            arm.miprov2_num_candidates,
            arm.miprov2_minibatch,
            arm.train_size is None,
            arm.val_size is None,
            arm.miprov2_minibatch_size is None,
        )
        for arm in protocol.arms
    )


#: Every registered protocol beside its own toy.
#:
#: The sized-field guards run **within** a pair, never across two. c19 and
#: c18 are different designs -- different populations, splits, run counts
#: and minibatch settings -- so comparing them under ``SIZED_FIELDS`` would
#: assert a sameness the pre-registration deliberately does not claim. What
#: each guard checks is that a *toy* rehearses the study it stands in for.
PROTOCOL_PAIRS = (
    pytest.param(STEP10_C19, STEP10_C19_TOY, id="c19"),
    pytest.param(STEP10_C18, STEP10_C18_TOY, id="c18"),
)


@pytest.mark.parametrize(("real", "toy"), PROTOCOL_PAIRS)
def test_the_toy_differs_from_the_real_design_only_in_sized_fields(
    real: StudyProtocol, toy: StudyProtocol
) -> None:
    """The load-bearing guard: a toy cannot rehearse a different study.

    Anything the toy is allowed to differ on is named in ``SIZED_FIELDS``.
    Every other field is compared directly, so a value that drifts between
    the two -- a model, a control pin, the correction, an arm -- fails
    here rather than silently making the tests measure a design the real
    study does not run.

    Run for each protocol against its own toy. A c18 toy that batched when
    the c18 study does not, or that ran its arms a different number of
    times, would be rehearsing a design the second family never registered
    -- and would fail here on ``miprov2_minibatch`` or ``design_k_run``,
    neither of which is a sized field.
    """
    unsized = [
        field.name
        for field in fields(StudyProtocol)
        if field.name not in SIZED_FIELDS and field.name != "arms"
    ]
    for name in unsized:
        assert getattr(real, name) == getattr(toy, name), name
    # ``arms`` carries the sized train/val and minibatch inside it, so it
    # is compared on the shape that is *not* sized: which arms exist, what
    # each one optimizes, and every flag that is design rather than size.
    assert _arm_shape(real) == _arm_shape(toy)


@pytest.mark.parametrize(("real", "toy"), PROTOCOL_PAIRS)
def test_every_sized_field_actually_differs(
    real: StudyProtocol, toy: StudyProtocol
) -> None:
    """A field licensed to differ that does not is licence without use."""
    real_values = real.sized_values()
    toy_values = toy.sized_values()
    assert set(real_values) == set(SIZED_FIELDS)
    for name in SIZED_FIELDS:
        assert real_values[name] != toy_values[name], name


@pytest.mark.parametrize(("real", "toy"), PROTOCOL_PAIRS)
def test_the_toy_is_sized_for_a_test(
    real: StudyProtocol, toy: StudyProtocol
) -> None:
    """Both toys shrink to the same test-affordable size.

    The toy sizes are a property of what a unit test can run, not of the
    family, so the two toys agree here while the two studies do not.
    """
    assert toy.split_sizes == (4, 4, 6)
    assert (toy.train_size, toy.val_size) == (2, 2)
    assert real.split_sizes != toy.split_sizes


def test_study_protocol_selects_the_variant_of_one_protocol() -> None:
    assert study_protocol("step10-c19") is STEP10_C19
    assert study_protocol("step10-c19", toy=True) is STEP10_C19_TOY
    assert study_protocol("step10-c18") is STEP10_C18
    assert study_protocol("step10-c18", toy=True) is STEP10_C18_TOY
    with pytest.raises(ValueError, match="unknown protocol"):
        study_protocol("step10-c17")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_train_val_partition_that_misses_the_internal_split_is_refused() -> (
    None
):
    """GEPA requires train + val to cover internal exactly, so a protocol
    that declared otherwise would be refused on its GEPA arm's turn."""
    with pytest.raises(ValueError, match="must equal the internal split"):
        replace(STEP10_C19, train_size=40)


def test_a_minibatch_larger_than_the_validation_split_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds the validation split"):
        replace(STEP10_C19, miprov2_minibatch_size=45)


# --------------------------------------------------------------------------
# The document digest
# --------------------------------------------------------------------------


def test_the_protocol_document_digest_is_read_from_the_file(
    tmp_path: Path,
) -> None:
    """Computed, never pinned: the manifest names the revision in force."""
    doc = tmp_path / "protocol.md"
    doc.write_bytes(b"# protocol\n")
    assert (
        protocol_doc_sha256(doc) == hashlib.sha256(b"# protocol\n").hexdigest()
    )


def test_a_missing_protocol_document_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the protocol document"):
        protocol_doc_sha256(tmp_path / "absent.md")


def test_the_protocol_names_the_document_it_ships() -> None:
    """The registered text is in the package, and it is readable here.

    A default pointing into one machine's durable notes made ``init``
    unusable from any other checkout, because ``init`` hashes this file and
    refuses to author a study without it. The document ships in the
    package, so the digest a manifest records is checkable by whoever reads
    the manifest.
    """
    assert STEP10_C19.protocol_doc_path == PROTOCOL_DOC_PATH
    assert PROTOCOL_DOC_PATH.endswith("step10-c19-protocol.md")
    assert Path(PROTOCOL_DOC_PATH).is_file()


def test_the_shipped_document_still_hashes_to_the_registered_digest() -> None:
    """The golden that makes the in-repo copy a pre-registration.

    ``PROTOCOL_DOC_SHA256`` is the digest of the text the study was
    registered on, byte-identical to the durable authoring copy. A
    pre-registration whose text could change without anything failing
    would pre-register nothing.
    """
    assert protocol_doc_sha256(Path(PROTOCOL_DOC_PATH)) == PROTOCOL_DOC_SHA256


# --------------------------------------------------------------------------
# The per-family effort pin moves exactly one design hash
# --------------------------------------------------------------------------


def _registered_design_hash(protocol: StudyProtocol) -> str:
    """The pre-registration hash of ``protocol`` as registered.

    Built through the same chain a real study takes -- protocol to
    manifest to spec to hash -- rather than by re-listing the payload here,
    so a change to how a study derives its pinning is caught by these
    goldens instead of being reproduced by them. The three arguments the
    hash takes that a protocol does not carry (the correction rule, its
    family size, and the completeness backstop) are the same for every
    registration and are passed as the study's pinned values.
    """
    from whetstone_envs.optim.study import stages
    from whetstone_envs.optim.study.init import study_manifest_for
    from whetstone_envs.optim.study.manifest import (
        COMPLETENESS_BACKSTOP,
        CORRECTION_FAMILY_SIZE,
        CORRECTION_HOLM_BONFERRONI,
        pre_registration_design_hash,
    )
    from whetstone_envs.optim.study.spec import spec_from_manifest

    spec = spec_from_manifest(study_manifest_for(protocol))
    return pre_registration_design_hash(
        k_repeat=spec.k_repeat,
        k_run_by_arm={arm.arm_id: arm.k_run for arm in spec.arms},
        kind_by_arm={arm.arm_id: arm.kind.value for arm in spec.arms},
        split_by_arm=stages._split_by_arm(spec),
        minibatch_by_arm=stages._minibatch_by_arm(spec),
        search_by_arm=stages._search_by_arm(spec),
        task_reasoning_effort=spec.task_reasoning_effort,
        ci_level=spec.ci_level,
        resamples=spec.resamples,
        bootstrap_seed=spec.bootstrap_seed,
        correction=CORRECTION_HOLM_BONFERRONI,
        m=CORRECTION_FAMILY_SIZE,
        completeness_backstop=COMPLETENESS_BACKSTOP,
    )


def test_the_c19_design_hash_is_unchanged_by_the_c18_effort_pin() -> None:
    """C19's pre-registration survives item 23 byte for byte.

    Item 23 makes the task-route reasoning effort a per-family pin so c18
    can drop to ``minimal``. The effort is hashed into the design, so the
    change had one way to go wrong that no other test here would catch: if
    the builder's new parameter reached c19's registrations with any value
    other than ``TASK_REASONING_EFFORT`` -- or if making it a parameter
    perturbed the payload -- c19's design hash would move, and the running
    c19 study's manifest would no longer validate against the design it
    pre-registered.

    The literals are the hashes ``main`` produced at v0.2.13, before this
    change. They are pinned rather than recomputed from both sides,
    because a test that compared the code against itself would pass no
    matter what the code did.
    """
    assert _registered_design_hash(STEP10_C19) == (
        "0aaf21e9ef74ffe03ebe1e3131f7f5ae90752815c92cebb51e4bba7fcc31586e"
    )
    assert _registered_design_hash(STEP10_C19_TOY) == (
        "0bac099ca90b29d2bf5ce7885d8a6a13cd46e272ff6a94fcdc2a04b754da5897"
    )


def test_the_c18_design_hash_moved_with_its_effort_pin() -> None:
    """C18's pre-registration is a *new* design, and says so.

    The other half of item 23: re-pinning the effort is a design change, so
    c18's hash must not be what it was at ``low``. Both literals are
    pinned -- the current one, and the superseded one it must not equal --
    because "the hash changed" and "the hash changed to the intended value"
    are different claims, and only the second one makes a c18 study
    initialised under this code checkable against its manifest.

    ``acc31b8f...`` and ``6f29ec90...`` are historical: they name the c18
    designs run at ``low``, which attempt 1's Stage-0 gate refused by
    saturation. No c18 result should be carried forward under them.
    """
    at_low_real = (
        "acc31b8f234f6ca9dd5c8611b73630bdff6decb02ab3399fcd74ed21dccc8e9d"
    )
    at_low_toy = (
        "6f29ec90a1e9371b2758f6cc44883441ad98fd468018358c46e204d818c4e465"
    )
    real = _registered_design_hash(STEP10_C18)
    toy = _registered_design_hash(STEP10_C18_TOY)
    assert real == (
        "ba939591324c5644b97eebfdab14e3c22c5f9efb106ef1b774dfa150ffc6ccc3"
    )
    assert toy == (
        "4d919b79a4eb1a51eef14b6accb89062e7e70e6939eaae3abbe18fc559b62a4c"
    )
    assert real != at_low_real
    assert toy != at_low_toy
    # And the two families still cannot be confused for one another.
    assert real != _registered_design_hash(STEP10_C19)


# --------------------------------------------------------------------------
# The codex-free projection
# --------------------------------------------------------------------------


#: What a projection is permitted to differ from the design on: its arm
#: list, and the two fields that exist so it cannot be mistaken for the
#: design it projects.
_PROJECTION_FIELDS = frozenset({"arms", "study_id", "codex_agent_model"})


def test_without_codex_drops_exactly_the_codex_arm() -> None:
    """A fake-transport rehearsal drops the billed arm and nothing else."""
    rehearsal = without_codex(STEP10_C19)
    assert all(arm.optimizer != "codex" for arm in rehearsal.arms)
    assert len(rehearsal.arms) == len(STEP10_C19.arms) - 1
    for field in fields(StudyProtocol):
        if field.name not in _PROJECTION_FIELDS:
            assert getattr(rehearsal, field.name) == getattr(
                STEP10_C19, field.name
            ), field.name


def test_the_projection_cannot_be_mistaken_for_the_design() -> None:
    """A rehearsal names itself, on every axis a reader checks.

    A ``--without-codex`` manifest was byte-indistinguishable from the
    pre-registration on ``study_id`` and on the ``models`` block, so its
    artifacts could land in the study's directory and its numbers could be
    cited as the study's. Both now say what they are.
    """
    rehearsal = without_codex(STEP10_C19)
    assert rehearsal.study_id == f"{STEP10_C19.study_id}-without-codex"
    assert rehearsal.study_id not in PROTOCOL_IDS
    assert rehearsal.codex_agent_model == CODEX_AGENT_OMITTED
    assert rehearsal.codex_agent_model != STEP10_C19.codex_agent_model


# --------------------------------------------------------------------------
# Arms cross into runnable specs
# --------------------------------------------------------------------------


def test_every_arm_becomes_a_runnable_spec() -> None:
    """``ArmSpec`` is where a design that cannot run is refused, so the
    protocol is validated by crossing into it."""
    specs = STEP10_C19.arm_specs(stage=StageId.STAGE2)
    assert len(specs) == len(STEP10_C19.arms)
    by_id = {spec.arm_id: spec for spec in specs}
    assert by_id["miprov2"].miprov2_minibatch is True
    assert by_id["miprov2"].miprov2_minibatch_size == 35
    assert by_id["gepa"].train_size == 44


def test_fidelity_modes_run_once_and_the_efficacy_arm_five_times() -> None:
    """Protocol 5.3: ``zeroshot`` and ``ground_only`` are fidelity
    evidence at ``K_RUN = 1``, not efficacy arms at the full count."""
    by_id = {
        spec.arm_id: spec
        for spec in STEP10_C19.arm_specs(stage=StageId.STAGE2)
    }
    assert by_id["miprov2"].k_run == 5
    assert by_id["miprov2-zeroshot"].k_run == 1
    assert by_id["miprov2-ground_only"].k_run == 1


def test_no_two_arms_share_a_seed() -> None:
    """Disjoint seeds across arms, so no two arms share an RNG stream.

    The three MIPROv2 arms run the same optimizer, so a seed table keyed
    only by optimizer would hand all three the same seeds and the study
    would report three arms that were never independent.
    """
    specs = STEP10_C19.arm_specs(stage=StageId.STAGE2)
    seen: dict[int, str] = {}
    for spec in specs:
        for seed in spec.seeds:
            assert seed not in seen, (
                f"{spec.arm_id} reuses seed {seed} already held by "
                f"{seen.get(seed)}"
            )
            seen[seed] = spec.arm_id


def test_the_holm_family_is_exactly_the_four_real_optimizers() -> None:
    """The correction family is a structural fact, not a coincidence.

    ``HOLM_FAMILY_SIZE`` is pre-registered at four, and the arms carrying
    ``ArmKind.REAL`` are what the analysis corrects together. Adding a fifth
    efficacy arm -- or promoting a fidelity arm into the family -- would
    make the pre-registered family size a lie, so the two are pinned against
    each other here rather than each asserted alone.
    """
    real = [arm for arm in STEP10_C19.arms if arm.kind is ArmKind.REAL]
    assert len(real) == HOLM_FAMILY_SIZE
    assert [arm.arm_id for arm in real] == list(REAL_OPTIMIZER_ARM_IDS)


def test_miprov2s_non_efficacy_modes_are_fidelity_arms() -> None:
    """The two extra MIPROv2 modes carry no efficacy claim.

    They pass their audits and are measured, which is exactly the shape
    that earned an unearned ``validated`` verdict before the role existed.
    """
    by_id = {arm.arm_id: arm for arm in STEP10_C19.arms}
    for arm_id in FIDELITY_ARM_IDS:
        assert by_id[arm_id].kind is ArmKind.FIDELITY
    assert by_id["miprov2"].kind is ArmKind.REAL


def test_every_miprov2_mode_registers_the_same_trial_count() -> None:
    """All three demo modes run at 10 trials, per revision 2 of the protocol.

    Revision 1 derived per-mode counts (10 for ``fewshot``, 9 for the other
    two) from DSPy's ``_recommended_num_trials`` at six candidates. Both
    halves of that derivation are gone: the design pins three candidates,
    where the same formula gives 7 and 5, and the study sets ``num_trials``
    on the control directly, so auto-mode never runs. A count the code does
    not derive is not a pre-registration, so the protocol registers one
    number for all three modes.
    """
    miprov2_arms = [
        arm for arm in STEP10_C19.arms if arm.optimizer == "miprov2"
    ]
    assert len(miprov2_arms) == 3
    assert {arm.demo_mode for arm in miprov2_arms} == {
        "fewshot",
        "zeroshot",
        "ground_only",
    }
    assert {arm.miprov2_num_trials for arm in miprov2_arms} == {
        MIPROV2_NUM_TRIALS
    }
    assert {arm.miprov2_num_candidates for arm in miprov2_arms} == {
        MIPROV2_NUM_CANDIDATES
    }
