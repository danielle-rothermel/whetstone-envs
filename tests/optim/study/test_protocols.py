"""The committed protocol is the pre-registration, so it is golden-tested.

Every assertion here is about a value the study fixes before it spends. A
change to one of these literals is a change to the design, and it should
cost a failing test and a deliberate edit rather than passing silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest

pytest.importorskip("whetstone.experiment.env")

from pathlib import Path

from whetstone_envs.optim.study.manifest import CODEX_AGENT_OMITTED
from whetstone_envs.optim.study.protocols import (
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
    STEP10_C19,
    STEP10_C19_TOY,
    TASK_MODEL,
    StudyProtocol,
    protocol_doc_sha256,
    study_protocol,
    without_codex,
)
from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    PROTOCOL_SPLIT_SIZES,
    PROTOCOL_TRAIN_SIZE,
    PROTOCOL_VAL_SIZE,
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


def test_the_real_protocol_pins_its_models() -> None:
    assert STEP10_C19.task_model == TASK_MODEL == "openai/gpt-5-nano"
    assert STEP10_C19.proposer_model == PROPOSER_MODEL == "openai/gpt-5.4-nano"
    assert STEP10_C19.codex_agent_model == CODEX_AGENT_MODEL == "gpt-5.6-sol"


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


def test_the_toy_differs_from_the_real_design_only_in_sized_fields() -> None:
    """The load-bearing guard: a toy cannot rehearse a different study.

    Anything the toy is allowed to differ on is named in ``SIZED_FIELDS``.
    Every other field is compared directly, so a value that drifts between
    the two -- a model, a control pin, the correction, an arm -- fails
    here rather than silently making the tests measure a design the real
    study does not run.
    """
    unsized = [
        field.name
        for field in fields(StudyProtocol)
        if field.name not in SIZED_FIELDS and field.name != "arms"
    ]
    for name in unsized:
        assert getattr(STEP10_C19, name) == getattr(STEP10_C19_TOY, name), name
    # ``arms`` carries the sized train/val and minibatch inside it, so it
    # is compared on the shape that is *not* sized: which arms exist, what
    # each one optimizes, and every flag that is design rather than size.
    assert _arm_shape(STEP10_C19) == _arm_shape(STEP10_C19_TOY)


def test_every_sized_field_actually_differs() -> None:
    """A field licensed to differ that does not is licence without use."""
    real = STEP10_C19.sized_values()
    toy = STEP10_C19_TOY.sized_values()
    assert set(real) == set(SIZED_FIELDS)
    for name in SIZED_FIELDS:
        assert real[name] != toy[name], name


def test_the_toy_is_sized_for_a_test() -> None:
    assert STEP10_C19_TOY.split_sizes == (4, 4, 6)
    assert (STEP10_C19_TOY.train_size, STEP10_C19_TOY.val_size) == (2, 2)


def test_study_protocol_selects_the_variant_of_one_protocol() -> None:
    assert study_protocol("step10-c19") is STEP10_C19
    assert study_protocol("step10-c19", toy=True) is STEP10_C19_TOY
    with pytest.raises(ValueError, match="unknown protocol"):
        study_protocol("step10-c18")


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
