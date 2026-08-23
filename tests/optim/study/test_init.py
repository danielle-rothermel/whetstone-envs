"""``whetstone-study init`` authors the pre-registration, so it is checked.

The manifest ``init`` writes is what every later stage reads its design
from, and three of its fields are *recomputed* rather than declared -- the
split task hashes, the pool manifest hash, and the protocol document's
digest. Those are the ones a test has to hold, because a wrong task hash
is not caught until a stage binds an engine, and a stale document digest
is never caught at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.study.init import (
    PENDING_EVAL_CONFIG_HASH,
    init_study,
    study_manifest_for,
)
from whetstone_envs.optim.study.manifest import (
    STUDY_MANIFEST_SCHEMA,
    ManifestExistsError,
    read_study_manifest,
)
from whetstone_envs.optim.study.protocols import (
    StudyProtocol,
    study_protocol,
    without_codex,
)
from whetstone_envs.optim.study.spec import (
    StageId,
    load_study_spec,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def protocol_doc(tmp_path: Path) -> Path:
    """A stand-in pre-registration document with a knowable digest."""
    doc = tmp_path / "protocol.md"
    doc.write_bytes(b"# step 10\n")
    return doc


@pytest.fixture
def toy() -> StudyProtocol:
    """The real design at test size: same arms, same pins, small splits."""
    return study_protocol("step10-c19", toy=True)


# --------------------------------------------------------------------------
# What init records
# --------------------------------------------------------------------------


def test_init_writes_a_pre_stage0_manifest(
    tmp_path: Path, toy: StudyProtocol, protocol_doc: Path
) -> None:
    """The design is complete; everything a stage measures is absent."""
    path = init_study(
        tmp_path / "study", protocol=toy, protocol_doc=protocol_doc
    )
    manifest = read_study_manifest(path)

    assert manifest.schema_ == STUDY_MANIFEST_SCHEMA
    assert manifest.study_id == "step10-c19-toy"
    assert len(manifest.arms) == len(toy.arms)
    # Pre-Stage-0: nothing that a stage measures has been invented.
    assert manifest.design is None
    assert manifest.pre_registration is None
    assert manifest.stages == ()
    assert manifest.selection == ()
    assert manifest.held_out == ()
    assert all(arm.runs == () for arm in manifest.arms)


def test_the_protocol_document_digest_is_of_the_file_read(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    """The manifest names the revision that was actually in force."""
    manifest = study_manifest_for(toy, protocol_doc=protocol_doc)
    assert (
        manifest.protocol_doc_sha256
        == hashlib.sha256(b"# step 10\n").hexdigest()
    )


def test_the_recorded_splits_are_the_ones_a_stage_regenerates(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    """Task hashes are recomputed, so the population check has something
    truthful to compare against."""
    manifest = study_manifest_for(toy, protocol_doc=protocol_doc)
    internal, official, held_out = toy.split_sizes
    assert len(manifest.splits.internal.task_hashes) == internal
    assert len(manifest.splits.official.task_hashes) == official
    assert len(manifest.splits.held_out.task_hashes) == held_out
    # Disjointness is the manifest's own validator, but it is the property
    # L5 rests on, so it is asserted where the record is authored too.
    everything = (
        set(manifest.splits.internal.task_hashes)
        | set(manifest.splits.official.task_hashes)
        | set(manifest.splits.held_out.task_hashes)
    )
    assert len(everything) == internal + official + held_out


def test_init_is_deterministic(toy: StudyProtocol, protocol_doc: Path) -> None:
    """Two initialisations agree on everything but the timestamp."""
    first = study_manifest_for(
        toy, protocol_doc=protocol_doc, created_at="2026-08-23T00:00:00+00:00"
    )
    second = study_manifest_for(
        toy, protocol_doc=protocol_doc, created_at="2026-08-23T00:00:00+00:00"
    )
    assert first == second


def test_the_eval_config_hashes_read_as_unset(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    """Stage 0 derives them; a pre-Stage-0 manifest says so rather than
    carrying a plausible-looking digest."""
    manifest = study_manifest_for(toy, protocol_doc=protocol_doc)
    for split in (
        manifest.splits.internal,
        manifest.splits.official,
        manifest.splits.held_out,
    ):
        assert split.eval_config_hash == PENDING_EVAL_CONFIG_HASH


def test_the_models_block_pins_the_designs_models(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    manifest = study_manifest_for(toy, protocol_doc=protocol_doc)
    assert manifest.models.task_model == toy.task_model
    assert manifest.models.proposer_model == toy.proposer_model
    assert manifest.models.codex_agent_model == toy.codex_agent_model


def test_the_minibatch_design_reaches_the_arm_record(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    """Minibatching is design, so the record carries it (schema v9).

    ``spec_from_manifest`` rebuilds every stage's runnable spec from the
    arm record. A record that said ``minibatch=False`` while the
    pre-registration hashed a batch size would run MIPROv2 on the whole
    valset under a design hash claiming it batched -- the exact drift the
    round-trip exists to stop.
    """
    manifest = study_manifest_for(toy, protocol_doc=protocol_doc)
    by_id = {arm.arm_id: arm for arm in manifest.arms}
    for arm in toy.arms:
        recorded = by_id[arm.arm_id]
        assert recorded.minibatch == arm.miprov2_minibatch, arm.arm_id
        assert recorded.minibatch_size == arm.miprov2_minibatch_size, (
            arm.arm_id
        )
    # The MIPROv2 arms are the ones that carry it at all.
    assert by_id["miprov2"].minibatch is True
    assert by_id["miprov2"].minibatch_size == toy.miprov2_minibatch_size
    assert by_id["copro"].minibatch is False
    assert by_id["copro"].minibatch_size is None


def test_the_written_manifest_rebuilds_the_minibatch_design(
    tmp_path: Path, toy: StudyProtocol, protocol_doc: Path
) -> None:
    """What init writes is what the stage that runs MIPROv2 reads back."""
    study_dir = tmp_path / "study"
    init_study(study_dir, protocol=toy, protocol_doc=protocol_doc)
    spec = load_study_spec(study_dir, stage=StageId.STAGE2)
    by_id = {arm.arm_id: arm for arm in spec.arms}
    assert by_id["miprov2"].miprov2_minibatch is True
    assert (
        by_id["miprov2"].miprov2_minibatch_size == toy.miprov2_minibatch_size
    )


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_init_refuses_to_overwrite_an_existing_study(
    tmp_path: Path, toy: StudyProtocol, protocol_doc: Path
) -> None:
    """A second init over a study that holds evidence would reset a design
    that evidence refers to."""
    study_dir = tmp_path / "study"
    init_study(study_dir, protocol=toy, protocol_doc=protocol_doc)
    with pytest.raises(ManifestExistsError):
        init_study(study_dir, protocol=toy, protocol_doc=protocol_doc)


def test_a_missing_protocol_document_refuses_before_writing(
    tmp_path: Path, toy: StudyProtocol
) -> None:
    """No manifest is left behind when the digest cannot be taken."""
    study_dir = tmp_path / "study"
    with pytest.raises(ValueError, match="cannot read the protocol document"):
        init_study(
            study_dir, protocol=toy, protocol_doc=tmp_path / "absent.md"
        )
    assert not (study_dir / "study.json").exists()


def test_an_arm_whose_design_cannot_run_is_refused_at_authoring(
    toy: StudyProtocol, protocol_doc: Path
) -> None:
    """``ArmSpec``'s refusals run at init, not on the arm's paid turn.

    A minibatch with no size is the case the manifest's own validators do
    not catch: ``ArmRecord`` has nowhere to put either field, so such a
    manifest would validate, write, and fail at the first MIPROv2 arm.
    """
    broken = replace(
        toy,
        arms=tuple(
            replace(arm, miprov2_minibatch_size=None)
            if arm.optimizer == "miprov2"
            else arm
            for arm in toy.arms
        ),
    )
    with pytest.raises(ValueError, match="minibatch"):
        study_manifest_for(broken, protocol_doc=protocol_doc)


# --------------------------------------------------------------------------
# The manifest reads back as the design
# --------------------------------------------------------------------------


def test_the_written_manifest_rebuilds_the_pre_registered_spec(
    tmp_path: Path, toy: StudyProtocol, protocol_doc: Path
) -> None:
    """What ``init`` writes is what ``load_study_spec`` reads: the run
    counts and seeds a stage would execute come back unchanged."""
    study_dir = tmp_path / "study"
    init_study(study_dir, protocol=toy, protocol_doc=protocol_doc)
    spec = load_study_spec(study_dir, stage=StageId.STAGE2)

    authored = {arm.arm_id: arm for arm in toy.arm_specs(stage=StageId.STAGE2)}
    assert set(spec.arm_ids) == set(authored)
    for arm in spec.arms:
        assert arm.k_run == authored[arm.arm_id].k_run, arm.arm_id
        assert arm.seeds == authored[arm.arm_id].seeds, arm.arm_id
        assert arm.optimizer == authored[arm.arm_id].optimizer, arm.arm_id
        assert arm.demo_mode == authored[arm.arm_id].demo_mode, arm.arm_id


def test_a_codex_free_design_pins_no_codex_agent(
    tmp_path: Path, toy: StudyProtocol, protocol_doc: Path
) -> None:
    """The spec carries the pin only when an arm honours it, but the
    manifest records the field either way."""
    study_dir = tmp_path / "study"
    init_study(
        study_dir, protocol=without_codex(toy), protocol_doc=protocol_doc
    )
    spec = load_study_spec(study_dir)
    assert spec.codex_agent_model is None
    assert read_study_manifest(study_dir).models.codex_agent_model
