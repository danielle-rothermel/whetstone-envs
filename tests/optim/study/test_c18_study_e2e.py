"""The c18 study end to end, and the two ways it must not be confusable.

Three things are proved here, all on a **fake transport with zero provider
calls**:

* A ``step10-c18`` study initialises from the registered protocol, plans,
  and runs its stages through the real CLI, leaving a manifest whose arms
  ran the pre-registered ``K_RUN = 1`` -- not the c19 ladder's two-then-five.
* That manifest carries a ``c18`` block: the runs and the adapter-swap
  verdict, which is the C3 evidence the report renders. Before this, the
  block was a shape nothing produced, so a c18 study's own report said no
  second family had been run.
* A c18 study and a c19 study cannot be mistaken for one another. Their
  designs hash differently, and neither can be driven against the other's
  population or protocol id.

The splits are the toy's rather than section 4.1's ``(24, 48, 48)``, but
nothing else is a stand-in: the runs go through the same
:func:`~whetstone_envs.optim.run.run_optimizer`, and the adapter-swap
verdict is computed over the shipped package rather than asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.families import FamilyId
from whetstone_envs.optim.study.adapter_swap import differing_modules
from whetstone_envs.optim.study.cli import EXIT_ERROR, EXIT_OK, main
from whetstone_envs.optim.study.init import init_study, study_manifest_for
from whetstone_envs.optim.study.manifest import (
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.protocols import (
    STEP10_C18,
    STEP10_C18_ID,
    STEP10_C18_TOY,
    STEP10_C19_ID,
    STEP10_C19_TOY,
    study_protocol,
)
from whetstone_envs.optim.study.spec import K_RUN_C18, StageId

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.manifest import StudyManifest
    from whetstone_envs.optim.study.protocols import StudyProtocol

#: The arms the end-to-end run drives. The registered design declares all
#: eight; this run keeps the two that exercise both paths -- one real
#: optimizer through run/audit/cost/evidence-copy, and one control that
#: runs no optimizer at all -- because the other six add stage minutes
#: without adding a code path, and which arms exist is already golden-tested
#: in ``test_protocols``.
E2E_ARMS = ("copro", "null-identity")


#: What a narrowed run appends to the study id.
#:
#: Dropping arms makes this a strictly smaller design than the registered
#: one, and the stage harness refuses a reduced design that still carries a
#: registered protocol's id -- its runs would be recorded against a
#: pre-registration it does not hold. So the fixture says so in its id,
#: exactly as a real rehearsal must.
NARROWED_SUFFIX = "-two-arm-rehearsal"


def _narrowed(protocol: StudyProtocol, arms: tuple[str, ...]) -> StudyProtocol:
    """``protocol`` with only ``arms``, under an id of its own.

    Built by *removing* arms from the registered design rather than by
    hand-writing a smaller one, so every value this study runs under is
    still the protocol's: a hand-built fixture could disagree with the
    pre-registration on the very fields the test is checking.
    """
    from dataclasses import replace

    return replace(
        protocol,
        study_id=f"{protocol.study_id}{NARROWED_SUFFIX}",
        arms=tuple(arm for arm in protocol.arms if arm.arm_id in arms),
    )


@pytest.fixture
def c18_study_dir(tmp_path: Path) -> Path:
    """A pre-Stage-0 c18 toy study, authored from the registered protocol."""
    directory = tmp_path / "c18-study"
    write_study_manifest(
        directory, study_manifest_for(_narrowed(STEP10_C18_TOY, E2E_ARMS))
    )
    return directory


def _run_stage(study_dir: Path, stage: str) -> int:
    return main(["run", "--study-dir", str(study_dir), "--stage", stage])


def _run_every_stage(study_dir: Path) -> StudyManifest:
    for stage in (StageId.STAGE0, StageId.STAGE1, StageId.STAGE2):
        assert _run_stage(study_dir, stage.value) == EXIT_OK, stage
    return read_study_manifest(study_dir)


# --------------------------------------------------------------------------
# init and plan
# --------------------------------------------------------------------------


def test_init_authors_a_c18_study_through_the_cli(tmp_path: Path) -> None:
    """``--protocol step10-c18`` is accepted and writes the c18 design.

    Fails before this change at the argument parser: ``--protocol`` offered
    ``step10-c19`` as its only choice, which is what made the second family
    unreachable from the command line no matter what the module registered.
    """
    study_dir = tmp_path / "study"
    assert (
        main(
            [
                "init",
                "--study-dir",
                str(study_dir),
                "--protocol",
                STEP10_C18_ID,
                "--toy",
            ]
        )
        == EXIT_OK
    )
    manifest = read_study_manifest(study_dir)
    assert manifest.study_id == "step10-c18-toy"
    assert manifest.population.family == FamilyId.C18.value
    assert len(manifest.arms) == len(STEP10_C18_TOY.arms) == 8


def test_the_real_c18_protocol_authors_section_4_1s_design(
    tmp_path: Path,
) -> None:
    """The unsized study, over c18's own population and splits."""
    path = init_study(tmp_path / "study", protocol=STEP10_C18)
    manifest = read_study_manifest(path)
    assert manifest.study_id == STEP10_C18_ID
    assert manifest.population.family == FamilyId.C18.value
    assert manifest.population.n_per_stratum == 30
    assert manifest.population.pool_seed_start == 1_000_000_000
    sizes = (
        manifest.splits.internal.size,
        manifest.splits.official.size,
        manifest.splits.held_out.size,
    )
    assert sizes == (24, 48, 48)
    # Real task hashes from a real c18 pool, not invented identifiers: a
    # manifest of invented hashes validates and then fails at the first
    # engine binding.
    assert len(manifest.splits.internal.task_hashes) == 24
    assert len(manifest.splits.held_out.task_hashes) == 48


def test_plan_prices_the_c18_study_at_one_run_per_arm(
    c18_study_dir: Path,
) -> None:
    """``plan`` reads the design, so it prices ``K_RUN = 1``."""
    assert main(["plan", "--study-dir", str(c18_study_dir)]) == EXIT_OK


# --------------------------------------------------------------------------
# The stages, and what they record
# --------------------------------------------------------------------------


def test_every_stage_runs_a_c18_study_on_a_fake_transport(
    c18_study_dir: Path,
) -> None:
    """The load-bearing assertion: the second family is operational."""
    manifest = _run_every_stage(c18_study_dir)

    assert manifest.design is not None
    assert manifest.pre_registration is not None
    for arm in manifest.arms:
        assert arm.runs, arm.arm_id
        assert len(arm.runs) == manifest.design.k_run_by_arm[arm.arm_id]


def test_the_c18_design_runs_every_arm_exactly_once(
    c18_study_dir: Path,
) -> None:
    """``K_RUN = 1``, and Stage 2 does not extend it to five.

    Fails before this change: ``k_run_for`` read the count off the stage
    alone, so a c18 study inherited c19's powered ladder and Stage 2 would
    have bought five runs per arm of a design that pre-registers one.
    """
    manifest = _run_every_stage(c18_study_dir)
    assert manifest.design is not None
    assert set(manifest.design.k_run_by_arm.values()) == {K_RUN_C18} == {1}
    for arm in manifest.arms:
        assert len(arm.runs) == 1, arm.arm_id


def test_the_manifest_records_the_c18_block(c18_study_dir: Path) -> None:
    """C3's evidence reaches the artifact.

    Fails before this change: ``C18Record`` was never constructed anywhere
    in ``src``, so a c18 study spent its budget and left a manifest whose
    generality section read "No second family was run" -- true of the
    artifact and false of the study.
    """
    manifest = _run_every_stage(c18_study_dir)
    assert manifest.c18 is not None
    # Every run the arms recorded is in the block, by id.
    assert {run.run_id for run in manifest.c18.runs} == {
        run.run_id for arm in manifest.arms for run in arm.runs
    }
    assert manifest.c18.runs


def test_the_c18_block_carries_the_adapter_swap_verdict(
    c18_study_dir: Path,
) -> None:
    """The assertion is computed at record time, not cited from a test.

    A manifest whose C3 claim rested on a green CI job at an unrecorded
    commit would be citing evidence the artifact does not carry.
    """
    manifest = _run_every_stage(c18_study_dir)
    assert manifest.c18 is not None
    swap = manifest.c18.adapter_swap
    assert swap.passed is True
    assert swap.differing_modules == () == differing_modules()


def test_a_c19_study_records_no_c18_block(tmp_path: Path) -> None:
    """The block is keyed on the population, so c19 never grows one.

    A c19 study that recorded a C3 block would be claiming generality
    evidence from a run of the primary family -- the report would render a
    second-family section for a study that ran one family.
    """
    directory = tmp_path / "c19-study"
    write_study_manifest(
        directory, study_manifest_for(_narrowed(STEP10_C19_TOY, E2E_ARMS))
    )
    manifest = _run_every_stage(directory)
    assert manifest.population.family == FamilyId.C19.value
    assert manifest.c18 is None


def test_the_c18_block_survives_a_resumed_stage(c18_study_dir: Path) -> None:
    """A re-run that executes nothing does not erase the recorded block.

    The block is the study's whole C3 evidence rather than the last
    invocation's, so an invocation with no new runs leaves it alone rather
    than replacing it with an empty run list.
    """
    _run_every_stage(c18_study_dir)
    before = read_study_manifest(c18_study_dir).c18
    assert before is not None

    assert _run_stage(c18_study_dir, StageId.STAGE2.value) == EXIT_OK
    after = read_study_manifest(c18_study_dir).c18
    assert after is not None
    assert {run.run_id for run in after.runs} >= {
        run.run_id for run in before.runs
    }
    assert after.adapter_swap.passed


# --------------------------------------------------------------------------
# The two studies cannot be confused
# --------------------------------------------------------------------------


def test_the_two_designs_hash_differently(tmp_path: Path) -> None:
    """A c18 study and a c19 study pre-register different designs.

    The pre-registration hash is what makes "this study ran the design it
    registered" checkable. If the two families hashed alike, a c18 result
    could be presented under the c19 study's pinning and validate.

    They differ on ``k_run_by_arm`` and on the MIPROv2 minibatch, both of
    which are hashed -- so this is a property of the design rather than of
    the population, which the hash deliberately does not cover.
    """
    hashes = {}
    for name, protocol in (
        ("c19", STEP10_C19_TOY),
        ("c18", STEP10_C18_TOY),
    ):
        directory = tmp_path / name
        write_study_manifest(
            directory, study_manifest_for(_narrowed(protocol, E2E_ARMS))
        )
        assert _run_stage(directory, StageId.STAGE0.value) == EXIT_OK
        pinned = read_study_manifest(directory).pre_registration
        assert pinned is not None
        hashes[name] = pinned.design_hash

    assert hashes["c18"] != hashes["c19"]


def test_a_c18_study_cannot_be_reinitialised_as_c19(tmp_path: Path) -> None:
    """``init`` refuses to overwrite a study that already has a design.

    The two protocols write to a directory the same way, so the only thing
    standing between a c18 study and a c19 manifest landing on top of it is
    this refusal -- and a study whose manifest changed family underneath
    its artifacts would cite runs measured on tasks it no longer names.
    """
    study_dir = tmp_path / "study"
    assert (
        main(
            [
                "init",
                "--study-dir",
                str(study_dir),
                "--protocol",
                STEP10_C18_ID,
                "--toy",
            ]
        )
        == EXIT_OK
    )
    assert (
        main(
            [
                "init",
                "--study-dir",
                str(study_dir),
                "--protocol",
                STEP10_C19_ID,
                "--toy",
            ]
        )
        == EXIT_ERROR
    )
    # Unchanged: the refusal is before the write, not after a partial one.
    assert (
        read_study_manifest(study_dir).population.family == FamilyId.C18.value
    )


def test_neither_protocol_claims_the_others_study_id(tmp_path: Path) -> None:
    """A study id that names the other protocol is refused.

    ``--study-id`` exists so a rehearsal can be told apart from the study.
    Handing a c18 study the c19 study's id would make every artifact
    downstream cite a design the run is not.
    """
    for protocol_id, claimed in (
        (STEP10_C18_ID, STEP10_C19_ID),
        (STEP10_C19_ID, STEP10_C18_ID),
    ):
        with pytest.raises(SystemExit):
            main(
                [
                    "init",
                    "--study-dir",
                    str(tmp_path / f"{protocol_id}-as-{claimed}"),
                    "--protocol",
                    protocol_id,
                    "--toy",
                    "--study-id",
                    claimed,
                ]
            )


def test_each_protocols_toy_is_its_own_populations_toy() -> None:
    """The two toys generate different pools, from different generators.

    A toy that shared the c19 toy's population would rehearse the second
    family's *design* over the first family's *tasks* -- which is neither
    study, and would make the C3 claim about a pool c18 never generated.
    """
    c18 = study_manifest_for(study_protocol(STEP10_C18_ID, toy=True))
    c19 = study_manifest_for(study_protocol(STEP10_C19_ID, toy=True))
    assert c18.population.family != c19.population.family
    assert c18.population.pool_seed_start != c19.population.pool_seed_start
    # Different tasks, not merely different labels.
    assert set(c18.splits.internal.task_hashes).isdisjoint(
        c19.splits.internal.task_hashes
    )
