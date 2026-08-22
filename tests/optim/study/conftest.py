"""A toy study manifest over a real c19 population.

The splits here are tiny -- (4, 4, 6) against the study's own (88, 132, 440)
-- but they are *real*: the task hashes come from an actually-generated pool
through the same experiment builder a paid stage would use, so the manifest
the harness reads describes tasks the harness can then evaluate. A manifest
of invented hashes would pass its own validation and then fail at the first
engine binding, which is exactly the failure a fixture should not hide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.c19 import generate_pool
from whetstone_envs.optim.experiment import prepare_c19_experiment
from whetstone_envs.optim.rows import task_rows_from_instances
from whetstone_envs.optim.study.manifest import (
    ArmRecord,
    ModelsRecord,
    PopulationRecord,
    SplitRecord,
    SplitsRecord,
    StudyManifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from whetstone_envs.instances import Instance

#: Toy sizes: enough tasks per role for a real calibration, small enough to
#: stay a unit test. The study's own sizes are (88, 132, 440).
TOY_SPLIT_SIZES = (4, 4, 6)
#: The toy train/val partition of the toy internal 4, standing in for the
#: protocol's 44/44 of 88 at a size a unit test can afford.
TOY_TRAIN_SIZE = 2
TOY_VAL_SIZE = 2
TOY_N_PER_STRATUM = 1
TOY_POOL_SEED_START = 765_432

INTERNAL_CONFIG = "toy-internal-config"
OFFICIAL_CONFIG = "toy-official-config"
HELD_OUT_CONFIG = "toy-held-out-config"


def _task_hashes(instances: Iterable[Instance]) -> tuple[str, ...]:
    return tuple(row.task_hash for row in task_rows_from_instances(instances))


def toy_splits() -> SplitsRecord:
    """The three splits of one deterministically generated toy pool."""
    pool = generate_pool(
        n_per_stratum=TOY_N_PER_STRATUM, seed_start=TOY_POOL_SEED_START
    )
    split = prepare_c19_experiment(
        pool, split_sizes=TOY_SPLIT_SIZES, num_seeds=1
    ).split
    internal, official, held_out = TOY_SPLIT_SIZES
    return SplitsRecord(
        internal=SplitRecord(
            size=internal,
            task_hashes=_task_hashes(split.internal_eval),
            eval_config_hash=INTERNAL_CONFIG,
        ),
        official=SplitRecord(
            size=official,
            task_hashes=_task_hashes(split.official),
            eval_config_hash=OFFICIAL_CONFIG,
        ),
        held_out=SplitRecord(
            size=held_out,
            task_hashes=_task_hashes(split.held_out),
            eval_config_hash=HELD_OUT_CONFIG,
        ),
    )


def toy_arms() -> tuple[ArmRecord, ...]:
    """Two arms with no runs yet: one real optimizer and one null.

    An arm is declared before it runs, which is what makes the design a
    pre-registration; ``runs`` fills in as stages execute.
    """
    return (
        ArmRecord(
            arm_id="copro",
            optimizer="copro",
            demo_mode=None,
            control_identity_hash="d" * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        ),
        ArmRecord(
            arm_id="null-identity",
            optimizer="null-identity",
            demo_mode=None,
            control_identity_hash="e" * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        ),
    )


def toy_manifest(
    *, arms: tuple[ArmRecord, ...] | None = None
) -> StudyManifest:
    """A pre-Stage-0 manifest: design, selection, and held-out unset."""
    return StudyManifest(
        study_id="step10-toy",
        created_at="2026-08-22T12:00:00+00:00",
        protocol_doc_path="~/drotherm/data/.claude/protocol.md",
        protocol_doc_sha256="a" * 64,
        assignment_doc_sha256="b" * 64,
        population=PopulationRecord(
            family="c19",
            generator_version="whetstone-envs toy",
            n_per_stratum=TOY_N_PER_STRATUM,
            pool_seed_start=TOY_POOL_SEED_START,
            pool_manifest_content_hash="c" * 64,
            stratum_counts={"toy": sum(TOY_SPLIT_SIZES)},
        ),
        splits=toy_splits(),
        models=ModelsRecord(
            task_model="fake",
            proposer_model="fake",
            temperature="unset",
            provider="fake",
            seed_control="none",
            codex_agent_model="uncontrolled",
        ),
        arms=toy_arms() if arms is None else arms,
    )
