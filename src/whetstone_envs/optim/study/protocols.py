"""The committed pre-registration: every design value the study fixes.

A study's design is only a pre-registration if it existed, in full, before
any provider was reached. This module is where the Step 10 c19 design is
written down -- as named constants with one authoring path, not as flags a
launch command happens to carry -- so ``whetstone-study init`` mints the
same ``study.json`` every time and a reviewer can diff the design against
the protocol document rather than against a shell history.

Two protocols are registered, and they are the *same* protocol at two
sizes. :data:`STEP10_C19` is the real study; :data:`STEP10_C19_TOY` is the
sized-down variant the tests and the dry runs use. Both are built by
:func:`_step10_c19` from one body of pinned values, so the only fields that
can differ between them are the sized ones -- splits, per-arm train/val,
the pool's ``n_per_stratum`` and seed, and the MIPROv2 minibatch size.
:data:`SIZED_FIELDS` names exactly those, and a golden test asserts nothing
else drifts. A toy that could disagree with the real design on ``K_REPEAT``,
on the arm list, or on the correction rule would be a toy that tests a
different study than the one that spends.

What this module does **not** own: run counts and seeds, which
:mod:`whetstone_envs.optim.study.spec` already pins per stage and which
would be a second source of truth if restated here; and the values Stage 0
*measures* rather than declares -- ``tau^2``, ``sigma^2``, the realized
MDE. Declaring a measurement would make the manifest state a result before
the study had one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from whetstone_envs.optim.families import FamilyId
from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    PROTOCOL_SPLIT_SIZES,
    PROTOCOL_TRAIN_SIZE,
    PROTOCOL_VAL_SIZE,
    ArmKind,
    ArmSpec,
    StageId,
    arm_seeds,
    k_run_for,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "CODEX_AGENT_MODEL",
    "GEPA_MAX_METRIC_CALLS",
    "MIPROV2_MINIBATCH_SIZE",
    "MIPROV2_NUM_CANDIDATES",
    "MIPROV2_NUM_TRIALS",
    "PROPOSER_MODEL",
    "PROTOCOL_DOC_PATH",
    "PROTOCOL_IDS",
    "SIZED_FIELDS",
    "STEP10_C19",
    "STEP10_C19_ID",
    "STEP10_C19_TOY",
    "STEP10_C19_TOY_ID",
    "TASK_MODEL",
    "TOY_MIPROV2_MINIBATCH_SIZE",
    "TOY_N_PER_STRATUM",
    "TOY_POOL_SEED_START",
    "TOY_SPLIT_SIZES",
    "TOY_TRAIN_SIZE",
    "TOY_VAL_SIZE",
    "ArmDesign",
    "StudyProtocol",
    "protocol_doc_sha256",
    "study_protocol",
    "without_codex",
]

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

#: The protocol this module encodes, by name.
STEP10_C19_ID = "step10-c19"
STEP10_C19_TOY_ID = "step10-c19-toy"
PROTOCOL_IDS: tuple[str, ...] = (STEP10_C19_ID, STEP10_C19_TOY_ID)

#: The pre-registration document itself. ``init`` hashes the file at this
#: path and records the digest, so a manifest names a *specific* revision of
#: the protocol rather than a filename that could be rewritten afterwards.
PROTOCOL_DOC_PATH = (
    "~/drotherm/data/.claude/whetstone-ai/2026-08-22/"
    "0430-step10-validation-protocol.md"
)

# --------------------------------------------------------------------------
# Models (D14, and note 26 for the agent)
# --------------------------------------------------------------------------

#: The task model every arm evaluates on, and the proposer that writes
#: candidates. Both are OpenRouter routes; the transport that serves them is
#: run-time evidence, not design, so it is not pinned here.
TASK_MODEL = "openai/gpt-5-nano"
PROPOSER_MODEL = "openai/gpt-5.4-nano"

#: The Codex arm's agent, pinned as design (Phase E item 6). The runner's
#: own ``CODEX_DEFAULT_AGENT_MODEL`` is a *run* default; a study that let it
#: stand in would be running whichever proposer the runner defaulted to on
#: the day, which is the drift the pin exists to refuse.
CODEX_AGENT_MODEL = "gpt-5.6-sol"

#: Recorded rather than claimed: ``CoproControl`` refuses a proposer
#: temperature and OpenRouter ignores it on nano routes, so the manifest
#: says the control was left unset instead of asserting a zero it cannot
#: enforce.
TEMPERATURE_NOTE = "unset (provider-default)"
PROVIDER_NOTE = "openrouter"
SEED_CONTROL_NOTE = "advertised"

# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------

#: The family this protocol studies, named through the registry's own
#: identifier rather than as a literal: the study reaches every piece of
#: family knowledge -- the pool, the experiment, the pool manifest -- through
#: :func:`~whetstone_envs.optim.families.family_spec`, and spelling the name
#: here as a bare string would be the one place it did not.
FAMILY = FamilyId.C19.value

#: The pinned population: 22 strata at 32 instances each, 704 total, of
#: which the three splits consume 660 and the tail stays unassigned.
N_PER_STRATUM = 32
POOL_SEED_START = 1_000_000

# --------------------------------------------------------------------------
# Per-optimizer control pins
# --------------------------------------------------------------------------

#: GEPA's metric-call ceiling (D3, note 14). ``auto="light"`` resolves to
#: 732, which ran for 22 minutes and produced a 1.73 GB store; 200 is the
#: pinned budget the power design and the Stage-1 gate are both built on.
GEPA_MAX_METRIC_CALLS = 200

#: MIPROv2's search shape. ``num_candidates`` is at the minibatch floor
#: rather than below it: two candidates with minibatching exhausts the
#: search space and raises inside the durable run boundary on releases
#: before the fix (note 25d), so the design pins the shape that runs.
MIPROV2_NUM_TRIALS = 10
#: Stated as a literal rather than read from
#: ``whetstone_envs.optim.run.MIPROV2_MINIBATCH_MIN_CANDIDATES``: the
#: runner imports the study package for its spend record, so importing the
#: runner back from here closes a cycle. ``test_protocols`` asserts the two
#: agree, which is the check the import was standing in for.
MIPROV2_NUM_CANDIDATES = 3

#: The minibatch size, pinned as a design field alongside trials and
#: candidates (Phase E item 3). Left unset the batch is the whole validation
#: split and minibatching is on in name only.
MIPROV2_MINIBATCH_SIZE = 35

#: MIPROv2's efficacy arm. ``zeroshot`` and ``ground_only`` are fidelity
#: evidence at one run each, pre-registered here so a later result cannot
#: promote one of them into the efficacy slot (R2).
MIPROV2_EFFICACY_DEMO_MODE = "fewshot"
MIPROV2_FIDELITY_DEMO_MODES: tuple[str, ...] = ("zeroshot", "ground_only")

# --------------------------------------------------------------------------
# Toy sizes
# --------------------------------------------------------------------------

#: The toy variant's sized fields. Small enough for a unit test, real
#: enough that the tasks exist and the harness can evaluate them.
TOY_SPLIT_SIZES: tuple[int, int, int] = (4, 4, 6)
TOY_TRAIN_SIZE = 2
TOY_VAL_SIZE = 2
TOY_N_PER_STRATUM = 1
TOY_POOL_SEED_START = 765_432
TOY_MIPROV2_MINIBATCH_SIZE = 2

#: Exactly the fields :data:`STEP10_C19` and :data:`STEP10_C19_TOY` are
#: permitted to differ on. The golden test reads this tuple, so adding a
#: field here is a deliberate widening of the toy's licence to diverge and
#: shows up in review as one.
SIZED_FIELDS: tuple[str, ...] = (
    "protocol_id",
    "study_id",
    "n_per_stratum",
    "pool_seed_start",
    "split_sizes",
    "train_size",
    "val_size",
    "miprov2_minibatch_size",
)


@dataclass(frozen=True, slots=True)
class ArmDesign:
    """One pre-registered arm, before any stage has run it.

    This is the design of an arm, not its progress: it names the optimizer,
    the demo mode, and the control pins, and it carries no runs. The run
    count and the seeds come from :mod:`spec` at the stage being priced,
    which is why they are absent here.
    """

    arm_id: str
    optimizer: str
    kind: ArmKind
    demo_mode: str | None = None
    train_size: int | None = None
    val_size: int | None = None
    miprov2_num_trials: int | None = None
    miprov2_num_candidates: int | None = None
    miprov2_minibatch: bool = False
    miprov2_minibatch_size: int | None = None

    def to_spec(self, *, stage: StageId) -> ArmSpec:
        """The runnable arm at ``stage``, with that stage's run count.

        Built through :class:`ArmSpec` rather than around it so the design
        is refused *here* -- at authoring time, for free -- if it names a
        minibatch with no size, a train/val partition on an optimizer with
        no such concept, or a MIPROv2 setting on an arm that is not
        MIPROv2. A manifest that validated and then failed at the first
        paid arm is the failure this crossing exists to prevent.
        """
        return ArmSpec(
            arm_id=self.arm_id,
            optimizer=self.optimizer,
            kind=self.kind,
            k_run=k_run_for(self.arm_id, stage=stage),
            seeds=arm_seeds(self.arm_id, stage=stage),
            demo_mode=self.demo_mode,
            miprov2_num_trials=self.miprov2_num_trials,
            miprov2_num_candidates=self.miprov2_num_candidates,
            miprov2_minibatch=self.miprov2_minibatch,
            miprov2_minibatch_size=self.miprov2_minibatch_size,
            train_size=self.train_size,
            val_size=self.val_size,
        )


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    """A complete pre-registered design, at one size.

    Every field is a value the study fixes before Stage 0. What Stage 0
    *measures* -- the variance components and the realized MDE -- is
    deliberately absent: a protocol that declared them would let the
    manifest state a result the study had not yet earned.
    """

    protocol_id: str
    study_id: str
    family: str
    n_per_stratum: int
    pool_seed_start: int
    split_sizes: tuple[int, int, int]
    train_size: int
    val_size: int
    task_model: str
    proposer_model: str
    codex_agent_model: str
    temperature: str
    provider: str
    seed_control: str
    protocol_doc_path: str
    gepa_max_metric_calls: int
    codex_evaluate_call_cap: int
    miprov2_num_trials: int
    miprov2_num_candidates: int
    miprov2_minibatch_size: int
    arms: tuple[ArmDesign, ...]

    def __post_init__(self) -> None:
        if self.protocol_id not in PROTOCOL_IDS:
            raise ValueError(
                f"unknown protocol id {self.protocol_id!r}; "
                f"registered protocols are {PROTOCOL_IDS}"
            )
        internal, official, held_out = self.split_sizes
        if min(self.split_sizes) < 1:
            raise ValueError(
                f"{self.protocol_id}: every split must hold at least one "
                f"task, got {self.split_sizes}"
            )
        if self.train_size + self.val_size != internal:
            # GEPA requires train + val to cover the internal split
            # exactly, and the study runs GEPA at this partition. Refusing
            # here means the protocol cannot declare a partition its own
            # GEPA arm would reject on its turn.
            raise ValueError(
                f"{self.protocol_id}: train {self.train_size} + val "
                f"{self.val_size} must equal the internal split "
                f"{internal}, which is what the GEPA arm requires"
            )
        if self.miprov2_minibatch_size > self.val_size:
            raise ValueError(
                f"{self.protocol_id}: MIPROv2 minibatch "
                f"{self.miprov2_minibatch_size} exceeds the validation "
                f"split {self.val_size}"
            )
        if official < 1 or held_out < 1:
            raise ValueError(
                f"{self.protocol_id}: official and held-out splits must be "
                f"non-empty, got {official} and {held_out}"
            )
        if not self.arms:
            raise ValueError(f"{self.protocol_id}: a study declares arms")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError(f"{self.protocol_id}: arm ids must be unique")

    @property
    def internal_size(self) -> int:
        return self.split_sizes[0]

    @property
    def official_size(self) -> int:
        return self.split_sizes[1]

    @property
    def held_out_size(self) -> int:
        return self.split_sizes[2]

    def arm_specs(self, *, stage: StageId) -> tuple[ArmSpec, ...]:
        """Every arm as a runnable spec at ``stage``."""
        return tuple(arm.to_spec(stage=stage) for arm in self.arms)

    def sized_values(self) -> Mapping[str, object]:
        """The fields the toy is licensed to differ on, by name."""
        return {name: getattr(self, name) for name in SIZED_FIELDS}


def _arms(
    *,
    train_size: int,
    val_size: int,
    minibatch_size: int,
) -> tuple[ArmDesign, ...]:
    """The study's arms, in report order: four real, then the two nulls.

    MIPROv2 appears three times. Only ``fewshot`` is an efficacy arm; the
    other two modes are fidelity evidence for the zeroshot-grounding and
    ground-only-deviation invariants, and they are declared here so that
    which mode carries the efficacy claim is pre-registered rather than
    chosen once the deltas are visible (R2).
    """

    def miprov2(demo_mode: str) -> ArmDesign:
        efficacy = demo_mode == MIPROV2_EFFICACY_DEMO_MODE
        return ArmDesign(
            arm_id="miprov2" if efficacy else f"miprov2-{demo_mode}",
            optimizer="miprov2",
            kind=ArmKind.REAL,
            demo_mode=demo_mode,
            train_size=train_size,
            val_size=val_size,
            miprov2_num_trials=MIPROV2_NUM_TRIALS,
            miprov2_num_candidates=MIPROV2_NUM_CANDIDATES,
            miprov2_minibatch=True,
            miprov2_minibatch_size=minibatch_size,
        )

    return (
        ArmDesign(arm_id="copro", optimizer="copro", kind=ArmKind.REAL),
        miprov2(MIPROV2_EFFICACY_DEMO_MODE),
        *(miprov2(mode) for mode in MIPROV2_FIDELITY_DEMO_MODES),
        ArmDesign(
            arm_id="gepa",
            optimizer="gepa",
            kind=ArmKind.REAL,
            train_size=train_size,
            val_size=val_size,
        ),
        ArmDesign(arm_id="codex", optimizer="codex", kind=ArmKind.REAL),
        ArmDesign(
            arm_id="null-random", optimizer="null-random", kind=ArmKind.NULL
        ),
        ArmDesign(
            arm_id="null-identity",
            optimizer="null-identity",
            kind=ArmKind.NULL,
        ),
    )


def _step10_c19(  # noqa: PLR0913
    *,
    protocol_id: str,
    study_id: str,
    n_per_stratum: int,
    pool_seed_start: int,
    split_sizes: tuple[int, int, int],
    train_size: int,
    val_size: int,
    miprov2_minibatch_size: int,
) -> StudyProtocol:
    """Build one size of the Step 10 c19 design.

    Every unsized value is a module constant read here rather than a
    parameter, which is what makes the toy and the real study the same
    protocol: there is no argument by which they could disagree on the
    models, the arm list, or the control pins.
    """
    return StudyProtocol(
        protocol_id=protocol_id,
        study_id=study_id,
        family=FAMILY,
        n_per_stratum=n_per_stratum,
        pool_seed_start=pool_seed_start,
        split_sizes=split_sizes,
        train_size=train_size,
        val_size=val_size,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        codex_agent_model=CODEX_AGENT_MODEL,
        temperature=TEMPERATURE_NOTE,
        provider=PROVIDER_NOTE,
        seed_control=SEED_CONTROL_NOTE,
        protocol_doc_path=PROTOCOL_DOC_PATH,
        gepa_max_metric_calls=GEPA_MAX_METRIC_CALLS,
        codex_evaluate_call_cap=CODEX_EVALUATE_CALL_CAP,
        miprov2_num_trials=MIPROV2_NUM_TRIALS,
        miprov2_num_candidates=MIPROV2_NUM_CANDIDATES,
        miprov2_minibatch_size=miprov2_minibatch_size,
        arms=_arms(
            train_size=train_size,
            val_size=val_size,
            minibatch_size=miprov2_minibatch_size,
        ),
    )


#: The real Step 10 c19 study.
STEP10_C19 = _step10_c19(
    protocol_id=STEP10_C19_ID,
    study_id=STEP10_C19_ID,
    n_per_stratum=N_PER_STRATUM,
    pool_seed_start=POOL_SEED_START,
    split_sizes=PROTOCOL_SPLIT_SIZES,
    train_size=PROTOCOL_TRAIN_SIZE,
    val_size=PROTOCOL_VAL_SIZE,
    miprov2_minibatch_size=MIPROV2_MINIBATCH_SIZE,
)

#: The same design at test size.
STEP10_C19_TOY = _step10_c19(
    protocol_id=STEP10_C19_TOY_ID,
    study_id=STEP10_C19_TOY_ID,
    n_per_stratum=TOY_N_PER_STRATUM,
    pool_seed_start=TOY_POOL_SEED_START,
    split_sizes=TOY_SPLIT_SIZES,
    train_size=TOY_TRAIN_SIZE,
    val_size=TOY_VAL_SIZE,
    miprov2_minibatch_size=TOY_MIPROV2_MINIBATCH_SIZE,
)

_BY_ID: dict[str, StudyProtocol] = {
    STEP10_C19.protocol_id: STEP10_C19,
    STEP10_C19_TOY.protocol_id: STEP10_C19_TOY,
}


def study_protocol(protocol_id: str, *, toy: bool = False) -> StudyProtocol:
    """The registered protocol named by ``protocol_id``.

    ``toy`` selects the sized-down variant of the same protocol rather than
    a separate registration, so a caller cannot ask for a toy of one
    protocol and receive the real design of another.
    """
    if protocol_id not in _BY_ID:
        raise ValueError(
            f"unknown protocol {protocol_id!r}; registered protocols are "
            f"{PROTOCOL_IDS}"
        )
    if not toy:
        return _BY_ID[protocol_id]
    toy_id = f"{protocol_id}-toy"
    if toy_id not in _BY_ID:
        raise ValueError(f"protocol {protocol_id!r} has no toy variant")
    return _BY_ID[toy_id]


def protocol_doc_sha256(doc_path: Path) -> str:
    """The digest of the pre-registration document at ``doc_path``.

    Computed from the file at init time rather than pinned as a literal:
    the point of recording it is that the manifest names the revision of
    the protocol that was actually in force, and a hardcoded digest would
    keep validating after the document changed underneath it.
    """
    resolved = doc_path.expanduser()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read the protocol document at {resolved}: {error}. "
            "The manifest records the digest of the pre-registration that "
            "was in force, so a study cannot be initialised without it"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def without_codex(protocol: StudyProtocol) -> StudyProtocol:
    """The same design with its Codex arm dropped.

    The Codex arm's runs spawn a real, billed agent session, and the stage
    harness refuses a Codex-bearing design before any arm runs regardless
    of which transport the task model is on. A fake-transport rehearsal of
    the *rest* of the study therefore has to drop the arm rather than
    stub it, and doing that here -- from the registered design, by
    removing exactly one arm -- keeps the rehearsal a projection of the
    real protocol instead of a second hand-written one.

    The result is not the pre-registration: it is a strictly smaller
    design, and a study initialised from it records its own arm list.
    """
    return replace(
        protocol,
        arms=tuple(arm for arm in protocol.arms if arm.optimizer != "codex"),
    )
