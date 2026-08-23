"""The study's design record: what is fixed before any provider spend.

Everything here is pre-registered. A ``StudySpec`` names the population, the
splits, the repeat counts, and every arm the study will run, and it is written
before Stage 0 rather than derived after seeing results -- which is what makes
the Stage gates and the null-triggered downgrade honest rather than
post-hoc.

These are plain frozen dataclasses with no serialization of their own. The
study manifest owns the persisted form; this module owns the design and its
validity rules, so a spec that could not be run truthfully is refused here
rather than half-way through a paid stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from whetstone_envs.optim.split import TRAIN_VAL_OPTIMIZERS
from whetstone_envs.optim.study.manifest import (
    PreRegistrationViolationError,
    read_study_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from whetstone_envs.optim.study.manifest import (
        ArmRecord,
        DesignRecord,
        SplitRecord,
        StudyManifest,
    )

__all__ = [
    "CI_LEVEL",
    "CODEX_ARM_ID",
    "CODEX_EVALUATE_CALL_CAP",
    "CORRECTION_RULE",
    "HOLM_FAMILY_SIZE",
    "K_CAL_CAP",
    "K_CAL_INITIAL",
    "K_RUN_NULL_A",
    "K_RUN_NULL_B",
    "K_RUN_PILOT",
    "K_RUN_STAGE2",
    "NULL_ARM_IDS",
    "PROTOCOL_SPLIT_SIZES",
    "PROTOCOL_TRAIN_SIZE",
    "PROTOCOL_VAL_SIZE",
    "REAL_OPTIMIZER_ARM_IDS",
    "RESAMPLES",
    "SEED_RANGE_BY_OPTIMIZER",
    "ArmKind",
    "ArmSpec",
    "SplitSpec",
    "StageId",
    "StudySpec",
    "arm_seeds",
    "default_arms",
    "k_run_for",
    "load_study_spec",
    "next_k_cal",
    "require_pinned_arms",
    "require_pinned_codex_agent_model",
    "spec_from_manifest",
]


#: The Stage-0 calibration repeat count (D1). This is a *measurement input*,
#: never the design ``K_REPEAT``: the two are separate quantities and the
#: manifest names them separately. An even count is what enables the
#: split-half stability check the doubling rule reads.
K_CAL_INITIAL = 4

#: The doubling rule's ceiling (D1). Doubling runs 4 -> 8 -> 16 and stops;
#: past 16 the calibration costs more than the design it is calibrating.
K_CAL_CAP = 16

#: Runs per real optimizer at each stage. Stage 1's two runs count toward
#: Stage 2 -- same code, same splits, the first two seeds reused -- so Stage 2
#: adds three runs per arm rather than five.
K_RUN_PILOT = 2
K_RUN_STAGE2 = 5

#: Runs per null (D4). Null-A is a statistical control and needs the full
#: repeat count; null-B is a single pipeline-overhead assertion, so one run
#: says everything a second would.
K_RUN_NULL_A = 5
K_RUN_NULL_B = 1

#: The Codex arm's admitted evaluate-call cap (D2).
CODEX_EVALUATE_CALL_CAP = 8

#: Bootstrap interval settings, pre-registered so the interval is not chosen
#: after seeing the deltas.
CI_LEVEL = 0.95
RESAMPLES = 10_000

#: Multiplicity correction over the four real optimizers. Nulls are controls,
#: not hypotheses, so they are uncorrected and do not enter the family.
CORRECTION_RULE = "holm-bonferroni"
HOLM_FAMILY_SIZE = 4


@verify(UNIQUE)
class StageId(StrEnum):
    """The study's three spending stages, in order."""

    STAGE0 = "stage0"
    STAGE1 = "stage1"
    STAGE2 = "stage2"


@verify(UNIQUE)
class ArmKind(StrEnum):
    """What an arm is evidence *for*.

    The distinction is load-bearing for the statistics: only ``REAL`` arms are
    hypotheses and enter the Holm family, and only ``NULL`` arms can trigger
    the study-wide downgrade.
    """

    REAL = auto()
    NULL = auto()


#: Disjoint per-optimizer seed ranges. Disjointness is what lets a run's seed
#: identify its arm in a flat artifact directory, and null-B takes a single
#: seed because it runs once.
SEED_RANGE_BY_OPTIMIZER: dict[str, int] = {
    "copro": 1000,
    "miprov2": 2000,
    "gepa": 3000,
    "codex": 4000,
    "null-random": 5000,
    "null-identity": 6000,
}

#: The Codex arm, named where the study's spend guard reads it. It is the
#: only arm whose runs can bill a foreign subscription, so the id is an
#: owned constant rather than a literal repeated at each check.
CODEX_ARM_ID = "codex"

#: The four hypotheses. Order is the Holm family's input order.
REAL_OPTIMIZER_ARM_IDS: tuple[str, ...] = (
    "copro",
    "miprov2",
    "gepa",
    CODEX_ARM_ID,
)

#: The two controls, in the order the report presents them.
NULL_ARM_IDS: tuple[str, ...] = ("null-random", "null-identity")


def k_run_for(optimizer: str, *, stage: StageId) -> int:
    """Runs this optimizer gets at ``stage``.

    Null-B is the exception at every stage: one run is the whole control, so
    a pilot does not run it twice and Stage 2 does not run it five times.
    """
    if optimizer == "null-identity":
        return K_RUN_NULL_B
    if stage is StageId.STAGE1:
        return K_RUN_PILOT
    if stage is StageId.STAGE2:
        return K_RUN_NULL_A if optimizer == "null-random" else K_RUN_STAGE2
    raise ValueError(f"stage {stage.value!r} runs no optimizers")


def arm_seeds(optimizer: str, *, stage: StageId) -> tuple[int, ...]:
    """The seeds this optimizer runs at ``stage``, from its own range.

    Stage 2 returns the full seed set including Stage 1's, because Stage 1's
    runs count toward Stage 2 rather than being repeated. A caller that has
    already run Stage 1 executes the difference; the seeds are the same
    either way, which is what makes "reused" checkable rather than asserted.
    """
    try:
        base = SEED_RANGE_BY_OPTIMIZER[optimizer]
    except KeyError as error:
        raise ValueError(f"unknown optimizer {optimizer!r}") from error
    runs = k_run_for(optimizer, stage=stage)
    return tuple(base + offset for offset in range(runs))


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """One split's pre-registered size and its measured task identity.

    ``task_hashes`` is empty until Stage 0 builds the experiment and reads
    them off the engine; the size is fixed beforehand. Recording both lets
    the leakage check prove disjointness on identity rather than on size.
    """

    role: str
    size: int
    task_hashes: tuple[str, ...] = ()
    eval_config_hash: str | None = None

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"split {self.role!r} size must be non-negative")
        if self.task_hashes and len(self.task_hashes) != self.size:
            raise ValueError(
                f"split {self.role!r} carries {len(self.task_hashes)} task "
                f"hashes for a declared size of {self.size}"
            )
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError(f"split {self.role!r} repeats a task hash")


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm: an optimizer, its demo mode, and how many runs it gets."""

    arm_id: str
    optimizer: str
    kind: ArmKind
    k_run: int
    seeds: tuple[int, ...]
    demo_mode: str | None = None
    #: MIPROv2 search shape and split, when this arm sets them. ``None``
    #: keeps the runner's own default, which is what every arm the study
    #: builds today does; they are here so an arm can request the
    #: protocol's auto-light shape without the runner hardcoding it.
    #: Refused on any other optimizer, because a setting that looks
    #: honoured but is not is how a study comes to misdescribe its own arm.
    miprov2_num_trials: int | None = None
    miprov2_num_candidates: int | None = None
    #: Whether this arm minibatches, and at what size.
    #:
    #: Design, not a runtime knob, and pinned like the train/val split: an
    #: arm that evaluated every trial on the whole valset and an arm that
    #: evaluated on a sampled batch of it bought different evidence for
    #: the same claim, so a batch size chosen after a result is the
    #: post-hoc adjustment the pre-registration exists to forbid.
    #:
    #: The two travel together. ``miprov2_minibatch`` on without a size
    #: resolves the batch to the whole valset -- minibatching in name only
    #: -- so an arm that turns it on states the size, and an arm that
    #: leaves it off states neither.
    miprov2_minibatch: bool = False
    miprov2_minibatch_size: int | None = None
    #: The arm's explicit train/val partition of the internal split,
    #: required for every optimizer with a train/val concept and refused on
    #: the others. Design fields, not runtime knobs: they enter the
    #: pre-registration hash, so an arm cannot quietly change what it
    #: trained and scored on between pre-registration and the run.
    train_size: int | None = None
    val_size: int | None = None

    def _validate_miprov2(self) -> None:
        """Refuse MIPROv2 settings this arm could not honestly claim.

        Refused on another optimizer rather than silently ignored, for the
        runner's reason: a setting that looks honoured but is not is how a
        study comes to misdescribe its own arm.
        """
        miprov2_settings = (
            self.miprov2_num_trials,
            self.miprov2_num_candidates,
            self.miprov2_minibatch_size,
        )
        if (
            any(value is not None for value in miprov2_settings)
            or self.miprov2_minibatch
        ) and self.optimizer != "miprov2":
            raise ValueError(
                f"arm {self.arm_id!r} sets MIPROv2 settings but runs "
                f"optimizer {self.optimizer!r}"
            )
        if self.miprov2_num_trials is not None and self.miprov2_num_trials < 1:
            raise ValueError(
                f"arm {self.arm_id!r} miprov2_num_trials must be at least 1"
            )
        if (
            self.miprov2_num_candidates is not None
            and self.miprov2_num_candidates < 1
        ):
            raise ValueError(
                f"arm {self.arm_id!r} miprov2_num_candidates must be at "
                "least 1"
            )
        if self.miprov2_minibatch and self.miprov2_minibatch_size is None:
            # The runner's refusal, restated at the design level: an arm
            # that pre-registered "minibatch on" and no size registered a
            # shape whose batch is the whole valset.
            raise ValueError(
                f"arm {self.arm_id!r} sets miprov2_minibatch and must "
                "declare miprov2_minibatch_size; left unset the batch is "
                "the whole validation split"
            )
        if (
            not self.miprov2_minibatch
            and self.miprov2_minibatch_size is not None
        ):
            raise ValueError(
                f"arm {self.arm_id!r} declares a miprov2_minibatch_size "
                "without turning miprov2_minibatch on"
            )
        if (
            self.miprov2_minibatch_size is not None
            and self.miprov2_minibatch_size < 1
        ):
            raise ValueError(
                f"arm {self.arm_id!r} miprov2_minibatch_size must be at "
                "least 1"
            )

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise ValueError("arm ids must be nonblank")
        self._validate_miprov2()
        split_supplied = (
            self.train_size is not None or self.val_size is not None
        )
        if self.optimizer in TRAIN_VAL_OPTIMIZERS:
            if self.train_size is None or self.val_size is None:
                raise ValueError(
                    f"arm {self.arm_id!r} runs optimizer "
                    f"{self.optimizer!r} and must declare train_size and "
                    "val_size"
                )
            if self.train_size < 1:
                raise ValueError(
                    f"arm {self.arm_id!r} train_size must be at least 1"
                )
            if self.val_size < 1:
                raise ValueError(
                    f"arm {self.arm_id!r} val_size must be at least 1"
                )
        elif split_supplied:
            raise ValueError(
                f"arm {self.arm_id!r} sets a train/val split but runs "
                f"optimizer {self.optimizer!r}"
            )
        if self.k_run < 1:
            raise ValueError(f"arm {self.arm_id!r} must run at least once")
        if len(self.seeds) != self.k_run:
            raise ValueError(
                f"arm {self.arm_id!r} declares {self.k_run} runs but "
                f"{len(self.seeds)} seeds"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"arm {self.arm_id!r} repeats a seed")


#: The study protocol's three split sizes, in role order: internal,
#: official, held-out.
#:
#: This is the Step 10 study's own pre-registration, and it is **not** the
#: c19 generation default. ``whetstone_envs.c19.generation``'s
#: ``DEFAULT_SPLIT_SIZES`` is ``(88, 132, 132)``: it describes what the
#: generator hands back when nobody asks for anything, while the protocol
#: pre-registered a held-out split of 440 so the design could resolve an
#: MDE of 0.0622 at ``tau^2 = 0.05`` -- which 132 cannot. A study manifest
#: records these three sizes in its ``splits`` block and every stage reads
#: them from there, so this constant is the protocol's *declaration* of
#: them, pinned by a golden test.
PROTOCOL_SPLIT_SIZES: tuple[int, int, int] = (88, 132, 440)

#: The protocol's train/val partition of the internal 88: half and half.
#:
#: An even split is the cheapest partition that keeps both halves
#: meaningful. MIPROv2 evaluates every trial on the whole valset by
#: default and GEPA scores its Pareto frontier there, so the valset is the
#: per-trial cost driver -- 44 keeps one full pass affordable while still
#: leaving the bootstrap a 44-task trainset to draw demonstrations from.
#: Splitting 88 any further would buy trainset size at the price of a
#: valset too small to separate arms. The two also *cover* the internal
#: split exactly, which GEPA requires: its data registry is built from the
#: whole internal split and its trainset and valset must partition it.
PROTOCOL_TRAIN_SIZE = 44
PROTOCOL_VAL_SIZE = 44


def default_arms(*, stage: StageId) -> tuple[ArmSpec, ...]:
    """Every arm the study runs at ``stage``, in report order.

    Nulls are never dropped, so they are built from the same table as the
    real optimizers rather than appended conditionally.

    The optimizers with a train/val concept carry the protocol's pinned
    partition; the others must not carry one at all, so the sizes are
    forwarded per arm rather than set for every arm.
    """
    arms: list[ArmSpec] = []
    for optimizer in (*REAL_OPTIMIZER_ARM_IDS, *NULL_ARM_IDS):
        kind = (
            ArmKind.REAL
            if optimizer in set(REAL_OPTIMIZER_ARM_IDS)
            else ArmKind.NULL
        )
        splits = (
            {
                "train_size": PROTOCOL_TRAIN_SIZE,
                "val_size": PROTOCOL_VAL_SIZE,
            }
            if optimizer in TRAIN_VAL_OPTIMIZERS
            else {}
        )
        arms.append(
            ArmSpec(
                arm_id=optimizer,
                optimizer=optimizer,
                kind=kind,
                k_run=k_run_for(optimizer, stage=stage),
                seeds=arm_seeds(optimizer, stage=stage),
                **splits,
            )
        )
    return tuple(arms)


@dataclass(frozen=True, slots=True)
class StudySpec:
    """The whole pre-registered design of one study.

    ``k_cal`` is Stage 0's measurement input and ``k_repeat`` is the design
    repeat count every reported evaluation uses. They are separate fields
    because conflating them is exactly the error the Stage-0 gate's
    arithmetic is sensitive to: ``tau^2``'s contamination correction is
    ``2 sigma^2 / k_cal``, so borrowing the design K there biases the gate
    optimistically.
    """

    study_id: str
    family: str
    n_per_stratum: int
    pool_seed_start: int
    internal: SplitSpec
    official: SplitSpec
    held_out: SplitSpec
    task_model: str
    proposer_model: str
    #: The Codex agent's own model, pre-registered rather than defaulted.
    #:
    #: This is **design**: the agent model decides what the Codex arm's
    #: proposer *is*, so a study that reported one agent and ran another
    #: would be comparing a different treatment against its own anchors.
    #: The manifest's hand-authored ``models`` block names it, this field
    #: carries it, and :func:`require_pinned_codex_agent_model` refuses a
    #: stage whose resolved control disagrees.
    #:
    #: ``None`` on a study that declares no Codex arm -- there is no agent
    #: to pin -- and required on one that does.
    #: :data:`~whetstone_envs.optim.codex.CODEX_DEFAULT_AGENT_MODEL` stays
    #: what it always was: the *runner's* default for a single run nobody
    #: pre-registered, never the study's.
    codex_agent_model: str | None = None
    k_cal: int = K_CAL_INITIAL
    k_repeat: int = 3
    bootstrap_seed: int = 0
    ci_level: float = CI_LEVEL
    resamples: int = RESAMPLES
    arms: tuple[ArmSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonblank")
        if self.n_per_stratum < 1:
            raise ValueError("n_per_stratum must be at least 1")
        if not K_CAL_INITIAL <= self.k_cal <= K_CAL_CAP:
            raise ValueError(
                f"k_cal must be between {K_CAL_INITIAL} and {K_CAL_CAP}"
            )
        if self.k_cal % 2:
            # Odd counts cannot be split in half, and the doubling rule's
            # stopping check is a split-half stability check.
            raise ValueError("k_cal must be even to allow a split-half check")
        if self.k_repeat < 1:
            raise ValueError("k_repeat must be at least 1")
        if self.held_out.size < 1:
            raise ValueError("a study reports from held-out; its size is > 0")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must be in (0, 1)")
        if self.resamples < 1:
            raise ValueError("resamples must be at least 1")
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("study arms must be unique by arm_id")
        declares_codex = any(
            arm.optimizer == CODEX_ARM_ID for arm in self.arms
        )
        if declares_codex and not (self.codex_agent_model or "").strip():
            # A Codex arm whose agent model the design never named would
            # take the runner's default, and the study would then report a
            # proposer it never pre-registered.
            raise ValueError(
                "a study declaring the Codex arm pre-registers its "
                "codex_agent_model; the runner's default is a run default, "
                "not a design"
            )
        if not declares_codex and self.codex_agent_model is not None:
            # A pinned agent for an arm that does not run is a design field
            # nothing honours, which is how a manifest comes to describe a
            # study it did not perform.
            raise ValueError(
                "this study declares no Codex arm, so it pre-registers no "
                "codex_agent_model"
            )

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        """The runner's ``(internal, official, held_out)`` triple."""
        return (self.internal.size, self.official.size, self.held_out.size)

    @property
    def arm_ids(self) -> tuple[str, ...]:
        """Every arm this study runs, in report order.

        The CLI's ``plan`` reads this and ``k_run_by_arm`` rather than the
        ``ArmSpec`` records themselves: a run matrix is arms and their run
        counts, and naming only that keeps the pre-spend answer independent
        of how an arm is configured.
        """
        return tuple(arm.arm_id for arm in self.arms)

    @property
    def k_run_by_arm(self) -> dict[str, int]:
        """Each arm's run count, keyed by arm id."""
        return {arm.arm_id: arm.k_run for arm in self.arms}

    @property
    def real_arms(self) -> tuple[ArmSpec, ...]:
        """The hypotheses, in Holm-family order."""
        return tuple(arm for arm in self.arms if arm.kind is ArmKind.REAL)

    @property
    def null_arms(self) -> tuple[ArmSpec, ...]:
        """The controls."""
        return tuple(arm for arm in self.arms if arm.kind is ArmKind.NULL)

    def splits(self) -> tuple[SplitSpec, ...]:
        """Every split, in role order."""
        return (self.internal, self.official, self.held_out)


def next_k_cal(k_cal: int) -> int:
    """The doubling rule's next calibration count.

    Raises at the cap rather than returning it unchanged: a caller that
    doubled up to 16 and still failed the stability check must report the
    calibration as unstable, not silently recalibrate at the same K forever.
    """
    doubled = k_cal * 2
    if doubled > K_CAL_CAP:
        raise ValueError(
            f"k_cal is capped at {K_CAL_CAP}; an unstable split-half check "
            f"at {k_cal} is a finding, not a reason to keep doubling"
        )
    return doubled


# --------------------------------------------------------------------------
# Reading the design back off a study directory
# --------------------------------------------------------------------------


def spec_from_manifest(
    manifest: StudyManifest, *, stage: StageId = StageId.STAGE2
) -> StudySpec:
    """Recover the pre-registered design from a persisted manifest.

    The manifest is the study's own record of its design, so it is also the
    spec's source of truth once a study exists: reading the spec back from
    ``study.json`` means there is one persisted design rather than a second
    file that could disagree with it.

    ``stage`` supplies the run counts a manifest written **before** Stage 0
    does not yet carry. Those counts are the pre-registration -- five runs
    per real optimizer at Stage 2, one for null-B -- so they come from
    :func:`k_run_for`, never from how many runs an arm happens to have
    executed. Reading them off the runs would make a study that has not
    started look like a study designed to run each arm once, and ``plan``
    would then under-report the budget by the whole factor of ``K_RUN``.
    Stage 2 is the default because it is the full design; a caller pricing
    the pilot passes ``StageId.STAGE1``.

    Once Stage 0 has recorded a ``design`` block, that block wins: it is
    what the study actually pre-registered, including any adjustment the
    Stage-0 gate permitted.

    **The rebuilt split is checked against the pinned one.** Each arm's
    ``train_size``/``val_size`` are ordinary mutable record fields, while
    ``pre_registration.split_by_arm`` is immutable and hashed. Reading the
    runnable spec off the mutable side without comparing it to the pinned
    side is what would let an edited ``ArmRecord`` run MIPROv2 or GEPA at a
    partition the design never registered, under a design hash that still
    validates. :func:`_require_pinned_split` refuses that.
    """
    _require_pinned_split(manifest)
    design = manifest.design
    arms = tuple(
        ArmSpec(
            arm_id=arm.arm_id,
            optimizer=arm.optimizer,
            kind=(
                ArmKind.REAL
                if arm.optimizer in set(REAL_OPTIMIZER_ARM_IDS)
                else ArmKind.NULL
            ),
            k_run=_k_run_from(arm, design=design, stage=stage),
            seeds=_arm_seeds_from(arm, design=design, stage=stage),
            demo_mode=arm.demo_mode,
            # Read back rather than re-derived from the protocol defaults:
            # the manifest records the partition each arm was actually
            # pre-registered at, and a spec that substituted today's default
            # would let a rerun quietly measure a different design.
            train_size=arm.train_size,
            val_size=arm.val_size,
            # Read back for the split's reason: minibatching is design,
            # it is hashed into the pre-registration, and a spec rebuilt
            # without it would run an arm unbatched under a design hash
            # that says it batched.
            miprov2_minibatch=arm.minibatch,
            miprov2_minibatch_size=arm.minibatch_size,
        )
        for arm in manifest.arms
    )
    return StudySpec(
        study_id=manifest.study_id,
        family=manifest.population.family,
        n_per_stratum=manifest.population.n_per_stratum,
        pool_seed_start=manifest.population.pool_seed_start,
        internal=_split_spec("internal", manifest.splits.internal),
        official=_split_spec("official", manifest.splits.official),
        held_out=_split_spec("held_out", manifest.splits.held_out),
        task_model=manifest.models.task_model,
        proposer_model=manifest.models.proposer_model,
        # Read off the hand-authored ``models`` block, which is where the
        # study pre-registers it. Carried only when a Codex arm exists,
        # because the spec refuses a pin nothing honours -- and every
        # manifest records the field, Codex arm or not.
        codex_agent_model=(
            manifest.models.codex_agent_model
            if any(arm.optimizer == CODEX_ARM_ID for arm in manifest.arms)
            else None
        ),
        k_cal=K_CAL_INITIAL if design is None else design.k_cal,
        k_repeat=3 if design is None else design.k_repeat,
        bootstrap_seed=0 if design is None else design.bootstrap_seed,
        ci_level=CI_LEVEL if design is None else design.ci_level,
        resamples=RESAMPLES if design is None else design.resamples,
        arms=arms,
    )


def _require_pinned_split(manifest: StudyManifest) -> None:
    """Refuse arm records that disagree with the pinned ``split_by_arm``.

    The pre-registration is the truth about what partition each arm was
    registered at, and it is immutable: ``write_study_manifest`` refuses any
    write that does not carry the block back byte for byte. An
    ``ArmRecord``'s ``train_size``/``val_size`` carry no such protection --
    they are rewritten every time a stage merges runs -- so the two can
    drift apart, and every stage after Stage 0 rebuilds its runnable spec
    from the *unprotected* side.

    That drift is the same class of error
    :class:`~whetstone_envs.optim.study.manifest.PreRegistrationViolationError`
    exists for -- a study running a design other than the one it registered
    -- so it is refused as one rather than as a generic value error. A
    manifest with no pre-registration yet has nothing to disagree with,
    which is the pre-Stage-0 state.

    An arm the pinned block does not name is a *different* case and is
    left to the caller. Adding an arm and then re-pinning is exactly how
    ``stage0 --replace-design`` records an amendment, and Stage 0 rebuilds
    the spec before it writes the new block -- so refusing an unnamed arm
    here would break the one legitimate path that produces one.
    :func:`require_pinned_arms` is the check for the stages that spend,
    where an unpinned arm really would run unregistered.
    """
    pinned = manifest.pre_registration
    if pinned is None:
        return
    disagreements = [
        f"{arm.arm_id}: records {(arm.train_size, arm.val_size)}, "
        f"pre-registered {pinned.split_by_arm[arm.arm_id]}"
        for arm in manifest.arms
        if arm.arm_id in pinned.split_by_arm
        and _recorded_split(arm) != pinned.split_by_arm[arm.arm_id]
    ]
    if disagreements:
        raise PreRegistrationViolationError(
            "these arm records disagree with the pre-registered "
            "split_by_arm, so the spec they rebuild is not the design this "
            "study registered: " + "; ".join(disagreements)
        )
    batched = [
        f"{arm.arm_id}: records {arm.minibatch_size}, pre-registered "
        f"{pinned.minibatch_by_arm[arm.arm_id]}"
        for arm in manifest.arms
        if arm.arm_id in pinned.minibatch_by_arm
        and arm.minibatch_size != pinned.minibatch_by_arm[arm.arm_id]
    ]
    if batched:
        # The same class of error as the split, for the same reason: an
        # arm that evaluated each trial on a sampled batch bought
        # different evidence for the same claim than one that evaluated
        # on the whole valset, and ``minibatch_by_arm`` is hashed while
        # the arm record is not.
        raise PreRegistrationViolationError(
            "these arm records disagree with the pre-registered "
            "minibatch_by_arm, so the spec they rebuild is not the design "
            "this study registered: " + "; ".join(batched)
        )


def require_pinned_arms(manifest: StudyManifest) -> None:
    """Refuse a spending stage whose arms the pre-registration never named.

    ``split_by_arm`` names exactly the arms the design declared, so an arm
    present in the manifest and absent from it appeared *after* the design
    was pinned. Running it would spend on an arm no pre-registration ever
    fixed a partition, a run count, or a place in the correction family
    for -- which is the drift the pinned block exists to prevent, in its
    purest form.

    Separate from :func:`_require_pinned_split` and called only by the arm
    stages, because Stage 0 legitimately sees this state: adding an arm and
    re-pinning is how ``--replace-design`` records an amendment, and Stage 0
    rebuilds the spec before writing the new block.
    """
    pinned = manifest.pre_registration
    if pinned is None:
        return
    unpinned = sorted(
        arm.arm_id
        for arm in manifest.arms
        if arm.arm_id not in pinned.split_by_arm
    )
    if unpinned:
        raise PreRegistrationViolationError(
            f"these arms are not named by the pre-registration: {unpinned}. "
            "They were declared after the design was pinned, so running "
            "them would spend on a design this study never registered; "
            "re-pin with stage0 --replace-design to record the amendment"
        )


def require_pinned_codex_agent_model(
    spec: StudySpec, *, resolved: str
) -> None:
    """Refuse a Codex arm whose resolved agent differs from the design.

    The agent model is what the Codex arm's *proposer* is, so it is
    pre-registered like the splits and the run matrix rather than taken
    from whatever the runner happens to default to. The runner's
    :data:`~whetstone_envs.optim.codex.CODEX_DEFAULT_AGENT_MODEL` remains a
    run default -- the right answer for a single run nobody registered --
    and this check is what stops it from silently becoming the study's.

    ``resolved`` is what the arm's control will actually carry, resolved
    through the runner's own helper rather than assumed, so the two cannot
    drift apart if an arm ever gains an override.

    Refused as a
    :class:`~whetstone_envs.optim.study.manifest.PreRegistrationViolationError`
    rather than a generic value error: running a proposer the design never
    named is the same class of error as running an unregistered split.

    A study with no Codex arm pins nothing and is left alone.
    """
    pinned = spec.codex_agent_model
    if pinned is None:
        return
    if resolved != pinned:
        raise PreRegistrationViolationError(
            f"the Codex arm would run agent model {resolved!r}, but this "
            f"study pre-registered {pinned!r} in models.codex_agent_model. "
            "The agent model is the arm's proposer, so running another one "
            "measures a treatment this study never registered; re-pin the "
            "manifest, or run the agent the design names"
        )


def _recorded_split(arm: ArmRecord) -> tuple[int, int] | None:
    """The arm record's partition in the pinned block's own shape.

    An arm whose optimizer has no train/val concept records both as
    ``None`` and is pinned as ``None``; a half-set record is neither, and
    compares unequal to any pinned value rather than being coerced into
    one.
    """
    if arm.train_size is None and arm.val_size is None:
        return None
    if arm.train_size is None or arm.val_size is None:
        # Reported through the caller's message as a disagreement: a record
        # naming one half of a partition names no partition at all.
        return (-1, -1)
    return (arm.train_size, arm.val_size)


def _split_spec(role: str, record: SplitRecord) -> SplitSpec:
    # ``SplitRecord`` and ``SplitSpec`` carry the same three facts under the
    # same names; the manifest owns the persisted form and the spec owns the
    # design, so the translation lives here rather than either type
    # depending on the other.
    return SplitSpec(
        role=role,
        size=record.size,
        task_hashes=record.task_hashes,
        eval_config_hash=record.eval_config_hash,
    )


def _arm_seeds_from(
    arm: ArmRecord, *, design: DesignRecord | None, stage: StageId
) -> tuple[int, ...]:
    """The arm's seeds, preferring the ones its runs actually used.

    A run whose optimizer carries no control seed records ``None``, so a
    partially-run arm falls back to its pre-registered range rather than
    inventing a seed the run never had. An arm whose optimizer is not in
    the seed table is refused rather than seeded from zero: every other
    lookup of that table raises, and a silently-seeded unknown arm would
    collide with whatever else defaulted the same way.
    """
    recorded = tuple(run.seed for run in arm.runs if run.seed is not None)
    k_run = _k_run_from(arm, design=design, stage=stage)
    if len(recorded) == k_run and len(set(recorded)) == k_run:
        return recorded
    try:
        base = SEED_RANGE_BY_OPTIMIZER[arm.optimizer]
    except KeyError as error:
        raise ValueError(
            f"arm {arm.arm_id!r} names unknown optimizer {arm.optimizer!r}; "
            f"seeded optimizers are {tuple(SEED_RANGE_BY_OPTIMIZER)}"
        ) from error
    return tuple(base + offset for offset in range(k_run))


def _k_run_from(
    arm: ArmRecord, *, design: DesignRecord | None, stage: StageId
) -> int:
    """The arm's pre-registered run count at ``stage``.

    A recorded design wins, because it is what the study pre-registered.
    Absent one, the count comes from :func:`k_run_for` -- the design table
    -- and never from ``len(arm.runs)``: how many runs an arm has executed
    is progress, not design, and reading it as design makes an unstarted
    study look like a one-run-per-arm study.
    """
    if design is not None and arm.arm_id in design.k_run_by_arm:
        return design.k_run_by_arm[arm.arm_id]
    return k_run_for(arm.optimizer, stage=stage)


def load_study_spec(
    study_dir: Path, *, stage: StageId = StageId.STAGE2
) -> StudySpec:
    """Load a study directory's pre-registered design.

    This is the CLI's default spec loader: ``plan`` reads the design from
    the same ``study.json`` every other subcommand reads, so a planned
    budget and a recorded run cannot describe different studies.
    """
    return spec_from_manifest(read_study_manifest(study_dir), stage=stage)
