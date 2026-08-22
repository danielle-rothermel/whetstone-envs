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

__all__ = [
    "CI_LEVEL",
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
    "next_k_cal",
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

#: The four hypotheses. Order is the Holm family's input order.
REAL_OPTIMIZER_ARM_IDS: tuple[str, ...] = (
    "copro",
    "miprov2",
    "gepa",
    "codex",
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

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise ValueError("arm ids must be nonblank")
        if self.k_run < 1:
            raise ValueError(f"arm {self.arm_id!r} must run at least once")
        if len(self.seeds) != self.k_run:
            raise ValueError(
                f"arm {self.arm_id!r} declares {self.k_run} runs but "
                f"{len(self.seeds)} seeds"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"arm {self.arm_id!r} repeats a seed")


def default_arms(*, stage: StageId) -> tuple[ArmSpec, ...]:
    """Every arm the study runs at ``stage``, in report order.

    Nulls are never dropped, so they are built from the same table as the
    real optimizers rather than appended conditionally.
    """
    arms: list[ArmSpec] = []
    for optimizer in (*REAL_OPTIMIZER_ARM_IDS, *NULL_ARM_IDS):
        kind = (
            ArmKind.REAL
            if optimizer in set(REAL_OPTIMIZER_ARM_IDS)
            else ArmKind.NULL
        )
        arms.append(
            ArmSpec(
                arm_id=optimizer,
                optimizer=optimizer,
                kind=kind,
                k_run=k_run_for(optimizer, stage=stage),
                seeds=arm_seeds(optimizer, stage=stage),
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

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        """The runner's ``(internal, official, held_out)`` triple."""
        return (self.internal.size, self.official.size, self.held_out.size)

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
