from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify

RESERVED_SEED_MAX = 100_000_000
PRONTOQA_SEED_MAX = (1 << 32) - 1


@verify(UNIQUE)
class DistractorMode(StrEnum):
    """The closed set of PrOntoQA distractor modes used by C18."""

    NONE = "none"
    RELEVANT = "relevant"


@dataclass(frozen=True, slots=True)
class DepthCategory:
    """One deduction-depth category and its upstream distractor policy."""

    hops: int
    distractors: DistractorMode

    def __post_init__(self) -> None:
        if type(self.hops) is not int:
            msg = "C18 depth hops must be an int"
            raise TypeError(msg)
        if self.hops <= 0:
            msg = f"C18 depth hops must be positive, got {self.hops}"
            raise ValueError(msg)
        if not isinstance(self.distractors, DistractorMode):
            msg = "C18 distractors must be a DistractorMode"
            raise TypeError(msg)

    @property
    def label(self) -> str:
        return f"D{self.hops}"


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Per-category split proportions for the three evaluation cohorts."""

    internal_eval: int
    official: int
    held_out: int

    def __post_init__(self) -> None:
        values = (self.internal_eval, self.official, self.held_out)
        if any(type(value) is not int for value in values):
            msg = "C18 split counts must be integers"
            raise TypeError(msg)
        if any(value < 0 for value in values):
            msg = f"C18 split counts must be non-negative, got {values!r}"
            raise ValueError(msg)
        if sum(values) == 0:
            msg = "C18 split counts must have a positive total"
            raise ValueError(msg)

    def scale(self, n_per_category: int) -> tuple[int, int, int]:
        """Scale cumulative split proportions to an actual category size."""
        _validate_positive_count(n_per_category, name="n_per_category")
        total = self.internal_eval + self.official + self.held_out
        internal = n_per_category * self.internal_eval // total
        official_boundary = (
            n_per_category * (self.internal_eval + self.official) // total
        )
        return (
            internal,
            official_boundary - internal,
            n_per_category - official_boundary,
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """A complete immutable C18 dataset-generation configuration."""

    generator_version: str
    seed_start: int
    n_per_category: int
    categories: tuple[DepthCategory, ...]
    split: SplitPlan

    def __post_init__(self) -> None:
        if not isinstance(self.generator_version, str):
            msg = "C18 generator_version must be a string"
            raise TypeError(msg)
        if not self.generator_version:
            msg = "C18 generator_version must be nonempty"
            raise ValueError(msg)
        if type(self.seed_start) is not int:
            msg = "C18 seed_start must be an int"
            raise TypeError(msg)
        if self.seed_start <= RESERVED_SEED_MAX:
            msg = (
                f"C18 seed_start must be above {RESERVED_SEED_MAX}, "
                f"got {self.seed_start}"
            )
            raise ValueError(msg)
        _validate_positive_count(
            self.n_per_category,
            name="n_per_category",
        )
        if not isinstance(self.categories, tuple):
            msg = "C18 categories must be a tuple of DepthCategory values"
            raise TypeError(msg)
        if not self.categories:
            msg = "C18 requires at least one depth category"
            raise ValueError(msg)
        if any(
            not isinstance(category, DepthCategory)
            for category in self.categories
        ):
            msg = "C18 categories must contain only DepthCategory values"
            raise TypeError(msg)
        depths = tuple(category.hops for category in self.categories)
        if len(depths) != len(set(depths)):
            msg = f"C18 requires distinct depth categories, got {depths!r}"
            raise ValueError(msg)
        final_seed = self.seed_start + len(self.categories) - 1
        if final_seed > PRONTOQA_SEED_MAX:
            msg = (
                f"C18 allocated seeds must not exceed {PRONTOQA_SEED_MAX}, "
                f"got final seed {final_seed}"
            )
            raise ValueError(msg)
        if not isinstance(self.split, SplitPlan):
            msg = "C18 split must be a SplitPlan"
            raise TypeError(msg)

    @property
    def seed_range(self) -> tuple[int, int]:
        return (self.seed_start, self.seed_start + len(self.categories))


def _validate_positive_count(value: int, *, name: str) -> None:
    if type(value) is not int:
        msg = f"{name} must be an int"
        raise TypeError(msg)
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


DEFAULT_CONFIG = GenerationConfig(
    generator_version="c18-generate-1",
    seed_start=1_000_000_000,
    n_per_category=30,
    categories=tuple(
        DepthCategory(hops, DistractorMode.RELEVANT) for hops in (1, 2, 3, 5)
    ),
    split=SplitPlan(internal_eval=6, official=12, held_out=12),
)

HARD_CONFIG = GenerationConfig(
    generator_version="c18-generate-1+hard",
    seed_start=2_000_000_000,
    n_per_category=20,
    categories=(
        DepthCategory(5, DistractorMode.RELEVANT),
        DepthCategory(8, DistractorMode.NONE),
        DepthCategory(10, DistractorMode.NONE),
    ),
    split=SplitPlan(internal_eval=2, official=6, held_out=12),
)


__all__ = [
    "DEFAULT_CONFIG",
    "HARD_CONFIG",
    "DepthCategory",
    "DistractorMode",
    "GenerationConfig",
    "SplitPlan",
]
