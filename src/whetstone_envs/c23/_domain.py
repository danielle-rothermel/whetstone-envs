from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class RuleFamily(StrEnum):
    """Supported single-rule transducer families."""

    ISL = "ISL"
    L_OSL = "L-OSL"
    R_OSL = "R-OSL"


@dataclass(frozen=True, slots=True)
class RuleConfiguration:
    family: RuleFamily
    context_length: int


@dataclass(frozen=True, slots=True)
class StratumConfiguration:
    label: str
    rule: RuleConfiguration
    seed: int


@dataclass(frozen=True, slots=True)
class GenerationConfiguration:
    vocab: tuple[str, ...]
    strata: tuple[StratumConfiguration, ...]
    demonstrations_per_instance: int
    maximum_query_length: int
    attempts_per_instance: int


@dataclass(frozen=True, slots=True)
class Hypothesis:
    configuration: RuleConfiguration
    context: str
    replacement: str


@dataclass(frozen=True, slots=True)
class Demonstration:
    input: str
    output: str


@dataclass(frozen=True, slots=True)
class GeneratedTask:
    hypothesis: Hypothesis
    demonstrations: tuple[Demonstration, ...]
    query: str
    gold: str
