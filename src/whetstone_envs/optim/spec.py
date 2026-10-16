from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone_envs.evaluation.concurrency import DEFAULT_PROVIDER_CONCURRENCY
from whetstone_envs.evaluation.runtime import TransportKind
from whetstone_envs.optim.codex import (
    CODEX_DEFAULT_BINARY,
    CodexReasoningEffort,
)
from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_FULL_EVAL_STEPS,
    DEFAULT_MIPROV2_MINIBATCH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    Miprov2DemoMode,
)
from whetstone_envs.optim.split import MIN_COPRO_BREADTH, MIN_COPRO_DEPTH

DEFAULT_COPRO_BREADTH = 2
DEFAULT_COPRO_DEPTH = 1
GEPA_DEFAULT_SEED = 0
MIPROV2_DEFAULT_SEED = 9


@verify(UNIQUE)
class OptimizerKind(StrEnum):
    CODEX = auto()
    COPRO = auto()
    GEPA = auto()
    MIPROV2 = auto()


class _RunInput(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )


class TrainValSizes(_RunInput):
    train_size: int = Field(ge=1)
    val_size: int = Field(ge=1)


class CoproSettings(_RunInput):
    kind: Literal[OptimizerKind.COPRO] = OptimizerKind.COPRO
    breadth: int = Field(default=DEFAULT_COPRO_BREADTH, ge=MIN_COPRO_BREADTH)
    depth: int = Field(default=DEFAULT_COPRO_DEPTH, ge=MIN_COPRO_DEPTH)


class GepaSettings(_RunInput):
    kind: Literal[OptimizerKind.GEPA] = OptimizerKind.GEPA
    partition: TrainValSizes
    seed: int = GEPA_DEFAULT_SEED
    max_metric_calls: int | None = Field(default=None, ge=1)
    reflection_minibatch_size: int | None = Field(default=None, ge=1)


class Miprov2Settings(_RunInput):
    kind: Literal[OptimizerKind.MIPROV2] = OptimizerKind.MIPROV2
    partition: TrainValSizes
    seed: int = MIPROV2_DEFAULT_SEED
    demo_mode: Miprov2DemoMode = Miprov2DemoMode.FEWSHOT
    minibatch: bool = DEFAULT_MIPROV2_MINIBATCH
    minibatch_size: int | None = Field(default=None, ge=1)
    minibatch_full_eval_steps: int = Field(
        default=DEFAULT_MIPROV2_FULL_EVAL_STEPS, ge=1
    )
    num_trials: int = Field(default=DEFAULT_MIPROV2_NUM_TRIALS, ge=1)
    num_candidates: int = Field(default=DEFAULT_MIPROV2_NUM_CANDIDATES, ge=1)

    @model_validator(mode="after")
    def _validate_minibatch(self) -> Miprov2Settings:
        if self.minibatch and self.minibatch_size is None:
            raise ValueError(
                "minibatching requires an explicit minibatch_size"
            )
        return self


class CodexSettings(_RunInput):
    kind: Literal[OptimizerKind.CODEX] = OptimizerKind.CODEX
    capacity: int | None = Field(default=None, ge=1)
    binary: str = CODEX_DEFAULT_BINARY
    model: str | None = None
    reasoning_effort: CodexReasoningEffort = CodexReasoningEffort.MEDIUM
    wall_seconds: float | None = Field(default=None, gt=0)
    allow_real_codex: bool = False


OptimizerSettings = Annotated[
    CoproSettings | GepaSettings | Miprov2Settings | CodexSettings,
    Field(discriminator="kind"),
]


class RunSpec(_RunInput):
    optimizer: OptimizerSettings
    transport: TransportKind
    output_dir: Path | None = None
    run_id: str | None = None
    proposer_model: str | None = None
    provider_concurrency: int = Field(
        default=DEFAULT_PROVIDER_CONCURRENCY, ge=1
    )


__all__ = [
    "DEFAULT_COPRO_BREADTH",
    "DEFAULT_COPRO_DEPTH",
    "GEPA_DEFAULT_SEED",
    "MIPROV2_DEFAULT_SEED",
    "CodexSettings",
    "CoproSettings",
    "GepaSettings",
    "Miprov2Settings",
    "OptimizerKind",
    "OptimizerSettings",
    "RunSpec",
    "TrainValSizes",
]
