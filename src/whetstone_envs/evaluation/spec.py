from __future__ import annotations

from pathlib import Path
from typing import Literal

from dr_providers import ReasoningEffort
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from whetstone_envs.evidence import EvalRoleName
from whetstone_envs.optim.concurrency import DEFAULT_PROVIDER_CONCURRENCY


class _EvaluationInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CandidateInput(_EvaluationInput):
    name: str
    source: Literal["naive", "ceiling", "custom"]
    template: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate names must be nonblank")
        return value


class C19EvalSpec(_EvaluationInput):
    transport: Literal["fake", "openrouter"]
    role: EvalRoleName
    candidates: tuple[CandidateInput, ...] = ()
    repeats: int = Field(default=1, ge=1)
    split_sizes: tuple[int, int, int] = (20, 20, 0)
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"
    task_reasoning_effort: ReasoningEffort | None = None
    provider_concurrency: int = Field(
        default=DEFAULT_PROVIDER_CONCURRENCY, ge=1
    )

    @field_validator("split_sizes")
    @classmethod
    def _validate_split_sizes(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(size < 0 for size in value):
            raise ValueError("split sizes must be nonnegative")
        if value[0] == 0 or value[1] == 0:
            raise ValueError(
                "whetstone_envs.c19 experiment requires non-empty "
                "internal and official splits"
            )
        return value

    @model_validator(mode="after")
    def _validate_spec(self) -> C19EvalSpec:
        if self.role == "held_out" and self.split_sizes[2] == 0:
            raise ValueError(
                "this experiment has no held_out split: "
                "split sizes must give the held_out role a positive size"
            )
        names: set[str] = set()
        for candidate in self.candidates:
            if candidate.name in names:
                raise ValueError(
                    f"duplicate candidate name {candidate.name!r}"
                )
            names.add(candidate.name)
        return self


__all__ = ["C19EvalSpec", "CandidateInput"]
