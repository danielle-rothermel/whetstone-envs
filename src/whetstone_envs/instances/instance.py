from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


class Instance(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    id: str = Field(min_length=1)
    seed: int
    strata: tuple[str, ...] = Field(min_length=1)
    prompt_inputs: Mapping[str, str] = Field(default_factory=dict)
    gold: str = ""

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if type(value) is not int:
            msg = "Instance.seed must be an int"
            raise ValueError(msg)
        return value

    @field_validator("strata")
    @classmethod
    def _validate_strata(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not label.strip() for label in value):
            msg = "Instance stratum labels must be non-empty and not blank"
            raise ValueError(msg)
        return tuple(dict.fromkeys(value))

    @field_validator("prompt_inputs")
    @classmethod
    def _freeze_prompt_inputs(
        cls,
        value: Mapping[str, str],
    ) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("prompt_inputs")
    def _serialize_prompt_inputs(
        self,
        value: Mapping[str, str],
    ) -> dict[str, str]:
        return dict(value)

    @property
    def public_prompt_identity(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.prompt_inputs.items()))

    def __hash__(self) -> int:
        # MappingProxyType is unhashable, so hash its public identity instead.
        return hash(
            (
                self.id,
                self.seed,
                self.strata,
                self.public_prompt_identity,
                self.gold,
            ),
        )
