from __future__ import annotations

from typing import Literal

from pydantic import StrictFloat, StrictInt, model_validator

from whetstone_envs.evidence._models import _StrictModel

SPEND_SCHEMA_VERSION = 1
RoleSpendName = Literal["task_model", "proposer"]


class RoleSpend(_StrictModel):
    role: RoleSpendName
    calls: StrictInt
    cached_calls: StrictInt
    input_tokens: StrictInt
    output_tokens: StrictInt
    priced_calls: StrictInt
    unpriced_calls: StrictInt
    rows_missing_token_breakdown: StrictInt
    # None means a complete USD total is unknown.
    usd: StrictFloat | None

    @model_validator(mode="after")
    def _validate_role_spend(self) -> RoleSpend:
        counts = (
            self.calls,
            self.cached_calls,
            self.input_tokens,
            self.output_tokens,
            self.priced_calls,
            self.unpriced_calls,
            self.rows_missing_token_breakdown,
        )
        if any(value < 0 for value in counts):
            raise ValueError("spend counters must be non-negative")
        if self.priced_calls + self.unpriced_calls != self.calls:
            raise ValueError(
                "priced and unpriced calls must exhaust billable calls"
            )
        if self.usd is not None and self.usd < 0:
            raise ValueError("reported spend must be non-negative")
        if self.usd is not None and self.unpriced_calls:
            raise ValueError(
                "a run with unpriced calls reports no total spend"
            )
        return self


class RunSpend(_StrictModel):
    schema_version: Literal[1]
    task_model: RoleSpend
    proposer: RoleSpend

    @model_validator(mode="after")
    def _validate_roles(self) -> RunSpend:
        if self.task_model.role != "task_model":
            raise ValueError("task model spend must carry the task_model role")
        if self.proposer.role != "proposer":
            raise ValueError("proposer spend must carry the proposer role")
        return self


class EvalSpend(_StrictModel):
    schema_version: Literal[1]
    task_model: RoleSpend

    @model_validator(mode="after")
    def _validate_roles(self) -> EvalSpend:
        if self.task_model.role != "task_model":
            raise ValueError("task model spend must carry the task_model role")
        return self
