from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ReportRef(_StrictModel):
    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate_ref(self) -> ReportRef:
        if not self.schema_name.strip() or not self.content_hash.strip():
            raise ValueError("report refs require nonblank schema and hash")
        return self
