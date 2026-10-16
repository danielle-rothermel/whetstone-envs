from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.evidence import ReportRef

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef


def _ref(value: TypedRef | None) -> ReportRef | None:
    if value is None:
        return None
    return ReportRef(
        schema_name=value.schema_name,
        content_hash=str(value.content_hash),
    )


def _required_ref(value: TypedRef) -> ReportRef:
    result = _ref(value)
    assert result is not None
    return result
