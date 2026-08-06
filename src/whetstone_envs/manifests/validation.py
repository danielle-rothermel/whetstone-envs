from collections.abc import Mapping
from types import MappingProxyType

MANIFEST_SCHEMA_VERSION = 1
_SEED_RANGE_LEN = 2


def _require_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"manifest {field_name} must be an integer"
        raise TypeError(msg)
    return value


def validate_schema_version(value: object) -> int:
    schema_version = _require_integer(
        value,
        field_name="schema_version",
    )
    if schema_version != MANIFEST_SCHEMA_VERSION:
        msg = (
            f"manifest schema_version {schema_version} is unsupported; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return schema_version


def validate_generator_version(value: object) -> str:
    if not isinstance(value, str):
        msg = "manifest generator_version must be a string"
        raise TypeError(msg)
    if not value:
        msg = "manifest generator_version must be a nonempty string"
        raise ValueError(msg)
    return value


def validate_seed_range(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != _SEED_RANGE_LEN:
        msg = "manifest seed_range must be a two-element tuple"
        raise TypeError(msg)
    start = _require_integer(value[0], field_name="seed_range")
    end = _require_integer(value[1], field_name="seed_range")
    if start >= end:
        msg = "manifest seed_range must be ordered with start less than end"
        raise ValueError(msg)
    return (start, end)


def validate_stratum_counts(
    value: object,
) -> MappingProxyType[str, int]:
    """Validate positive counts and return a detached, read-only mapping."""
    if not isinstance(value, Mapping):
        msg = "manifest stratum_counts must be an object"
        raise TypeError(msg)

    counts: dict[str, int] = {}
    for name, raw_count in value.items():
        if not isinstance(name, str):
            msg = "manifest stratum_counts names must be strings"
            raise TypeError(msg)
        if not name.strip():
            msg = "manifest stratum_counts names must be nonblank strings"
            raise ValueError(msg)
        count = _require_integer(
            raw_count,
            field_name="stratum_counts",
        )
        if count < 0:
            msg = "manifest stratum_counts values must be non-negative"
            raise ValueError(msg)
        if count == 0:
            msg = "manifest stratum_counts values must be positive"
            raise ValueError(msg)
        counts[name] = count
    if sum(counts.values()) == 0:
        msg = "manifest stratum_counts must have a positive total"
        raise ValueError(msg)
    return MappingProxyType(counts)
