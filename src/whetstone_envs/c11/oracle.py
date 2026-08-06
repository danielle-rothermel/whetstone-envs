"""RFC 8785 canonicalization for generated C11 inputs."""

import json

import rfc8785


def _object_without_duplicate_names(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, member in pairs:
        if name in value:
            msg = f"duplicate JSON property name {name!r}"
            raise ValueError(msg)
        value[name] = member
    return value


def canonicalize(input_json: str) -> str:
    """Return the RFC 8785 representation of one JSON text."""
    if not isinstance(input_json, str):
        msg = "input_json must be a string"
        raise TypeError(msg)
    value = json.loads(
        input_json,
        object_pairs_hook=_object_without_duplicate_names,
        parse_int=float,
    )
    return rfc8785.dumps(value).decode("utf-8")
