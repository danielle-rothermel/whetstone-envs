"""RFC 8785 canonicalization for generated C11 inputs."""

import json

import rfc8785


def canonicalize(input_json: str) -> str:
    """Return the RFC 8785 representation of one JSON text."""
    if not isinstance(input_json, str):
        msg = "input_json must be a string"
        raise TypeError(msg)
    value = json.loads(input_json)
    return rfc8785.dumps(value).decode("utf-8")
