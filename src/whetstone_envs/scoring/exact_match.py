"""Deterministic binary exact-match scoring."""

from whetstone_envs.probes import normalize


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if ``prediction`` equals ``gold`` after normalization.

    Both sides are passed through :func:`whetstone_envs.probes.normalize` so
    fence and whitespace handling is identical everywhere. The result is
    exactly ``0`` or ``1`` -- no partial credit.
    """
    return int(normalize(prediction) == normalize(gold))
