from whetstone_envs.probes import normalize


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 for a normalized match, otherwise 0."""
    return int(normalize(prediction) == normalize(gold))
