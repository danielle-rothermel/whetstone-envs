from whetstone_envs.probes import normalize


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))
