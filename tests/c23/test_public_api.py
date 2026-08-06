import inspect

from whetstone_envs import c23


def test_public_api_is_owned_by_c23() -> None:
    assert c23.__all__ == [
        "GENERATOR_VERSION",
        "PROBES",
        "default_split_sizes",
        "generate_pool",
        "score_gold",
    ]
    assert tuple(inspect.signature(c23.generate_pool).parameters) == (
        "n_per_stratum",
    )
