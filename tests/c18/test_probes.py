from __future__ import annotations

from whetstone_envs.c18 import PROBES
from whetstone_envs.instances import Instance, make_instance

_QUESTION = "ZZ_PUBLIC_QUESTION_ZZ"
_QUERY = "ZZ_PUBLIC_QUERY_ZZ"


def _instance(gold: str = "True") -> Instance:
    return make_instance(
        id="c18-probe",
        seed=1_000_000_000,
        strata="D1",
        prompt_inputs={"question": _QUESTION, "query": _QUERY},
        gold=gold,
    )


def test_probes_render_public_inputs_without_gold() -> None:
    sentinel = "ZZ_SECRET_LABEL_ZZ"
    instance = _instance(gold=sentinel)
    for rendered in (
        PROBES.render_naive(instance),
        PROBES.render_ceiling(instance),
    ):
        assert _QUESTION in rendered
        assert _QUERY in rendered
        assert sentinel not in rendered
