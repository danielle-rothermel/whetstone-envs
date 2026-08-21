from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from whetstone_envs.scoring import exact_match

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

# Module flag so in-process eval threads see the fake GEPA ceiling bias.
_PREFER_C19_CEILING_SCORE = {"on": False}


@contextmanager
def prefer_c19_ceiling_score() -> Iterator[None]:
    previous = _PREFER_C19_CEILING_SCORE["on"]
    _PREFER_C19_CEILING_SCORE["on"] = True
    try:
        yield
    finally:
        _PREFER_C19_CEILING_SCORE["on"] = previous


class ExactMatchEvalProcedureRunner:
    """Zero-arg eval-node runner that scores generations with exact match.

    Workers reconstruct this type via ``runner_type()`` and transfer no
    constructor state.
    """

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: object,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        generation = node_inputs.get("provider_generation")
        text = (
            generation
            if isinstance(generation, str)
            else str(generation or "")
        )
        raw_gold = getattr(task, "gold", "")
        gold = raw_gold if isinstance(raw_gold, str) else ""
        if _PREFER_C19_CEILING_SCORE["on"]:
            return 1.0, {"text": text}, {}
        score = float(exact_match(text, gold))
        return score, {"text": text}, {}


__all__ = ["ExactMatchEvalProcedureRunner", "prefer_c19_ceiling_score"]
