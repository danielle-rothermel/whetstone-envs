from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.scoring import exact_match

if TYPE_CHECKING:
    from collections.abc import Mapping


class ExactMatchEvalProcedureRunner:
    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: object,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        text = node_inputs.get("provider_generation")
        gold = getattr(task, "gold", None)
        if not isinstance(text, str) or not isinstance(gold, str):
            raise TypeError(
                "exact-match evaluation requires string output and gold"
            )
        return float(exact_match(text, gold)), {"text": text}, {}


__all__ = ["ExactMatchEvalProcedureRunner"]
