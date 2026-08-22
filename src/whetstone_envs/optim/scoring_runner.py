from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone_envs.scoring.families import family_score

if TYPE_CHECKING:
    from collections.abc import Mapping


class ExactMatchEvalProcedureRunner:
    """Zero-arg eval-node runner that scores c19 generations.

    Workers reconstruct this type via ``runner_type()`` and transfer no
    constructor state.

    The scoring rule itself is not restated here: it comes from
    :func:`whetstone_envs.scoring.families.family_score`, which a report's
    own score check also calls. One owner for the rule is what keeps a
    recorded score and its later re-derivation from drifting apart.
    """

    #: The family whose rule this runner applies, resolved through the
    #: shared scorer registry rather than reimplemented.
    family_id = "c19"

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
        score = family_score(
            family=self.family_id, output_text=text, gold=gold
        )
        return score, {"text": text}, {}


__all__ = ["ExactMatchEvalProcedureRunner"]
