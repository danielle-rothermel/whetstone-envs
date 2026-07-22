"""The serializable per-instance constraint spec shared by the oracle.

An :class:`Instance` carries its constraint stack in ``gold`` as a
canonical JSON string produced by :meth:`ConstraintSpec.to_gold`. The
oracle reads that string back (:meth:`ConstraintSpec.from_gold`) and
re-runs the vendored checkers -- it never touches the generator's
internal RNG or atom tables. This is the boundary that keeps the oracle
an independent function of an instance's *public* fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """The oracle-checkable constraint stack for one instance.

    Parameters
    ----------
    base_task:
        The trivial base micro-task line (e.g. ``"Name a color."``).
    constraint_descriptions:
        The per-atom natural-language constraint descriptions, in stack
        order, exactly as emitted by IFEval ``build_description``.
    instruction_id_list:
        The vendored IFEval registry ids, in stack order.
    kwargs_list:
        The explicit ``build_description`` kwargs for each atom, in stack
        order. Reapplying these reproduces the same checker state, so the
        oracle's ``check_following`` is deterministic.
    """

    base_task: str
    constraint_descriptions: tuple[str, ...]
    instruction_id_list: tuple[str, ...]
    kwargs_list: tuple[dict[str, object], ...]

    def constraints_block(self) -> str:
        """The base task plus numbered constraint lines, as the prompt sees it.

        This is the ``{BASE_TASK_AND_CONCATENATED_CONSTRAINTS}`` slot the
        probe prompts interpolate. It contains only what the model is
        allowed to see -- never the raw kwargs or the registry ids.
        """
        lines = [self.base_task, ""]
        lines.extend(
            f"{i}. {desc}"
            for i, desc in enumerate(self.constraint_descriptions, start=1)
        )
        return "\n".join(lines)

    def to_gold(self) -> str:
        """Serialize to a canonical JSON string for ``Instance.gold``."""
        return json.dumps(
            {
                "base_task": self.base_task,
                "constraint_descriptions": list(
                    self.constraint_descriptions,
                ),
                "instruction_id_list": list(self.instruction_id_list),
                "kwargs_list": [dict(k) for k in self.kwargs_list],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_gold(cls, gold: str) -> ConstraintSpec:
        """Reconstruct a spec from its :meth:`to_gold` JSON string."""
        data = json.loads(gold)
        return cls(
            base_task=str(data["base_task"]),
            constraint_descriptions=tuple(data["constraint_descriptions"]),
            instruction_id_list=tuple(data["instruction_id_list"]),
            kwargs_list=tuple(dict(k) for k in data["kwargs_list"]),
        )
