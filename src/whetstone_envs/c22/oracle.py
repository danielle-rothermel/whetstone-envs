"""The strict all-pass oracle for c22.

A pure function of an instance's *public* fields: it reads the
constraint stack back from ``Instance.gold`` (never from the generator's
RNG or atom tables) and re-runs the vendored IFEval checkers, returning 1
iff **every** constraint passes -- strict all-pass, no partial credit
(rubric criteria 2 and 4).

The all-pass reduction reuses Google Research's
``evaluation_lib.test_instruction_following_strict`` verbatim; this
module only marshals a :class:`ConstraintSpec` into the ``InputExample``
that function consumes and reports the per-atom verdicts as a diagnostic
fact (rubric criteria 10, 11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from instruction_following_eval import evaluation_lib

from whetstone_envs.c22.spec import ConstraintSpec
from whetstone_envs.core.probes import normalize

_PROMPT_KEY = "c22"


@dataclass(frozen=True, slots=True)
class OracleResult:
    """The strict 0/1 score plus its per-atom diagnostic verdicts.

    ``score`` is 1 iff every atom passed. ``per_atom`` pairs each
    ``instruction_id`` with its individual pass/fail so a failure is
    diagnosable to the exact constraint that broke (rubric 10/11).
    """

    score: int
    per_atom: tuple[tuple[str, bool], ...]

    @property
    def follow_all(self) -> bool:
        """Whether every constraint passed (equivalent to ``score == 1``)."""
        return self.score == 1


def check(spec: ConstraintSpec, response: str) -> OracleResult:
    """Score ``response`` against a :class:`ConstraintSpec`.

    The response is normalized with the shared
    :func:`whetstone_envs.core.probes.normalize` (strip whitespace and a
    single wrapping code fence) before checking, so scoring differences
    come from the model, not from per-candidate string handling. The
    all-pass reduction is delegated to the vendored
    ``test_instruction_following_strict``.
    """
    normalized = normalize(response)
    # ``InputExample.kwargs`` is annotated ``dict[str, str | int | None]``
    # upstream, but keyword atoms legitimately pass ``Sequence[str]``
    # values (e.g. ``keywords=[...]``); the runtime handles them. Cast at
    # this boundary rather than widening the vendored annotation.
    kwargs = cast(
        "list[dict[str, str | int | None]]",
        [dict(k) for k in spec.kwargs_list],
    )
    inp = evaluation_lib.InputExample(
        key=0,
        instruction_id_list=list(spec.instruction_id_list),
        prompt=_PROMPT_KEY,
        kwargs=kwargs,
    )
    out = evaluation_lib.test_instruction_following_strict(
        inp,
        {_PROMPT_KEY: normalized},
    )
    per_atom = tuple(
        zip(
            spec.instruction_id_list,
            out.follow_instruction_list,
            strict=True,
        ),
    )
    return OracleResult(
        score=int(out.follow_all_instructions),
        per_atom=per_atom,
    )


def score_gold(gold: str, response: str) -> int:
    """Return the strict 0/1 score from a raw ``Instance.gold`` string.

    This is the pool-facing entry point: given only the instance's public
    ``gold`` field and a model response, return 0 or 1. It is a pure
    function of public state -- it never consults the generator.
    """
    return check(ConstraintSpec.from_gold(gold), response).score
