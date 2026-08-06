from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.c22 import _ifeval
from whetstone_envs.c22.constraints import ConstraintStack
from whetstone_envs.probes import normalize


@dataclass(frozen=True, slots=True)
class OracleResult:
    score: int
    per_constraint: tuple[tuple[str, bool], ...]

    @property
    def follow_all(self) -> bool:
        return self.score == 1


def evaluate(stack: ConstraintStack, response: str) -> OracleResult:
    if not isinstance(response, str):
        msg = "response must be a string"
        raise TypeError(msg)
    verdicts = _ifeval.check(stack.constraints, normalize(response))
    return OracleResult(
        score=int(all(verdict for _, verdict in verdicts)),
        per_constraint=verdicts,
    )


def score(gold: str, response: str) -> int:
    """Return one only when every serialized C22 constraint passes."""
    return evaluate(ConstraintStack.from_gold(gold), response).score
