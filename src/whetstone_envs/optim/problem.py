from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from runpy import run_path
from types import MappingProxyType
from typing import TYPE_CHECKING

from whetstone.eval.eval_procedure import EvalProcedureRunner

from whetstone_envs.evaluation import (
    ExperimentContract,
    PreparedSplitExperiment,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    prepared: PreparedSplitExperiment
    contract: ExperimentContract
    eval_runner: EvalProcedureRunner
    task_context: str
    rendering_rules: str
    example_execution: str
    proposal_bodies: tuple[str, ...] = ()
    fake_replies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prepared.experiment.env_name != self.contract.namespace:
            raise ValueError(
                "experiment and prompt contract namespaces disagree"
            )
        object.__setattr__(
            self, "fake_replies", MappingProxyType(dict(self.fake_replies))
        )


def load_problem(path: Path) -> OptimizationProblem:
    # This is executable local input, also reconstructed by Codex subprocesses.
    problem = run_path(str(path.resolve())).get("problem")
    if not isinstance(problem, OptimizationProblem):
        raise TypeError(
            "problem file must define an OptimizationProblem named 'problem'"
        )
    return problem


__all__ = ["OptimizationProblem", "load_problem"]
