from __future__ import annotations

from whetstone.experiment.candidate import candidate_reference

from whetstone_envs.c19 import PROBES
from whetstone_envs.evaluation.spec import CandidateInput
from whetstone_envs.optim.experiment import c19_candidate
from whetstone_envs.optim.extraction.evaluation import EvaluationCandidate


def default_candidates() -> tuple[CandidateInput, ...]:
    return (
        CandidateInput(
            name="naive", source="naive", template=PROBES.naive_template
        ),
        CandidateInput(
            name="ceiling", source="ceiling", template=PROBES.ceiling_template
        ),
    )


def prepare_candidates(
    candidates: tuple[CandidateInput, ...],
) -> tuple[EvaluationCandidate, ...]:
    return tuple(
        EvaluationCandidate(
            name=item.name,
            source=item.source,
            candidate=candidate_reference(
                c19_candidate(
                    candidate_id=f"c19-{item.name}", template=item.template
                )
            ),
        )
        for item in candidates
    )


__all__ = ["default_candidates", "prepare_candidates"]
