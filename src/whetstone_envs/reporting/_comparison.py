from whetstone_envs.evidence import ComparisonOutcome


def comparison_label(
    outcome: ComparisonOutcome, candidate_a: str, candidate_b: str
) -> str:
    return {
        ComparisonOutcome.BOTH_CORRECT: "both correct",
        ComparisonOutcome.BOTH_WRONG: "both wrong",
        ComparisonOutcome.LEFT_ONLY: f"{candidate_a} only",
        ComparisonOutcome.RIGHT_ONLY: f"{candidate_b} only",
        ComparisonOutcome.EXECUTION_MISMATCH: "execution mismatch",
    }[outcome]
