from whetstone_envs.c23._domain import (
    Hypothesis,
    RuleConfiguration,
    RuleFamily,
)
from whetstone_envs.c23._transducers import apply_reference


def hypothesis(
    family: RuleFamily,
    context: str,
    replacement: str,
) -> Hypothesis:
    return Hypothesis(
        RuleConfiguration(family, len(context)),
        context,
        replacement,
    )


def test_literal_reference_transducer_fixtures() -> None:
    fixtures = (
        (hypothesis(RuleFamily.ISL, "ab", "c"), "abab", "acac"),
        (hypothesis(RuleFamily.ISL, "ab", ""), "aabb", "aab"),
        (hypothesis(RuleFamily.L_OSL, "ab", "c"), "abbb", "acbb"),
        (hypothesis(RuleFamily.R_OSL, "ab", "c"), "bbba", "bbca"),
        (hypothesis(RuleFamily.ISL, "abc", "d"), "aabca", "aabda"),
    )
    for rule, value, expected in fixtures:
        assert apply_reference(rule, value) == expected
