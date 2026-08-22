"""How each task family turns one generation into one score.

This is the reporting-safe half of the family registry. The per-family
*scoring rule* is a pure function of ``(output_text, gold)`` -- c19 compares
normalized exact match, c18 extracts the terminal verdict first -- and
nothing about that rule needs an optimizer, a provider, or an experiment.

It lives here rather than behind ``whetstone_envs.optim.families`` because
both an eval run *and* a report consumer must apply the identical rule. The
optimizer registry imports whetstone-ai, which the ``optim`` extra installs
only on Python 3.13+, so routing a report validator through it makes
validating a scored report fail on a base install. A report is read in far
more places than it is written, and reading one must not require the stack
that wrote it.

The drift this file exists to prevent is two spellings of one family's
rule. Both the eval-node runners under ``whetstone_envs.optim`` and
:class:`~whetstone_envs.reporting.schema.EvalReport`'s score check call
:func:`family_score` here, and ``tests/scoring/test_families.py`` pins that
the runners' scores equal this registry's for representative outputs of
both families.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from whetstone_envs.c18.oracle import score_gold as _c18_score_gold
from whetstone_envs.scoring.exact_match import exact_match as _c19_exact_match

__all__ = [
    "FAMILY_SCORERS",
    "FamilyScorer",
    "family_score",
    "scorable_family_ids",
]

#: One family's scoring rule: what a generation scores against frozen gold.
type FamilyScorer = Callable[[str, str], float]


def _c19_score(output_text: str, gold: str) -> float:
    """c19 scores a reply by normalized exact match against the oracle."""
    return float(_c19_exact_match(output_text, gold))


def _c18_score(output_text: str, gold: str) -> float:
    """c18 scores the terminal verdict extracted from a reasoned reply.

    C18's ceiling probe asks for reasoning ending in a lone True/False
    line, so exact match over the whole reply would score every reasoned
    answer zero. :func:`whetstone_envs.c18.oracle.score_gold` owns the
    extraction and the comparison.
    """
    return float(_c18_score_gold(output_text, gold))


#: Every family's scoring rule, keyed by the family name a persisted report
#: carries. Registered by name here rather than derived from the optimizer
#: registry: this table must resolve without whetstone-ai installed, and a
#: new family adds its rule deliberately alongside a golden test.
FAMILY_SCORERS: Final[Mapping[str, FamilyScorer]] = {
    "c19": _c19_score,
    "c18": _c18_score,
}


def scorable_family_ids() -> tuple[str, ...]:
    """Every family this registry can score, in registration order."""
    return tuple(FAMILY_SCORERS)


def family_score(*, family: str, output_text: str, gold: str) -> float:
    """What ``family``'s own scorer yields for one observation.

    The single owner of "how a generation becomes a score" for every
    consumer -- the eval-node runners that produce a report and the schema
    that validates one both route here, so a report's recorded score and
    its re-derivation cannot drift apart.
    """
    scorer = FAMILY_SCORERS.get(family)
    if scorer is None:
        msg = (
            f"unsupported family {family!r}; "
            f"scorable families are {scorable_family_ids()}"
        )
        raise ValueError(msg)
    return scorer(output_text, gold)
