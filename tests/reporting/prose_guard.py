"""The rendered-prose number guard, shared by the report tests.

A number in a paragraph or a prose cell reaches a reader exactly as a number
in a figure cell does, but carries no provenance mark. This module owns the
rule: every digit the report renders outside a :class:`Figure` must belong
to one of the structural identifiers named below.

It lives beside the tests rather than in ``src`` because it is a *test*
rule -- an assertion about how the report is built -- and the whitelist is
maintained by the same review that adds prose to the report.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from whetstone_envs.reporting.study_report import rendered_text_in

if TYPE_CHECKING:
    from whetstone_envs.reporting.study_report import StudyReport

__all__ = [
    "DIGIT",
    "NON_EVIDENCE_PATTERNS",
    "strip_non_evidence",
    "unbacked_numbers_in",
]

#: Digits that are structure rather than evidence, and why each is allowed.
#: A number matching one of these is not a measurement, so it has nothing to
#: point at; anything else in rendered prose must be a Figure. The list is
#: deliberately explicit -- a broad pattern would let a real number through
#: by resembling a label.
NON_EVIDENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\bL[1-7]\b",
        "leakage rule names",
    ),
    (
        r"\bF\d+\b",
        "prerequisite ids from the protocol",
    ),
    (
        r"\bStages [0-2]-[0-2]\b|\bStages? [0-2]\b|\bStage-[0-2]\b",
        "stage names",
    ),
    (
        r"\bC[1-3]\b",
        "claim ids",
    ),
    (
        r"\bO\d+\b|\bD[1-9]\b|\bR\d+\b",
        "decision and open-question ids from the protocol",
    ),
    (
        (
            r"\b[\w.-]*(?:copro|miprov2|gepa|codex|null[A-Za-z]*|c18)"
            r"[\w.-]*\d[\w.-]*"
        ),
        "run and arm ids, which name a run rather than measure one",
    ),
    (
        r"\bMIPROv2\b",
        "the optimizer's own name",
    ),
    (
        r"\bc1[89]\b|\bc1[89]-[\w-]+",
        "task-family ids, and run ids built from one",
    ),
    (
        r"\bstudy\.json\b",
        "the manifest's own filename",
    ),
    (
        r"n_per_stratum \d+",
        "a population setting, printed beside the pool it generated",
    ),
    (
        r"\b[\d,]+ resamples\b",
        "the pre-registered resample count, named as a setting",
    ),
    (
        r"/[\w./-]*\d[\w./-]*",
        "filesystem paths, which locate evidence rather than state it",
    ),
    (
        r"\d{4}-\d{2}-\d{2}T[\d:+.-]+|\d{4}-\d{2}-\d{2}",
        "timestamps",
    ),
    (
        r"\bsha256\b",
        "the hash algorithm's name",
    ),
    (
        r"\b[0-9a-f]{12,64}\b",
        "content hashes and their prefixes",
    ),
    (
        r"\bstep10\S*",
        "the study's own id",
    ),
    (
        r"@[0-9a-f]{12}\b",
        "content-hash prefixes, which are provenance marks themselves",
    ),
    (
        r"/v\d+\b",
        "schema versions inside a cited schema name",
    ),
    (
        r"MDE\(T, K\) = [^\n]*",
        "the MDE formula, printed as the study's own auditable arithmetic",
    ),
    (
        r"tau\^2|sigma\^2|z_\{1-alpha/2\}",
        "variance-component names in the design table",
    ),
    (
        r"\bseed \d+\b",
        "the pre-registered bootstrap seed, named as a setting",
    ),
    (
        r"2/resamples",
        "the p-value floor, stated as a formula rather than a value",
    ),
    (
        r"\b95%",
        "the pre-registered CI level, named as a setting",
    ),
    (
        r"\b3-5 points\b",
        "an illustrative magnitude in a threats note, not a measurement",
    ),
    (
        r"\bgpt-[\d.]+\S*",
        "model names",
    ),
    (
        r"whetstone[_a-z-]*[.\w/-]*\d[\w./-]*",
        "schema names, which identify a record format rather than measure",
    ),
)

DIGIT = re.compile(r"\d")


def strip_non_evidence(text: str) -> str:
    """Remove every allowed non-evidence digit run from ``text``."""
    remaining = text
    for pattern, _why in NON_EVIDENCE_PATTERNS:
        remaining = re.sub(pattern, " ", remaining)
    return remaining


def unbacked_numbers_in(
    report: StudyReport,
) -> list[tuple[str, str, str]]:
    """Every rendered string carrying a digit the whitelist does not excuse.

    Returned as ``(kind, location, text)`` triples so a failure names where
    the number is rather than only that one exists.
    """
    return [
        (entry.kind, entry.location, entry.text)
        for entry in rendered_text_in(report)
        if DIGIT.search(strip_non_evidence(entry.text))
    ]
