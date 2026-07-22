r"""The independent oracle for c23, reusing the vendored transducers.

Ground truth for a c23 instance is "apply the latent subregular rule to the
held-out query." That application is delegated **unmodified** to the
vendored InductionBench transducers -- ``apply_ISL_rule`` /
``apply_L_OSL_rule`` / ``apply_R_OSL_rule`` (see
``_vendor/inductionbench/PROVENANCE.md``) -- via
:func:`whetstone_envs.c23.upstream.apply_rule`. The c23 oracle never
reimplements rule application; it reuses the reference the vendored code
already provides (the danielle-code-quality norm against re-deriving what a
dependency guarantees, and the repos review's finding that this oracle is
genuinely independent of the sampler and verifiably self-consistent).

This module is genuinely independent of the *generator's internal state*:
:func:`apply_to_query` is a pure function of the latent ``rule`` dict, the
``rule_type``, ``k``, and the ``query`` string -- never a re-derivation of
how the generator sampled that rule or built its characteristic sample. The
generator uses it to derive each instance's frozen ``gold`` and to
cross-check that gold at construction; the fixture tests use it directly on
hand-picked rules; and :func:`score` / :func:`score_gold` are the
pool-facing 0/1 exact-match entry points.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from whetstone_envs.c23 import upstream
from whetstone_envs.core.probes import normalize

if TYPE_CHECKING:
    from collections.abc import Mapping

# --- Output extraction (baseline spec Sections 2 / 3) ----------------------
# Spec Section 2: "Output extraction for both [probes]: take the text after
# the last Output: line, strip surrounding whitespace and markdown fences,
# compare for exact string equality to the oracle output." The ceiling
# probe (Section 2.2) tells the model to emit "the transformed string, on a
# single line prefixed with 'Output:'", so a well-formed reply answers on an
# ``Output: <string>`` line -- possibly preceded by reasoning or trailed by
# commentary, and a model may emit several ``Output:`` lines, of which the
# LAST is the answer. The scorer previously exact-matched the WHOLE reply
# against the gold, so every ``Output:``-prefixed reply scored 0 (live:
# ceiling 0/30). The naive probe gives no format instruction, so a bare
# emission has no ``Output:`` line -- for that shape the whole (normalized)
# reply is the string to compare, matching the one bare naive hit (1/30).
# A ``Output:`` prefix may appear case-insensitively with any run of spaces
# after the colon; only the text on that line, after the prefix, is taken.
_OUTPUT_LINE_RE = re.compile(
    r"^\s*output\s*:\s*(?P<answer>.*)$",
    re.IGNORECASE,
)


def extract_output(text: str) -> str:
    """Extract the answer string from a reply per the spec extraction rule.

    Returns the text after the LAST ``Output:``-prefixed line (case-
    insensitive), which is where the ceiling probe instructs the model to
    place the transformed string. If no ``Output:`` line is present -- the
    naive probe's bare-emission shape -- the whole ``text`` is returned
    unchanged. Surrounding whitespace / markdown fences are handled by the
    shared :func:`whetstone_envs.core.probes.normalize` at the call site.
    """
    last: str | None = None
    for line in text.splitlines():
        m = _OUTPUT_LINE_RE.match(line)
        if m is not None:
            last = m.group("answer")
    return text if last is None else last


def apply_to_query(
    rule_type: str,
    k: int,
    rule: Mapping[str, str],
    query: str,
) -> str:
    """Return the oracle output: the latent rule applied to ``query``.

    Delegates to the vendored transducer for ``rule_type`` unmodified. This
    is the independent ground-truth derivation the generator's frozen
    ``gold`` must equal, and the function the fixture tests exercise on
    hand-picked rules.
    """
    return upstream.apply_rule(rule_type, k, rule, query)


def score(
    prediction: str,
    rule_type: str,
    k: int,
    rule: Mapping[str, str],
    query: str,
) -> int:
    """Return 1 iff ``prediction`` matches the re-derived gold, else 0.

    Re-derives the gold from the public ``query`` and the latent ``rule``
    via the vendored transducer, then compares against the model's
    ``prediction`` after :func:`extract_output` (the spec Section 2 rule:
    take the text after the last ``Output:`` line) and the shared
    :func:`whetstone_envs.core.probes.normalize` (strip surrounding
    whitespace / one code fence) -- exact match, no partial credit (rubric
    criterion 2).
    """
    gold = apply_to_query(rule_type, k, rule, query)
    predicted = normalize(extract_output(prediction))
    return int(predicted == normalize(gold))


def score_gold(prediction: str, gold: str) -> int:
    """Return the 0/1 score of ``prediction`` against a frozen ``gold``.

    The pool-facing entry point (mirrors the other candidates'
    ``score_gold``): given an instance's already-derived ``gold`` string and
    a model response, return 0 or 1. The prediction is run through
    :func:`extract_output` (spec Section 2: the text after the last
    ``Output:`` line) and the shared normalization before exact match, so an
    ``Output:``-prefixed reply is scored on its emitted string, not its whole
    text. Use :func:`score` instead to re-derive straight from the public
    query + the latent rule.
    """
    predicted = normalize(extract_output(prediction))
    return int(predicted == normalize(gold))
