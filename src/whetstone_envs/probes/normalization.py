"""Shared prediction normalization for exact-match evaluation.

:func:`normalize` is the single normalization step shared by every task
environment's exact-match scoring. Keeping it here means every oracle extracts
predictions identically, so scoring differences come from the model rather
than task-family-specific string handling.
"""

import re

_MIN_FENCED_LINES = 2
_OPENING_CODE_FENCE = re.compile(r"```[^\s`]*")


def _strip_code_fence(text: str) -> str:
    """Remove one matching pair of triple-backtick fences, if present.

    A leading fence line (```` ``` ```` optionally followed by a language
    tag) paired with a trailing fence line is removed. Text without a matched
    pair is returned unchanged, so a stray backtick inside a real answer is
    never eaten.
    """
    lines = text.split("\n")
    if len(lines) < _MIN_FENCED_LINES:
        return text
    first = lines[0].strip()
    last = lines[-1].strip()
    if _OPENING_CODE_FENCE.fullmatch(first) and last == "```":
        return "\n".join(lines[1:-1])
    return text


def normalize(prediction: str) -> str:
    """Normalize a raw model prediction for exact-match comparison.

    Strips surrounding whitespace, then removes complete wrapping code fences
    to a fixed point, re-stripping whitespace after each pair. Unmatched
    fences and backticks within the remaining answer are preserved. No other
    content is changed, so exact-match semantics are deterministic and
    normalization is idempotent.
    """
    normalized = prediction.strip()
    while True:
        unfenced = _strip_code_fence(normalized)
        if unfenced == normalized:
            return normalized
        normalized = unfenced.strip()
