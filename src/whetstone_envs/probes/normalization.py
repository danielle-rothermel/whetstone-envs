import re

_MIN_FENCED_LINES = 2
_OPENING_CODE_FENCE = re.compile(r"```[^\s`]*")


def _strip_code_fence(text: str) -> str:
    """Strip one matched outer triple-backtick fence with an optional tag."""
    lines = text.split("\n")
    if len(lines) < _MIN_FENCED_LINES:
        return text
    first = lines[0].strip()
    last = lines[-1].strip()
    if _OPENING_CODE_FENCE.fullmatch(first) and last == "```":
        return "\n".join(lines[1:-1])
    return text


def normalize(prediction: str) -> str:
    """Strip whitespace and matched outer fences to a fixed point.

    Unmatched fences and internal backticks are preserved.
    """
    normalized = prediction.strip()
    while True:
        unfenced = _strip_code_fence(normalized)
        if unfenced == normalized:
            return normalized
        normalized = unfenced.strip()
