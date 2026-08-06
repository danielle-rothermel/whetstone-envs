"""Tests for shared prediction normalization."""

import pytest

from whetstone_envs.probes import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  yes  ", "yes"),
        ("yes\n", "yes"),
        ("```\nyes\n```", "yes"),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("  ```\n  answer  \n```  ", "answer"),
        ("no fence here", "no fence here"),
        ("```only one backtick line", "```only one backtick line"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    payload = '{"a": 1}'
    raw = f"  \n```text\n  ```json\n{payload}\n```  \n```  \n"
    once = normalize(raw)
    assert once == payload
    assert normalize(once) == payload


@pytest.mark.parametrize(
    "raw",
    [
        "````\nanswer\n```",
        "```\nanswer\n````",
        "```json`\nanswer\n```",
        "```json extra\nanswer\n```",
        "```json\nanswer\n~~~",
    ],
    ids=[
        "longer-opening",
        "longer-closing",
        "backtick-in-language-tag",
        "non-tag-opening-suffix",
        "different-closing-marker",
    ],
)
def test_normalize_preserves_non_exact_fence_wrappers(raw: str) -> None:
    assert normalize(raw) == raw


def test_normalize_accepts_exact_opening_with_language_tag() -> None:
    assert normalize("```c++\nanswer\n```") == "answer"


def test_normalize_preserves_internal_backticks() -> None:
    # A lone backtick inside the answer must not be eaten.
    assert normalize("a `b` c") == "a `b` c"


@pytest.mark.parametrize("raw", ["```\nanswer", "answer\n```"])
def test_normalize_preserves_unmatched_fence_lines(raw: str) -> None:
    assert normalize(raw) == raw
