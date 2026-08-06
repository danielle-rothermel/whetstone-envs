"""Tests for shared prediction normalization."""

import pytest

from whetstone_envs.probes import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("  yes  ", "yes", id="surrounding-spaces"),
        pytest.param("yes\n", "yes", id="trailing-newline"),
        pytest.param("```\nyes\n```", "yes", id="plain-fence"),
        pytest.param(
            '```json\n{"a": 1}\n```',
            '{"a": 1}',
            id="language-tagged-fence",
        ),
        pytest.param(
            "```text\nfirst line\n\n    indented line\nlast line\n```",
            "first line\n\n    indented line\nlast line",
            id="internal-blank-line-and-indentation",
        ),
        pytest.param(
            "  ```\n  answer  \n```  ",
            "answer",
            id="whitespace-around-fence-and-payload",
        ),
        pytest.param("no fence here", "no fence here", id="unfenced-text"),
        pytest.param(
            "```only one backtick line",
            "```only one backtick line",
            id="inline-backticks",
        ),
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


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("```\nanswer", id="opening-only"),
        pytest.param("answer\n```", id="closing-only"),
    ],
)
def test_normalize_preserves_unmatched_fence_lines(raw: str) -> None:
    assert normalize(raw) == raw
