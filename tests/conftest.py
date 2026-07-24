"""Session-wide test determinism.

The vendored google-research IFEval checkers use ``langdetect`` for their
English-language checks (``english_capital_checker`` /
``english_lowercase_checker`` and the language-response instruction).
``langdetect`` is *non-deterministic by design*: it seeds its own
detector from the global RNG on each call, so a check can flip between
runs depending on interleaving global-random state. That surfaces as a
flaky ``test_english_capital_checker`` in the vendored upstream suite.
C22's supported oracle atoms exclude these language checkers.

langdetect's own prescribed fix for reproducibility is to pin
``DetectorFactory.seed``. We do exactly that once per test session so
the pinned upstream language-checker tests are deterministic.
"""

from __future__ import annotations

import langdetect
import pytest


@pytest.fixture(scope="session", autouse=True)
def _pin_langdetect_seed() -> None:
    """Make ``langdetect`` deterministic for the whole test session."""
    langdetect.DetectorFactory.seed = 0
