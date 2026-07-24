"""Run the namespaced vendored google-research IFEval test suites.

The upstream tests pin all checker behavior outside C22's documented
import-path and exact-word-count patches.
"""

from __future__ import annotations

import unittest

from whetstone_envs.c22._vendor.instruction_following_eval import instructions


def _run_suite(module_name: str) -> unittest.TestResult:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    runner = unittest.TextTestRunner(verbosity=0)
    return runner.run(suite)


def test_vendored_instructions_test_passes() -> None:
    result = _run_suite(
        "whetstone_envs.c22._vendor.instruction_following_eval."
        "instructions_test",
    )
    assert result.wasSuccessful(), result.errors + result.failures
    assert result.testsRun > 0


def test_vendored_instructions_util_test_passes() -> None:
    result = _run_suite(
        "whetstone_envs.c22._vendor.instruction_following_eval."
        "instructions_util_test",
    )
    assert result.wasSuccessful(), result.errors + result.failures
    assert result.testsRun > 0


def test_exactly_is_local_to_number_of_words() -> None:
    assert "exactly" not in instructions._COMPARISON_RELATION
