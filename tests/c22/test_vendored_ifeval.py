"""Run the vendored google-research IFEval test suites, verbatim.

The c22 baseline spec reuses the IFEval checker library unmodified for
both generation and the oracle; its own upstream unit tests are the
evidence that the vendored copy is intact and correct. This wrapper
loads those absltest suites (ships with the vendored tree) and asserts
they pass -- so any accidental edit to the byte-for-byte vendored source
fails CI here rather than silently changing oracle behavior.
"""

from __future__ import annotations

import unittest

import whetstone_envs.c22  # noqa: F401  (installs the vendor sys.path shim)


def _run_suite(module_name: str) -> unittest.TestResult:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    runner = unittest.TextTestRunner(verbosity=0)
    return runner.run(suite)


def test_vendored_instructions_test_passes() -> None:
    result = _run_suite(
        "instruction_following_eval.instructions_test",
    )
    assert result.wasSuccessful(), result.errors + result.failures
    assert result.testsRun > 0


def test_vendored_instructions_util_test_passes() -> None:
    result = _run_suite(
        "instruction_following_eval.instructions_util_test",
    )
    assert result.wasSuccessful(), result.errors + result.failures
    assert result.testsRun > 0
