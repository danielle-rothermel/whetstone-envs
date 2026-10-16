from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

from whetstone_envs.c22 import Preset, generate_pool
from whetstone_envs.c22.constraints import ConstraintStack
from whetstone_envs.c22.oracle import evaluate

ROOT = Path(__file__).resolve().parents[2]


class TaskViewerTests(TestCase):
    def test_task_specific_displays_and_rendered_probes(self) -> None:
        expected = {
            "c11": (
                "Public input JSON",
                "Exact input with escapes visible",
                "canonical JSON",
            ),
            "c18": (
                "Public facts and rules",
                "Public query",
                "Deduction depth",
            ),
            "c19": (
                "Public initial grid",
                "Public action script",
                "Public question",
            ),
            "c22": (
                "Public numbered constraints",
                "scoring specification",
                "Parameters",
                "Passing example | Score: 1",
                "Failing example | Score: 0",
                "Per-constraint outcomes",
            ),
            "c23": ("Public demonstrations", "Public query", "Context length"),
        }
        for task, labels in expected.items():
            with self.subTest(task=task):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / task / "viz.py"),
                        "--count",
                        "2",
                        "--probes",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                )
                for label in labels:
                    self.assertIn(label, result.stdout)
                self.assertEqual(result.stdout.count("Seed:"), 2)
                self.assertEqual(result.stdout.count("Private gold:"), 2)
                self.assertEqual(result.stdout.count("Naive probe"), 2)
                self.assertEqual(result.stdout.count("Ceiling probe"), 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_c22_scoring_examples_cover_every_preset_instance(self) -> None:
        viewer = runpy.run_path(str(ROOT / "scripts" / "c22" / "viz.py"))
        for preset in Preset:
            for instance in generate_pool(preset).instance_group.instances:
                with self.subTest(preset=preset, instance=instance.id):
                    stack = ConstraintStack.from_gold(instance.gold)
                    passing = viewer["passing_response"](stack)
                    failing = viewer["failing_response"](stack, passing)
                    self.assertEqual(evaluate(stack, passing).score, 1)
                    self.assertEqual(evaluate(stack, failing).score, 0)
                    self.assertTrue(passing.strip())
                    self.assertTrue(failing.strip())

    def test_invalid_counts_are_cli_errors(self) -> None:
        for task in ("c11", "c18", "c19", "c22", "c23"):
            with self.subTest(task=task):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / task / "viz.py"),
                        "--count",
                        "0",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("positive", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_hard_configurations(self) -> None:
        for task, flags, label in (
            ("c18", ["--hard"], "Deduction depth: 5"),
            ("c22", ["--preset", "hard"], "Preset: hard"),
        ):
            with self.subTest(task=task):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / task / "viz.py"),
                        *flags,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                )
                self.assertIn(label, result.stdout)
                self.assertEqual(result.stdout.count("Private gold:"), 1)
                self.assertNotIn("Naive probe", result.stdout)
