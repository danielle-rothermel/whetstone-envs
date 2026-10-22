import hashlib
import json
from importlib.resources import files
from unittest import TestCase

from whetstone_envs import ifbench

INPUT_SHA256 = (
    "a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85"
)
TEMPLATE = "format:output_template"
NO_WHITESPACE = "format:no_whitespace"


def gold(instruction_ids: list[str]) -> str:
    return ifbench.gold_of(
        {
            "key": "1",
            "prompt": "Describe the ocean.",
            "instruction_id_list": instruction_ids,
            "kwargs": [{"N": None} for _ in instruction_ids],
        }
    )


class IfbenchDataTests(TestCase):
    def test_packaged_prompts_match_pinned_upstream_copy(self) -> None:
        raw = files("whetstone_envs.ifbench").joinpath(
            "data", "IFBench_test.jsonl"
        )
        self.assertEqual(
            hashlib.sha256(raw.read_bytes()).hexdigest(), INPUT_SHA256
        )

    def test_pool_carries_prompt_and_pruned_checker_spec(self) -> None:
        instances = ifbench.generate_pool().instance_group.instances
        self.assertEqual(len(instances), 300)
        first = instances[0]
        spec = json.loads(first.gold)
        self.assertEqual(first.id, f"ifbench-{spec['key']}")
        self.assertEqual(first.prompt_inputs["prompt"], spec["prompt"])
        self.assertEqual(first.categories, tuple(spec["instruction_id_list"]))
        for kwargs in spec["kwargs"]:
            self.assertNotIn(None, kwargs.values())


class IfbenchScoringTests(TestCase):
    def test_strict_requires_every_instruction(self) -> None:
        spec = gold([TEMPLATE, NO_WHITESPACE])
        passing = ifbench.check_strict(
            "MyAnswer:x,My-Conclusion:y", gold([NO_WHITESPACE])
        )
        self.assertTrue(passing.all_followed)
        failing = ifbench.check_strict(
            "My Answer: a My Conclusion: b Future Outlook: c", spec
        )
        self.assertEqual(failing.failed, (NO_WHITESPACE,))
        self.assertFalse(ifbench.check_strict(" ", spec).all_followed)

    def test_loose_retries_without_first_line_and_think_blocks(self) -> None:
        spec = gold([NO_WHITESPACE])
        for response in ("Sure:\nwaves", "<think>a b</think>waves"):
            with self.subTest(response=response):
                self.assertFalse(
                    ifbench.check_strict(response, spec).all_followed
                )
                self.assertTrue(
                    ifbench.check_loose(response, spec).all_followed
                )
