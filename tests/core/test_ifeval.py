import hashlib
import json
from importlib.resources import files
from unittest import TestCase

from whetstone_envs import ifeval

INPUT_SHA256 = (
    "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
)


def gold(instruction_ids: list[str], kwargs: list[dict[str, object]]) -> str:
    return ifeval.gold_of(
        {
            "key": 1,
            "prompt": "Write about cats.",
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs,
        }
    )


class IfevalDataTests(TestCase):
    def test_packaged_prompts_match_pinned_hub_copy(self) -> None:
        raw = files("whetstone_envs.ifeval").joinpath(
            "data", "ifeval_input_data.jsonl"
        )
        digest = hashlib.sha256(raw.read_bytes())
        self.assertEqual(digest.hexdigest(), INPUT_SHA256)

    def test_pool_carries_prompt_and_checker_spec(self) -> None:
        instances = ifeval.generate_pool().instance_group.instances
        self.assertEqual(len(instances), 541)
        first = instances[0]
        spec = json.loads(first.gold)
        self.assertEqual(first.id, f"ifeval-{spec['key']}")
        self.assertEqual(first.prompt_inputs["prompt"], spec["prompt"])
        self.assertEqual(first.categories, tuple(spec["instruction_id_list"]))
        self.assertEqual(len(spec["instruction_id_list"]), len(spec["kwargs"]))
        self.assertEqual(ifeval.NAIVE_TEMPLATE, "{prompt}")


class IfevalScoringTests(TestCase):
    def test_strict_requires_every_instruction(self) -> None:
        spec = gold(
            ["punctuation:no_comma", "change_case:english_lowercase"],
            [{}, {}],
        )
        passing = ifeval.check_strict(
            "the cats sleep for most of the day and wake up hungry", spec
        )
        self.assertTrue(passing.all_followed)
        failing = ifeval.check_strict(
            "The cats sleep for most of the day and wake up hungry", spec
        )
        self.assertFalse(failing.all_followed)
        self.assertEqual(failing.failed, ("change_case:english_lowercase",))
        self.assertFalse(ifeval.check_strict("  ", spec).all_followed)

    def test_loose_retries_without_first_line_and_markdown(self) -> None:
        spec = gold(
            ["startend:end_checker"], [{"end_phrase": "Any other questions?"}]
        )
        response = "cats are great.\nAny other questions?\n**"
        self.assertFalse(ifeval.check_strict(response, spec).all_followed)
        self.assertTrue(ifeval.check_loose(response, spec).all_followed)

    def test_loose_ignores_think_blocks(self) -> None:
        spec = gold(["punctuation:no_comma"], [{}])
        response = "<think>a, b</think>cats nap"
        self.assertFalse(ifeval.check_strict(response, spec).all_followed)
        self.assertTrue(ifeval.check_loose(response, spec).all_followed)
