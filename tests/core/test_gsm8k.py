import gzip
import hashlib
from importlib.resources import files
from unittest import TestCase

from whetstone_envs import gsm8k

TEST_JSONL_SHA256 = (
    "752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac"
)


class Gsm8kDataTests(TestCase):
    def test_packaged_split_matches_pinned_hub_copy(self) -> None:
        raw = files("whetstone_envs.gsm8k").joinpath("data", "test.jsonl.gz")
        digest = hashlib.sha256(gzip.decompress(raw.read_bytes()))
        self.assertEqual(digest.hexdigest(), TEST_JSONL_SHA256)

    def test_pool_has_every_test_item_with_numeric_gold(self) -> None:
        instances = gsm8k.generate_pool().instance_group.instances
        self.assertEqual(len(instances), 1319)
        first = instances[0]
        self.assertEqual(first.id, "gsm8k-test-0000")
        self.assertTrue(first.prompt_inputs["question"].startswith("Janet"))
        self.assertEqual(first.gold, "18")
        self.assertNotIn(",", "".join(i.gold for i in instances))

    def test_gold_number_drops_thousands_separators(self) -> None:
        self.assertEqual(gsm8k.gold_number("steps\n#### 1,450,000"), "1450000")
        self.assertEqual(gsm8k.gold_number("x\n#### -3"), "-3")


class Gsm8kPromptTests(TestCase):
    def test_template_has_eight_exemplars_and_one_question_slot(self) -> None:
        template = gsm8k.NAIVE_TEMPLATE
        self.assertEqual(len(gsm8k.FEWSHOT_EXEMPLARS), 8)
        self.assertEqual(template.count("\nA: "), 8)
        self.assertEqual(template.count("{question}"), 1)
        self.assertTrue(template.endswith("Q: {question}\nA:"))
        for exemplar in gsm8k.FEWSHOT_EXEMPLARS:
            self.assertIn(f"\n#### {exemplar.answer}\n", template)
            self.assertTrue(
                exemplar.rationale.endswith(
                    f"The answer is {exemplar.answer}."
                )
            )


class Gsm8kScoringTests(TestCase):
    def test_strict_requires_a_final_marker_line(self) -> None:
        self.assertEqual(gsm8k.strict_answer("6 + 12 = 18.\n#### 18"), "18")
        self.assertEqual(gsm8k.strict_answer("#### 1,450,000\n\n"), "1450000")
        self.assertEqual(gsm8k.strict_answer("#### 18.00"), "18")
        self.assertIsNone(gsm8k.strict_answer("The answer is 18."))
        self.assertIsNone(gsm8k.strict_answer("#### 18\nDone."))
        self.assertIsNone(gsm8k.strict_answer("#### $18"))
        self.assertIsNone(gsm8k.strict_answer("####18"))
        self.assertIsNone(gsm8k.strict_answer(""))

    def test_forgiving_reads_the_last_number(self) -> None:
        self.assertEqual(
            gsm8k.forgiving_answer("9 * 2 = $18. The answer is 18."), "18"
        )
        self.assertEqual(
            gsm8k.forgiving_answer("It costs $1,450,000."), "1450000"
        )
        self.assertEqual(
            gsm8k.forgiving_answer("<think>maybe 7</think>So 3 - 5 = -2"), "-2"
        )
        self.assertEqual(gsm8k.forgiving_answer("<think>42 and more"), None)
        self.assertIsNone(gsm8k.forgiving_answer("no digits"))

    def test_scores_compare_canonical_numbers(self) -> None:
        self.assertEqual(gsm8k.score_strict("#### 2,125", "2125"), 1)
        self.assertEqual(gsm8k.score_strict("The answer is 2125", "2125"), 0)
        self.assertEqual(
            gsm8k.score_forgiving("The answer is 2125", "2125"), 1
        )
        self.assertEqual(gsm8k.score_forgiving("#### 2124", "2125"), 0)
