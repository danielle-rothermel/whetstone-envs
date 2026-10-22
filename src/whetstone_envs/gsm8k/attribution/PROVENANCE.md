# GSM8K test split and 8-shot exemplars

`data/test.jsonl.gz` is the `main` configuration test split of the Hugging Face
dataset [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) at
revision `740312add88f781978c0658806c59bc2815b9866`
(`main/test-00000-of-00001.parquet`, sha256
`ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`). The 1319
parquet rows were written in order as JSON lines with the `question` and
`answer` fields unchanged (decompressed sha256
`752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac`) and
compressed with `gzip -9 -n`. The dataset is released by OpenAI under the MIT
license in `LICENSE` (from `openai/grade-school-math`).

`FEWSHOT_EXEMPLARS` holds the eight chain-of-thought exemplars of Wei et al.
(2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
copied from `lm_eval/tasks/gsm8k/gsm8k-cot.yaml` in
[`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)
at commit `d6de81643928d653435c431bae19945d41d32520` (MIT license). Each
exemplar's rationale is followed by a final `#### <answer>` line, the GSM8K
answer convention read by the strict scorer.
