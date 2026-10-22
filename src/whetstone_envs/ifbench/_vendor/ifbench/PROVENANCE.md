# Pinned AllenAI IFBench runtime and test prompts

This directory vendors the instruction checkers of
[`allenai/IFBench`](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d)
at commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d` under the upstream
Apache-2.0 license in `LICENSE`: `ifbench/instructions.py`,
`ifbench/classic_instructions.py`, `ifbench/instructions_registry.py`,
`ifbench/instructions_util.py`, and the repository-root `evaluation_lib.py`.
Upstream sha256 values:

| file | sha256 |
| --- | --- |
| classic_instructions.py | d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973 |
| evaluation_lib.py | e681a4b03a6a9dfb540fbd9ef6b963de503a5ffac557ed611c326ee385850cf6 |
| instructions.py | a76a3b300b531b95200e53427c6b1fbb72b671cf6d358796797c0fa4b6c7c44f |
| instructions_registry.py | 5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9 |
| instructions_util.py | 6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930 |

`VENDORED_DIFF.patch` records the complete local patch: intra-package imports
become relative, and two import-time side effects are removed (creating a
`.nltk_data` directory inside the package and downloading NLTK resources). The
checkers therefore need the NLTK `punkt`, `punkt_tab`, `stopwords`, and
`averaged_perceptron_tagger_eng` resources installed in a standard NLTK data
location for the instructions that tokenize, count stopwords, or tag parts of
speech.

`../../data/IFBench_test.jsonl` is `ifbench/data/IFBench_test.jsonl` from the
same commit, copied byte for byte (sha256
`a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85`, 300
prompts, 58 instruction types).
