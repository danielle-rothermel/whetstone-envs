# IFEval prompts

`data/ifeval_input_data.jsonl` is `ifeval_input_data.jsonl` from the Hugging
Face dataset [`google/IFEval`](https://huggingface.co/datasets/google/IFEval)
at revision `966cd89545d6b6acfd7638bc708b98261ca58e84`, copied byte for byte
(sha256 `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`,
541 prompts). The dataset is released under the Apache-2.0 license in
`LICENSE`.

Responses are checked by the Google Research `instruction_following_eval`
runtime that `whetstone_envs.c22` vendors at commit
`37ffb72669bc762fe899d5eaec83d28be2c882cc`; its local patch only adds an exact
word-count relation, which no IFEval prompt uses. Strict and loose accuracy are
the upstream `test_instruction_following_strict` and
`test_instruction_following_loose` functions.
