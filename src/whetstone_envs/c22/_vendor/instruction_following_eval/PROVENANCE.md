# Pinned Google Research IFEval runtime

C22 vendors the runtime checker modules from
[`google-research/google-research`](https://github.com/google-research/google-research/tree/37ffb72669bc762fe899d5eaec83d28be2c882cc/instruction_following_eval)
at commit `37ffb72669bc762fe899d5eaec83d28be2c882cc` under the
upstream Apache-2.0 license.

The runtime snapshot contains `evaluation_lib.py`, `instructions.py`,
`instructions_registry.py`, and `instructions_util.py`. Upstream tests are not
packaged: C22 supports a closed subset of the registry and tests every supported
checker through hand-built first-party fixtures.

`VENDORED_DIFF.patch` records the complete local patch. It makes intra-package
imports relative so the snapshot is isolated below `whetstone_envs.c22`, and it
adds the C22 exact-word relation to `NumberOfWords`. The shared upstream
comparison-relation tuple remains unchanged.

The first-party vendor test reverse-applies this patch, checks the pinned
upstream file hashes, reapplies it, and byte-compares the result with the shipped
runtime. No network access or tokenizer data download is required.
