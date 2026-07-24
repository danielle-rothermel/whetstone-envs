# Pinned google-research IFEval snapshot

C22 vendors the checker and test modules from
[`google-research/google-research`](https://github.com/google-research/google-research/tree/37ffb72669bc762fe899d5eaec83d28be2c882cc/instruction_following_eval)
at commit `37ffb72669bc762fe899d5eaec83d28be2c882cc` (2026-07-20).
The snapshot was prepared for C22 on 2026-07-22.

The upstream code is Apache-2.0. `LICENSE` is copied from the upstream
repository root, and every upstream-derived Python file retains its
copyright and license header.

## Snapshot scope

The runtime and upstream regression-suite files pinned here are:

- `evaluation_lib.py`
- `instructions.py`
- `instructions_registry.py`
- `instructions_util.py`
- `instructions_test.py`
- `instructions_util_test.py`

`__init__.py` makes those files a private package at
`whetstone_envs.c22._vendor.instruction_following_eval`. First-party C22
code and the vendored modules use that namespace exclusively.

## Local patch

The complete patch against the pinned upstream files is
[`VENDORED_DIFF.patch`](VENDORED_DIFF.patch). It has two functional
parts:

1. Absolute intra-package imports in production and test modules are
   package-relative. Importing C22 is isolated from top-level
   `instruction_following_eval` modules and does not change module search
   paths.
2. `NumberOfWords` accepts the C22-local relation `"exactly"` and checks
   equality. The shared upstream `_COMPARISON_RELATION` tuple remains
   unchanged, so existing `"less than"` and `"at least"` behavior and all
   other checker semantics stay pinned to upstream.

## Runtime dependencies

The upstream requirements used by these modules (`absl`, `langdetect`,
`nltk`, and `immutabledict`) are declared in this repository's
`pyproject.toml`. C22 uses complete explicit kwargs for every supported
atom. Its word counter is the pure-regex
`nltk.tokenize.RegexpTokenizer`; supported C22 generation and scoring do
not require a network download or module-global random sampling.
