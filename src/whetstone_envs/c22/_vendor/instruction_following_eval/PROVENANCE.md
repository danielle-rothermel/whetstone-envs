# Vendored: google-research IFEval (`instruction_following_eval`)

- **Upstream:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Vendored commit:** `37ffb72669bc762fe899d5eaec83d28be2c882cc` (2026-07-20)
- **License:** Apache-2.0 (see `LICENSE` in this directory; per-file
  headers read `Copyright 2026 The Google Research Authors`).
- **Vendored on:** 2026-07-22, for whetstone-envs candidate **c22**
  (stacked IFEval instruction-following constraints).

## Why vendored

The c22 baseline spec reuses this checker library **verbatim** for both
the generation-side constraint selection (`build_description`) and the
scoring oracle (`check_following` /
`evaluation_lib.test_instruction_following_strict`). Vendoring pins the
exact checker version so a regenerated pool and a re-scored response are
reproducible against a frozen manifest.

## What was and was not changed

Byte-for-byte copies of upstream (unmodified):

- `instructions.py`
- `instructions_util.py`
- `instructions_registry.py`
- `evaluation_lib.py`
- `instructions_test.py`
- `instructions_util_test.py`
- `LICENSE`

Added by this vendor (not present upstream):

- `__init__.py` — a package marker only. Upstream imports this directory
  as a path under the repo root and ships no `__init__.py`; the marker
  lets the vendored tree be imported as `instruction_following_eval`
  once `whetstone_envs.c22._vendor` has put this directory on
  `sys.path`. The marker adds no logic.
- `PROVENANCE.md` — this file.

No line of upstream checker logic was edited. The module-global
`random` seed plumbing flagged in the repo review is handled entirely in
`whetstone_envs.c22.generate` (which seeds `random` before each
`build_description` and passes explicit nonce kwargs), never by editing
the vendored source.

## Runtime dependencies

The upstream `requirements.txt` lists `absl`, `langdetect`, `nltk`,
`immutabledict`; these are declared in the repo's `pyproject.toml`.
`instructions_util.count_words` uses only `nltk.tokenize.RegexpTokenizer`
(a pure-regex tokenizer, no `punkt` download). The c22 atom pool
deliberately excludes every atom whose `check_following` calls
`langdetect` (casing/language atoms) or the `punkt` sentence tokenizer
(sentence-count atoms), per the spec's determinism exclusions, so no
network download is ever required at generation or scoring time.
