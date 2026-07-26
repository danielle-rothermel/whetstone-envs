# Vendored: Wenyueh/inductive_reasoning_benchmark (InductionBench)

- **Upstream:** https://github.com/Wenyueh/inductive_reasoning_benchmark
- **Vendored commit:** `e0b839221a8509b351b324dfb247b35a434b7fd5`
  (2025-06-26; ACL 2025 long paper "InductionBench", Hua et al.,
  arXiv:2502.15823).
- **License:** Apache-2.0 (see `LICENSE` in this directory).
- **Vendored on:** 2026-07-22, for whetstone-envs candidate **c23**
  (subregular ISL/OSL rule induction, InductionBench-style).

## Why vendored — and why *patched*, not verbatim

The c23 baseline spec regenerates fresh subregular ISL/OSL instances with
this generator and reuses its rule-application transducers
(`apply_ISL_rule` / `apply_L_OSL_rule` / `apply_R_OSL_rule`) unmodified as
the independent oracle. Unlike c22's IFEval library (reused byte-for-byte),
the InductionBench active path **does not import or run as shipped** — the
repo review verified three import-time crashes plus a
`PYTHONHASHSEED`-dependent nondeterminism. So this vendor is applied as a
small, reviewable series of **named patches** on top of the byte-for-byte
originals, each recorded as its own git commit, and the whole delta is
captured in `VENDORED_DIFF.patch` for one-file review.

Vendoring pins the exact generator + oracle version so a regenerated pool
is byte-reproducible against a frozen manifest, and so the generation path
has no network or upstream-checkout dependency.

## What was vendored

Byte-for-byte copies of upstream (the minimal on-path set), committed
**first, unmodified**, before any patch:

- `standard_benchmark/synthetic_data_generation.py` → `synthetic_data_generation.py`
  (~330 LOC: rule sampling, characteristic-sample generation, and the
  three `apply_*_rule` oracle transducers).
- `utils.py` → `utils.py` (`generate_all_k_strings`,
  `translate_input_output_pairs`, and the answer extractors; ~215 LOC).
- `LICENSE`.

Not vendored (upstream files only the model-evaluation / few-shot /
exploration paths use, never reached by our generation + oracle path):
`standard_benchmark/inference.py`, `standard_benchmark/standard_run.py`,
`model.py`, the entire `exploration_benchmark/`, `run_inference.sh`,
`requirements.txt`, the bundled `result/` archives, and the upstream
`README.md`/PDF.

Added by this vendor (not present upstream):

- `config.py` — a config stub (see the config-stub patch below). Upstream
  ships no `config.py` at all.
- `__init__.py` — a package marker so the vendored bare-import modules load
  once `whetstone_envs.c23` has put this directory on `sys.path`.
- `PROVENANCE.md` — this file.
- `VENDORED_DIFF.patch` — the full unified diff of every patch below versus
  the byte-for-byte upstream, so the delta is reviewable in one place.

## The patches applied (each a separate commit)

Every edit below is a minimal, named change; none alters the sampling
*algorithm* or the three oracle transducers' *logic*. The
`apply_ISL_rule` / `apply_L_OSL_rule` / `apply_R_OSL_rule` functions are
**untouched** — the c23 oracle reuses them exactly as written.

1. **config stub** — add `config.py` exposing a mutable `vocab` list, so
   `import config` (upstream `synthetic_data_generation.py:4`) resolves.
   Upstream shipped no such file → `ModuleNotFoundError` as cloned.

2. **drop the broken import + `sys.path.append` hack** — upstream
   `synthetic_data_generation.py:8` imports
   `translate_fewshot_input_output_pairs`, a function that **does not exist
   in `utils.py`** → `ImportError` at import time. It is used only by
   `generate_few_shot_data`, which is off our path. Remove that name from
   the import; also remove the `import sys; sys.path.append('..')` lines
   (upstream lines 6–7): the vendored copy resolves its bare `import
   config` / `from utils import ...` via the `sys.path` entry the c23
   package installs, so the fragile relative-path append is unnecessary
   (and `sys` is re-imported where the code actually needs it, inside the
   two `sys.exit` guards). Also drop the top-level `from tqdm import tqdm`
   and its lone `tqdm(...)` wrapper in `generate_data` (a progress bar over
   an off-path code path for us): `tqdm` is a hard import-time dependency
   upstream, not otherwise required by the generation path we use.

3. **thread a real seed parameter** — upstream seeds only via a
   module-global `random.seed(0)` set once in the (un-vendored)
   `inference.py` / `standard_run.py`, with no way to reseed per stratum
   /instance. Add an explicit `seed` parameter to `generate_rules` and
   `generate_data` (and a shared `_seed(seed)` helper) that calls
   `random.seed(seed)` at entry, so distinct strata draw distinct,
   reproducible instances without editing source. The module-global
   `__main__` demo block's `random.seed(0)` is left as-is (it is a demo,
   off our path).

4. **`list(set(...))` → `sorted(set(...))` at the 6 nondeterminism call
   sites** — upstream lines 53, 61, 107, 132, 208, 244 build lists by
   iterating a `set` of **strings**. Python randomizes string hashing per
   process, so set-iteration order — and therefore which rule / output /
   sample `random.choice` picks — depends on `PYTHONHASHSEED`. Verified:
   two runs at the same `seed` produced different instance hashes until
   fixed. Replacing each `list(set(...))` with `sorted(set(...))` makes
   iteration order canonical and kills the hash-order dependence at the
   source (the spec's preferred fix over pinning `PYTHONHASHSEED`). This is
   why the determinism regenerate-twice test passes under a randomized
   `PYTHONHASHSEED`.

No line of the three `apply_*_rule` oracle transducers was edited.
