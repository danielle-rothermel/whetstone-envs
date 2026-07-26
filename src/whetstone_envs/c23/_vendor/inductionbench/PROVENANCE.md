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
originals. The whole delta is captured in `VENDORED_DIFF.patch` for
one-file review.

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
- `__init__.py` — a package marker for isolated package-relative imports.
- `PROVENANCE.md` — this file.
- `VENDORED_DIFF.patch` — the complete, byte-exact, applicable unified diff
  versus the pinned upstream, including `/dev/null` sections for `config.py`
  and `__init__.py`, so the delta is reviewable in one place.

## The patches applied

Every edit below is a minimal, named change; none alters the sampling
*algorithm* or the three oracle transducers' *logic*. The
`apply_ISL_rule` / `apply_L_OSL_rule` / `apply_R_OSL_rule` functions are
**untouched** — the c23 oracle reuses them exactly as written.

1. **config stub** — add `config.py` exposing a mutable `vocab` list for the
   generator's package-relative config import. Upstream shipped no such file
   and its bare import fails with `ModuleNotFoundError` as cloned.

2. **package-relative isolation; drop the broken import and path hack** —
   the vendored generator imports `config` and `utils` package-relative, so
   it cannot reuse or mutate unrelated top-level modules and c23 does not
   change the embedding process's `sys.path`. Upstream
   `synthetic_data_generation.py:8` imports
   `translate_fewshot_input_output_pairs`, a function that **does not exist
   in `utils.py`** → `ImportError` at import time. It is used only by
   `generate_few_shot_data`, which is off our path. Remove that name from
   the import and remove `sys.path.append('..')`. Keep `sys` because the
   two `sys.exit` guards use it. Also drop the top-level `from tqdm import tqdm`
   and its lone `tqdm(...)` wrapper in `generate_data` (a progress bar over
   an off-path code path for us): `tqdm` is a hard import-time dependency
   upstream, not otherwise required by the generation path we use.

3. **thread a real seed parameter through a private RNG** — upstream seeds
   Python's process-global RNG, with no way to reseed per stratum or avoid
   consuming the embedding application's random stream. The vendored module
   owns a private `Random` instance. An explicit `seed` parameter on
   `generate_rules` and `generate_data`, plus the shared `_seed(seed)`
   helper, seeds only that private RNG. `utils.translate_input_output_pairs`
   receives the same RNG explicitly. Distinct strata remain reproducible,
   unrelated threads cannot perturb generation through the process-global
   RNG, and generation does not mutate that global state.

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
