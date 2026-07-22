# Build report: PR 0 — shared harness

- **Repo:** `whetstone-envs`
- **Branch:** `envs/00-harness` (off `main`)
- **Authority:** `PLAN.md` "Shared harness" section; quick-test rubric
  criteria 2, 5, 7, 8, 13 and the aggregation-crosses-strata callout.
- **Scope:** the five model-call-free core primitives every quick-test
  candidate reuses. No real task logic; fully unit-tested against
  synthetic fixtures.

## Deliverables

Package layout follows the existing repo convention (`src/` layout,
`py.typed`, one module per concern, numpy-style docstrings, ruff
line-length 79).

| Module | Contents |
| --- | --- |
| `whetstone_envs.core.instance` | Frozen `Instance` (`id`, `seed`, `strata`, read-only `prompt_inputs`, `gold`); `make_instance` constructor normalizing a bare-string stratum and freezing inputs. |
| `whetstone_envs.core.probes` | `ProbePair` (naive + ceiling templates + render fn); shared `normalize` (strip surrounding whitespace + one wrapping code fence, idempotent); `render_with_prompt_inputs` restricted to public inputs. |
| `whetstone_envs.core.scoring` | `exact_match(prediction, gold) -> int` (0/1, no partial credit); `Observation`/`Outcome` (SCORED/FAILED/MISSING); `Aggregate`; the `repeat -> task -> stratum -> overall` ladder (`aggregate` plus level helpers). |
| `whetstone_envs.core.pool` | `TaskPool` (dedup-checked instances, derived stratum membership, `stratum_counts`) with `.split(internal_eval_n, official_n, held_out_n)` returning a `PoolSplit` that asserts disjointness. |
| `whetstone_envs.core.manifest` | `Manifest` JSON writer/reader recording generator version, seed-range pins, per-stratum counts, and a stable SHA-256 `content_hash`; `matches_pool` for diffing a regenerated pool against a frozen manifest. |

`whetstone_envs.core.__init__` re-exports the public surface.

## Rubric mapping

- **Criterion 2 (exact check, no partial credit):** `exact_match`
  returns exactly 0/1 after shared normalization; there is no
  per-instance partial-credit path anywhere in scoring.
- **Criterion 5 (determinism):** `content_hash` is order-independent
  (sorted prompt-input keys, canonical JSON) and reproducible;
  `test_content_hash_is_deterministic` regenerates a pool twice and
  asserts an identical hash.
- **Criterion 7 (generated set with known ground truth + splits):**
  `TaskPool` carries pinned instances with gold labels and derived
  stratum membership; `.split` produces the disjoint
  internal/official/held-out subsets.
- **Criterion 8 (identity/seed pinning):** `Instance` pins `seed`
  distinct from `id`; `Manifest` records the seed range, so a candidate
  generator can assert fresh-seed ranges and audit them. (The concrete
  contamination assertion belongs in each candidate's generator, per
  the plan; the harness provides the pinned fields it checks against.)
- **Criterion 13 (failure paths, never silent zeros):** an
  `Observation` is either a scored 0/1 or a FAILED/MISSING marker. Any
  non-scored observation forces `Aggregate.mean = None` and surfaces
  `failed_count`/`missing_count`, and incompleteness propagates up every
  level — a failure in one stratum makes the overall aggregate visibly
  incomplete rather than a silent zero. Covered by
  `test_failed_repeat_makes_task_incomplete_not_zero`,
  `test_failed_observation_propagates_to_overall`, and
  `test_empty_stratum_is_incomplete_not_zero`.
- **Aggregation-crosses-strata callout:** `aggregate` reduces repeats
  within task, then tasks within stratum, then strata — the mean of
  stratum means, never a raw pooled mean.
  `test_aggregation_crosses_strata_not_raw_mean` pins an imbalanced pool
  where the two differ (0.5 vs 0.75) and asserts the strata-crossing
  value.

## Tests

`tests/core/` — 55 tests, all against hand-built synthetic fixtures
(`conftest.py`), no real task logic:

- `test_instance.py` — freezing, caller-dict detachment, validation,
  value-equality and hashing.
- `test_probes.py` — normalization table (whitespace, fences,
  language-tagged fences, backtick-safety, idempotence) and render
  isolation (gold cannot be interpolated; missing field raises).
- `test_scoring.py` — exact-match table, binary-score enforcement, the
  full aggregation ladder, and the criterion-13 incomplete cases.
- `test_pool.py` — stratum counts, contiguous disjoint split, oversize
  and negative-size rejection, overlap assertion.
- `test_manifest.py` — deterministic/order-independent hash, JSON and
  file round-trips, matches-regenerated-pool and detects-drift, and
  malformed-input rejection.

## Quality gates (all green)

Run in `whetstone-envs`:

- `uv run ruff check .` — All checks passed
- `uv run ruff format --check .` — all files formatted
- `uv run ty check` — All checks passed
- `uv run pytest` — 55 passed
- `uv run pre-commit run --all-files` — every hook passed

## Decisions recorded (autonomous calls)

- **`Observation`/`Outcome` typed markers** rather than sentinel scores:
  makes "failed vs missing vs scored" structural, so criterion 13's
  "never a silent zero" is enforced by the type, not by convention.
- **`Aggregate.mean is None` on any incompleteness**, with
  `failed_count`/`missing_count`/`complete` exposed, so the shortfall is
  visible at every level and propagates to the root.
- **Instance kept hashable** by widening `prompt_inputs` to `Mapping`
  and defining an explicit `__hash__` over sorted items (the wrapped
  `MappingProxyType` is itself unhashable). Instances remain frozen and
  read-only; this lets them be used in sets/dict keys downstream.
- **`.split` is contiguous in pool order** (internal, then official,
  then held-out) and allows leaving a tail unassigned; sizes are caller
  arguments (the per-spec proposed numbers), never hardcoded.
- **Report location:** written to `reports/build-envs-pr0.md` inside the
  repo under change (the task's `undefined/reports/...` path had an
  unpopulated prefix).

## Constraints honored

- All work on `envs/00-harness`, created off `main`. Nothing committed
  to `main`; nothing pushed to any remote.
- Surgical additions only — no changes to existing modules beyond adding
  `core/`; existing `test_package.py` untouched.
