# Step 10 — Four-optimizer validation protocol (c19 primary, c18 second family)

**Status.** Draft for adversarial statistical review, then for implementers.
**Companion to** `../2026-08-21/1124-whetstone-next-steps-sequence.md` (Step 10 and
decisions D14–D18) and `../2026-08-09/1756-ed1-baseline-calibration-power-design.md`
+ `../2026-08-09/1836-bootstrap-power-options.md` (the power backbone).
**Scope.** This document is the protocol only. It does not authorize spend; the
pilot gate in §7 is where spend authorization attaches.

Everything below is pre-registered. Once the pilot gate closes, the only
permitted mid-run change is the staged `K` escalation in §3.6 and the abort
rules in §3.9. Any other change invalidates the confirmatory claim and forces a
fresh run on a new `study_id`.

---

## Revision 2 — 2026-08-23

Revision 1 (2026-08-22, sha256 `a311de47411412c7a02b7e83c46874f927f17eb9c173c0b53b207499c6d0e85e`)
was written before Danielle ratified the decisions below on 2026-08-23. The
executed design — `whetstone_envs/optim/study/protocols.py` — follows those
decisions; this text did not. **The decisions are the authority**, so the body
below is updated to match the code, and this block is the provenance for every
change. Plan-note numbers refer to the follow-up notes in
`../2026-08-21/1124-whetstone-next-steps-sequence.md`; `D2` refers to
`1200-step10-implementation-assignment.md`.

**Digests.** Revision 1 hashed to
`a311de47411412c7a02b7e83c46874f927f17eb9c173c0b53b207499c6d0e85e`, which is now
a *historical* value: it is the digest a manifest initialised before 2026-08-23
recorded, and no study should be run against it. Revision 2's digest is recorded
in `protocols.py` as `PROTOCOL_DOC_SHA256` and recomputed from the file at
`init` time, so a manifest always names the revision actually in force.

The body sections are rewritten in place rather than annotated: a
pre-registration a reader has to reconcile against a changelog is not
pre-registered. Nothing elsewhere in the document is rewritten to hide that
revision 1 said something different.

| # | What changed | From → to | Decided by |
|---|---|---|---|
| 1 | Held-out split | 220 → **440**; splits `(88, 132, 440)` | note 19 |
| 2 | Pinned MDE table row | recomputed at T=440: 0.0622 / 0.0690 | note 19 |
| 3 | Optimizer train/val | implicit → **explicit disjoint 44/44** of the internal split, required, refused if missing or overlapping | note 18 |
| 4 | Codex admitted evaluate-call cap | 20 → **8** | assignment D2 |
| 5 | GEPA `max_metric_calls` | `auto="light"` (732) → pinned **200** | note 14 |
| 6 | null-B (identity) | an optimizer run → **the seed candidate evaluated through the study's report harness**, no optimizer step | note 13 |
| 7 | MIPROv2 `num_candidates` | 6 → **3** (the minibatch floor) | note 25d |
| 8 | MIPROv2 `num_trials` | derived per mode (10 / 9) → **10, pinned uniformly** | this revision, §5.1 |
| 9 | Codex agent model | "whatever the CLI session provides" → pinned **`gpt-5.6-sol`** | note 26 |
| 10 | MIPROv2 minibatch size | resolved at run time → pinned **35** | note 28 (Phase E item 3) |
| 11 | COPRO search shape | assumed → pinned **breadth 6 / depth 3**, forwarded to the run and the estimator | note 28 (#24 review) |
| 12 | GEPA reflection minibatch | unstated (ran at 1) → pinned **3** | note 28 (#24 review) |
| 13 | `K_REPEAT = 3` scope | ambiguous → **applies to in-search evaluations too**; provider rows ×3, GEPA's metric-call budget not | note 28 |
| 14 | Cost model | 630 tokens/call, "small tens of dollars" → **measured ≈4.5k completion tokens/call, ≈$0.00168/call**; stage totals from the regenerated plan | note 24 |
| 15 | null-A | bypassed the runner → **routed through the ordinary runner with COPRO's shape** | note 25b |
| 16 | §6 Codex evidence | unbounded → **one real Codex-direct run**, artifacts historical evidence only | note 20 |
| 17 | L1's mechanical form | "the internal Eval Config ref" → **the evaluated task set is contained in the internal split**, checked on all three evaluation surfaces | this revision (2026-08-23), §3.2 |
| 18 | Missing-row handling | aggregation `missing_data="propagate"` (one failed row voids the whole evaluation) → **`"skip"` with `max_skip_fraction = 0.10`**, paired with a **per-task completeness floor**: an evaluation is refused (`missing_data`) if any task lost every repeat, or if fewer than 90% of planned tasks were measured. The row tolerance alone cannot see a fully-lost task — at 76 tasks × 4 repeats one lost task is 1.3% of rows, inside the bound, yet it is dropped from the task mean's denominator and biases the reported mean upward. Paid transports additionally retry transient failures (429/5xx/timeout, **5 attempts total per logical call**, 2–32 s exponential backoff with jitter, `Retry-After` **delta-seconds** honoured and bounded at 120 s; the HTTP-date form is ignored by design, since resolving it against two clocks is least reliable exactly when it matters) and raise the per-call timeout 30 s → 300 s to cover reasoning-token calls. The 5 attempts are spent inside the transport wrapper, which is the sole owner of the retry budget; whetstone's driver is pinned to a single attempt so the two loops cannot multiply. | this revision (2026-08-23), §3.9 and O7 |
| 19 | Task-model reasoning effort | unpinned ("this protocol accepts the spend and does not pin reasoning effort") → **pinned `minimal`** on the task route, hashed into the pre-registered design and **enforced**: every paid bind — the reporting pass and the in-search evaluations alike — reports its effort into the manifest, and a bind disagreeing with the design is refused before it bills. The proposer route stays unpinned. | Danielle, 2026-08-24 |

**On item 8.** Revision 1 derived per-mode trial counts from DSPy's
`_recommended_num_trials(component_count=1, searches_demos, n)` at `n = 6`
candidates: `fewshot` (`num_vars = 2`) → `max(2·2·log₂6, 1.5·6) = 10`, and
`zeroshot`/`ground_only` (`num_vars = 1`) → `max(2·1·log₂6, 9) = 9`. That
derivation no longer applies, for two independent reasons. First, its input
changed: note 25d pins `num_candidates = 3`, and at `n = 3` the same formula
yields 7 and 5 — so keeping 9 would be neither the old derivation nor the new
one. Second, and decisively, the study **pins** `num_trials` on the control
(`run.py` passes `num_trials=spec.miprov2_num_trials`), so
`_recommended_num_trials` never executes; auto-mode's recommendation is not the
quantity the run uses. A derived value that the code does not derive is not a
pre-registration. The protocol therefore registers **`num_trials = 10`
uniformly across all three demo modes**, which is what `protocols.py`
(`MIPROV2_NUM_TRIALS = 10`, one constant, no per-mode table) and the estimator
both already use. The two fidelity modes run once each, so the uniform count
costs at most the difference between 9 and 10 trials on two single runs.

**On item 13.** `K_REPEAT = 3` multiplies every *provider-row* count, including
the evaluations an optimizer performs inside its own search. GEPA is the one
exception in *unit*, not in policy: its 200-call pin is denominated in metric
calls, and each metric call is one instance evaluation, so the pin bounds rows
directly and is not multiplied again.

**On the fidelity arms.** MIPROv2's `zeroshot` and `ground_only` modes are
fidelity evidence for the `MIPRO_ZEROSHOT_GROUNDING` and
`MIPRO_GROUND_ONLY_DEVIATION` invariants. They are **not efficacy arms**: they
carry no held-out claim, they do not enter the Holm family, and the report lists
them without a verdict. The Holm family is exactly the four real optimizers
(§3.7). This was always the intent (§5.3 pre-registers `fewshot` as the mode
carrying the MIPROv2 claim); revision 2 makes it structural rather than
conventional, via an arm role recorded in the manifest.

---

## 0. What this stage must prove

Three independent claims, each with its own evidence. "Runs to completion" is
evidence for none of them.

| Claim | Short name | Evidence type | Failure is |
|---|---|---|---|
| C1 | **Fidelity** — each optimizer adheres to its reference algorithm | Pure-function trace audits over run evidence, per optimizer, run in CI on fake artifacts and on every real run | a whetstone implementation defect |
| C2 | **Efficacy** — each optimizer improves held-out accuracy over the naive seed | Paired bootstrap CI on held-out per-task scores excluding zero; null optimizers show none | a negative result about that optimizer at this scale, not necessarily a defect |
| C3 | **Toolchain generality** — a second family runs through the identical runner with only the env adapter swapped | c18 PrOntoQA runs end-to-end for all four optimizers; a diff-shaped artifact showing which modules changed | a domain leak in whetstone-ai or in the shared envs runner |

C1 gates C2: an optimizer whose fidelity audit fails is reported as
**not validated**, and its efficacy number is reported as descriptive only,
never as a claim.

C3 is deliberately not a second statistical study (D15). It is one budgeted run
per optimizer plus a mechanical no-domain-leak assertion.

---

## 1. Prerequisites (hard blockers)

Step 10 cannot start until all of these are true. Each is checkable.

1. **whetstone-ai `v0.1.5`** released with #125 (MIPROv2) and #126 (GEPA
   platform) merged, plus the Codex half of Step 8 released.
2. **Step 11 lockstep bump** landed (dr-exec 0.1.14 / dr-store 0.2.6 /
   dr-platform 0.2.7).
3. **`OptimResult.cost` populated on real runs.** Today `cost` is
   `ImmutableJsonObject({})`
   (`src/whetstone/optim/contracts.py:1285`) and the 0.1.2 real runs recorded
   "Spend: not visible"
   (`../../whetstone-envs/2026-08-21/1937-c19-optimizer-reruns-on-0.1.2.md`).
   Required shape (§5.4): per-role calls and tokens always; `usd` only when
   every counted call carries a price.
4. **whetstone-envs Step 12a + 12b** merged: `build_c19_miprov2_adapter`,
   `build_c19_codex_adapter`, `--optimizer {copro,gepa,miprov2,codex}`,
   demonstrations slot in the c19 render contract, GEPA ported onto
   `build_gepa_harness_adapter`, reporting projection verified against 0.1.5.
5. **envs `reporting/` on main.** `src/whetstone_envs/reporting/` exists at
   `574036c` (PR #15) but the working checkout head is `dee0150` (PR #16);
   confirm the merge order left both on `main` before building on
   `reporting/projection.py`.
6. **7a `eval/analysis/` tests** merged (#121) — they are.

If (3) or (4) slip, Step 10 does not start early with a partial matrix. A
half-populated cost column silently breaks the budget accounting in §5.

---

## 2. Claim C1 — fidelity: the trace audits

### 2.1 Shape and placement

New package `whetstone-envs/src/whetstone_envs/optim/audit/`, one module per
optimizer plus a shared evidence-reader:

```
optim/audit/__init__.py         # public: audit_run(run_dir, optimizer) -> AuditReport
optim/audit/_evidence.py        # opens runtime.sqlite via dr_store.sync.open_sqlite,
                                # dereferences TypedRefs through public whetstone APIs
optim/audit/copro.py
optim/audit/miprov2.py
optim/audit/gepa.py
optim/audit/codex.py
optim/audit/schema.py           # AuditReport / AuditFinding pydantic models (persisted)
```

Contract, matching the plan's "pure function over evidence, fails loudly":

- `audit_run` takes only `(result_json_path, sqlite_path, control_json)` — no
  network, no re-execution, no re-scoring.
- Every invariant returns an `AuditFinding` with `invariant_id`
  (a `StrEnum`, `@verify(UNIQUE)`), `status ∈ {pass, fail, not_applicable}`,
  and the exact evidence refs it read (`schema_name` + `content_hash`).
- `AuditReport.passed` is `all(f.status is not FAIL)`. The report is written to
  `audit.json` beside `result.json` and cited by content hash in the manifest.
- Audits import only public `whetstone.*` surface; the existing envs
  `test_public_imports` guard (empty `ALLOWED_PRIVATE_IMPORTS`, extended in Step
  2b to catch `cast("Any", …)._name`) must cover `optim/audit/` too.

### 2.2 Evidence records each audit may read

All of these are already on the public surface and already carried by real runs
(0.1.2 note confirms search evidence and reward refs resolve):

| Record | Path | Fields the audits use |
|---|---|---|
| `OptimResult` | `result.json`, schema `OPTIM_RESULT_SCHEMA` | `run`, `proposals`, `step_results`, `cost`, `seed_retained`, `terminal_failure` |
| `OptimStepResult` | `contracts.py:906` | `request`, `proposed_candidates`, `accepted_candidates`, `resolved_intents`, `search_evidence`, `tool_evidence`, `state_ref`, `budget_delta`, `budget`, `status`, `seed_retained`, `retained_candidate_ref` |
| `IntentResolution` | `contracts.py:558` | `optim_eval_request`, `outcome`, `eval_result_ref`, `reward_ref`, `reward_evidence_refs`, `resolved_eval_config` |
| `SearchEvidence` | `contracts.py` (GEPA/MIPROv2 in-search evals) | same eval/reward refs, `optim_step_index` |
| `ToolEvidence` | `contracts.py` | admission-ledger linkage for Codex |
| `EvalEvidence` | `whetstone.eval.schema`, schema `whetstone.eval_evidence` | `eval_config_ref`, `eval_role`, `task_hashes`, `num_seeds`, aggregate + per-row output refs |
| adapter state | `state_ref` under `GEPA_STATE_KEY` / `MIPROV2_STATE_KEY` | GEPA Pareto front + `skipped_mutations`; MIPROv2 `StudyTranscript` (schema v5), `Miprov2RngCheckpoint`, `demo_mode` |
| tool-admission ledger | `whetstone_tool_admission_*` tables in `runtime.sqlite` (`optim/tools/admission.py`) | Codex: every admitted/refused call, `capacity_debit_ordinal` |

### 2.3 COPRO audit — `copro.py`

Reference: `COPRO_ALGORITHM_VERSION = "dspy_copro_single_prompt/v1"`, reference
commit `6f68dcdb…` (`optim/copro/control.py`).

| id | Invariant | Evidence read |
|---|---|---|
| `COPRO_BREADTH_PER_DEPTH` | Every non-terminal step proposes exactly `control.breadth` candidates; the terminal step proposes `breadth` or terminalizes with `seed_retained` | `step.proposed_candidates`, `control.breadth` |
| `COPRO_DEPTH_STEPS` | Step count is exactly `control.depth + 1` (depth search rounds plus the terminal report), or fewer only with a `terminal_failure` | `result.step_results`, `control.depth` |
| `COPRO_INTERNAL_ONLY` | Every `IntentResolution.resolved_eval_config` equals the internal Eval Config ref and every `EvalEvidence.eval_role is INTERNAL` | `resolved_intents[*].resolved_eval_config`, dereferenced `EvalEvidence.eval_role` |
| `COPRO_BEST_SO_FAR` | The candidate accepted at step *i* has internal reward ≥ every candidate evaluated at steps ≤ *i*; ties broken by the earlier candidate | `resolved_intents[*].reward_ref` dereferenced, `accepted_candidates` |
| `COPRO_DISTINCT_BASES` | Proposed candidates within one step have pairwise distinct `base_ref` | `proposed_candidates[*].record.base_ref` |
| `COPRO_NO_SEARCH_EVALS` | `search_evidence` is empty on every step (COPRO does not drive evals inside its own search — confirmed on 0.1.2) | `step.search_evidence` |
| `COPRO_TERMINAL_PROVENANCE` | The terminal candidate is either a proposal minted in this run or the run's `initial_candidate_ref` under `seed_retained` — never the ceiling probe | `result.proposals`, `retained_candidate_ref`, compare against `PROBES.ceiling_template` |

`COPRO_TERMINAL_PROVENANCE` is the direct regression guard for the
`_with_distinct_mutation` forgery retired in Step 2b.

### 2.4 MIPROv2 audit — `miprov2.py`

Reference: `MIPROV2_ALGORITHM_VERSION = "dspy_miprov2/v2"`, pinned
`optuna==4.8.0`, `MIPROV2_REFERENCE_COMMIT`.

| id | Invariant | Evidence read |
|---|---|---|
| `MIPRO_BOOTSTRAP_BEFORE_PROPOSAL` | In `fewshot` and `ground_only` and `zeroshot`, bootstrap eval intents (`bootstrap_eval_source` config ref) all precede the first instruction-proposal call, ordered by `optim_step_index` then resolution index | `resolved_intents`, `search_evidence`, `control.bootstrap_eval_source` |
| `MIPRO_ZEROSHOT_GROUNDING` | `zeroshot` performs exactly DSPy's 3 bootstrapped / 0 labeled grounding demos (`BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3`) and ships **no** `demo_set` on any candidate — the D7 correction landed in `117d4416` | bootstrap intent count; `Miprov2ComponentSelection.demo_set is None` on every proposed candidate |
| `MIPRO_GROUND_ONLY_DEVIATION` | `ground_only` runs bootstrap, exclude the demo dimension from `Miprov2ParameterSpace`, ship no `demo_set`, and the report is flagged as a whetstone deviation (`algorithm_version` marker) | `StudyTranscript.demo_mode`, parameter-space record, candidate payloads |
| `MIPRO_TPE_SELECTION` | Every trial's parameter assignment appears in the `StudyTranscript` in order; replaying the transcript into a fresh `TPESampler(seed=control.seed, multivariate=True)` reproduces the recorded suggestion sequence | `state_ref` → `StudyTranscript` (schema v5), `control.seed`, `control.num_trials` |
| `MIPRO_MINIBATCH_SIZING` | Every minibatch eval intent's task set is a subset of `control.valset_task_hashes` of size `min(control.minibatch_size, len(valset))`, drawn by the recorded RNG | `resolved_intents[*].optim_eval_request` task sets, `Miprov2RngCheckpoint` |
| `MIPRO_PERIODIC_FULL_EVAL` | A full-valset evaluation of the incumbent occurs every `control.minibatch_full_eval_steps` trials and once at the end | intent task-set sizes vs `len(valset_task_hashes)` |
| `MIPRO_BOOTSTRAP_THROUGH_ENGINE` | No bootstrap evaluation appears on the proposer transport; every one is an `IntentResolution`/`SearchEvidence` entry with a resolvable `eval_result_ref` (D7) | absence of proposer-transport eval records; presence of engine intents |
| `MIPRO_TRIALS_MATCH_CONTROL` | Observed trial count equals `control.num_trials` (auto-mode: `_recommended_num_trials`, `light` → n=6) unless a `terminal_failure` truncated the run | `StudyTranscript`, `control.num_trials` |

### 2.5 GEPA audit — `gepa.py`

| id | Invariant | Evidence read |
|---|---|---|
| `GEPA_PARETO_FRONT` | The candidate pool at each step is exactly the Pareto-nondominated set over per-instance internal scores; no dominated candidate is retained | `state_ref` under `GEPA_STATE_KEY`; per-instance scores from `search_evidence[*].eval_result_ref` → `EvalEvidence` per-row outputs |
| `GEPA_MUTATION_TRACES_TO_REFLECTION` | Every accepted candidate has a reflection record in step state whose input traces are exactly execution traces of its base candidate, and whose output text equals the accepted candidate's `mutation_field` payload | step state reflection records, `accepted_candidates[*].record.payload` |
| `GEPA_METRIC_CALL_BUDGET` | Total metric calls ≤ `control.resolved_max_metric_calls`; for `auto` mode the resolved value equals `gepa_auto_budget(...)` recomputed from the control (`gepa/control.py:65`) | count of eval intents + search evidence entries; `control` fields |
| `GEPA_REFLECTION_MINIBATCH` | Each reflection consumes exactly `control.reflection_minibatch_size` instance traces | reflection records |
| `GEPA_SKIPPED_MUTATIONS_RECORDED` | Every rejected/unparseable reflection response produces a `GepaSkippedMutation` in step state (D5: one bounded retry, then surface); the key is present and durable on every step even when empty | step state `skipped_mutations` |
| `GEPA_STEP_EVIDENCE_PRESENT` | Every step carries `resolved_intents` (Step 2b item 6) and each `eval_result_ref` resolves in `runtime.sqlite` | `step.resolved_intents` |
| `GEPA_NO_FORGED_TERMINAL` | The terminal candidate is a live reflection draft or an honest `seed_retained` result whose `retained_candidate_ref` content hash equals `OptimRun.initial_candidate_ref` — never `PROBES.ceiling_template` | `retained_candidate_ref`, `result.proposals`, probe comparison |
| `GEPA_PLATFORM_RESUME_IDENTITY` | (platform dispatch only) For each deferral episode, the re-run's paid prefix is served from the effect cache and no `GepaEffectConflictError` was raised; the `platform_stage_index` salt differs per episode | effect-cache records, step provenance |

`GEPA_NO_FORGED_TERMINAL` is the second forgery regression guard and is the
invariant that the 0.1.2 rerun note demonstrated by hand.

### 2.6 Codex-direct audit — `codex.py`

| id | Invariant | Evidence read |
|---|---|---|
| `CODEX_NO_EVAL_OUTSIDE_TOOLS` | Every `EvalEvidence` record produced during the run has a matching admitted entry in `whetstone_tool_admission_*`; the count of admitted evaluate-calls equals the count of eval evidence records | admission ledger rows, `step.tool_evidence`, eval evidence set |
| `CODEX_FINAL_CANDIDATE_EVALUATED` | The returned candidate's ref appears in at least one admitted evaluation before the run's terminal step | admission ledger, `resolved_intents`, terminal candidate ref |
| `CODEX_CAPACITY_RESPECTED` | No `capacity_debit_ordinal` exceeds the run's admission capacity cap; every refusal is recorded as `refused` with no ordinal (`tools/admission.py:201`) | admission ledger |
| `CODEX_TOOL_SURFACE_MINIMAL` | Only `evaluate` and `read-scores` tool names appear in the ledger (D12 option (i) plus per-task readback) | admission ledger `tool_name` column |
| `CODEX_CONTAINMENT` | The dr-exec spawn recorded its wall-time and process-count budgets and terminated within them (D13: network allowed, filesystem scratch-only) | dr-exec outcome record |
| `CODEX_INTERNAL_ONLY` | Every admitted evaluation targeted the internal Eval Config ref | ledger + `EvalEvidence.eval_role` |

### 2.7 Where the audits run

- **CI, on fakes.** Every audit runs against the fake-transport CLI e2e
  artifacts already produced per optimizer/mode. This is the plan's "must pass
  on fakes too" requirement and is what makes the audits regression tests rather
  than one-shot scripts.
- **Every real run.** `audit.json` is written unconditionally; a failing audit
  does **not** abort the run (the paid evidence is still worth keeping) but does
  mark that run `fidelity=fail` in the manifest, which propagates to the report.
- **Negative tests.** Each audit ships at least one hand-mutated evidence
  fixture that must make it fail — e.g. a COPRO step with `breadth+1` proposals,
  a GEPA terminal candidate swapped to the ceiling template, a Codex eval
  evidence record with no ledger row. An audit with no failing fixture is not
  yet an audit.

---

## 3. Claim C2 — statistical design

### 3.1 Population and splits

c19's pool is generated, not sampled from a fixed corpus: 22 strata
(`scenario|size|fact`, `c19/generation.py:strata_labels`), `n_per_stratum`
instances each, `DEFAULT_N_PER_STRATUM = 16`, `MAX_N_PER_STRATUM = 128`,
`DEFAULT_SPLIT_SIZES = (88, 132, 132)` over a 352-instance pool.

This differs from the 76-task HumanEval population in the power plan, and the
difference is favorable: the binding resource there was tasks
(`1836-…: "tasks are the binding resource"`). Here tasks are cheap to generate
and the binding resource is provider spend. The power plan's *structure*
transfers; its *task ceiling* does not.

**Pinned population for Step 10.**

```
pool = generate_pool(n_per_stratum=32, seed_start=1_000_000)   # 704 instances
```

`n_per_stratum=32` doubles the default and keeps the manifest's existing
generator version (`c19-custom-v2`). The pool manifest content hash is recorded
in the study manifest; any change to it is a new `study_id`.

**Split sizes** (`TaskPool.split` is deterministic, strata-balanced, and
disjoint by construction — `pools/splitting.py:PoolSplit.__post_init__` asserts
disjointness):

| Split | Role | Size | Rationale |
|---|---|---|---|
| internal | optimizer's own search signal | **88** (4/stratum) | matches `DEFAULT_SPLIT_SIZES[0]`; large enough that COPRO/GEPA/MIPROv2 see a non-degenerate ranking signal, small enough that per-step search cost stays bounded |
| official | model selection across the K repeats of one optimizer | **132** (6/stratum) | selection needs less resolution than the confirmatory test |
| held-out | reporting only, touched once per optimizer | **440** (20/stratum) | sized in §3.4; raised from 220 by note 19 |

Remaining pool tail (704 − 660 = 44) stays unassigned;
`split_pool` explicitly permits an unused tail.

**Train/val inside the internal split (note 18, pre-registered).** Every
optimizer that has the concept runs with an **explicit, disjoint train/val
partition of the internal split**: `train = 44`, `val = 44`. This is a required
input with no default — a run is refused if the partition is missing, if the two
overlap, or if they exceed the internal split. MIPROv2 takes it as
bootstrap/search; GEPA takes it as trainset/valset and additionally requires
`train + val == internal` exactly. The sizes are carried on the study's arm spec
and hashed into the pre-registration, so an arm rerun at a different partition
is a different arm.

Rationale: with a disjoint partition, in-search improvement cannot be demo
memorization, which is what makes a debugging signal trustworthy. This
supersedes the `--miprov2-split {single-task,internal}` switch, which is
deleted. Selection on official and reporting on held-out are unchanged.

**c18 splits** use `default_split_sizes(pool, DEFAULT_CONFIG)` unchanged:
`SplitPlan(internal_eval=6, official=12, held_out=12)` scaled over 4 depth
strata at `n_per_stratum=30` → `(24, 48, 48)`. c18 is C3 only; it gets no power
analysis.

### 3.2 Leakage rules (pre-registered, mechanically checked)

L1. **The optimizer never sees official or held-out.** Every optimizer control
    binds `eval_role = INTERNAL`, and every evaluation it causes must evaluate a
    **task set contained in the internal split**, under the internal role,
    against a config that is neither the official nor the held-out one.
    `CoproControl` and `Miprov2Control` already refuse a non-internal role
    (`if self.eval_role is not EvalRole.INTERNAL: raise`). The audits assert it
    independently over evidence (`COPRO_INTERNAL_ONLY`, `MIPRO_*`,
    `CODEX_INTERNAL_ONLY`, GEPA equivalent).

    Containment, not equality of Eval Config identity: MIPROv2 minibatches the
    internal split and GEPA scores single tasks and Pareto subsets, so each such
    evaluation mints its own *derived* Eval Config over a subset of the internal
    split. That is the registered design, not a leak. The mechanical check reads
    all three evaluation surfaces — resolved intents, `search_evidence`, and
    `tool_evidence` — because which one an optimizer uses is its own
    implementation detail; a run yielding evidence on none of them leaves L1
    **unchecked**, which fails the study rather than passing it vacuously. How
    many evaluations used the full internal config versus a derived subset is
    recorded as an observation beside the verdict.

L2. **Selection happens on official, exactly once per optimizer.** For each
    optimizer, the K run outputs are scored on official; the arg-max run's
    terminal candidate is the optimizer's *representative candidate*. No other
    quantity may be computed on official and used to choose anything.

L3. **Held-out is evaluated exactly once per reported candidate.** One eval per
    (candidate × held-out split) at the design repeat count. The held-out
    evaluation happens *after* L2 selection is frozen and recorded. Held-out
    results may not feed back into any choice — not the model, not the split,
    not the demo mode, not K, not the number of steps.

L4. **The naive, ceiling, and both null candidates are evaluated on held-out
    under the identical procedure** (same `eval_config_hash`, same repeats, same
    provider controls), so the paired comparison is genuinely paired.

L5. **Split identity is content-addressed.** `EvalConfigs.held_out_task_hashes`
    and the internal/official `derive_eval_split` hashes are recorded in the
    study manifest. Any held-out task hash appearing in an internal or official
    eval config for the same `study_id` is a hard protocol violation and voids
    the study.

L6. **Mechanical check.** A `study_leakage_check` function over all run
    artifacts asserts L1 and L5 by set intersection of task hashes across roles,
    and asserts L3 by counting held-out `EvalEvidence` records per candidate
    (exactly one). It runs before the report is generated and fails the study
    loudly.

### 3.3 What "improvement" means

For both families: **exact-match accuracy**, the family's own scoring.

- c19: `scoring/exact_match.py` under `normalize`; `Observation` in the envs
  reporting schema already refuses any score outside `{0.0, 1.0}`. Reward policy
  `c19-exact-match`, single term `score` weight 1.0.
- c18: entailment label True/False, `oracle.score_gold`, same binary shape.

The reported statistic is the **per-task mean over repeats**, giving a per-task
score in `[0, 1]`. The study statistic is the mean of those per-task scores over
the held-out split, i.e. the split-level accuracy.

**Primary estimand per optimizer:**

```
Δ = mean_held-out( score(representative_candidate) ) − mean_held-out( score(naive_seed) )
```

paired by task.

### 3.4 CI method and held-out sizing

**Method.** Paired task-level percentile bootstrap,
`whetstone.eval.analysis.bootstrap_paired_delta_ci`
(`worktrees/step6-ai/src/whetstone/eval/analysis/statistics.py`), with
`level=0.95`, `resamples=10_000` (`DEFAULT_RESAMPLES`), and an explicitly
recorded `seed`. It resamples *tasks* with replacement and recomputes the paired
delta — the correct unit, since tasks are the exchangeable unit and repeats are
nested within them.

This is Option A of `1836-bootstrap-power-options.md` used for what Option A is
good for: "did the arm move at all" on a frozen split. It is not used for
design; design uses Option C (below). The doc's objection to Option A ("no K in
it") does not apply to the confirmatory step, where K is fixed by design.

**Recorded caveat (pre-registered):** percentile intervals under-cover at small
task counts. At T=440 with binary per-repeat outcomes this is minor, but the
report states it rather than hiding it, and §3.8's null optimizers are the
empirical check — if a null's CI excludes zero, the interval is miscalibrated
and every efficacy claim in the study is downgraded.

**Sizing held-out.** Option C backbone (`1836`, settled): with per-task deltas
of variance `τ² + 2σ²/K`,

```
MDE(T, K) ≈ (z_{α/2} + z_power) · sqrt( (τ² + 2σ²/K) / T )
```

Under a conservative binary worst case (`σ² ≤ 0.25`, `τ²` unknown pre-pilot),
at K=3 repeats and T=440:

```
sd(Δ̂) ≤ sqrt( (τ² + 2(0.25)/3) / 440 ) = sqrt( (τ² + 0.167)/440 )
```

With `τ² = 0.05` (a plausible between-task heterogeneity for a strata-balanced
generated pool), `sd(Δ̂) ≈ 0.0222`, giving a two-sided 95%/80%-power MDE of
`2.80 × 0.0222 ≈ 0.062`. With `τ² = 0.10`, MDE ≈ 0.069.

**Pinned MDE table (pre-registered, T=440, K_REPEAT=3, worst-case σ²=0.25).**
This is the row the study registers and `whetstone-study plan` prints; Stage 0
measures `τ̂²`/`σ̂²` and reports the realized MDE against it.

| τ² | T | K | MDE |
|---:|---:|---:|---:|
| 0.05 | 440 | 3 | 0.0622 |
| 0.10 | 440 | 3 | 0.0690 |

**This is the honest headline: at T=440, K=3, the study can only detect
improvements of roughly 6–7 accuracy points.** That is still a large effect,
and it is the resolution note 19 bought by raising held-out from 220 (where the
MDE was 0.087–0.104) for roughly 12% more study spend. The
pilot (§3.6) replaces `τ²` and `σ²` with measured values and re-inverts; if the
observed naive→ceiling headroom on c19 with gpt-5-nano is smaller than ~2× the
MDE, the design is underpowered and §3.9's abort rule fires *before* the full
spend, not after.

`whetstone.eval.analysis.analyze_power` computes this directly from the pilot's
naive/ceiling per-task vectors (`analyze_power(naive_per_task=…,
ceiling_per_task=…, pool_ceiling=…, anchor_samples=K)`), and
`run_anchor_calibration` produces the naive/ceiling anchor pair through the eval
engine with evidence validation. Note that `analyze_power`'s `DEFAULT_ALPHA =
0.25` is a *headroom fraction*, not a significance level — the target gap is
`0.25 × (ceiling − naive)`. For Step 10 we set `PowerConfig(alpha=0.25,
target_prob=0.80, sample_cap=8)` and report both the analytic MDE above and
`analyze_power`'s surface, flagging any disagreement.

**Option B check.** A hierarchical bootstrap (resample tasks, then repeats
within task) at the achieved `(T, K)` validates the closed-form `sd(Δ̂)`. Where
they disagree, the bootstrap wins and the report says so. This is ~30 lines over
the already-persisted per-row outputs; it costs no provider calls.

### 3.5 Repeats: K, and the temperature caveat

Two distinct counts, never conflated (per `1756`):

- **`K_REPEAT`** — repeats per (candidate, task) within one evaluation. This is
  `num_seeds` in `derive_eval_split` / `repeats` in the envs eval CLI.
- **`K_RUN`** — independent optimizer runs per optimizer with distinct RNG
  seeds. This is the plan's "K repeated runs per optimizer".

**`K_REPEAT` is a design requirement, not an optional robustness measure.**
Recorded memory: *temperature-0 agreement is not testable on OpenRouter nano
models — OpenRouter ignores the temperature control on nano routes*. The seeded
provider path advertises `SEED` and c19 uses `openrouter_seeded_call_config`,
but seed advertisement is not a determinism guarantee at the provider. The
protocol therefore **never asserts repeat agreement** and handles
non-determinism by design: every reported score is a mean over `K_REPEAT`
repeats, and the within-task repeat variance `σ²` is estimated rather than
assumed zero.

Staged values (§3.6): pilot `K_REPEAT = 3`, full `K_REPEAT = 3` unless the pilot
shows `2σ̂²/K` still dominating `τ̂²`, in which case escalate to `K_REPEAT = 5`
(the `1836` stopping logic: repeats stop paying once `2σ²/K ≪ τ²`). Stage 0's
anchor calibration uses its own `K_CAL = 4`, which is a measurement input, not
the design `K_REPEAT`.

**Scope of `K_REPEAT` (note 28).** `K_REPEAT` applies to **every** evaluation
the study pays for, including the ones an optimizer performs inside its own
search — not only the official-selection and held-out reporting passes. Every
per-run row count in §5.1 is therefore multiplied by 3. GEPA is the sole
exception, and in *unit* rather than in policy: its pinned budget of 200
**metric calls** is already denominated in instance evaluations, so it bounds
task-model rows directly and is not multiplied again.

**`K_RUN` = 5 per optimizer** for the full stage, `K_RUN = 2` for the pilot.
Rationale: `K_RUN` exists to characterize run-to-run variability of the
optimizer (a property of the algorithm under a stochastic proposer), not to
increase the precision of the held-out estimate — held-out precision comes from
T and `K_REPEAT`. Five runs give a usable spread and a defensible arg-max under
L2. Seeds are `control.seed = 1000 + j` for `j ∈ [0, K_RUN)`, recorded per run,
and are distinct across optimizers as well (so `1000..1004` for COPRO,
`2000..2004` for MIPROv2, etc.) to avoid any accidental shared-RNG coupling.

**Seed handling, pre-registered:** the run seed is set on the optimizer control
(`CoproControl` has no seed field — COPRO's stochasticity is the proposer LM, so
its "seed" is the provider `SEED` control plus proposal ordering; `GepaControl.seed`,
`Miprov2Control.seed` / `run_seed` are explicit). Where an optimizer has no
algorithmic seed, that fact is recorded in the manifest rather than faked.

### 3.6 Staged plan: pilot then full

Mirrors the `K_CAL` staged procedure of `1836`, adapted: the pilot is a
*decision* stage, not a warmup.

**Stage 0 — anchor calibration (no optimizers).**
Evaluate naive and ceiling on internal, official, and held-out at
`K_REPEAT = 3`, on c19 with `n_per_stratum=32`. Uses the existing envs eval CLI
(`whetstone_envs.reporting.cli run`), not the optimizer runner.
Outputs: `Δ_headroom = ceiling − naive` on held-out, per-task vectors for
`analyze_power`, `τ̂²`, `σ̂²`, and the re-inverted MDE.
Cost: 2 candidates × (88 + 132 + 440) tasks × `K_CAL` 4 repeats =
**5,280 rows**, one provider call each.

**Stage 0 gate.** Proceed only if all hold:
- `Δ_headroom ≥ 0.20` on held-out (there is something to optimize);
- `naive ≤ 0.60` and `ceiling ≥ 0.30` (neither anchor is at a ceiling/floor
  that flattens the reward);
- re-inverted `MDE(440, K_REPEAT) ≤ Δ_headroom / 2`.
If the gate fails: adjust `K_REPEAT` and/or held-out size once, recompute, and
if it still fails, report the design as underpowered and stop before optimizer
spend. Do not silently proceed.

**Stage 1 — pilot.** `K_RUN = 2` per optimizer (COPRO, MIPROv2 `fewshot`, GEPA,
Codex) plus both nulls, full c19 splits. Purpose: verify every audit passes on
real evidence, verify cost accounting is populated, measure per-run call counts
against the §5 estimates, and produce a preliminary Δ per optimizer.

**Stage 1 gate.** Proceed to full only if: every fidelity audit passes on at
least one real run per optimizer; `OptimResult.cost` is non-empty with per-role
calls and tokens on every run; actual per-run calls are within 1.5× the §5
estimate; and no optimizer's pilot Δ point estimate is negative by more than the
Stage-0 MDE (a strongly negative optimizer is a defect signal worth
investigating before paying for three more runs of it).

**Stage 2 — full.** `K_RUN = 5` per optimizer (Stage 1's two runs count toward
it — same code, same splits, same seeds 1000/1001 reused), plus the second
family (C3).

### 3.7 Multiple comparisons

Six arms are tested against naive on held-out: COPRO, MIPROv2, GEPA, Codex,
null-A (perturbation), null-B (identity). MIPROv2's `zeroshot` and `ground_only`
arms are **not** among them: they are fidelity checks, they produce no held-out
claim and no verdict, and they are excluded from the family by role rather than
by convention. Each of the six is a separate scientific claim
("this optimizer works"), not a family from which a winner is picked, so the
default is **per-optimizer 95% CIs reported without correction**, with the
correction reported alongside:

- **Primary (pre-registered):** per-optimizer 95% paired bootstrap CI on Δ,
  uncorrected. Claim C2 for optimizer *o* holds iff its CI excludes zero **and**
  its fidelity audit passes.
- **Secondary, always reported:** Holm–Bonferroni over the four real optimizers
  (m = 4; the nulls are controls, not hypotheses) applied to bootstrap p-values
  (`p = 2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0))` over the resamples). Any optimizer whose
  claim survives uncorrected but not corrected is explicitly labeled
  **"significant uncorrected only."**
- **"Best optimizer" is not a claim.** If the report ranks optimizers, the
  ranking is descriptive and carries the pairwise CIs, not a significance test.
  A pairwise optimizer-vs-optimizer test would need its own correction and its
  own power analysis; it is out of scope.
- The **nulls are not corrected** and are not hypotheses: they are calibration
  probes. If either null's uncorrected CI excludes zero, that is a finding about
  the interval, and §3.4's downgrade fires.

### 3.8 The null optimizers (D16 — both options, per the recommendation)

The decisions appendix records D16's recommendation as **both**, cheaply:
option (ii) shows the pipeline adds nothing by itself, option (i) shows selection
does not reward noise. Both are implemented.

**null-B (identity, option (ii)) — `--optimizer null-identity`.** The **seed
candidate evaluated through the study's own selection and report harness, with
no optimizer step at all** (note 13). A byte-identical proposer cannot be run:
`diff_check` rejects a no-op proposal, and `seed_retained` exists only on
contracts carrying `terminal_proposal_count` (GEPA/MIPROv2), so under COPRO's
shape a byte-identical proposal fails as `copro_proposal_cardinality` — the
control would be indistinguishable from a defect. Routing the seed through
`report_arm` instead gives the same evidence without a run that cannot succeed.
Expected: `Δ = 0` exactly, up to repeat noise on the held-out evaluation. This is
the pure pipeline-overhead control: any nonzero Δ here is either evaluation
nondeterminism (quantifiable — it is `2σ²/K` over T) or a bug.
`optim/nulls.py` keeps the failed-draft transport only as a contract pin.

**null-A (perturbation, option (i)) — `--optimizer null-random`.** A proposer
that applies seeded token-level perturbations to the seed template: with the run
RNG, swap/delete/duplicate whitespace-delimited tokens at a fixed rate (5% of
tokens), preserving the render contract's required fields
(`{grid}`, `{command}`, `{question}` for c19 — the perturber must never touch a
format placeholder or `TemplateRenderContract.validate_template` rejects the
candidate). It produces `breadth` perturbed candidates per step and the harness
selects best-on-internal exactly as COPRO does. Expected: `Δ ≈ 0` or negative.

**Why null-A matters more than null-B:** null-A shares COPRO's *selection*
machinery. If null-A shows a positive held-out Δ with a CI excluding zero, then
best-on-internal selection over random noise is enough to "improve" held-out
accuracy at this split size — which would mean the study cannot distinguish
optimization from selection-on-noise, and every efficacy claim is void. This is
the single most important negative control in the design.

**null-A gets the full `K_RUN = 5`.** Cutting it to save budget destroys its
function: it is the selection-on-noise control, and its whole value is the
run-to-run spread of best-on-internal selection. **null-B runs once** (`K_RUN =
1`), because with no optimizer step there is no run-to-run variability to
characterize — repeating it would re-measure the same deterministic seed
candidate.

### 3.9 Stopping, abort, and budget rules

- **Hard budget:** the OpenRouter key balance. Recorded memory: the key is the
  hard budget and is fully spendable. Before Stage 2 the remaining balance is
  recorded in the manifest; if the Stage-2 estimate exceeds it, arms are dropped
  in this pre-registered order: (1) c18 MIPROv2 `ground_only` and `zeroshot`
  variants, (2) c18 down to one run per optimizer, (3) `K_RUN` 5 → 3 for the
  real optimizers (never for the nulls), (4) held-out 440 → 220. Dropping in a
  different order, or dropping the nulls, requires a new pre-registration.
- **Per-run wall/spend cap:** each run carries a dr-exec batch wall deadline and
  the eval-budget cap already in the control. A run that exceeds it terminalizes
  with `terminal_failure` and is recorded as a failed run, not retried silently.
- **Retry policy:** a run that fails on infrastructure (provider 5xx storm,
  rate-limit exhaustion, executor crash) is re-run **at the same seed on a new
  `run_id`**, and both runs are recorded; the failed run's artifacts are kept.
  A run that fails on an algorithm error is a finding, not a retry.
- **Row-level transient failures** are handled *below* the run-level rule
  above, so an isolated provider hiccup never escalates into a re-run. On a
  paid transport each task call is attempted up to **5 times in total**, with
  exponential backoff of **2 s → 32 s** plus jitter, honouring the provider's
  `Retry-After` where it gives one as a **delta-seconds** hint, bounded at
  120 s; the HTTP-date form of that header is deliberately ignored, because
  resolving it requires agreement between the provider's clock and ours and a
  disagreement is worst precisely when the header matters. 429s, transient
  5xx, and timeouts are retried, and permanent rejections are not. The budget
  has exactly one owner: the retrying transport wrapper, which is the layer
  that actually waits between attempts. whetstone's own call driver is pinned
  to a single attempt, because two nested retry loops over the same failure
  multiply rather than compose — 5 × 5 = 25 billed invocations for one row,
  while the persisted attempt record would show only 5. The per-call timeout is **300 s**,
  set from the measured completion-token distribution (median ≈4.5k, max
  ≈12.3k) rather than the 30 s default, which a reasoning-token call routinely
  outruns.
- **Rows that fail anyway reduce completeness; they do not void the
  evaluation.** Aggregation runs with `missing_data="skip"` and
  `max_skip_fraction = 0.10`: the present rows are averaged and the shortfall
  is reported through the per-task achieved counts the analysis already weights
  by (§8, O7). Beyond that fraction the aggregate returns to `None` and the
  evaluation refuses to report a number, so the tolerance is a floor against
  losing a whole evaluation to one bad row — not permission to average a
  biased subset. The 10% bound is the complement of the 90% completeness
  backstop this section already pre-registers, so there is one threshold
  rather than two.
- **A task that loses *every* repeat voids the evaluation**, whatever fraction
  of rows it represents. The row tolerance above counts rows, and a fully-lost
  task is dropped from the task mean's *denominator* rather than counted as a
  zero — so at 76 tasks × 4 repeats one lost task is 4 of 304 rows (1.3%,
  comfortably inside the 10% bound) and the evaluation would report a mean over
  75 tasks as though it covered 76. The resulting bias is upward and
  systematic, not noise: a task that loses every repeat is a slow,
  long-generation one, which is the task that would have scored low. The
  evaluation is therefore refused if any task has zero present rows, and
  refused if fewer than **90%** of planned tasks were measured — the same
  backstop this section already states, applied to tasks as well as rows.
  Presence is read from the per-task vectors the evidence reports (a task
  with no successful reduction reports no score and a count of zero), never
  inferred from arithmetic that would assume a missing row scores 0.0. The
  refusal is applied before an evaluation is accepted, so a fully-lost task
  cannot reach calibration — which requires complete per-task counts for its
  anchors and rejects an absent per-task value outright.
- **The floor applies to every evaluation a claim is read from**, and only
  those: the official selection score, the held-out measurement, and the
  standalone `whetstone-eval` report all apply it through one shared owner, so
  the command that publishes a held-out number is subject to the same refusal
  as the stage that records one. Evaluations *inside* a search are deliberately
  exempt. Under whetstone 0.1.13 a fully-lost task reports an absent per-task
  value rather than a zero, and what a candidate is worth mid-search is the
  optimizer's reward policy to decide; aborting a run over a transient loss the
  search itself is entitled to tolerate would make the floor a stopping rule
  rather than a reporting one.
- **"Measured" means measured to depth.** The 90% bound counts tasks that
  produced a full `K_REPEAT` of present rows, not merely tasks that produced
  something. Counting only fully-lost tasks would make the bound unreachable —
  the zero-present rule above already refuses those, so the remaining fraction
  would be 100% by construction and the 90% threshold could never bind. Counting
  short tasks gives it the population it is written for: a split whose tasks
  broadly ran three of four repeats is measured more shallowly than this
  protocol pre-registers, even though every task contributed a value.
- **Never resume a partial run.** Recorded memory: clean reruns over stale
  partials — never resume or repair pre-stabilization partial experiment runs.
- **No mid-run design changes.** Per `1756`'s execution rules, the only
  pre-registered exception is the §3.6 staged escalation.

---

## 4. Claim C3 — toolchain generality (c18)

The claim is mechanical, so the evidence is mechanical.

1. **One budgeted run per optimizer on c18** (`--family c18`), splits
   `(24, 48, 48)` from `DEFAULT_CONFIG` at `n_per_stratum=30`, `K_REPEAT = 3`,
   `K_RUN = 1`. Report each run's held-out accuracy vs naive and ceiling with a
   CI — descriptive, not a claim; T=48 has an MDE around 0.20.
2. **The adapter-swap assertion.** A test asserts that the c18 path reaches the
   optimizer through the same `run_optimizer` entry point, differing only in the
   env-adapter module. Concretely: the c18 additions are
   `whetstone_envs/optim/c18_experiment.py` (a `build_c18_experiment` mirroring
   `build_c19_experiment`), a `c18_render_contract()` over `{question}`,
   `{query}`, and the family branch in the CLI. The test asserts that no module
   under `whetstone_envs/optim/` other than the c18 adapter and the CLI family
   dispatch differs between the two paths, and that no `whetstone.*` private
   import was needed to add the family. **If adding c18 requires a whetstone-ai
   change, that change is the finding** — it is a domain leak, and C3 is
   reported as failed-and-fixed with the fix named, not silently absorbed.
3. **The audits run unchanged on c18 artifacts.** The fidelity audits are pure
   functions over optimizer evidence and contain no c19 vocabulary. If any audit
   needs a c18 special case, that is a design defect in the audit, recorded.

Refactoring note for implementers: `run.py` today is c19-specific
(`run_c19_optimizer`, `C19RunSpec`, `_ceiling_gold_by_prompt`, the hardcoded
`generate_pool`). Step 12/Step 10 should generalize it to
`run_optimizer(spec)` with a family registry, so C3's claim is structurally
true rather than argued. That generalization is the cleanest way to make the
adapter-swap assertion checkable.

---

## 5. Run matrix and budget

### 5.1 Per-run call derivation

Notation: `T_int = 88` (internal), `T_off = 132`, `T_held = 440`,
`K_REPEAT = 3`. One eval "row" = one provider call for the task model. Proposer
calls are counted separately (different model, different price).

**COPRO** — `breadth × depth` structure (`CoproControl.breadth=10, depth=3`
defaults; Step 10 pins **breadth=6, depth=3**).
- Proposer calls: 1 per candidate proposal = `breadth × depth` = 18, plus 1
  initial-instruction call ≈ **19 proposer calls**.
- Task-model calls: each proposed candidate is evaluated once on internal:
  `breadth × depth × T_int × K_REPEAT` = `6 × 3 × 88 × 3` = **4,752**.
- Total per run ≈ 4,752 task + 19 proposer.

**MIPROv2** — the search shape is **pinned, not auto-derived**: `num_trials =
10`, `num_candidates = 3`, `minibatch = True`, `minibatch_size = 35`, over the
44-task validation half of the internal split (§3.1). Auto-mode is not used, so
`_recommended_num_trials` never runs; revision 1's per-mode 10/9 derivation is
retired (revision block, item 8) and **all three demo modes register 10 trials**.

`num_candidates = 3` is the minibatch floor rather than a tuning choice: two
candidates with minibatching exhausts the search space and raises inside the
durable run boundary (note 25d), so the design pins the smallest shape that
runs.

Task-model calls per run, at `K_REPEAT = 3` (identical for all three modes,
since trials, candidates, and batch size no longer vary by mode):
- minibatch trials: `num_trials × minibatch_size × K_REPEAT` = `10 × 35 × 3` = **1,050**.
- periodic full evals: `⌈num_trials / minibatch_full_eval_steps⌉ + 1` = `⌈10/5⌉ + 1 = 3`,
  each `T_int × K_REPEAT = 264` → **792**.
- bootstrap: mode-dependent and bracketed rather than point-estimated —
  **28 rows** best case to **616** worst case (the F10 bound).
- Total per run ≈ **1,870–2,458 task calls** + ≈ **10 proposer calls**.

This bracket is what `whetstone-study plan` prints for `miprov2`,
`miprov2-zeroshot`, and `miprov2-ground_only` alike.

**GEPA** — `max_metric_calls` is **pinned at 200** (D3, note 14), not left to
`auto="light"`. Auto-mode resolves to 732 metric calls, which ran for 22 minutes
and produced a 1.73 GB `runtime.sqlite`; 200 is the budget the power design and
the Stage-1 gate are both built on, and `protocols.py` is its single owner.

Metric calls are per-instance evaluations, so the pin bounds task-model rows
**directly**: **200 rows per run**, not multiplied by `K_REPEAT` again (§3.5).
For provenance, the measured 732-call run scaled to the pin is 73 rows, so 200
is a ceiling the run is not expected to reach.

GEPA's trainset/valset is the required 44/44 partition of the internal split
(§3.1), and it requires `train + val == internal` exactly. Reflection (proposer)
calls consume `reflection_minibatch_size = 3` traces per round — **pinned as
design**, since the implementation's own default of 1 is a single-trace
reflection, a different proposer input than the one registered ≈ **10–16
proposer calls**.

**Codex direct** — the admission cap is the budget. Pin the cap so Codex's eval
spend matches the others' order of magnitude (D12: "cap defaults from the same
eval budget other optimizers get, so comparisons are fair"). Set
**admission capacity = 8 evaluate-calls per run** (assignment D2; revision 1
specified 20), each an internal-split eval:
`8 × T_int × K_REPEAT` = `8 × 88 × 3` = **up to 2,112 task calls**. The cap is a
ceiling, not a floor — the agent chooses how much of it to spend, so the
estimate brackets **0–2,112**. The Codex arm's agent model is pinned to
**`gpt-5.6-sol`** (note 26); it is not "whatever the CLI session provides". The
Codex
agent's own model calls run on the Codex CLI subscription, not the OpenRouter
key, and are recorded as a separate cost role (`codex_agent`) with calls but no
USD — §5.4's rule then leaves `usd` absent for Codex runs.

**null-B (identity)**: **no optimizer run at all** (note 13). A byte-identical
proposer is unreachable — `diff_check` rejects a no-op proposal, and
`seed_retained` exists only on contracts carrying `terminal_proposal_count`, so
under COPRO's shape it fails as `copro_proposal_cardinality`. Null-B is instead
the **seed candidate evaluated through the study's own selection and report
harness**, with no optimizer step. Its cost is therefore the report harness's
passes: `1 official pass × 132 + 1 held-out pass × 440`, at `K_REPEAT = 3` =
**1,716 rows**, **0 proposer calls**. `optim/nulls.py` retains the failed-draft
transport only as a contract pin.

**null-A (perturbation)**: COPRO's structure with a local (non-LM) proposer,
**routed through the ordinary runner** (note 25b) rather than a bespoke path —
the shipped control evaluated nothing, which made it no control at all. It takes
COPRO's pinned shape exactly, because it is COPRO's search with an
uninformative proposer: `6 × 3 × 88 × 3` = **4,752 task calls**, **0 proposer
calls**. A control at a different breadth or depth would control for a search
the study never ran.

### 5.2 Reporting evaluations (per candidate, outside the optimizer run)

Selection on official: `T_off × K_REPEAT = 396` calls per run's terminal
candidate. Held-out on the representative candidate:
`T_held × K_REPEAT = 1,320` calls (at the 440-task held-out split).

### 5.3 Full matrix

**Stage 0 (anchors), c19:** 2 candidates × (88+132+440) × `K_CAL` 4 = **5,280** calls.

**Stage 2 (c19 optimizers), `K_RUN = 5`.** This table is the one
`whetstone-study plan` prints from the committed protocol module; the ranges are
the estimator's own low–high bracket. MIPROv2's spread is the F10 bootstrap
bound; Codex's low end is zero because its cap is a ceiling the agent may
underspend.

| arm | optimizer | K_RUN | optimizer rows/run | arm total | official | held-out |
|---|---|---:|---:|---:|---:|---:|
| copro | copro | 5 | 4,752 | 23,760 | 1,980 | 1,320 |
| miprov2 | miprov2 (fewshot) | 5 | 1,870–2,458 | 9,350–12,290 | 1,980 | 1,320 |
| miprov2-zeroshot | miprov2 (zeroshot) | 1 | 1,870–2,458 | 1,870–2,458 | 396 | 1,320 |
| miprov2-ground_only | miprov2 (ground_only) | 1 | 1,870–2,458 | 1,870–2,458 | 396 | 1,320 |
| gepa | gepa | 5 | 200 | 1,000 | 1,980 | 1,320 |
| codex | codex | 5 | 0–2,112 | 0–10,560 | 1,980 | 1,320 |
| null-random | null-random | 5 | 4,752 | 23,760 | 1,980 | 1,320 |
| null-identity | null-identity | 1 | 1,716 | 1,716 | 396 | 1,320 |

Total official rows **11,088**; total held-out rows **10,560**; total
selection + report rows **21,648**. Total optimizer-side calls
**63,326–78,002**.

`miprov2-zeroshot` and `miprov2-ground_only` run once each: they are evidence
for the `MIPRO_ZEROSHOT_GROUNDING` and `MIPRO_GROUND_ONLY_DEVIATION` audit
invariants, **not efficacy arms**. They carry no held-out claim and do not enter
the Holm family (§3.7); `fewshot` is the pre-registered mode carrying the
MIPROv2 claim.

**Budget by stage** (rows are task-model evaluation rows, one provider call
each; USD at the measured rate in §5.4):

| stage | rows | $ at $0.00168/call | what it buys |
|---|---:|---:|---|
| stage0 | 5,280 | $9 | 2 anchors × 660 tasks × `K_CAL` 4, all three roles |
| stage1 | 44,312–50,888 | $74–$85 | pilot: 2 runs per real arm, 1 per single-run arm, + selection + held-out |
| stage2 | 84,974–99,650 | $143–$167 | full design: 5 runs per real arm (Stage 1's two reused) |
| **total** | **90,254–104,930** | **$152–$176** | stage0 + stage2 |

Stage 2's figure prices the full five-run design; Stage 1's two runs count
toward it rather than adding to it, so the incremental cost of Stage 2 after a
completed Stage 1 is the difference, not the sum.

**Stage 2 (c18, C3), `K_RUN = 1`, splits (24, 48, 48):** internal is 24, so
MIPROv2's `resolved_minibatch = 24 > 50` is **False** — full-valset evals, and
`minibatch_size=35 > 24` would raise, so pin `minibatch=False` explicitly for
c18. Rough per-run task calls: COPRO `6×3×24×3 = 1,296`; MIPROv2
`~10 trials × 24 × 3 + bootstrap ≈ 792`; GEPA
GEPA at the pinned 200-metric-call ceiling → **200**; Codex `8×24×3 = 576`;
null-A `1,296` and null-B the report harness only.
Plus 4 candidates × (48 official + 48 held-out) × 3 = 1,152 for anchors, and
`48×3 = 144` official + `48×3 = 144` held-out per arm.
**c18 total ≈ 8,000 task-model calls.**

**Grand total ≈ 90,254–104,930 c19 task-model rows (the stage table above)
plus ≈ 8,000 for c18, and ~230 proposer calls.** Revision 1's ≈127,000 figure
predated the 440 held-out split, the GEPA and Codex pins, and null-B's move off
the runner.

### 5.4 Tokens and USD

**The cost model is measured, not assumed (note 24).** Revision 1 budgeted 30
completion tokens per call and concluded the study was a "small tens of dollars"
run. A toy Stage 0 on OpenRouter measured otherwise: gpt-5-nano billed
**≈4,165–4,500 completion (reasoning) tokens per call** — roughly 150× the
assumed figure — because the route emits reasoning tokens the protocol did not
anticipate. The measured rate is **$0.188 for 112 calls = $0.00168 per call**,
already inclusive of those reasoning tokens.

Every dollar figure in §5.3 is that single measured rate times the row count. It
is an order-of-magnitude figure measured at toy prompt sizes, not a quote. The
consequence is registered rather than hidden: the study is a **$152–$176** run,
not a small-tens-of-dollars one.

**The task model's `reasoning.effort` is pinned to `minimal` (item 19).** A
reasoning effort changes the task model's capability, so it changes the thing
being measured — which is exactly why it is fixed **before** Stage 0 measures
the anchors and hashed into the pre-registered design rather than chosen once
the anchor scores are visible. It applies to the task route only; the proposer
route is deliberately unpinned, because it writes candidates rather than
answering tasks and the study makes no claim about it.

The measured rate above was taken at the route's *default* effort. The pin is
expected to reduce it, but the protocol registers **no cost prediction from the
pin**: the figures in §5.3 stand as the pre-registered upper bound, and what
Stage 0 actually bills is what the manifest records. If Stage 0's gate fails
under `minimal`, that is a finding for humans to act on, not a condition the
protocol resolves automatically — there is no fallback to a higher effort.

**How the pin is verified: request correctness, not token statistics.** A
provider is free to spend whatever reasoning it likes at any effort, and
OpenRouter is known to silently ignore controls on nano routes (it does exactly
that with `temperature`), so billed reasoning tokens are not evidence that a
control was honoured, and a token comparison could not distinguish "the pin was
not sent" from "the pin was sent and ignored". What is checkable is the request
this study sends. Two artifacts carry it:

- **The bind is refused, not merely recorded.** Every paid task route this
  study binds is reported into the manifest at
  `models.provider_calls[].reasoning`, and the recording path checks it against
  `models.task_reasoning_effort` first: a paid bind whose effort disagrees with
  the pre-registered one raises before the write, so the stage fails *before it
  bills* rather than leaving a discrepancy for a reader to notice afterwards.
  Recording alone would not be enough — it makes a mismatch visible, but the
  looking happens after the spend.
- **Both paid paths report themselves**, which is what makes that refusal
  cover the study rather than a corner of it. The reporting pass (the two
  scored evaluations and the held-out verdict) binds through the stage
  environment; the *in-search* evaluations — the `K_REPEAT`-multiplied majority
  of the study's calls, driven by COPRO, MIPROv2, GEPA, null-A, and the Codex
  arm — bind through a `RunSpec` the study's optimizer runner builds per arm.
  Both report into the same witness, so both are gated. The fake transport is
  exempt: it binds whetstone's reference default and never reaches a provider,
  so its recorded effort is not a claim about the study's treatment.
- The request body itself is pinned by a unit test over the same translation
  the live transport uses: the OpenRouter chat preset declares
  `ReasoningRequestShape.REASONING_OBJECT`, so a pinned effort is emitted as
  `{"reasoning": {"effort": "minimal"}}` and an unpinned route sends no
  `reasoning` key at all.

The outgoing HTTP body is not itself persisted by a study run, so
`models.provider_calls[].reasoning` — and the refusal that guards it — is what
a runner relies on to know a live stage bound the design's effort.

```
~105,000 calls × $0.00168 ≈ $176 (task model, gpt-5-nano, measured)
230 proposer calls × ~2,500 tokens ≈ 0.6M tokens (gpt-5.4-nano)
```

USD is not estimated here — it is *measured*. `OptimResult.cost` records, per
role (`task_model`, `proposer`, `codex_agent`), the call count and prompt /
completion / total tokens; `usd` appears **only when every counted call in that
role carried a price from the provider response**. A run where any call lacked
pricing reports `usd: null` with `priced_calls` / `unpriced_calls` counts, and
the study total reports a lower bound plus the unpriced share. This is the
pre-registered rule; it is the reason follow-up note 2 is a hard prerequisite.

Recorded balance check: the OpenRouter key balance is read and recorded in the
manifest before Stage 0, before Stage 1, and before Stage 2. The §3.9 drop order
exists precisely so this estimate being wrong by a further multiple does not
require a new decision mid-run.

### 5.5 Models (D14)

- **Task model:** `openai/gpt-5-nano` via OpenRouter, seeded call config
  (`openrouter_seeded_call_config`). Note this changes from the 0.1.2 reruns'
  `openai/gpt-4.1-nano`, which scored 0.0 on both anchors at a 2-task split —
  Stage 0 exists to confirm gpt-5-nano is not also at the floor.
- **Task-model reasoning effort:** **`minimal`, pinned** (item 19). Carried as
  `TASK_REASONING_EFFORT` in `protocols.py`, recorded as
  `models.task_reasoning_effort` in the manifest, and hashed into
  `pre_registration_design_hash` — a design that changed the effort could not
  keep its design hash. It is **not** a sized field: the toy and the real study
  run at the same effort or they are not the same protocol.
- **Proposer / reflection model:** `openai/gpt-5.4-nano`, at the route's
  default reasoning effort. The task-model pin does not reach it.
- **Temperature:** left unset. `CoproControl` refuses a proposer temperature
  outright; OpenRouter ignores it on nano routes anyway. The manifest records
  "temperature: unset (provider-default)" rather than claiming 0.
- **Codex agent model:** **`gpt-5.6-sol`, pinned** (note 26). The runner's own
  `CODEX_DEFAULT_AGENT_MODEL` is a *run* default; a study that let it stand in
  would be running whichever proposer the runner defaulted to on the day. The
  agent runs on Danielle's Codex subscription rather than the OpenRouter key,
  so its calls are recorded under the `codex_agent` cost role with no USD.
  Defaulting the agent to the task model's OpenRouter route was a real defect:
  the subscription refuses it.

### 5.6 Invocation shapes

Environment: the envs CLI needs the pinned uv and Python, and the OpenRouter key
comes from `mise`.

Anchor calibration (Stage 0):
```
cd ~/drotherm/repos/whetstone-envs
mise exec -- uvx --from 'uv==0.11.25' uv run --python 3.13 --extra optim \
  python -m whetstone_envs.reporting.cli run \
  --family c19 --candidate naive --candidate ceiling \
  --transport openrouter --model openai/gpt-5-nano \
  --role official --repeats 3 --split-sizes 88,132,440 \
  --run-id step10-c19-anchor-official
```
(one invocation per role; the `run` subcommand's `--role` accepts
`internal|official`, so the held-out anchor needs the held-out role added in
Step 12a or a dedicated `--role held_out` — **see open decision O3**.)

Optimizer run (Stage 1/2):
```
mise exec -- uvx --from 'uv==0.11.25' uv run --python 3.13 --extra optim \
  python -m whetstone_envs.optim.cli \
  --family c19 --optimizer miprov2 --demo-mode fewshot \
  --transport openrouter --model openai/gpt-5-nano \
  --proposer-model openai/gpt-5.4-nano \
  --split-sizes 88,132,440 --repeats 3 --seed 2000 \
  --run-id step10-c19-miprov2-fewshot-s2000
```

Audit (offline, no key needed):
```
uvx --from 'uv==0.11.25' uv run --python 3.13 --extra optim \
  python -m whetstone_envs.optim.audit \
  ~/drotherm/data/runs/whetstone-envs/step10-c19-miprov2-fewshot-s2000
```

Report build:
```
uvx --from 'uv==0.11.25' uv run --python 3.13 --extra optim \
  python -m whetstone_envs.reporting.cli trajectory-html <run_dir>
```

New CLI flags this protocol requires (all in Step 12 or Step 10's own envs PR):
`--family c18`, `--proposer-model`, `--repeats`, `--seed`, `--demo-mode`,
`--optimizer {…,codex,null-random,null-identity}`.

---

## 6. Artifacts and report skeleton

### 6.1 Run artifacts (existing conventions)

Per run, under `~/drotherm/data/runs/whetstone-envs/<run_id>/`:
`runtime.sqlite`, `result.json`, plus new `audit.json` and `cost.json`
(the latter a projection of `OptimResult.cost` for convenience; the authority is
`result.json`).

**The single real Codex-direct run (note 20).** Before Stage 1, **one** real
Codex-direct run is authorized, to establish that nothing reaches the real
experiments having been tested only against mocks. It is billed to Danielle's
Codex **subscription** (the ChatGPT login in `~/.codex/auth.json`), so the
runner must not require `OPENAI_API_KEY` for the agent process; the task model
still goes through `--transport openrouter`. It is opt-in twice over:
`WHETSTONE_ENVS_ALLOW_REAL_CODEX=1` in the environment **and**
`--allow-real-codex` on the command line, the latter being run-time spend
authorization that is deliberately **not** part of the pre-registration hash.
A study whose design includes a Codex arm fails fast, before any arm runs, if
that authorization is absent.

**Its artifacts are historical evidence only.** They are not a Stage-1 run, they
carry no held-out claim, and they enter no statistic. They exist to prove the
real transport works.

### 6.2 Study manifest — `study.json`

The single artifact that makes every number in the report checkable. Pinned
schema `whetstone_envs.step10_study/v1`, golden-tested literals.

```
study_id, created_at, protocol_doc_path (this file), protocol_doc_sha256
population: {family, generator_version, n_per_stratum, seed_start,
             pool_manifest_content_hash, stratum_counts}
splits: {internal: {size, task_hashes[], eval_config_hash},
         official: {...}, held_out: {size, task_hashes[]}}
models: {task_model, proposer_model, temperature: "unset",
         provider: "openrouter", seed_control: "advertised"}
design: {k_repeat, k_run, ci_level, resamples, bootstrap_seed,
         correction: "holm-bonferroni", m: 4}
arms: [{arm_id, optimizer, kind, demo_mode?, train_size, val_size,
        minibatch, minibatch_size, copro_breadth, copro_depth,
        miprov2_num_trials, miprov2_num_candidates, control_identity_hash,
        runs: [{run_id, seed, artifact_dir, result_ref, audit_ref,
                audit_passed, cost}]}]
   # `kind` is the arm's role: `real` (an efficacy hypothesis, in the Holm
   # family), `fidelity` (an audit check with no held-out claim and no
   # verdict), or `null` (a control). It is part of the pre-registration.

selection: [{arm_id, selected_run_id, official_score, rule: "argmax-official"}]
held_out: [{candidate_name, eval_evidence_ref, per_task_scores_ref,
            mean, ci_low, ci_high, delta_vs_naive, p_bootstrap, p_holm}]
balance: {before_stage0_usd, before_stage1_usd, before_stage2_usd, after_usd}
leakage_check: {passed, checks: [...]}
```

Every number printed in the report is either a field of this manifest or a
deterministic function of the referenced evidence — the manifest names the
`(schema_name, content_hash)` pair for each. Report generation reads only the
manifest and the store; it never recomputes a score from a loose file.

### 6.3 Report — Markdown + polished HTML

Location per report conventions:
`~/drotherm/data/.claude/whetstone-ai/<YYYY-MM-DD>/<HHMM>-step10-validation-report/`
as a multi-file packet (`report.md`, `report.html`, `study.json`, `doc.css`,
figures). Long human-facing HTML follows the `html-doc-polish` skill at
`~/drotherm/repos/dotfiles/agents/skills/html-doc-polish/`, with the Markdown
source retained beside it.

Skeleton:

1. **Verdict table** — one row per optimizer: fidelity pass/fail, held-out Δ with
   95% CI, uncorrected and Holm-corrected significance, spend. Three-state
   verdict: *validated* / *not validated (fidelity)* / *no detected improvement*.
2. **Design summary** — splits, sizes, K, models, MDE, and the pre-registered
   leakage rules with the mechanical check's result.
3. **Per optimizer** (×4, plus a shorter section per null):
   - best-so-far internal-split trajectory by step (from `TrajectoryReport`,
     already produced by `reporting/projection.py`);
   - held-out score vs naive / ceiling / null with the CI, on one axis;
   - cost: calls and tokens per role, wall time, USD or the explicit
     lower-bound-plus-unpriced statement;
   - the fidelity-audit table: every `invariant_id`, status, and evidence refs;
   - a sample of proposed prompts (first, best, last) with diffs against the seed.
4. **Second family (c18)** — one table: per optimizer, held-out accuracy vs
   naive/ceiling, audit status, spend; plus the adapter-swap assertion result
   and the list of modules that differed.
5. **Threats to validity** — §7's risks, each with what the study actually did
   about it and what remains unaddressed.
6. **Validation checklist for Danielle** — explicitly what needs a human eye:
   - do the sampled optimized prompts read as genuine improvements or as
     overfitting to the internal split's phrasing?
   - is the Codex arm's uncontrolled agent model a fair comparison?
   - does the null-A result look like selection-on-noise?
   - are the anchors (naive / ceiling) behaving as intended on gpt-5-nano, or is
     either at a floor/ceiling?
   - does the spend match the intent, and is the remaining key balance as
     expected?

Per review-patterns.md this study's claims are experimental and require manual
judgment, so this checklist is mandatory, not optional.

---

## 7. Risks — for the adversarial reviewer to attack

Listed with the design's current answer, so the reviewer can attack the answer
rather than rediscover the risk.

**R1 — Split leakage.** *Answer:* §3.2 L1–L6, with L6 a mechanical set-intersection
check over persisted task hashes. *Attack surface:* `derive_eval_split` derives
internal and official from the same `PoolSplit`; if `split_pool`'s strata
balancing ever assigned an instance to two destinations the `PoolSplit`
`__post_init__` assertion catches it — but the assertion is on instance IDs,
while the eval configs key on **task hashes**. If two distinct instances ever
produced the same task hash, L5 would pass while the splits overlapped
semantically. Is the hash collision-free by construction across strata?

**R2 — Selection on the reporting split.** *Answer:* L2/L3, and held-out
evaluated exactly once, after selection is frozen. *Attack surface:* the report
is written by the same people who will read the held-out numbers. If Stage-2
results prompt "let's try `demo_mode=ground_only` as the MIPROv2 arm instead,"
that is selection on held-out through the back door. The protocol's answer is
that `fewshot` is pre-registered as the MIPROv2 efficacy arm and the other two
modes are fidelity-only, at `K_RUN = 1`. A reviewer should check that this
pre-registration is actually binding.

**R3 — Seed handling.** *Answer:* explicit per-run seeds, disjoint ranges across
optimizers, recorded in the manifest; optimizers without an algorithmic seed
(COPRO) record that fact. *Attack surface:* COPRO's only stochasticity is the
proposer LM, whose seed is a provider control that OpenRouter may ignore. So
COPRO's `K_RUN = 5` may be measuring provider nondeterminism rather than
algorithmic variability — and if the provider is *more* deterministic than
assumed, COPRO's five runs may be near-identical, making its arg-max-on-official
selection vacuous and its run-to-run spread misleadingly tight.

**R4 — Temperature and provider nondeterminism.** *Answer:* never assert repeat
agreement; every score is a mean over `K_REPEAT = 3`; `σ²` is measured.
*Attack surface:* if the provider silently changes routing or model revision
mid-study, `σ²` measured at Stage 0 no longer describes Stage 2, and the CI is
wrong in an unquantified direction. The manifest records the model string but
OpenRouter can route the same string to different backends. Mitigation candidate
the reviewer should weigh: re-run the naive anchor at the end of Stage 2 and
compare to Stage 0 as a drift probe (cost: 1,320 calls at T=440).

**R5 — Provider rate limits and timeouts.** *Answer:* per-row and batch
deadlines through the dr-exec pool; `RowDispatchStatus` discriminates
`NOT_DISPATCHED` from `OPERATION_DEADLINE`; failed rows are visible in
`RowAccounting` and the envs reporting schema refuses accounting that does not
exhaust the planned matrix. *Attack surface:* the envs `StratumSummary`
validator sets `score = None` whenever `present != denominator`. A study where
1% of rows time out therefore produces `None` scores for whole strata, and a
naive report generator could quietly drop them, biasing toward whichever strata
completed. The protocol needs an explicit rule: **any arm with less than 98%
row completeness on held-out is reported as incomplete and its CI is not
claimed.** (This is added here; the reviewer should check whether 98% is the
right threshold and whether partial-completeness should instead be handled by
per-task weighting as `1756`'s ragged-cell rule does.)

**R6 — Minibatch cost accounting.** *Answer:* §5.1 derives MIPROv2's and GEPA's
call counts from `minibatch_size` and `num_trials`, and D9 notes the platform
fan-out's row count is `intents × tasks × seeds` and must take per-intent task
sets. *Attack surface:* if the deferral row-count formula was not generalized to
per-intent task sets, every minibatch intent fans out over the *full* split, and
actual spend is `88/35 = 2.5×` the estimate for MIPROv2 and GEPA. Stage 1's
"actual within 1.5× estimate" gate is the detector, but the reviewer should
confirm the generalization actually landed rather than trusting the gate.

**R7 — Underpowered by design.** *Answer:* stated openly in §3.4 (MDE ≈ 9–10
accuracy points at T=440, K=3) with the Stage-0 gate as the abort. *Attack
surface:* prompt-optimization gains on a deterministic simulation family may
plausibly be 3–5 points, in which case the study is guaranteed to report "no
detected improvement" for every optimizer and the result is uninformative about
the optimizers. Is a bigger held-out split (the pool tail has 264 spare
instances, and `n_per_stratum` can go to 128 → 2,816 instances) the better trade
against spend? This is the design's most consequential open trade.

**R8 — Nulls as calibration, and the identity null's degenerate structure.**
*Answer:* both nulls run at full `K_RUN`. *Attack surface:* null-B terminalizes
at step 0–1 with `seed_retained`, so it does not exercise the selection
machinery at all; null-A does, but with a *non-LM* proposer, so it does not
exercise the proposer transport. Neither null is a complete pipeline control.
A reviewer might argue for a third null: an LM proposer prompted to produce
semantically-null rewrites, which would exercise every stage.

**R9 — Codex is not comparable.** *Answer:* the admission cap equalizes eval
budget (D12). *Attack surface:* Codex's agent model is uncontrolled, its own
reasoning tokens are unbilled to the study key, and its 20-evaluation budget
buys 20 *full-internal-split* evaluations while MIPROv2's budget buys 10 × 35-task
minibatches. "Equal eval budget" is doing a lot of work across very different
consumption shapes. Should the cap be in *rows* (88×3 = 264 per evaluation, so a
row-equivalent cap) rather than in evaluate-calls?

**R10 — Fidelity audits validate against whetstone's own reading of DSPy/GEPA.**
*Answer:* the audits check invariants against pinned reference commits and
version strings. *Attack surface:* an audit written by the same effort that
wrote the adapter can encode the adapter's misreading as the invariant. The
negative-fixture requirement (§2.7) mitigates but does not eliminate this. The
one genuinely independent check available is MIPROv2's TPE replay determinism
(D10), which compares against Optuna's own behavior rather than whetstone's.

**R11 — `analyze_power`'s variance decomposition is bespoke.**
*Attack surface:* `_decompose_variance` (`eval/analysis/power.py`) uses
midpoint variance minus `within_obs/(2r)` for `between`, floors `interaction` at
`0.1 × within`, and sets `within = base_rate(1 − base_rate)` from the *naive*
arm only — which is not the same estimator as `1756`'s method-of-moments on the
task-by-repeat table. The `0.1 × within` floor in particular is an arbitrary
regularizer that will *inflate* the MDE when the true interaction variance is
small. §3.4 therefore reports the closed-form MDE alongside `analyze_power`'s;
a reviewer should decide whether the two should be reconciled before the study
or whether reporting both is honest enough.

**R12 — One study, many degrees of freedom.** Splits, `n_per_stratum`, K, demo
mode, breadth/depth, GEPA auto level, Codex cap — all chosen here, none forced.
*Answer:* this document pre-registers all of them and the golden-hashed
`study.json` records them. *Attack surface:* pre-registration written by the
same agent that will run the study is weak pre-registration. The strongest
available fix is that this document is reviewed adversarially and its sha256 is
recorded in `study.json` before Stage 0 spends anything — which §6.2 does.

---

## 8. Open decisions

Only where genuinely open. Each has a recommendation.

**O1 — Held-out size vs spend (R7's trade). DECIDED 2026-08-23 (note 19):
option (b), T=440, pre-registered now rather than at the Stage-0 gate.** The
splits are `(88, 132, 440)`; §3.4 carries the recomputed MDE row. Stage 0 still
measures `τ̂²`/`σ̂²` and reports the realized MDE, but it no longer *chooses* the
held-out size — a size chosen after seeing the anchors would not be
pre-registered. The original options are retained below as the reasoning behind
the choice.

Options: (a) T=220 as specified,
MDE ≈ 9–10 points, ~127k calls; (b) T=440 (20/stratum, `n_per_stratum=48`),
MDE ≈ 6–7 points, adds ~4,000 calls for anchors and ~1,300 per reported
candidate — roughly +15k calls total, since held-out is evaluated once per arm,
not per run; (c) keep T=220 but raise `K_REPEAT` to 5, which only shrinks the
`2σ²/K` term and is the *worse* lever if `τ²` dominates.
**Recommendation: (b).** Held-out is evaluated once per arm, so it is the
cheapest place in the whole matrix to buy resolution — roughly 12% more spend
for a ~30% smaller MDE.

**O2 — COPRO breadth/depth. DECIDED: option (b), breadth 6 / depth 3**, now
pinned in `protocols.py` and forwarded to both the run and the estimator.
Options: (a) library defaults `breadth=10,
depth=3` → 7,920 task calls/run; (b) `breadth=6, depth=3` → 4,752 (assumed
above); (c) `breadth=6, depth=5` → 7,920, trading candidates-per-round for
rounds. **Recommendation: (b)**, and record the deviation from DSPy's default
explicitly in the fidelity report — `COPRO_BREADTH_PER_DEPTH` checks adherence
to the *configured* breadth, so a non-default breadth is a budget choice, not an
infidelity. Reviewer should confirm that framing is acceptable.

**O3 — Held-out evaluation path.** The envs eval CLI's `run --role` accepts only
`internal|official` (`reporting/cli.py`), and `EvalConfigs` carries held-out as
bare `held_out_task_hashes` with no derived eval split
(`optim/experiment.py`). Options: (a) add a third `derive_eval_split` for
held-out and a `--role held_out` CLI choice; (b) evaluate held-out by
constructing a one-off experiment whose "official" split is the held-out tasks.
**Recommendation: (a).** (b) makes the held-out eval config hash indistinguishable
from an official one in the evidence, which directly undermines L5's mechanical
leakage check. This is a small Step 12a/Step 10 envs change and should be sized
into the implementation.

**O4 — Codex cap unit (R9). DECIDED (assignment D2): the cap stays in
evaluate-calls, pinned at 8** — option (a)'s unit at a lower number, not option
(c). The admission ledger debits calls, which is the unit it can enforce
per-admission; 8 evaluate-calls × 88 internal tasks × 3 repeats brackets the arm
at 0–2,112 rows, the same order of magnitude as the other arms. Options
considered: (a) 20 evaluate-calls; (b) a row-
equivalent cap of ~5,300 rows, letting the agent spend it on partial splits if
the tool surface allows; (c) match GEPA's `resolved_max_metric_calls` (732
metric calls × 3 repeats). The original recommendation was (c), on the grounds that it
makes "same eval budget" literally true in GEPA's own unit; it was not adopted,
because it requires the MCP eval tool to debit capacity in rows rather than
calls.

**O5 — Third null (R8).** Options: (a) two nulls as specified; (b) add an
LM-proposer semantic-null arm at `K_RUN = 3` (+~14k calls).
**Recommendation: (a) for Stage 2, revisit only if null-A shows a positive Δ.**
If null-A is clean, the pipeline-plus-selection control has done its job and a
third null adds spend without a distinct question.

**O6 — Drift probe (R4).** Options: (a) none; (b) re-run the naive anchor on
held-out at the end of Stage 2 (1,320 calls at T=440) and report the delta
against Stage 0.
**Recommendation: (b).** At the 440-task held-out split this is 1,320 calls,
still around 1% of the study, to convert an unquantified threat into a measured
number.

**O7 — Row-completeness threshold (R5).** The 98% figure in R5 is proposed here,
not derived. Options: (a) 98% hard threshold, CI not claimed below it;
(b) `1756`'s ragged-cell rule — weight each task's delta by its achieved sample
count and report the excluded tasks. **Recommendation: (b) with an (a) backstop
at 90%**, since (b) is already the pre-registered handling for the same problem
in the ED1 design and reusing it keeps one rule rather than two.

---

## 9. Implementation checklist

Ordered, each item independently verifiable.

1. envs: generalize `run.py` to `run_optimizer` + family registry (enables C3).
2. envs: `--role held_out` and a held-out `derive_eval_split` (O3).
3. envs: `optim/audit/` package, 4 audits + shared evidence reader + schema,
   with one failing negative fixture per invariant.
4. envs: `null-identity` and `null-random` proposers behind
   `--optimizer null-{identity,random}`; null-random must preserve render-contract
   placeholders.
5. envs: c18 optim adapter (`build_c18_experiment`, `c18_render_contract`,
   family dispatch) and the adapter-swap assertion test.
6. envs: `study.json` schema + writer + `study_leakage_check` (L6).
7. envs: report generator reading only `study.json` + the store; Markdown and
   polished HTML per §6.3.
8. whetstone-ai: `OptimResult.cost` per-role calls/tokens with the
   `usd`-only-when-fully-priced rule (§5.4) — prerequisite, not Step 10 work.
9. CI: every audit runs against the existing fake-transport e2e artifacts.
10. Stage 0 run → gate; Stage 1 pilot → gate; Stage 2 full → report.
