# Implementation plan: quick-test envs

Status: draft. Tracks the work needed to implement, verify, and land all
five quick-test task-family envs (c11, c18, c19, c22, c23) as stacked PRs
against `main`, so whetstone-ai can consume them to validate its
optimizer implementations (Eval, COPRO, MIPROv2, GEPA, Codex CLI agent).

Source material this plan implements (all in
[danielle-rothermel/whetstone-ai](https://github.com/danielle-rothermel/whetstone-ai)):

- [`design/quick-test-rubric.html`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/design/quick-test-rubric.html)
  — the 14 core/additional criteria every task family must satisfy.
- Per-candidate docs under
  [`research/quick-test-tasks/`](https://github.com/danielle-rothermel/whetstone-ai/tree/main/research/quick-test-tasks):

  | Candidate | Candidate page | Related work | Baseline spec |
  | --- | --- | --- | --- |
  | c11 — JSON canonicalization (RFC 8785 / JCS) | [c11.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c11.html) / [c11.md](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c11.md) | [c11-related-work.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c11-related-work.html) | [c11-baseline-spec.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c11-baseline-spec.html) |
  | c18 — PrOntoQA deductive entailment | [c18.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c18.html) / [c18.md](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c18.md) | [c18-related-work.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c18-related-work.html) | [c18-baseline-spec.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c18-baseline-spec.html) |
  | c19 — Minigrid state prediction | [c19.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c19.html) / [c19.md](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c19.md) | [c19-related-work.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c19-related-work.html) | [c19-baseline-spec.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c19-baseline-spec.html) |
  | c22 — stacked IFEval instruction-following constraints | [c22.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c22.html) / [c22.md](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c22.md) | [c22-related-work.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c22-related-work.html) | [c22-baseline-spec.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c22-baseline-spec.html) |
  | c23 — subregular rule induction (InductionBench-style) | [c23.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c23.html) / [c23.md](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/candidates/c23.md) | [c23-related-work.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c23-related-work.html) | [c23-baseline-spec.html](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c23-baseline-spec.html) |

  Each baseline spec is the primary implementation reference (strata
  design in §1, probe prompts in §2, decision rule in §3, model
  evidence in §4, cost estimate in §5, rubric mapping in §6, and open
  owner decisions in §7). The candidate page and related-work doc are
  background/justification, not additional requirements.
- Upstream repos each candidate wraps or vendors, referenced from
  [`research/quick-test-tasks/repos/`](https://github.com/danielle-rothermel/whetstone-ai/tree/main/research/quick-test-tasks/repos):
  [`json-schema-faker`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/json-schema-faker-json-schema-faker.md) +
  [`rfc8785-py`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/trailofbits-rfc8785-py.md) (c11);
  [`asaparov/prontoqa`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/asaparov-prontoqa.md) (c18);
  [`Farama-Foundation/Minigrid`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/farama-foundation-minigrid.md) (c19);
  [`google-research` IFEval](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/google-research-instruction-following-eval-ifeval.md) (c22);
  [InductionBench](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/repos/inductionbench-wenyueh-inductive-reasoning-benchma.md) (c23).

## Scope boundary (recap)

This repo owns generators, independent oracles, probe prompts, and
scoring — pure env concerns with zero dependency on whetstone's
execution contract (Optimization Requests, Tool Specs, Graph Config,
Materialization Records, Rollout Execution Keys, Evaluation Contexts).
That contract-facing adapter layer stays in whetstone-ai and depends on
this package. Nothing in the PRs below should import or assume anything
about whetstone-ai's runtime.

## Owner decisions that block a full run (not this repo's blocker)

Every baseline spec ships as a DRAFT with owner-gated open items: model
slate, decision-rule thresholds, N per stratum, a few wording/version
pins. **None of these block building and unit-testing the env code
below** — generator, oracle, and probe-prompt correctness is verifiable
independent of which models or thresholds get chosen. Model slate and
thresholds only matter once a *baseline calibration run* is executed,
which happens after whetstone-ai's adapter layer exists. This plan
treats those as parameters threaded through config, not prerequisites.

## Shared harness (build once, before the first candidate)

All five specs use the identical outer shape: generate a pinned
instance pool → render one of two probe prompts per instance → (later,
in whetstone-ai) one LLM call → score 0/1 against an independent oracle
→ aggregate by repeat → task → stratum. The parts of that shape with no
model-call dependency belong here and should be built once, shared
across all five candidate modules, not duplicated per-candidate:

- **`whetstone_envs.core.instance`** — a minimal frozen `Instance` type:
  stable task identity (id/seed), stratum label(s), rendered prompt
  inputs, and gold/oracle-checkable state. Every candidate module
  produces a `list[Instance]` of this shape.
- **`whetstone_envs.core.probes`** — a `ProbePair` type (naive prompt
  template + ceiling prompt template + a render function) and a
  standard normalization step (strip surrounding whitespace/code
  fences) shared by every candidate's exact-match scoring.
- **`whetstone_envs.core.scoring`** — `exact_match(prediction, gold) ->
  int` (0/1) plus the aggregation helpers: mean-by-repeat-within-task,
  then across tasks within stratum, then across strata (never partial
  credit inside an instance, per the rubric's aggregation-crosses-strata
  callout).
- **`whetstone_envs.core.pool`** — a `TaskPool` container: pinned
  instances plus stratum membership, with `.split(internal_eval_n,
  official_n, held_out_n)` producing disjoint subsets and asserting no
  overlap. Each candidate's generator returns a `TaskPool`; the split
  sizes are the per-spec proposed numbers (owner-adjustable, not
  hardcoded assumptions).
- **`whetstone_envs.core.manifest`** — a small manifest writer/reader
  (JSON) recording: generator version/commit, schema/seed-range pin,
  per-stratum counts, and a content hash of the instance pool. Every
  candidate's generation script writes one of these alongside the
  pool it produces, so a regenerated pool can be diffed against a
  frozen one.

This is genuinely shared code (rubric criteria 2, 5, 7, 8, and the
aggregation-crosses-strata callout apply identically to all five), so
it lands as **PR 0**, before any candidate. Candidate PRs depend on it
but not on each other.

## Stack order and dependency rationale

PRs are ordered cheapest-and-least-external-dependency first, so early
PRs retire the most schema/tooling risk before the harder
external-dependency candidates (c18's vendored generator, c19's
Minigrid runtime, c23's patched upstream code) get built. Each candidate
PR is independent of the others once PR 0 lands — they can be built and
reviewed in parallel — but are listed here in the recommended landing
order to keep the stack shallow and reviewable.

1. **PR 0 — shared harness** (`core/`): instance/pool/probe/scoring/manifest primitives, fully unit-tested against synthetic fixtures (no real task logic yet).
2. **PR 1 — c22** ([baseline spec](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c22-baseline-spec.html), stacked IFEval constraints): cheapest build — reuses a pinned, namespaced `google-research` IFEval checker snapshot for both generation and oracle, with a documented exact-word-count patch. Best candidate to prove the shared harness's shape is right before other candidates build on it.
3. **PR 2 — c11** ([baseline spec](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c11-baseline-spec.html), JSON canonicalization / RFC 8785 JCS): reuses `json-schema-faker` + `trailofbits/rfc8785-py` unmodified. Self-contained, no vendored/patched upstream code.
4. **PR 3 — c19** ([baseline spec](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c19-baseline-spec.html), Minigrid state prediction): depends on the `minigrid` package as a real runtime (not just a checker library) — env instantiation, seeded rollout execution, object-model introspection for the oracle.
5. **PR 4 — c18** ([baseline spec](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c18-baseline-spec.html), PrOntoQA): wraps `asaparov/prontoqa`'s `run_experiment.py` as a subprocess/import boundary, plus a from-scratch ~30–50 line forward-chaining fixpoint oracle (the one candidate needing a hand-written oracle, not a reused one).
6. **PR 5 — c23** ([baseline spec](https://github.com/danielle-rothermel/whetstone-ai/blob/main/research/quick-test-tasks/related-work/c23-baseline-spec.html), subregular rule induction / InductionBench-style): requires vendoring and patching upstream (config stub, drop a broken import, thread a real seed param, replace `list(set(...))` with `sorted(...)` at 6 call sites for determinism) — the most invasive integration, done last so the shared harness and manifest/patch conventions are already proven.

Each PR is reviewable and mergeable on its own; "stacked" here means
sequential landing on `main` for reviewability, not that PR *n+1*
branches off PR *n*'s unmerged branch. Rebase each candidate branch onto
latest `main` before opening its PR so diffs stay small and
independent.

## Per-candidate PR checklist

Every candidate PR (1–5 above) must include, in this order:

1. **Generator module** (`whetstone_envs.<candidate>.generate`)
   - Wraps or reimplements the spec's §1 strata design and produces a
     `TaskPool` via the shared `core.pool` container.
   - Fresh-seed only: assert at construction time that no generated
     instance's seed coincides with any seed reserved/published by the
     upstream paper or dataset the candidate is adjacent to (rubric
     criterion 8). Where the spec names a concrete contamination risk
     (e.g. c18/c19's fixed nonce-ontology or fresh-seed-range
     requirement), encode that check as an assertion, not a comment.
   - Deterministic given `(generator_version, seed)` — verify by
     regenerating twice and diffing.

2. **Oracle module** (`whetstone_envs.<candidate>.oracle`)
   - Independent of the generator's internal state: a pure function of
     the instance's public fields (input JSON / facts+query / grid+moves
     / constraint spec / demo pairs), never a re-derivation from how the
     generator built the instance (oracle-independence, rubric
     criteria 2 and 8).
   - Where the spec names a specific reference implementation (c11:
     `rfc8785.dumps` unmodified; c22: `check_following` /
     `test_instruction_following_strict` from the pinned checker snapshot;
     c19: the Minigrid
     object-model walk; c23: `apply_ISL_rule`/`apply_L_OSL_rule`/
     `apply_R_OSL_rule` reapplied to the held-out query), use it
     unmodified — do not reimplement logic a reference already
     provides, per the danielle-code-quality norm against
     re-deriving what a dependency already guarantees.
   - c18 is the one candidate with no off-the-shelf oracle: implement
     the forward-chaining fixpoint checker from scratch (~30–50 LOC per
     spec estimate) and give it its own focused test file independent
     of the generator's tests.

3. **Probe prompts** (`whetstone_envs.<candidate>.prompts`)
   - Both prompts verbatim from the spec's §2, parameterized only by
     the instance's rendered fields (never by anything the model
     shouldn't see, e.g. c23's ceiling prompt must state conventions
     without leaking the latent rule — see that spec's own
     "legitimate ceiling, not cheating" note).
   - Any prompt text in a baseline spec marked "evidence note (thin)" or
     needing regeneration against a real oracle (c11's ceiling-prompt
     worked examples are hand-typed placeholders per that spec's §7.5)
     must be regenerated programmatically here, not copied verbatim
     from the HTML.

4. **Config surface**
   - Every owner-open numeric (N per stratum, strata included/excluded,
     repeats, model-agnostic prompt wording) is a constructor/CLI
     argument with the spec's *proposed* value as the default — never
     hardcoded such that changing it requires an env-code change.

5. **Tests** (see Verification below) and **manifest** written for the
   default-config pool.

## Verification (must pass before opening each candidate's PR)

Verification is split into what's checkable with zero LLM calls (must
pass before any PR opens) and what requires a small live-model pilot
(must pass before a PR is marked ready for the "used to validate
whetstone" milestone, but can follow the initial PR as a fast-follow if
the owner wants to unblock review sooner).

### A. No-LLM-call checks (blocking for every PR)

- [ ] **Determinism**: regenerating the pool twice from the same
      `(generator_version, seed_range)` produces byte-identical
      instances (content hash in the manifest matches).
- [ ] **Contamination guard**: an explicit test asserts the generated
      seed range/schema never intersects any seed or instance published
      by the upstream paper/dataset the candidate lists as adjacent
      evidence.
- [ ] **Oracle correctness on hand-built fixtures**: for each candidate,
      a handful of hand-constructed instance/gold pairs (not
      generator-produced) with independently verified expected output,
      asserting the oracle returns the exact expected string/label —
      this is what catches an oracle that's silently a re-derivation of
      generator internals rather than a true independent check.
  - c11: the worked examples from the spec's ceiling prompt,
    regenerated through `rfc8785.dumps` and compared byte-for-byte
    (per that spec's §7.5 evidence-thinness flag).
  - c18: at least one hand-traced multi-hop chain per stratum depth,
    confirming the fixpoint oracle matches manual derivation.
  - c19: at least one hand-traced command sequence per env,
    confirming final coordinate/heading/carrying/what's-in-front
    against manual grid-walking.
  - c22: at least one instance per atom in the easy and hard pools,
    confirming the checker's pass/fail matches manual inspection of
    the constraint text.
  - c23: at least one hand-picked rule per type (ISL/L-OSL/R-OSL)
    applied to a held-out query, confirming oracle output matches
    manual rule application.
- [ ] **Strata coverage**: every stratum in the spec's §1 table has
      the proposed N (or the config-overridden N) — a test asserts
      pool composition matches the manifest's declared per-stratum
      counts, not just the total.
- [ ] **Aggregation-crosses-strata**: a synthetic test confirms the
      shared `core.scoring` aggregation reduces repeat → task → stratum
      → overall correctly on constructed scores, including the
      exhausted/failed-observation case from rubric criterion 13
      (a missing/failed result must make the aggregate visibly
      incomplete, never silently zero).
- [ ] **Prompt rendering**: both probe prompts render byte-for-byte as
      drafted in the spec for a fixed fixture instance (guards against
      template drift), and neither prompt leaks gold/oracle-only fields
      (a static check over the rendered prompt string).
- [ ] **Lint/type/test clean**: `uv run ruff check .`, `uv run ty
      check`, `uv run pytest` all pass in CI for the PR branch.

### B. Small live-pilot checks (before "ready to validate whetstone")

These require a handful of real LLM calls (a "pilot," per each spec's
§5/§7 recommendation) and should be run once per candidate, cheaply,
using whichever single cheap model is fastest to wire up — the actual
model-slate decision from each spec's §4 open item is irrelevant here,
this is a plumbing check, not a calibration run.

- [ ] **Token estimate sanity**: run ~10 instances per probe through one
      model; confirm actual token counts are within ~2x of the spec's
      §5 per-instance estimate (every spec flags these as unconfirmed
      guesses). Update the estimate in this repo's own docs if it's
      off, rather than silently trusting the HTML spec's number.
- [ ] **Temp-0 repeat agreement**: run repeats={0,1,2} on the same ~10
      instances; confirm repeats agree (near-100%) as the rubric's
      determinism criterion (5) assumes. Disagreement here means either
      the provider isn't honoring temperature 0 or the prompt is
      underspecified enough to be non-deterministic — either is a
      blocking finding, not a note.
- [ ] **Naive-vs-ceiling direction sanity**: confirm the ceiling probe
      scores at or above the naive probe on the pilot slice (doesn't
      need to hit the spec's predicted magnitude yet — just the
      correct sign). A ceiling prompt that scores *below* naive on even
      a small pilot indicates a prompt bug (e.g. over-constraining
      output format) that should be fixed before the full calibration
      run.
- [ ] **Format-extraction robustness**: confirm the scoring glue's
      extraction step (last `Output:` line, strip fences, etc.) doesn't
      silently mis-score a real model response — spot-check the raw
      response text against the extracted/scored value for every pilot
      call, not just the aggregate.

## Integration handoff to whetstone-ai (out of scope for this repo's PRs)

Once a candidate's PR lands here (checklist A green; checklist B green
or explicitly deferred with a tracking note), whetstone-ai adds a thin
adapter that:

- Declares the candidate's generator+oracle+prompts as a single LLM
  Call Node + single Eval Node Rollout Definition (rubric criterion 1).
- Wraps `TaskPool` splits as the internal/official/held-out Evaluation
  Contexts and wires the oracle's 0/1 result into Metric
  Facts/Records.
- Resolves the candidate spec's remaining owner-open items (model
  slate, decision-rule thresholds) as part of that adapter's own
  config, not by changing anything in this repo.

This repo's contract to whetstone-ai is: a `TaskPool`, a `ProbePair`,
and a scoring function, all pinned by a manifest — nothing about how
whetstone-ai executes or authorizes rollouts belongs here.

## Definition of done (this repo, all five candidates)

- [ ] PR 0 (shared harness) merged.
- [ ] PRs 1–5 (c22, c11, c19, c18, c23) each merged with checklist A
      green; checklist B green or deferred with an explicit tracking
      issue.
- [ ] Every candidate's manifest committed alongside its pool-generation
      script, so whetstone-ai can pin an exact env version.
- [ ] README's task-family list updated from placeholder to link each
      entry to its module and test suite.
