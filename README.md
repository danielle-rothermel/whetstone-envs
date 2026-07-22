# whetstone-envs

Task generators, independent oracles, and probe prompts for
[whetstone-ai](https://github.com/danielle-rothermel/whetstone-ai)'s
quick-test optimizer benchmarks.

## Scope

This repo owns the parts of each quick-test task family that are pure
environment concerns, with no dependency on whetstone's optimizer or
execution-contract code:

- **Generators** — pinned, seeded instance generation (or thin wrappers
  around an upstream generator) for each task family.
- **Oracles** — independent, deterministic scoring functions used as the
  ground truth for exact-match evaluation.
- **Probe prompts** — the naive/ceiling prompt pairs used to measure
  headroom before any optimizer runs.

It deliberately does **not** own anything tied to whetstone's execution
contract (Optimization Requests, Tool Specs, Graph Config,
Materialization Records, Rollout Execution Keys, etc.) — that adapter
layer lives in whetstone-ai, which depends on this package.

See whetstone-ai's `design/quick-test-rubric.html` and
`research/quick-test-tasks/related-work/` for the rubric and baseline
specs each task family here implements.

## Task families

One module per candidate, added as each baseline spec is implemented:

- `c11` — JSON canonicalization (RFC 8785 / JCS) —
  [`whetstone_envs.c11`](src/whetstone_envs/c11)
  ([generate](src/whetstone_envs/c11/generate.py) ·
  [oracle](src/whetstone_envs/c11/oracle.py) ·
  [prompts](src/whetstone_envs/c11/prompts.py) ·
  [tests](tests/c11)). Adversarial seeded generator (one messy input per
  JCS sub-rule stratum); the oracle delegates canonicalization to
  trailofbits [`rfc8785`](https://github.com/trailofbits/rfc8785.py)
  (Apache-2.0) strictly unmodified, and the ceiling prompt's worked-
  example outputs are regenerated through `rfc8785.dumps`, never
  hand-typed.
- `c18` — PrOntoQA deductive entailment —
  [`whetstone_envs.c18`](src/whetstone_envs/c18)
  ([generate](src/whetstone_envs/c18/generate.py) ·
  [oracle](src/whetstone_envs/c18/oracle.py) ·
  [prompts](src/whetstone_envs/c18/prompts.py) ·
  [upstream](src/whetstone_envs/c18/upstream.py) ·
  [tests](tests/c18)). Depth-binned (D1/D2/D3/D5) True/False entailment
  over fictional nonce predicates, reseeded from the vendored
  [`asaparov/prontoqa`](https://github.com/asaparov/prontoqa) (Apache-2.0)
  generator through a subprocess boundary (`--model-name json`,
  `--ontology fictional`, fresh `--seed` per depth). The generator's
  stored label is definitional, so an independent from-scratch
  forward-chaining fixpoint oracle re-derives the label from the public
  facts + query alone, and construction asserts the two agree. Both probe
  prompts are verbatim from the baseline spec (Section 2); scored 0/1
  exact match.
- `c19` — Minigrid grid-world state prediction —
  [`whetstone_envs.c19`](src/whetstone_envs/c19)
  ([generate](src/whetstone_envs/c19/generate.py) ·
  [oracle](src/whetstone_envs/c19/oracle.py) ·
  [prompts](src/whetstone_envs/c19/prompts.py) ·
  [envs](src/whetstone_envs/c19/envs.py) ·
  [tests](tests/c19)). Seeded generator over four stochastic-layout
  Farama-Foundation [`minigrid`](https://github.com/Farama-Foundation/Minigrid)
  (Apache-2.0) envs — a real runtime dependency: it instantiates seeded
  envs, renders their ASCII grid, and walks the live object model for
  ground truth. The oracle reproduces that derived-fact walk
  independently from the public ASCII alone, and construction asserts
  the two agree. Predicts one derived fact (coordinate, heading,
  carrying-flag, or what-is-in-front) under vanilla Minigrid dynamics,
  scored 0/1 exact match.
- `c22` — stacked IFEval instruction-following constraints —
  [`whetstone_envs.c22`](src/whetstone_envs/c22)
  ([generate](src/whetstone_envs/c22/generate.py) ·
  [oracle](src/whetstone_envs/c22/oracle.py) ·
  [prompts](src/whetstone_envs/c22/prompts.py) ·
  [tests](tests/c22)). Reuses a pinned Google Research IFEval snapshot
  with namespaced imports and exact word counts (vendored under
  [`c22/_vendor`](src/whetstone_envs/c22/_vendor), Apache-2.0) for
  generation-side constraint selection and the scoring oracle.
- `c23` — subregular rule induction (InductionBench-style) —
  [`whetstone_envs.c23`](src/whetstone_envs/c23)
  ([generate](src/whetstone_envs/c23/generate.py) ·
  [oracle](src/whetstone_envs/c23/oracle.py) ·
  [prompts](src/whetstone_envs/c23/prompts.py) ·
  [upstream](src/whetstone_envs/c23/upstream.py) ·
  [tests](tests/c23)). Infer one latent subregular ISL/L-OSL/R-OSL
  string-transform rule from a few `IN -> OUT` demonstrations, then apply it
  to one held-out query; scored 0/1 exact match. Four strata (S1 ISL k2, S2
  L-OSL k2, S3 R-OSL k2, S4 ISL k3; |Σ|=4, single-rule) reseeded from the
  vendored [InductionBench](https://github.com/Wenyueh/inductive_reasoning_benchmark)
  (Apache-2.0) generator. The upstream active path does not import or run as
  shipped, so the vendor is applied as four named, individually-committed
  patches (config stub; drop the broken import; thread a real seed;
  `list(set(...))` → `sorted(...)` at the 6 nondeterminism sites) with a
  `VENDORED_DIFF.patch` capturing the whole delta — the `sorted()` fix makes
  the pool byte-identical across runs under a randomized `PYTHONHASHSEED`.
  The oracle reuses the vendored `apply_ISL_rule` / `apply_L_OSL_rule` /
  `apply_R_OSL_rule` transducers unmodified, re-applied to the held-out
  query; the ceiling prompt states the task conventions without leaking the
  latent rule.

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
```
