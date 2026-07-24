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

- `c11` — JSON canonicalization (RFC 8785 / JCS)
- `c18` — PrOntoQA deductive entailment
- `c19` — Minigrid state prediction
- `c22` — stacked IFEval instruction-following constraints —
  [`whetstone_envs.c22`](src/whetstone_envs/c22)
  ([generate](src/whetstone_envs/c22/generate.py) ·
  [oracle](src/whetstone_envs/c22/oracle.py) ·
  [prompts](src/whetstone_envs/c22/prompts.py) ·
  [tests](tests/c22)). Reuses a pinned Google Research IFEval snapshot
  with namespaced imports and exact word counts (vendored under
  [`c22/_vendor`](src/whetstone_envs/c22/_vendor), Apache-2.0) for
  generation-side constraint selection and the scoring oracle.
- `c23` — subregular rule induction (InductionBench-style)

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
```
