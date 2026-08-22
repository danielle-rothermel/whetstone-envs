# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The C19 optimizer runner is now a family-agnostic runner. `C19RunSpec`
  becomes `RunSpec` and `run_c19_optimizer` becomes `run_optimizer`; both
  read every family-specific decision -- pool generator, experiment builder,
  probes, mutation field, render contract, and scripted proposer bodies --
  from a new `whetstone_envs.optim.families` registry rather than from
  literals in the runner. `run_optimizer` now names no task family, which is
  what lets a second family reach the optimizers through the identical path.
  `C19_OPTIMIZERS` and `C19_TRANSPORTS` become `OPTIMIZERS` and
  `TRANSPORTS`, and `C19_PROPOSAL_BODIES` becomes `FamilySpec
  .proposal_bodies()`.
- Task-pool generation is no longer hardcoded. The runner used to call
  `generate_pool(n_per_stratum=2, seed_start=765_432)` with both values
  written into its body; they are now `RunSpec` fields defaulting to the
  family's own values, so an unparameterised run generates the pool it
  always generated.
- COPRO's breadth and depth are no longer hardcoded to `2` and `1`. Both are
  `RunSpec` fields with those values as defaults. A breadth below 2 is
  refused at spec validation rather than inside the durable run boundary,
  matching `CoproControl`'s own rule.

### Added

- A task-family registry at `whetstone_envs.optim.families`. `FamilySpec`
  carries one family's generator, experiment builder, probes, mutation
  field, prompt fields, render contract, and pool defaults;
  `register_family` admits it and `family_spec` resolves it. `c19` is
  registered; `c18` is a known identifier the registry already admits, so
  registering it later is a registration rather than a change to the
  registry's vocabulary. A known-but-unregistered family reports a wiring
  gap distinctly from an unrecognised name.
- `RunSpec.seed` plumbs one explicit algorithmic seed per run. GEPA and
  MIPROv2 carry it onto their controls as an explicit field; `CoproControl`
  has no seed, so a COPRO run's stochasticity remains the proposer LM and
  the provider `SEED` control. `seed_disposition` names that difference for
  the study manifest rather than faking a seed the control never reads. An
  omitted seed keeps each optimizer's own default, so an unseeded run keeps
  the control identity hash it always had.
- `RunSpec.proposer_model` separates the proposal route from the task route,
  so a study can run a cheap task model against a stronger proposer.
  `None` reuses the experiment's own route.
- `RunSpec.gepa_max_metric_calls` pins GEPA's paid metric-call ceiling, and
  `RunSpec.codex_capacity` carries the Codex arm's evaluate-call cap. Each
  is refused on an optimizer that cannot honour it, so a flag nothing reads
  cannot look respected in the study manifest.
- New `whetstone_envs.optim.cli` flags: `--proposer-model`, `--seed`,
  `--n-per-stratum`, `--pool-seed-start`, `--copro-breadth`,
  `--copro-depth`, `--gepa-max-metric-calls`, and `--codex-capacity`.
  `--family` now offers exactly the registered families.

- The C19 evaluation CLI and report accept `--role held_out`, evaluating the
  held-out split through the same path as `internal` and `official`. The
  persisted `EvalRun.role` records which of the three roles a report covers,
  and every role reports against its own tasks and its own
  `eval_config_hash`. An experiment prepared without a held-out split refuses
  the role by name before creating a run directory rather than falling back to
  another split.


- The study's two null optimizers, in `whetstone_envs.optim.nulls`, as
  `ProposerTransport` implementations so they reach the optimizer through the
  same surface a real proposer does. `NullRandomTransport` (null-A) is the
  selection-on-noise control: it perturbs the base candidate's template with a
  seeded RNG — swapping, deleting, and duplicating whitespace-delimited tokens
  at `NULL_PERTURBATION_RATE` — and returns the results as ordinary drafts, so
  best-on-internal selection runs over candidates carrying no information. The
  perturber treats every `{field}` placeholder as atomic and immovable and
  re-validates against the run's `TemplateRenderContract` before returning, so
  a required field is never dropped; a draw the contract rejects, that repeats
  the base, or that collides with an earlier slot in the same batch is retried
  from the same seeded stream, bounded, then falls back to the base unchanged
  with `identity_fallback` recorded on the draft. Perturbation is deterministic
  in the run seed and the proposal request, so a null-A run replays exactly.
  `NullIdentityTransport` (null-B) is the pipeline-overhead control: it
  proposes the base unchanged. Neither null makes a provider call, so every
  draft reports `proposer_calls: 0` with no call id and no price, and run cost
  records no proposer spend for a null arm rather than a zero-dollar phantom
  call.

  Two upstream contracts shape null-B and are pinned by its tests. A
  *successful* draft repeating its base is not representable — `diff_check`
  rejects a mutation equal to its base, because a no-op is not a proposal — so
  null-B reports an unfilled slot, which is how a real proposer that returned
  nothing reports it. What an optimizer does with that is the optimizer's
  contract: GEPA and MIPROv2 set `terminal_proposal_count` on their step
  contracts and so may terminalize `seed_retained` when their search accepts
  nothing over the seed, while COPRO does not set it, so under COPRO's control
  shape an unfilled round is a `copro_proposal_cardinality` terminal failure
  whose result still names the seed as the run's outcome.
- The C19 optimizer CLI and runner accept `--optimizer gepa` and
  `--optimizer miprov2` through the same shared runner path as COPRO.
- `--num-seeds` on the CLI and `RunSpec.num_seeds` make repeats per task
  (`K_REPEAT`) a runner parameter instead of a hardcoded 1.

## [0.2.2] - 2026-08-22

### Changed

- C19 optimizer runs record effect leases in a SQLite authority instead of an
  in-memory one. The authority shares the run directory's `runtime.sqlite`
  with the object store, mirroring whetstone-ai's platform CLI; the two
  components own disjoint tables. Leases now outlive the process, so a re-run
  against a completed run directory replays its recorded effects rather than
  re-executing them. The runtime is closed on every exit path, releasing the
  eval engine and the authority's connection.

### Added

- The C19 optimizer CLI and runner accept `--optimizer gepa` and
  `--optimizer miprov2` through the same `run_c19_optimizer` path as COPRO.
- MIPROv2 support for C19: `build_c19_miprov2_control`,
  `build_c19_miprov2_adapter`, and `build_c19_miprov2_state` assemble the
  control, adapter, and opening durable state on whetstone-ai's public
  MIPROv2 surface. `--demo-mode fewshot|zeroshot|ground_only` selects the
  demonstration regime. Demonstrations reach the candidate through MIPROv2's
  own composed `### Demonstrations` section rather than a C19 placeholder:
  `fewshot` searches over demo sets and renders the selected set, while
  `zeroshot` and `ground_only` bootstrap demos only to ground instruction
  proposals and leave the section empty. All three modes compose a template
  that still satisfies the C19 `{grid}`/`{command}`/`{question}` contract.
  Labeled demonstrations carry the task's oracle gold as the component's
  `response` output, and the proposer's prompt model binds the experiment's
  `ProviderCallConfig` reference, as COPRO and GEPA already do.
- `--num-seeds` on the CLI and `C19RunSpec.num_seeds` make repeats per task
  (`K_REPEAT`) a runner parameter instead of a hardcoded 1.
- Trajectory reports carry a `spend` block projected from
  `OptimResult.cost`: per-role billable calls, cached calls, token totals,
  the priced/unpriced split, and a USD total only when every contributing
  call carried a provider-reported price. Terminal and HTML trajectory views
  render an unpriced role as `unpriced (n/total)` rather than as a zero. The
  block's `schema_version` is pinned to the one whetstone-ai cost-report
  version this package projects, so an embedded report at any other version
  is rejected rather than reinterpreted.
- A C19 evaluation CLI for task-family information, standalone fake or
  OpenRouter execution, strict local report publication, summary/failure/task
  inspection, and paired candidate comparison, exposed as the installed
  `whetstone-eval` command and a thin source-checkout launcher.
- Strict bounded `eval-report.json` and `trajectory-report.json` contracts,
  including safe typed provider-failure projection, exact binary C19 score
  reconciliation, COPRO intent and GEPA effect-transcript trajectory
  publication, explicit
  delta/cumulative budgets, and exact full-text candidate views.
- Deterministic, portable, offline `eval-report.html` and
  `trajectory-report.html` renderings with strict embedded-data escaping,
  hashed inline CSP, complete C19 debugging surfaces, exact-ref trajectory
  lineage, exact per-resolution navigation and diagnosis, responsive layouts,
  and pinned Chromium interaction and screenshot-comparison checks.

### Changed

- Pin the `optim` extra to published `whetstone-ai==0.1.6` and `dr-store`
  to 0.2.6, which that release requires. 0.1.6 drives MIPROv2 through
  `state.control.mutation_field`, so C19 runs MIPROv2 in every demonstration
  mode, and it resolves the provider/`eval.drivers` import cycle, so the
  optimizer CLI imports whetstone-ai's modules in any order.
- C19 reaches whetstone-ai through its current production entry point:
  `build_runtime` with an explicit adapter registry and effect authority,
  replacing the removed `register_runtime`. Registry membership is part of
  controller identity, so each run registers exactly the adapter it drives.
- C19's GEPA adapter is built by `build_gepa_harness_adapter`, whetstone-ai's
  canonical GEPA constructor, rather than by a hand-assembled authority
  chain. `build_c19_gepa_adapter` is now the C19 prompt services and control
  plus that one call, and the trainset/valset partition follows the control
  instead of handing the search the union of both splits.
- `prepare_c19_experiment` derives held-out rows into a full `EvalSplit`
  under the `held_out` split role and passes it as `EvalConfigs.held_out`,
  replacing the removed `held_out_task_hashes=` constructor argument.
  `EvalConfigs` enforces split disjointness at construction, and
  `held_out_task_hashes` remains readable as a derived property. An
  experiment with no held-out rows leaves the split absent.
- A GEPA run that finds no improvement completes by reporting the
  retained seed instead of substituting a candidate, and its steps carry
  resolvable eval and reward evidence for the evaluations its search
  drove. The GEPA trainset remains the internal eval split, and fake GEPA
  still accepts the ceiling draft because the fake task model emits gold
  for ceiling-rendered prompts; exact-match scoring is unchanged.
- The public-surface guard runs with an empty allowlist and also rejects
  `getattr(obj, "_name")` reads of whetstone privates alongside the
  existing `cast("Any", …)._name` check.
- Simplify HTML reports to a stable white scientific layout with
  colorblind-safe blue navigation and green success emphasis.
- Preserve the authoritative C19 `PoolSplit` beside each prepared Whetstone
  experiment so report projection joins evidence to source instances exactly.
- `whetstone_envs.optim.experiment` owns the C19 provider-call-config
  reference and the task-hash-to-gold mapping, so COPRO, GEPA, and MIPROv2
  bind one derivation instead of per-optimizer copies.

## [0.2.1] - 2026-08-21

### Changed

- Pin `dr-store` to 0.2.5 and `dr-graph` to 0.1.3 so this package can share
  an environment with whetstone-ai.
- Pin the `optim` extra to published `whetstone-ai==0.1.1` from PyPI.

### Added

- An optional `optim` extra that maps C19 pools, probes, and exact-match
  scoring onto whetstone-ai experiments through the public optimizer surface.

## [0.2.0] - 2026-08-06

### Added

- Deterministic C11 RFC 8785 task generation across five adversarial strata.
- An exactly pinned independent canonicalization oracle, naive and known-good
  probes, and a canonical persisted pool manifest.
- The C22 instruction-following environment with fixed default and hard
  presets, canonical manifests, naive and ceiling probes, and strict all-pass
  scoring.
- A namespaced Google Research IFEval runtime pinned with provenance,
  reproducible patch verification, and hand-built fixtures for every supported
  constraint.
- The C23 single-rule subregular induction environment with four balanced
  ISL/OSL strata, determinate six-demonstration tasks, ceiling and naive
  probes, exact scoring, default split sizing, and a committed pool manifest.
- A private-RNG adaptation of the pinned InductionBench generation and
  reference-transducer path with packaged Apache-2.0 attribution.
- The C18 PrOntoQA task family with frozen default and hard generation
  configurations, independent surface-text entailment, two public probes, and
  checked-in pool manifests.
- A pinned vendored generator boundary, an optional `c18` dependency extra,
  and an explicit script for regenerating either canonical C18 manifest.
- Deterministic C19 MiniGrid state-prediction tasks across navigation, object
  manipulation, and door-interaction scenarios at two grid sizes.
- A supported answer-relevant physical-state oracle independently
  cross-checked against live MiniGrid transitions, naive and known-good probes,
  bounded regeneration, and a canonical persisted pool manifest protected from
  custom generation inputs.
- Distribution validation that checks artifact metadata, package contents, and
  isolated installed-wheel imports before publication.

### Changed

- Restrict Depot cache writes to trusted `main` pushes while allowing pull
  requests to restore the rotated cache namespace.
- Run repository safety hooks in CI and release validation in addition to the
  canonical format, lint, type, definition, test, and build gate.
- Validate binding-contract structure and require `.defs` mappings for every
  symbol exported by an owning public package.
- Require a finalized dated changelog entry before a version tag can publish
  distributions.
- Model C22 gold as a closed composition of concrete constraint variants and
  derive checker descriptions and arguments through one vendor adapter.
- Cut one package release from the combined `main` tip after all task families
  assigned to that version have merged and passed the release gate.
- Extend the owning-subpackage API model to the higher-layer
  `whetstone_envs.c23` environment.
- Package and test C18 through the published instances, pools, probes,
  scoring, and manifests capability boundaries.
- Expand the package scope to include task families and exactly pin C19's
  MiniGrid, Gymnasium, and NumPy transition semantics.

## [0.1.1] - 2026-08-06

### Added

- Immutable task instances and canonical public prompt identity.
- Validated task pools with deterministic, disjoint, destination-balanced
  splits.
- Probe rendering, idempotent prediction normalization, explicit evaluation
  observations, exact-match scoring, and complete-matrix aggregation.
- Frozen persisted pool manifests with versioned `dr-serialize` identities,
  `dr-store` atomic canonical publication and bounded descriptor-pinned reads,
  and explicit retained-pool matching.
- The authoritative `.defs` vocabulary and contracts reference, published by
  GitHub Pages from the current TOML sources.
- Locked, multi-version Depot CI, a canonical local pre-check hook, and
  tag-triggered PyPI trusted publishing.

### Changed

- Organize the package and test suite by the `instances`, `manifests`, `pools`,
  `probes`, and `scoring` capability boundaries.
- Express pool-split coverage and balance as explicit marginal-cost policy
  solved through `dr-graph`'s exact separable transport primitive.
- Pin foundational runtime dependencies and validate workflow syntax in the
  canonical local and CI gate.
- Update package metadata and repository guidance for the public harness and
  its release process.

### Removed

- Remove the former `whetstone_envs.core` package and its import paths without
  compatibility shims.
- Remove completed implementation-planning documentation from the live tree.

## [0.1.0] - 2026-07-22

### Added

- Initial Python package, test, lint, type-checking, and CI scaffold.
