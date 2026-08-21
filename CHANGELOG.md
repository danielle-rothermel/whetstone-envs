# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Pin `dr-store` to 0.2.5 and `dr-graph` to 0.1.3 so this package can share
  an environment with whetstone-ai.

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
