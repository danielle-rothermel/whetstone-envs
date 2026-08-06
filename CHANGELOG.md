# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-08-06

### Added

- Immutable task instances and canonical public prompt identity.
- Validated task pools with deterministic, disjoint, destination-balanced
  splits.
- Probe rendering, idempotent prediction normalization, explicit evaluation
  observations, exact-match scoring, and complete-matrix aggregation.
- Strict persisted pool manifests with canonical content hashing and explicit
  retained-pool matching.
- The authoritative `.defs` vocabulary and contracts reference, published by
  GitHub Pages from the current TOML sources.
- Locked, multi-version Depot CI, a canonical local pre-check hook, and
  tag-triggered PyPI trusted publishing.

### Changed

- Organize the package and test suite by the `instances`, `manifests`, `pools`,
  `probes`, and `scoring` capability boundaries.
- Update package metadata and repository guidance for the public harness and
  its release process.

### Removed

- Remove the former `whetstone_envs.core` package and its import paths without
  compatibility shims.
- Remove completed implementation-planning documentation from the live tree.

## [0.1.0] - 2026-07-22

### Added

- Initial Python package, test, lint, type-checking, and CI scaffold.
