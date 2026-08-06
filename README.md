# whetstone-envs

[![CI](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml)

**whetstone-envs provides the task-family-agnostic harness for reproducible
quick-test environments: immutable instances, deterministic pools and splits,
probe rendering, score aggregation, and pinned manifests.** Its functionality
is organized into five shared contract areas:

- **Instances** define immutable task inputs, private gold data, seeds, and
  stratum membership.
- **Pools and splits** validate collections of instances and partition them
  deterministically across evaluation destinations.
- **Probes** pair naive and ceiling prompts and normalize predictions
  consistently for evaluation.
- **Scoring** evaluates repeated observations and aggregates results through
  task, stratum, and overall levels.
- **Manifests** pin generation metadata and pool contents for reproducibility
  and drift detection.

The harness is independent of any particular task family and has no dependency
on whetstone's optimizer or execution contracts.
