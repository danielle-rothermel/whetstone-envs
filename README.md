# whetstone-envs

[![CI](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml)

**whetstone-envs provides the task-family-agnostic harness for reproducible
quick-test environments: immutable instances, deterministic pools and splits,
probe rendering, score aggregation, and pinned manifests.**

- `whetstone_envs.instances` defines immutable task inputs,
  private gold data, seeds, and stratum membership.
- `whetstone_envs.probes` pairs naive and ceiling prompts and
  normalizes predictions for evaluation.
- `whetstone_envs.scoring` evaluates repeated observations and
  aggregates results through task, stratum, and overall levels.
- `whetstone_envs.pools` validates instance collections and splits
  them across evaluation destinations.
- `whetstone_envs.manifests` pins generation metadata and pool
  contents for reproducibility and drift detection.

The harness is independent of any particular task family and has no dependency
on whetstone's optimizer or execution contracts.

## Behavioral boundaries

- Public prompt identity is the canonical identity of sorted `prompt_inputs`;
  private fields do not participate, and rendered-prompt uniqueness is owned
  by each renderer/template.
- The default probe renderer formats only against `prompt_inputs`. Custom
  renderers receive the full `Instance`, so their callers own that access
  boundary.
- Aggregation covers the complete planned task/repeat matrix. A failed or
  missing observation, or any incomplete child aggregate, yields no mean.
- Pool splitting is deterministic, disjoint, and stratified by each
  instance's complete strata tuple. Every destination preserves the original
  pool order.
- Manifests strictly validate their persisted schema. Pool matching explicitly
  checks retained seeds, stratum counts, and the canonical content hash.

## Consumer entry points

Import public contracts from their owning subpackages. The root
`whetstone_envs` package intentionally exports no API.

```python
from whetstone_envs.instances import Instance, public_prompt_identity
from whetstone_envs.manifests import Manifest, content_hash
from whetstone_envs.pools import PoolSplit, TaskPool
from whetstone_envs.probes import ProbePair, render_with_prompt_inputs
from whetstone_envs.scoring import Aggregate, Observation, aggregate
```

## Development

```sh
uv sync
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
