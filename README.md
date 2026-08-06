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
- `whetstone_envs.manifests` validates frozen manifests, derives versioned
  pool identities through `dr-serialize`, and persists exact canonical JSON.

The harness is independent of any particular task family and has no dependency
on whetstone's optimizer or execution contracts.

## Terms and contracts

The [published terms and contracts](https://danielle-rothermel.github.io/whetstone-envs/)
render the authoritative
[vocabulary](https://github.com/danielle-rothermel/whetstone-envs/blob/main/.defs/terms.toml)
and
[binding contracts](https://github.com/danielle-rothermel/whetstone-envs/blob/main/.defs/contracts.toml)
directly from their TOML sources. The
[changelog](https://github.com/danielle-rothermel/whetstone-envs/blob/main/CHANGELOG.md)
records notable changes.

## Consumer entry points

The public imports are organized by owning subpackage:

```python
from whetstone_envs.instances import Instance, public_prompt_identity
from whetstone_envs.manifests import Manifest, content_hash
from whetstone_envs.pools import PoolSplit, TaskPool
from whetstone_envs.probes import ProbePair, render_with_prompt_inputs
from whetstone_envs.scoring import Aggregate, Observation, aggregate
```

## Development

Install the locked development environment and commit hook once per clone:

```bash
uv sync --locked
uv run pre-commit install
```

The hook runs the same formatting, lint, type, definitions, test, and package
build gate used by CI. Run it directly at any time:

```bash
scripts/pre-check.sh
```
