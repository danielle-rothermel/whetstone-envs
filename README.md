# whetstone-envs

[![CI](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml)

Reproducible quick-test environment contracts and task families.

## Scope

This repo owns the environment data and evaluation rules shared by Whetstone's
quick-test task families, with no dependency on optimizer or execution-contract
code:

- [**Instances**][instances-source] define immutable task inputs,
  private gold data, generation seeds, task strata, and public prompt identity.
- [**Pools and splits**][pools-source] validate ordered instance
  collections and allocate deterministic internal, official, and held-out
  cohorts.
- [**Probes**][probes-source] pair naive and ceiling templates,
  render public prompt inputs, and normalize predictions for evaluation.
- [**Scoring**][scoring-source] represents scored, failed, and
  missing observations and aggregates complete repeat matrices through task,
  stratum, and overall levels.
- [**Manifests**][manifests-source] pin generated pools with
  versioned identities and bounded canonical persistence.
- [**C11 JSON canonicalization**][c11-source] provides deterministic RFC 8785
  tasks, an independent canonicalization oracle, and naive and known-good
  probes.

The task-family implementations and the adapter to Whetstone's optimizer live
above this shared harness rather than inside its contracts.

## Installation

```bash
uv add whetstone-envs
```

## C11 JSON canonicalization

[`whetstone_envs.c11`][c11-source] generates balanced adversarial tasks for
RFC 8785 whitespace removal, key ordering, number rendering, Unicode escaping,
and mixed inputs. The exactly pinned `rfc8785` package produces private gold
values independently from the input builders. Shared harness contracts own
pool splitting, prompt rendering, scoring, and manifest persistence.

```python
from whetstone_envs.c11 import (
    DEFAULT_SPLIT_SIZES,
    PROBES,
    build_manifest,
    generate_pool,
)

pool = generate_pool()
split = pool.split(*DEFAULT_SPLIT_SIZES)
manifest = build_manifest(pool)
prompt = PROBES.render_ceiling(split.internal_eval[0])
```

## Instances

[`whetstone_envs.instances`][instances-source] owns the immutable
unit passed through generation, prompting, scoring, splitting, and persistence.
Prompt inputs are public; `gold` remains private evaluation data.

```python
@dataclass(frozen=True, slots=True)
class Instance:
    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: Mapping[str, str] = field(default_factory=lambda: ...)
    gold: str = ""
```

```python
def make_instance(
    *,
    id: str,
    seed: int,
    strata: tuple[str, ...] | str,
    prompt_inputs: Mapping[str, str] | None = None,
    gold: str = "",
) -> Instance: ...

def public_prompt_identity(
    instance: Instance,
) -> tuple[tuple[str, str], ...]: ...
```

## Pools and splits

[`whetstone_envs.pools`][pools-source] owns validated ordered pools
and the deterministic policy for selecting three disjoint evaluation cohorts.
Split optimization is delegated to `dr-graph`; returned instances preserve pool
order.

```python
@dataclass(frozen=True, slots=True)
class PoolSplit:
    internal_eval: tuple[Instance, ...]
    official: tuple[Instance, ...]
    held_out: tuple[Instance, ...]
```

```python
@dataclass(frozen=True, slots=True)
class TaskPool:
    instances: tuple[Instance, ...]

    @property
    def strata(self) -> tuple[str, ...]: ...

    def stratum_counts(self) -> dict[str, int]: ...
    def in_stratum(self, label: str) -> tuple[Instance, ...]: ...
    def split(
        self,
        internal_eval_n: int,
        official_n: int,
        held_out_n: int,
    ) -> PoolSplit: ...
```

## Probes

[`whetstone_envs.probes`][probes-source] owns the floor/ceiling
prompt pair and the default renderer that can see only public prompt inputs.
Normalization strips whitespace and complete outer triple-backtick fences.

```python
def render_with_prompt_inputs(template: str, instance: Instance) -> str: ...
def normalize(prediction: str) -> str: ...
```

```python
@dataclass(frozen=True, slots=True)
class ProbePair:
    naive_template: str
    ceiling_template: str
    render: Callable[[str, Instance], str] = render_with_prompt_inputs

    def render_naive(self, instance: Instance) -> str: ...
    def render_ceiling(self, instance: Instance) -> str: ...
```

## Scoring

[`whetstone_envs.scoring`][scoring-source] keeps failures and absent
results distinct from binary scores. Aggregation exposes a mean only when the
complete planned task/repeat matrix is present and scored.

```python
@verify(UNIQUE)
class Outcome(StrEnum):
    SCORED = "scored"
    FAILED = "failed"
    MISSING = "missing"

@dataclass(frozen=True, slots=True)
class Observation:
    task_id: str
    repeat_id: int
    outcome: Outcome = Outcome.SCORED
    score: int | None = None
```

```python
@dataclass(frozen=True, slots=True)
class Aggregate:
    mean: float | None
    usable: int
    failed_count: int
    missing_count: int
    label: str | None = None
    children: tuple["Aggregate", ...] = field(default_factory=tuple)

def aggregate(
    observations: Iterable[Observation],
    task_strata: Mapping[str, tuple[str, ...]],
    *,
    expected_repeat_ids: Iterable[int],
) -> Aggregate: ...
```

`exact_match`, `scored`, `failed`, and `missing` provide the primary leaf-level
constructors. `aggregate_task`, `aggregate_stratum`, and `aggregate_overall`
expose the individual aggregation steps when callers already own the hierarchy.

## Manifests

[`whetstone_envs.manifests`][manifests-source] owns the serialized
boundary for regenerated pool identity. Manifests use a closed Pydantic schema,
`dr-serialize` identities, and `dr-store` canonical files.

```python
class Manifest(BaseModel):
    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: Mapping[str, int]
    content_hash: Sha256Digest
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_pool(
        cls,
        pool: TaskPool,
        *,
        generator_version: str,
        seed_range: tuple[int, int],
    ) -> "Manifest": ...

    def write(self, path: Path) -> None: ...
    @classmethod
    def read(cls, path: Path) -> "Manifest": ...
    def matches_pool(self, pool: TaskPool) -> bool: ...

def content_hash(pool: TaskPool) -> Sha256Digest: ...
```

## Terms and contracts

The [published terms and contracts](https://danielle-rothermel.github.io/whetstone-envs/)
render the authoritative
[vocabulary](https://github.com/danielle-rothermel/whetstone-envs/blob/main/.defs/terms.toml)
and
[binding contracts](https://github.com/danielle-rothermel/whetstone-envs/blob/main/.defs/contracts.toml)
directly from their TOML sources. The
[changelog](https://github.com/danielle-rothermel/whetstone-envs/blob/main/CHANGELOG.md)
records notable changes.

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

Regenerate the canonical C11 manifest after an intentional generator change:

```bash
uv run python -m whetstone_envs.c11.regenerate
```

[c11-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c11
[instances-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/instances
[manifests-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/manifests
[pools-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/pools
[probes-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/probes
[scoring-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/scoring
