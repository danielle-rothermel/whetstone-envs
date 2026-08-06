# whetstone-envs

[![CI](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml)

**whetstone-envs provides the task-family-agnostic harness for reproducible
quick-test environments: immutable instances, deterministic pools and splits,
probe rendering, score aggregation, and pinned manifests.** Its functionality
is organized around three functional areas and two supporting infrastructure
packages:

- **[Instances](https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/instances)**
  define immutable task inputs, private gold data, seeds, and stratum
  membership.
- **[Probes](https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/probes)**
  pair naive and ceiling prompts and normalize predictions consistently for
  evaluation.
- **[Scoring](https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/scoring)**
  evaluates repeated observations and aggregates results through task,
  stratum, and overall levels.
- **Infrastructure**
  - **[Pools and splits](https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/pools)**
    validate collections of instances and partition them deterministically
    across evaluation destinations.
  - **[Manifests](https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/manifests)**
    pin generation metadata and pool contents for reproducibility and drift
    detection.

The harness is independent of any particular task family and has no dependency
on whetstone's optimizer or execution contracts.

The shapes below summarize its stable public contracts. Validation and
algorithm details remain encapsulated in the linked packages.

## Instances

Instances are the immutable boundary between task generation, prompt
rendering, and scoring. Their public prompt identity excludes private
generation and evaluation metadata.

```python
@dataclass(frozen=True, slots=True)
class Instance:
    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: Mapping[str, str]
    gold: str
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

## Probes

Probes define the prompt boundary: a paired floor and ceiling interface, a
renderer restricted to public inputs, and shared prediction normalization.

```python
@dataclass(frozen=True, slots=True)
class ProbePair:
    naive_template: str
    ceiling_template: str
    render: Callable[[str, Instance], str] = render_with_prompt_inputs

    def render_naive(self, instance: Instance) -> str: ...
    def render_ceiling(self, instance: Instance) -> str: ...
```

```python
def render_with_prompt_inputs(
    template: str,
    instance: Instance,
) -> str: ...

def normalize(prediction: str) -> str: ...
```

## Scoring

Scoring combines binary exact match with explicit scored, failed, and missing
observations. Aggregation follows the stable repeat → task → stratum → overall
reduction over the complete planned task/repeat matrix.

```python
class Outcome(Enum):
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
    children: tuple[Aggregate, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool: ...
```

```python
def exact_match(prediction: str, gold: str) -> int: ...

def aggregate(
    observations: Iterable[Observation],
    task_strata: Mapping[str, tuple[str, ...]],
    *,
    expected_repeat_ids: Iterable[int],
) -> Aggregate: ...
```

## Infrastructure

### Pools and splits

Pools validate instance membership and expose deterministic, disjoint
evaluation splits while preserving the pinned instance order.

```python
@dataclass(frozen=True, slots=True)
class TaskPool:
    instances: tuple[Instance, ...]

    def split(
        self,
        internal_eval_n: int,
        official_n: int,
        held_out_n: int,
    ) -> PoolSplit: ...


@dataclass(frozen=True, slots=True)
class PoolSplit:
    internal_eval: tuple[Instance, ...]
    official: tuple[Instance, ...]
    held_out: tuple[Instance, ...]
```

### Manifests

Manifests pin the generation inputs and canonical pool contents needed to
detect drift across regenerations.

```python
@dataclass(frozen=True, slots=True)
class Manifest:
    generator_version: str
    seed_range: tuple[int, int]
    stratum_counts: Mapping[str, int]
    content_hash: str
    schema_version: int

    @classmethod
    def from_pool(
        cls,
        pool: TaskPool,
        *,
        generator_version: str,
        seed_range: tuple[int, int],
    ) -> Manifest: ...

    def matches_pool(self, pool: TaskPool) -> bool: ...


def content_hash(pool: TaskPool) -> str: ...
```
