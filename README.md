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
- [**C22 instruction constraints**][c22-source] provides fixed seeded pools of
  composed IFEval constraints and strict all-pass scoring.
- [**C23 subregular induction**][c23-source] provides determinate hidden-rule
  string transformations across four ISL and OSL strata.

Task-family implementations live in their owning subpackages alongside the
shared harness; the adapter to Whetstone's optimizer lives above this package.

## Installation

```bash
uv add whetstone-envs
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
```

```python
def content_hash(pool: TaskPool) -> Sha256Digest: ...
```

## C11 JSON canonicalization

[`whetstone_envs.c11`][c11-source] generates balanced adversarial tasks for
RFC 8785 whitespace removal, key ordering, number rendering, Unicode escaping,
and mixed inputs. An independent, exactly pinned oracle produces private gold;
the shared harness owns splitting, prompting, scoring, and persistence.

```python
@verify(UNIQUE)
class C11Stratum(StrEnum):
    WHITESPACE = "c11/whitespace"
    KEY_ORDER = "c11/key-order"
    NUMBER = "c11/number"
    UNICODE = "c11/unicode"
    MIXED = "c11/mixed"
```

```python
DEFAULT_SPLIT_SIZES: tuple[int, int, int]
PROBES: ProbePair

def generate_pool(
    *,
    n_per_stratum: int = ...,
    seed_start: int = ...,
) -> TaskPool: ...

def build_manifest(pool: TaskPool) -> Manifest: ...
def canonicalize(input_json: str) -> str: ...
```

## C22 instruction constraints

[`whetstone_envs.c22`][c22-source] provides two fixed, seeded pools of
composed Google Research IFEval constraints. The default preset crosses three,
four, and five constraints with easy and mixed strata; the hard preset uses
three, six, and eight constraints and includes every hard constraint in each
task. C22 scores only this model-visible stack and claims no separate semantic
task grading.

```python
@verify(UNIQUE)
class Preset(StrEnum):
    DEFAULT = "default"
    HARD = "hard"
```

```python
PROBES: ProbePair

def score(gold: str, response: str) -> int: ...
```

```python
def generate_pool(preset: Preset = Preset.DEFAULT) -> TaskPool: ...
def load_manifest(preset: Preset = Preset.DEFAULT) -> Manifest: ...
```

## C23 subregular induction

[`whetstone_envs.c23`][c23-source] is a higher-layer environment built on the
shared harness. It generates four balanced single-rule strata: ISL k=2,
left-OSL k=2, right-OSL k=2, and ISL k=3 over the fixed vocabulary `abcd`; each
task has six demonstrations and a distinct nontrivial query whose output is
determinate across the complete supported hypothesis class.

Internally, the stable rule vocabulary is represented by:

```python
@verify(UNIQUE)
class RuleFamily(StrEnum):
    ISL = "ISL"
    L_OSL = "L-OSL"
    R_OSL = "R-OSL"

@dataclass(frozen=True, slots=True)
class RuleConfiguration:
    family: RuleFamily
    context_length: int
```

```python
GENERATOR_VERSION: str
PROBES: ProbePair

def generate_pool(*, n_per_stratum: int = 50) -> TaskPool: ...
def default_split_sizes(pool: TaskPool) -> tuple[int, int, int]: ...
```

```python
def score_gold(prediction: str, gold: str) -> int: ...
```

Generation uses fixed fresh stratum seeds beginning at `555000000` and
private injected random-number generators. The adapted InductionBench
reference transducers and generation path are pinned and attributed inside
the package; no process-global random state is read or mutated.

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

The hook runs formatting, lint, type, definitions, the fast test suite, and
package validation. Run it directly at any time:

```bash
scripts/pre-check.sh
```

CI also runs the full-cohort integration checks. Run that exact gate locally
before release:

```bash
CI=true scripts/pre-check.sh
```

Regenerate the canonical C11 manifest after an intentional generator change:

```bash
uv run python -m whetstone_envs.c11.regenerate
```

[c11-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c11
[instances-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/instances
[c22-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c22
[manifests-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/manifests
[pools-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/pools
[probes-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/probes
[scoring-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/scoring
[c23-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c23
