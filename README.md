# whetstone-envs

[![CI](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-envs/actions/workflows/ci.yml)

Reproducible quick-test environment contracts and task families.

## Scope

This repo owns the environment data and evaluation rules shared by Whetstone's
quick-test task families. Core packages have no dependency on optimizer or
execution-contract code:

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
- [**C18 PrOntoQA**][c18-source] provides deterministic fictional-ontology
  entailment tasks with an independent forward-chaining oracle.
- [**C19 MiniGrid state prediction**][c19-source] provides deterministic
  grid-world tasks, a supported answer-relevant physical-state transition
  oracle, and naive and known-good probes.
- [**C22 instruction constraints**][c22-source] provides fixed seeded pools of
  composed IFEval constraints and strict all-pass scoring.
- [**C23 subregular induction**][c23-source] provides determinate hidden-rule
  string transformations across four ISL and OSL strata.
- [**Reporting**][reporting-source] owns strict local evaluation and
  optimization-trajectory reports plus read-only terminal and self-contained
  HTML inspection.

Task-family implementations live in their owning subpackages alongside the
shared harness. An optional [`whetstone_envs.optim`][optim-source] extra maps
those contracts onto whetstone-ai experiments; installing it requires Python
3.13 or 3.14 and pins published `whetstone-ai==0.1.7`.

## Installation

```bash
uv add whetstone-envs
```

Install C18's pinned generator dependencies when generating its pools:

```bash
uv add 'whetstone-envs[c18]'
```

Install the optimizer adapter extra when running COPRO, GEPA, MIPROv2, or
Codex against a task family. The extra pins published `whetstone-ai==0.1.7`
from PyPI:

```bash
uv add 'whetstone-envs[optim]'
```

The extra is Python 3.13/3.14 only. Every optimizer runs on whetstone-ai's
public surface, with no private imports and no adapter subclassing:

| Optimizer | Constructed by | Modes | Notes |
| --- | --- | --- | --- |
| `copro` | `configure_copro` + `CoproAdapter` | — | Proposal-only search over the mutation field. |
| `gepa` | `build_gepa_harness_adapter` | — | Reflection search; the trainset is the internal eval split. A run that finds no improvement reports the retained seed rather than substituting a candidate. |
| `miprov2` | `configure_miprov2` + `Miprov2Adapter` | `--demo-mode fewshot\|zeroshot\|ground_only` | Also binds an opening durable state (labeled trainset, proposal examples, RNG checkpoint). |
| `codex` | `configure_codex` + `CodexAdapter` | — | The foreign-agent arm: the Codex CLI searches out of process under dr-exec containment and reaches exactly one MCP tool, which evaluates a candidate on the internal split. One opaque step; whetstone runs no search of its own. macOS only — the containment profile is `sandbox-exec`. |

`--demo-mode` selects MIPROv2's demonstration regime and is ignored by COPRO
and GEPA. Demonstrations reach the candidate through MIPROv2's own composed
`### Demonstrations` section rather than through a family placeholder:
`fewshot` searches over demo sets and renders the selected set there, while
`zeroshot` and `ground_only` still bootstrap demos to ground instruction
proposals but leave the section empty. `--num-seeds` sets repeats per task
(`K_REPEAT`).

The Codex arm takes its own flags. `--codex-capacity` is the per-run
admitted evaluate-call cap, which is simultaneously the step's `tool_calls`
budget and the `ToolCapacity` the evaluation server admits against; it
defaults to the study's pre-registered 8. `--codex-binary` is the CLI to
spawn (default: the real `codex` on the run PATH), and `--codex-model`,
`--codex-reasoning-effort`, and `--codex-wall-seconds` configure the agent.
A Codex run proves its session with an authentication preflight before it
commits any capacity, and there is no flag that skips it. The preflight is
not a spend guard, though — it proves a session by spawning the CLI — so a
Codex run is refused outright unless it is scripted or deliberately opted
in; see [Real Codex runs](#real-codex-runs).

Two properties follow from the agent being foreign. A Codex run resolves no
evaluation intent — every paid evaluation is admitted through the tool and
cited from the step's tool evidence, which is where the trajectory report
and the audit read it. And the agent's own model spend runs on the Codex
subscription rather than the study's key, so the cost report prices the
task model and attributes nothing to a proposer; there is no `codex_agent`
cost role.

`--family` selects which task family the optimizers drive. Every optimizer
runs every family through the same `run_optimizer`; a family is admitted by
registering a `FamilySpec` in [`whetstone_envs.optim.families`][optim-source]
and nothing on the shared path names one:

| Family | Placeholders | Scoring | Protocol splits | Registered by |
| --- | --- | --- | --- | --- |
| `c19` | `{grid}`, `{command}`, `{question}` | exact match on the whole reply | `88,132,220` | `optim/experiment.py` |
| `c18` | `{question}`, `{query}` | terminal `True`/`False` verdict, via `c18.score_gold` | `24,48,48` | `optim/c18_experiment.py` |

C18's splits come from its own `SplitPlan` at `n_per_stratum=30` over four
depth strata. Its internal split of 24 is below MIPROv2's default minibatch
size of 35, so C18 runs keep `minibatch=False`, which is what the shared
control builder already does.

`tests/optim/test_c18_adapter_swap.py` is the mechanical evidence for that
claim: it traces both families through one optimizer and asserts the two runs
differ only inside the family-adapter file set.

Runs happen in-process; artifacts write under
`~/drotherm/data/runs/whetstone-envs/<run-id>/`, never inside the git tree:

```bash
uv run --extra optim python scripts/run-optim.py \
  --family c19 --optimizer copro --transport fake --split-sizes 2,2,0
uv run --extra optim python scripts/run-optim.py \
  --family c19 --optimizer gepa --transport fake --split-sizes 2,2,0
uv run --extra optim python scripts/run-optim.py \
  --family c19 --optimizer miprov2 --demo-mode fewshot \
  --transport fake --split-sizes 2,2,0
uv run --extra optim python scripts/run-optim.py \
  --family c18 --optimizer copro --transport fake --split-sizes 2,2,0 \
  --n-per-stratum 1
```

`--optimizer codex` is deliberately absent from that list — see below.

### Real Codex runs

A Codex run spawns the real Codex CLI, which costs money, so it is refused
by default. The authentication preflight is not the thing that stops an
accidental run: it proves a session by *spawning* the CLI, and on a machine
with a Codex login that spawn succeeds and is billed. So `run_optimizer`
refuses `--optimizer codex` outright, before any preflight, adapter,
admission authority, or subprocess exists, unless one of two things is true:

- a `CodexTestSeam` is supplied, which points the run at the scripted fake
  CLI. It is keyword-only, absent from `RunSpec`, and has no flag, so only a
  test reaches it; or
- both halves of the opt-in are present:
  `WHETSTONE_ENVS_ALLOW_REAL_CODEX=1` in the environment **and**
  `--allow-real-codex` (`RunSpec.allow_real_codex`). Requiring both means
  neither a serialized spec nor an exported variable authorizes spend on its
  own — a study arm or a copied command line carrying the flag still
  refuses.

Anything else raises `RealCodexRefusedError`, and the refused run leaves no
run directory behind. A session-scoped `conftest.py` fixture asserts the
environment variable is unset and clears it, so no test can opt in.

A real run therefore requires all of: the opt-in variable, the flag, a live
authenticated Codex session, macOS (the containment profile is
`sandbox-exec`), provider spend for the task-model evaluations, and a go
from Danielle. **No real Codex run has been performed yet** — every claim
about the arm rests on the scripted stand-in, which speaks real MCP to the
real evaluation server and so exercises the production admission, lease,
evaluation, and ledger path. Only the agent's own decisions are scripted.

```bash
WHETSTONE_ENVS_ALLOW_REAL_CODEX=1 uv run --extra optim \
  python scripts/run-optim.py \
  --family c19 --optimizer codex --transport fake --split-sizes 2,2,0 \
  --codex-capacity 8 --allow-real-codex
```

## Evaluation and trajectory reports

Explain C19, run a standalone fake evaluation, and inspect the saved report:

```bash
uv run --extra optim whetstone-eval info c19 --show-templates
uv run --extra optim whetstone-eval run \
  --family c19 --candidate naive --candidate ceiling \
  --transport fake --role internal --split-sizes 20,20,0 --repeats 1
uv run --extra optim whetstone-eval summary RUN_DIR
uv run --extra optim whetstone-eval failures RUN_DIR
uv run --extra optim whetstone-eval task RUN_DIR TASK_ID
uv run --extra optim whetstone-eval compare RUN_DIR naive ceiling
uv run --extra optim whetstone-eval html RUN_DIR
```

`--candidate-file NAME=PATH` adds a validated UTF-8 prompt template.
`--role` selects the split to evaluate: `internal`, `official`, or
`held_out`. Official and held-out evidence intentionally carries no reward,
and each role reports against its own tasks and eval config hash. The
held-out split is optional, so `--role held_out` requires a positive
third `--split-sizes` entry and is refused by name otherwise. Standalone runs publish `runtime.sqlite` and a bounded 128 MiB
canonical `eval-report.json`. Optimizer runs additionally publish
`trajectory-report.json` under the same bound, carrying a per-role spend
block (calls, cached calls, tokens, and a USD total only when every billable
call carried a provider-reported price). Inspect it with:

```bash
uv run --extra optim whetstone-eval trajectory RUN_DIR
uv run --extra optim whetstone-eval trajectory RUN_DIR --show-candidates
uv run --extra optim whetstone-eval trajectory-html RUN_DIR
```

Both HTML commands validate the strict JSON report and atomically replace a
deterministic `eval-report.html` or `trajectory-report.html`. Each output is a
portable single file that opens directly through `file://`; it embeds its CSS,
classic JavaScript, C19 guide, and report data and performs no network access.
Regenerate it at any time from the authoritative JSON.

Inspection reads report JSON only and never opens SQLite. JSON and HTML reports
are private local debugging artifacts: they contain gold, prompt inputs,
rendered prompts, model outputs, component traces, and complete candidate text.
They never contain credentials, authorization headers, ambient environment
values, or SQLite bytes. Provider failures retain only allowlisted
classification and typed transport status metadata; raw provider messages,
response bodies, headers, and metadata are excluded. Keep reports outside
every Git repository and do not publish them casually.

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

## C18 PrOntoQA

[`whetstone_envs.c18`][c18-source] provides deterministic fictional-ontology
deductive-entailment pools. An independent forward-chaining oracle derives each
label from public question and query text before an instance enters the pool.

```python
@verify(UNIQUE)
class DistractorMode(StrEnum):
    NONE = "none"
    RELEVANT = "relevant"

@dataclass(frozen=True, slots=True)
class DepthStratum:
    hops: int
    distractors: DistractorMode

@dataclass(frozen=True, slots=True)
class SplitPlan:
    internal_eval: int
    official: int
    held_out: int

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    generator_version: str
    seed_start: int
    n_per_stratum: int
    strata: tuple[DepthStratum, ...]
    split: SplitPlan
```

```python
DEFAULT_CONFIG: GenerationConfig
HARD_CONFIG: GenerationConfig
PROBES: ProbePair

def generate_pool(
    config: GenerationConfig = DEFAULT_CONFIG,
    *,
    n_per_stratum: int | None = None,
) -> TaskPool: ...

def default_split_sizes(
    pool: TaskPool,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> tuple[int, int, int]: ...

def build_manifest(
    pool: TaskPool,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> Manifest: ...

def score_gold(prediction: str, gold: str) -> int: ...
```

The frozen default and hard configurations use a pinned vendored PrOntoQA
generator. Their checked-in manifests pin the complete pool content; custom
validated configurations produce explicit, unpinned cohorts. Regeneration is a
repository operation:

```bash
uv run python scripts/regenerate-c18.py \
  --config default \
  --output src/whetstone_envs/c18/resources/default.manifest.json
```

## C19 MiniGrid state prediction

[`whetstone_envs.c19`][c19-source] generates balanced navigation, object-
manipulation, and door-interaction tasks on 5x5 and 8x8 MiniGrid worlds. Its
independent oracle simulates complete `LRFPDT` scripts from the public grid and
is checked against the pinned MiniGrid adapter after every action prefix.

```python
@verify(UNIQUE)
class Action(StrEnum):
    LEFT = "L"
    RIGHT = "R"
    FORWARD = "F"
    PICKUP = "P"
    DROP = "D"
    TOGGLE = "T"

@verify(UNIQUE)
class C19Fact(StrEnum):
    COORDINATE = "coordinate"
    HEADING = "heading"
    FRONT = "front"
    CARRYING = "carrying"
```

```python
@verify(UNIQUE)
class C19Scenario(StrEnum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    DOOR = "door"

@verify(UNIQUE)
class C19Size(IntEnum):
    SMALL = 5
    MEDIUM = 8
```

```python
DEFAULT_SPLIT_SIZES: tuple[int, int, int]
PROBES: ProbePair

def generate_pool(
    *,
    n_per_stratum: int = ...,
    seed_start: int = ...,
) -> TaskPool: ...

def build_manifest(
    *,
    n_per_stratum: int = ...,
    seed_start: int = ...,
) -> Manifest: ...

def derive_fact(grid_text: str, command: str, fact: C19Fact) -> str: ...
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
uv sync --locked --extra c18
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

Regenerate the canonical C19 manifest after an intentional generator change:

```bash
uv run python -m whetstone_envs.c19.regenerate
```

[c11-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c11
[c18-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c18
[c19-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c19
[instances-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/instances
[c22-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c22
[manifests-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/manifests
[pools-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/pools
[probes-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/probes
[scoring-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/scoring
[c23-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/c23
[optim-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/optim
[reporting-source]: https://github.com/danielle-rothermel/whetstone-envs/tree/main/src/whetstone_envs/reporting
