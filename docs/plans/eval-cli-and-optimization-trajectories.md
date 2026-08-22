# Evaluation CLI and optimization trajectory plan

Status: implemented

## Outcome

Add a terminal-first debugging tool to `whetstone-envs` that can:

- explain a task family and its terminology;
- run standalone evaluations for named prompt candidates;
- inspect summaries, failures, individual task rows, and paired candidate
  differences from a durable result artifact; and
- inspect an optimization trajectory, including the full text of every prompt
  candidate when explicitly requested.

The first supported family is C19. Both COPRO intent resolutions and GEPA's
persisted effect transcript are optimizer sources produced by
`scripts/run-optim.py`. The command is intentionally a scriptable CLI rather
than a server or interactive TUI. Rich owns terminal presentation; strict
versioned JSON remains the durable debugging boundary.

## Settled product decisions

1. The tool lives in `whetstone-envs`. It describes environment semantics and
   adapts the public `whetstone-ai` evaluation and optimization records without
   introducing a dependency on `whetstone-viewer`.
2. The installed `whetstone-eval` console script is the user-facing entry
   point. `scripts/eval.py` remains a thin source-checkout launcher. Both use
   subcommands so a saved run can be inspected without rerunning paid work.
3. Standalone eval runs write `runtime.sqlite` and `eval-report.json` under the
   existing off-repository run root. Optimization runs continue to write
   `runtime.sqlite` and `result.json` and additionally publish
   `trajectory-report.json` while the object store is open.
4. Inspection commands read only the versioned report JSON. They do not open
   `runtime.sqlite`: the current `dr-store` SQLite opener is not a read-only
   interface, and presentation must not mutate or reconstruct runtime state.
5. Candidate text is first-class debugging data. `trajectory
   --show-candidates` renders the exact string at the optimization run's
   declared mutation field. It never substitutes a candidate ID, digest,
   summary, or truncated prefix for the text.
6. Rich is an optional execution-tool dependency. Core task-family imports and
   the existing extra-free `whetstone_envs.optim` exports remain importable
   without Rich or `whetstone-ai`.
7. The implemented HTML renderer in `eval-html-report-design.md` consumes the
   same strict report models and defines no second result schema.

## Current implementation facts

The plan builds on these existing contracts rather than inventing another eval
engine:

- `prepare_c19_experiment` maps C19 pools, naive and ceiling probes,
  exact-match reward, and the internal/official split onto a Whetstone
  `Experiment`.
- `RuntimeEvalEngine.evaluate(EvalRequest)` produces content-addressed
  `EvalEvidence` or typed rejection/failure evidence.
- Successful `EvalEvidence` cites strict persisted output and trace records.
  Those records carry task/repeat coordinates, rendered prompts, raw outputs,
  scores, row states, provider failures, budgets, and component traces.
- `OptimResult` contains ordered exact step results. Each step contains its
  request candidates, proposed and accepted candidate records, intent
  resolutions, rewards, budgets, status, and failure evidence.
- A `Candidate` contains its full payload. An `OptimRun` names the exact
  `mutation_field`; C19 uses `prompt_template`. Candidate text can therefore be
  selected deterministically and validated as a strict string.
- `TaskRow` intentionally contains only task ID, prompt inputs, and gold. It
  does not carry seed or strata. Report construction must join evidence to the
  source `Instance`/`PoolSplit`; it must not parse semantic facts back out of a
  task ID.

## CLI surface

All commands use the existing optional execution environment:

```console
uv run --extra optim whetstone-eval <command> ...
```

### Explain the task

```console
whetstone-eval info c19
whetstone-eval info c19 --show-templates
```

`info c19` renders, in this order:

- the task objective and the fact that C19 is complete-script state
  prediction, not action planning;
- the public inputs (`grid`, `command`, and `question`) and private gold;
- the two grid sizes and three scenario families;
- every `LRFPDT` action and its preconditions/no-op behavior;
- the two-character grid tokens and coordinate convention;
- the four facts (`coordinate`, `heading`, `front`, and `carrying`) with their
  exact answer forms;
- exact-match normalization and binary scoring;
- the default 22-stratum, 352-instance pool and split roles; and
- candidate terminology:
  - **candidate**: one complete prompt template evaluated against a fixed task
    set;
  - **naive**: the intentionally sparse floor probe, not a claim that every
    model must fail;
  - **ceiling**: the known-good instruction-rich probe, not a guaranteed score
    upper bound.

`--show-templates` appends the full current naive and ceiling templates in
separate Rich panels. Task semantics come from a small C19-owned structured
description that references the actual enums and `PROBES`; the CLI must not
parse `.defs` files at runtime because they are repository documentation, not
packaged runtime data.

### Run an evaluation

```console
whetstone-eval run \
  --family c19 \
  --candidate naive \
  --candidate ceiling \
  --transport fake \
  --role internal \
  --split-sizes 20,20,0 \
  --repeats 1
```

Candidate selection is explicit and repeatable:

- `--candidate naive` and `--candidate ceiling` select the family probes.
- `--candidate-file NAME=PATH` adds a UTF-8 custom template under a stable
  nonblank candidate name.
- If neither option appears, the default is the pair `naive`, `ceiling`.
- Candidate names must be unique. Every custom template must satisfy the exact
  C19 render contract before any provider call.

Other run options mirror the existing optimizer CLI where their semantics are
identical: `--transport`, `--model`, `--run-id`, and `--output`.
`--role` accepts `internal` or `official`; held-out hashes are not an executable
split. `--repeats` maps to the Whetstone seed plan and must be positive.

The command prints a compact summary and the absolute run directory after
atomically publishing `eval-report.json`. A failed or rejected run exits
nonzero but still reports the runtime directory when durable evidence exists.
It never writes run artifacts inside any detected repository root.

### Inspect an eval report

```console
whetstone-eval summary RUN_DIR
whetstone-eval failures RUN_DIR
whetstone-eval task RUN_DIR TASK_ID
whetstone-eval compare RUN_DIR naive ceiling
```

Common filters are strict exact values, not fuzzy guesses:

```console
whetstone-eval failures RUN_DIR \
  --candidate ceiling \
  --scenario door \
  --size medium \
  --fact coordinate
```

The commands have these responsibilities:

- `summary`: run provenance, candidate totals, row accounting, overall score,
  and a scenario/size/fact table with both numerator and denominator.
- `failures`: incorrect, failed, missing, and invalid rows, grouped by outcome
  and ordered by the report's task/repeat plan.
- `task`: structured task material, full rendered prompt, raw and normalized
  output, gold, score, row state, provider failure, and component trace for
  each selected candidate/repeat.
- `compare`: paired classifications (`both correct`, `both wrong`, `A only`,
  `B only`, and execution mismatch), followed by the corresponding task rows.

Every inspection command supports `--no-color`. A later `--format json` may
expose filtered machine-readable rows, but the versioned report file—not
terminal scraping—is the initial machine interface.

### Inspect an optimization trajectory

```console
whetstone-eval trajectory OPTIM_RUN_DIR
whetstone-eval trajectory OPTIM_RUN_DIR --show-candidates
```

The default trajectory view groups resolutions into vertically stacked step
panels. Step-level status and budget accounting appear once, followed by one
wrapping detail block per evaluation resolution. This preserves repeated
evaluations without forcing long debugging values into a wide table:

- step and resolution ordinal;
- candidate ID and content hash prefix;
- base/parent candidate when resolvable;
- proposed and accepted disposition;
- eval outcome and typed classification;
- reward value or failure;
- budget consumed/remaining; and
- step status.

Long IDs, dispositions, messages, and other values wrap to the terminal width;
the trajectory view does not truncate or elide them.

`--show-candidates` then renders every distinct candidate once in first-seen
trajectory order. Each Rich panel includes step, candidate ID, exact content
hash, base candidate, and disposition. The body is the complete value of
`OptimRun.mutation_field` from the exact candidate payload.

Candidate rendering rules are binding behavior:

- require an actual string at the mutation field;
- preserve Unicode, whitespace, and newlines;
- construct `rich.text.Text` from the value instead of treating candidate text
  as Rich markup;
- wrap to terminal width but never truncate or elide content;
- show proposed candidates even when they were rejected;
- deduplicate only by exact candidate record reference; and
- fail with a candidate/ref-specific diagnostic when an exact record is
  malformed instead of falling back to `repr(payload)`.

The report also retains each candidate's full exact payload. The terminal view
selects the mutation field because that is the text being optimized, while a
future raw-record command may expose other payload fields.

## Durable report contracts

Persisted-format names and schema versions are explicit constants with golden
tests:

- `whetstone_envs.eval_report/v1`
- `whetstone_envs.trajectory_report/v1`

Strict frozen Pydantic models reject unknown fields, scalar coercion,
non-finite numbers, duplicate coordinates, and inconsistent references.

### Eval report

`eval-report.json` has five conceptual collections:

1. **Run**: schema version, run ID, family, transport/model identity, role,
   split sizes, repeats, dataset revision, graph/eval configuration identities,
   and source package version.
2. **Candidates**: stable ID, exact record reference, source
   (`naive`/`ceiling`/`custom`), full payload, and selected prompt template.
3. **Tasks**: task ID/hash, seed, ordered strata, prompt inputs, and gold from
   the source `Instance`.
4. **Observations**: candidate/task/repeat coordinate, rendered prompt, raw
  output, normalized output when present, exact binary score, row state, typed
  failure and safe allowlisted provider classification/transport fields, budget
  fields, submission result, and ordered component trace.
5. **Reported evidence**: exact aggregate/reward identities and values plus row
   accounting from Whetstone.

Tasks and candidates are not duplicated into every observation. Observation
coordinates must cover the exact planned matrix for successful evidence. The
projection recomputes candidate totals and C19 stratum summaries from
observations and verifies them against Whetstone's reported accounting and
aggregate before publication.

Private gold, prompts, generations, and candidate text are intentionally
present because this is a local debugging artifact. Raw provider messages,
response bodies, headers, and metadata are excluded; API keys, authorization
headers, and ambient environment values are never included.

### Trajectory report

`trajectory-report.json` contains:

- the exact terminal `OptimResult` reference and run identity;
- ordered step and resolution projections;
- every discovered exact candidate and its relation to a base candidate;
- proposed/accepted/final disposition;
- full candidate payload and selected mutation text;
- exact reward, evidence, and failure references;
- explicit per-step budget deltas, cumulative consumed/remaining state, and
  terminal status; and
- embedded candidate eval reports hydrated from each resolution while the
  optimizer's object store is open.

COPRO resolutions retain their ordered intent coordinates. GEPA evaluate
effects retain their original transcript invocation ordinals—including gaps
for intervening proposal effects—and map to the harness step stamped on the
exact recorded evaluation resolution. Replayed per-step search evidence does
not duplicate an already-recorded GEPA invocation.

The embedded eval records reuse the eval-report candidate/task/observation
models and bind the exact resolution candidate, evaluation result, and reward
references. Repeated evidence may reference shared task and candidate records
but keeps its own role, purpose, resolution coordinate, and then-current base
comparison snapshot. This makes future per-task trajectory regressions possible
without reopening the SQLite store.

Both report files are written through validated same-directory temporary files
and atomic replacement. A projection failure leaves `runtime.sqlite` and the
authoritative Whetstone result intact, removes no evidence, and causes the run
command to return nonzero with the projection error and durable run directory.

## Code ownership

The implementation should keep execution, projection, and presentation
separate:

- `pyproject.toml`: installed `whetstone-eval` console-script ownership.
- `scripts/eval.py`: five-line source-checkout launcher analogous to
  `run-optim.py`.
- `src/whetstone_envs/reporting/schema.py`: strict eval and trajectory report
  models and persisted-format constants; no Rich dependency.
- `src/whetstone_envs/reporting/publication.py`: validate and atomically write
  reports.
- `src/whetstone_envs/reporting/projection.py`: resolve public Whetstone records
  and join them to source instances; no terminal formatting.
- `src/whetstone_envs/reporting/cli.py`: argparse command dispatch with lazy
  imports for optional execution dependencies.
- `src/whetstone_envs/reporting/rich_views.py`: Rich tables, panels, diffs, and
  task detail rendering.
- `src/whetstone_envs/c19/_info.py`: structured C19 task descriptions backed by
  the actual C19 enums and probes; private implementation, not a second public
  task API.
- `src/whetstone_envs/optim/experiment.py`: prepare a C19 experiment together
  with its exact `PoolSplit`, so report projection receives seed/strata from
  authoritative `Instance` objects.
- `src/whetstone_envs/optim/run.py`: publish the trajectory projection before
  closing the run store.
- `tests/reporting/`: schema, projection, publication, CLI, and fixed-width Rich
  rendering tests.

Replace the internal experiment-construction path rather than maintaining two
ways to create the C19 experiment. A small frozen prepared-experiment value
should carry `Experiment` plus `PoolSplit`; update the optimizer and tests to
use it. Do not add a generic family registry while C19 is the only consumer;
CLI dispatch may name C19 explicitly.

Add an exact Rich pin to the `optim` extra and lockfile. Keep imports of Rich
and `whetstone-ai` below the optional CLI boundary so Python 3.12 core-package
and extra-free import checks remain valid.

## Implemented scope

The terminal CLI, both strict JSON report contracts, standalone C19 execution,
and optimization trajectory publication landed as one coherent implementation.
The implementation includes:

- Add report schemas, bounded validation, and atomic publication.
- Refactor C19 experiment preparation to retain the exact split.
- Add the standalone eval runner.
- Add `info`, `run`, `summary`, `failures`, `task`, and `compare`.
- Add Rich to the optional execution extra.
- Document commands and artifact privacy in the README/changelog/contracts.
- Prove fake naive/ceiling evaluation end to end without provider access.

- Project ordered `OptimResult` steps, candidates, rewards, failures, and
  budgets while the store is open.
- Reuse eval projection for every COPRO resolution and every unique GEPA
  transcript evaluation.
- Publish `trajectory-report.json` beside `result.json`.
- Add `trajectory` and the binding `--show-candidates` behavior.
- Add paired per-task gain/regression summaries when both candidate eval
  matrices are present.
- Update optimizer CLI documentation and end-to-end fake COPRO and GEPA
  coverage.

Python schema, Rich presentation, trajectory publication, self-contained HTML
presentation, and their tests share one ownership boundary. HTML consumes the
completed strict JSON contracts through their existing loaders.

## Verification

Focused tests must cover:

- strict report version and unknown-field rejection;
- exact persisted string literals and canonical candidate/task/observation
  ordering;
- task joins that preserve source seed and strata without parsing IDs;
- complete planned matrices, explicit failed/missing rows, and aggregate
  reconciliation;
- fake naive/ceiling eval publication and reloading;
- official versus internal role selection;
- custom UTF-8 template validation and duplicate candidate-name rejection;
- atomic publication preserving an existing report on validation/write
  failure;
- `info c19` definitions agreeing with action/fact/scenario/size enums and
  `PROBES`;
- compare buckets, including asymmetric execution failure;
- trajectory ordering, repeated evals, rejected candidates, failed terminal
  steps, and budget display;
- `--show-candidates` with multiline Unicode, braces, Rich-looking markup such
  as `[bold]`, very long lines, and no truncation;
- output paths inside the repository rejected from another current directory;
  and
- extra-free public imports on Python 3.12.

Capture Rich output with `Console(record=True)` at a fixed width and assert the
exported plain text. Do not test incidental ANSI escape sequences or terminal
color choices.

Final validation for each implementation PR:

```console
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
uv lock --check
uv build --clear --no-sources
uv run python scripts/check_distributions.py
git diff --check
```

Run the repository's CI-tier integration gate at the stack tip. Manually
inspect one fake eval report and fake COPRO/GEPA trajectories at both a narrow
and wide terminal width before relying on layout claims.

## Completion criteria

The immediate implementation is complete when:

- a user can understand C19 terminology with no source-code lookup;
- a fake or OpenRouter C19 eval can compare naive, ceiling, and custom prompt
  candidates and publish a strict local report;
- a user can move from an aggregate discrepancy to the exact prompt, output,
  gold, score, and trace row;
- a saved optimization run can show every ordered evaluation and the complete
  candidate text with `--show-candidates` without reopening mutable runtime
  storage;
- report corruption and unsupported versions fail loudly;
- no CLI rendering dependency leaks into core imports; and
- local artifacts remain outside repository trees and contain no credentials.

## Explicit deferrals

- Interactive TUI navigation and live-follow mode.
- A generic task-family plugin/registry.
- Animated C19 action-by-action state playback.
- Editing or rerunning candidates from the inspection command.
- Remote artifact hosting or a local web server.
- Statistical confidence intervals until runs use enough repeated binary
  observations for interval estimates to be meaningful.
