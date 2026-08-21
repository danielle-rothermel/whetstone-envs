# Self-contained evaluation and trajectory HTML design

Status: future product design; not part of the initial CLI implementation

## Purpose

Provide a portable, offline visual report for the same C19 evaluation and
optimization debugging records consumed by the terminal CLI. The report should
make aggregate behavior legible, but its primary job is to shorten the path
from “this run is surprising” to the exact candidate text, task material,
rendered prompt, model output, score, and failure provenance that explain it.

The HTML renderer does not define another result schema. It consumes the
versioned `eval-report.json` and `trajectory-report.json` contracts from the CLI
plan and embeds the validated report in one self-contained HTML file.

## Product principles

1. **Candidate text stays visible.** A score without the prompt candidate is
   insufficient for optimizer debugging. Candidate text and diffs are primary
   views, not hidden raw-record details.
2. **Aggregate to row is one interaction.** Every chart or matrix cell links to
   the exact contributing observations.
3. **Counts accompany rates.** A colored `80%` cell also says `8/10` and exposes
   failed/missing rows; color never replaces accounting.
4. **The report is an immutable local artifact.** It makes no network requests,
   mutates no run data, and requires no server.
5. **Task semantics travel with results.** The report includes the same C19
   terms and naive/ceiling definitions as `eval.py info c19`.
6. **Structured data remains authoritative.** HTML is a presentation of a
   validated report, not a persistence or interchange boundary.

## Artifact shape

Standalone evaluation:

```text
<run-dir>/
├── runtime.sqlite
├── eval-report.json
└── eval-report.html
```

Optimization:

```text
<run-dir>/
├── runtime.sqlite
├── result.json
├── trajectory-report.json
└── trajectory-report.html
```

The Python renderer validates the source report, serializes it as strict JSON,
and embeds it in a non-executable `application/json` script element. The
serializer must neutralize HTML closing sequences and verify round-trip bytes
before publication. CSS and JavaScript are inline and versioned with the Python
package. No CDN, analytics, fonts, images, or external source maps are loaded.

The renderer writes a validated same-directory temporary file and atomically
replaces the final HTML. A failed render leaves the JSON report intact and does
not replace an older valid HTML artifact.

## Information architecture

### Persistent header

The top bar identifies what is being viewed before showing a metric:

- run ID and terminal state;
- C19 family and dataset revision;
- evaluation role and task/repeat counts;
- transport, provider, and model;
- candidate count;
- aggregate status, including incomplete or failed runs; and
- copyable exact run/eval/graph identities in a provenance popover.

A local-artifact notice states that the file contains private gold, full model
outputs, and prompt candidate text and should not be published casually.

### Task guide drawer

An always-available “About C19” drawer presents:

- state-prediction objective versus action planning;
- grid token legend;
- `LRFPDT` action rules and no-op behavior;
- scenario and size definitions;
- coordinate convention;
- fact definitions and exact answer forms;
- scoring/normalization rules; and
- candidate, naive, and ceiling definitions.

The drawer has a “show templates” control that displays the complete current
naive and ceiling templates. It uses the same structured C19 info source as the
CLI so terminology cannot drift independently.

## Evaluation report layout

### 1. Candidate rail

A left rail lists every candidate with:

- stable candidate name;
- source (`naive`, `ceiling`, custom, or optimized);
- exact hash prefix with copy action;
- overall score and row accounting; and
- visibility/compare selection.

Selecting a candidate updates the summary and task table without navigating
away. Selecting exactly two candidates enables paired comparisons.

A sticky candidate inspector beside the rail has two tabs:

- **Full text**: the exact complete prompt template, preserving whitespace and
  never truncating; and
- **Diff**: a line-oriented diff against the other selected candidate or its
  optimization parent.

Candidate text is rendered as text, never injected as HTML. Long lines wrap by
default with a user toggle for horizontal scrolling. A copy button copies the
exact raw value, not a visually wrapped representation.

### 2. Outcome summary

Compact cards show:

- scored mean plus passed/planned count;
- completed, failed, missing, and invalid rows;
- tasks and repeats;
- provider-error count;
- deadline/guard-timeout status; and
- cost/token facts when the producer recorded them.

Incomplete evidence uses a visibly different status from a numeric zero. A
missing aggregate never renders as `0%`.

### 3. C19 stratum matrix

The principal overview is a small multiple for each grid size. Rows are the
three scenario families; columns are coordinate, heading, front, and carrying.
Non-applicable navigation/carrying cells are marked `N/A`.

Each applicable cell contains:

- `passed/planned`;
- percentage when complete;
- failed/missing indicators; and
- paired delta when two candidates are selected.

Color encodes score or paired delta, while text and icons encode the same
meaning for users who cannot distinguish the color. Clicking a cell filters
the task table to its exact observation coordinates.

### 4. Paired outcome buckets

When two candidates are selected, a horizontal summary shows:

- both correct;
- candidate A only;
- candidate B only;
- both wrong; and
- execution mismatch.

Each bucket is clickable. “Execution mismatch” remains separate from semantic
score disagreement so provider or infrastructure failures do not masquerade as
prompt regressions.

### 5. Task table

The task table defaults to surprising rows first:

1. execution mismatch;
2. candidate regression;
3. both wrong;
4. candidate gain; and
5. both correct.

Filters cover candidate, scenario, size, fact, row state, score, provider
failure, and literal task ID. Columns include task, strata, repeat, candidate,
normalized output, gold, score/state, and a prompt/output preview. Sorting is
stable and always falls back to planned task/repeat order.

No filter silently changes denominators. The page shows both the filtered row
count and the complete run accounting.

### 6. Task detail drawer

Opening a row presents:

- monospaced ASCII grid with spaces preserved;
- action script and question;
- task ID/hash, seed, ordered strata, and repeat;
- selected candidate's complete template;
- fully rendered prompt;
- raw generation and normalized answer;
- gold and exact score;
- row state, failure code, finish reason, provider error, and budgets;
- submission result; and
- ordered component trace with each step's exact inputs and outputs.

For a paired comparison, candidate A and B appear side by side where viewport
width permits and in labeled tabs on narrow screens. The initial design does
not animate MiniGrid transitions; the public ASCII task is the source display.

## Optimization trajectory layout

The trajectory report adds three linked surfaces above the ordinary eval
layout.

### 1. Step timeline

An ordered horizontal or vertical timeline shows each step's:

- index and status;
- proposed and accepted candidate counts;
- evaluation resolutions;
- reward values or typed failures;
- cumulative/remaining budget; and
- terminal selection.

Repeated evaluations remain separate points. Hover/focus reveals exact
resolution coordinates rather than collapsing them into an unexplained latest
score.

### 2. Candidate lineage

A compact lineage graph connects each candidate record to its exact base
candidate. Node styling distinguishes request seed, proposed, rejected,
accepted, and terminal candidates. The graph derives only from exact record
references; unresolved bases appear as explicit external roots.

Selecting a node updates the candidate inspector, timeline highlight, eval
summary, and task rows. The complete candidate text is visible immediately.

### 3. Change diagnosis

For a candidate and its parent, the report shows:

- full prompt diff;
- overall and per-stratum reward delta;
- fail-to-pass and pass-to-fail task counts;
- execution mismatches;
- budget spent to propose/evaluate the change; and
- links to every changed task row.

This view does not infer causal claims from correlation. It says which observed
rows changed alongside the prompt mutation and preserves all recorded failure
states.

## Visual language

The report should feel like a debugging instrument, not an experiment
marketing dashboard:

- neutral off-white or dark-neutral canvas with high-contrast text;
- restrained semantic colors for success, regression, failure, missing, and
  selection;
- monospaced type for IDs, grids, templates, prompts, and outputs;
- tabular numerals for counts and scores;
- compact density with generous whitespace inside task/candidate text panels;
  and
- no gradients, animation, or decorative chart chrome that obscure exact
  values.

Status is always expressed by text or icon in addition to color. Focus states,
table headers, drawers, tabs, and copy controls are keyboard accessible. The
document respects reduced-motion and light/dark browser preferences while
preserving semantic contrast.

## URL and state behavior

The file works under `file://`. Filter and selection state lives in the URL
fragment so a copied local URL can reopen the same candidate, task, step, and
filter state without rewriting the artifact. Invalid fragment coordinates are
ignored with a visible notice rather than selecting a nearby row.

The page performs all filtering client-side over the embedded bounded report.
It makes no fetch requests. A future report-size limit must be defined from
measured C19 runs before adding pagination or split assets; the canonical
352-task pool should first be tested with realistic candidates and repeats.

## Security and privacy

The report intentionally contains private evaluation data. Generation must:

- emit a prominent private-artifact notice;
- use no remote resources or telemetry;
- render every prompt/output/candidate field through text nodes;
- apply a restrictive content security policy compatible with the embedded
  script and styles;
- avoid embedding API keys, headers, ambient environment, or SQLite bytes;
- preserve typed provider errors without rendering arbitrary fields as HTML;
  and
- refuse unsupported report schema versions.

Opening the HTML has no effect on `runtime.sqlite`, JSON reports, prompt cache,
or provider state.

## Implementation direction

After the CLI report contracts stabilize:

1. Add an HTML renderer under `whetstone_envs.reporting` that consumes the same
   validated models as Rich views.
2. Keep a small repository-owned HTML shell, CSS, and vanilla JavaScript asset
   set; embed those exact assets at generation time. Do not introduce a web
   server or frontend framework for the first report.
3. Add `eval.py html RUN_DIR` and `eval.py trajectory-html RUN_DIR`, each
   printing the absolute generated file.
4. Reuse the C19 info model and candidate/eval comparison derivations rather
   than recomputing semantics in JavaScript. JavaScript owns interaction and
   filtering, not evaluation math.
5. Validate exact report-to-DOM accounting in browser tests and keep a small
   set of screenshot fixtures for wide and narrow layouts.

The HTML work should be one coherent PR after the two CLI implementation PRs.
It should not change the report JSON merely to simplify DOM code unless a real
missing debugging fact is demonstrated.

## Acceptance criteria

The HTML design is realized when:

- either report opens directly from disk with no network access;
- a user can identify the run, task, candidate, and completion state without
  opening raw JSON;
- full candidate text is visible and copyable without truncation;
- every aggregate/stratum/trajectory selection reaches its exact contributing
  task rows;
- naive, ceiling, candidate, scenario, action, and fact meanings are available
  in the report;
- failed and missing observations never appear as ordinary wrong answers;
- paired gains and regressions preserve denominators and execution mismatches;
- optimization lineage uses exact candidate/base references;
- prompt, output, and trace text cannot execute as markup; and
- the generated HTML is replaceable from the JSON source with deterministic
  substantive content.

## Non-goals

- A replacement for `whetstone-viewer`.
- A hosted experiment-tracking service.
- Live streaming or watching an active run.
- Editing candidates or launching provider calls from the report.
- Reconstructing data directly from SQLite in the browser.
- A generic visualization grammar for every future environment.
- MiniGrid animation in the first HTML version.
