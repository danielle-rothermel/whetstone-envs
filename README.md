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
3.13 or 3.14 and pins published `whetstone-ai==0.1.10`.

## Installation

```bash
uv add whetstone-envs
```

Install C18's pinned generator dependencies when generating its pools:

```bash
uv add 'whetstone-envs[c18]'
```

Install the optimizer adapter extra when running COPRO, GEPA, MIPROv2, or
Codex against a task family. The extra pins published `whetstone-ai==0.1.10`
from PyPI:

```bash
uv add 'whetstone-envs[optim]'
```

The extra is Python 3.13/3.14 only. Every optimizer runs on whetstone-ai's
public surface, with no private imports and no adapter subclassing:

| Optimizer | Constructed by | Modes | Notes |
| --- | --- | --- | --- |
| `copro` | `configure_copro` + `CoproAdapter` | — | Proposal-only search over the mutation field. |
| `gepa` | `build_gepa_harness_adapter` | — | Reflection search over a required disjoint train/val split of the internal eval split. A run that finds no improvement reports the retained seed rather than substituting a candidate. |
| `miprov2` | `configure_miprov2` + `Miprov2Adapter` | `--demo-mode fewshot\|zeroshot\|ground_only` | Bootstraps demonstrations from a required disjoint train/val split of the internal eval split. Also binds an opening durable state (labeled trainset, proposal examples, RNG checkpoint). |
| `codex` | `configure_codex` + `CodexAdapter` | — | The foreign-agent arm: the Codex CLI searches out of process under dr-exec containment and reaches exactly one MCP tool, which evaluates a candidate on the internal split. One opaque step; whetstone runs no search of its own. macOS only — the containment profile is `sandbox-exec`. |

`--train-size` and `--val-size` are **required** for `--optimizer miprov2`
and `--optimizer gepa`, and refused on the optimizers that have no train/val
concept. They partition the internal split deterministically -- the trainset
is its first `--train-size` tasks and the valset the next `--val-size` -- so
the two sets are disjoint and reproducible from the spec alone. There is no
default: MIPROv2 bootstraps demonstrations from the trainset and GEPA writes
its reflections from it, while both score on the valset, so an overlapping
split would let memorization read as an in-search improvement. The study
protocol pins 44/44 of the internal 88.

The two optimizers do not take the same partition. **MIPROv2** needs only a
disjoint pair inside the internal split, so a partition that leaves tasks
unused is legal. **GEPA** builds its data registry from the whole internal
split and then requires the trainset and valset to cover it, so a GEPA
partition must sum to the internal size exactly; `run_optimizer` refuses a
partial one at spec validation rather than letting it fail after the run
directory exists. The protocol's 44/44 covers its internal 88, which is why
the study's GEPA arm satisfies the rule.

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

| Family | Placeholders | Scoring | Protocol splits | Family registered by | Splits pinned by |
| --- | --- | --- | --- | --- | --- |
| `c19` | `{grid}`, `{command}`, `{question}` | exact match on the whole reply | `88,132,440` | `optim/experiment.py` | `optim/study/spec.py` (`PROTOCOL_SPLIT_SIZES`) |
| `c18` | `{question}`, `{query}` | terminal `True`/`False` verdict, via `c18.score_gold` | `24,48,48` | `optim/c18_experiment.py` | `optim/c18_experiment.py` (`C18_PROTOCOL_SPLIT_SIZES`) |

The two columns are separate because the family and its protocol splits have
different owners. Registering a family says which tasks exist and how they are
scored; it does not choose how many of them each role gets.

For `c19` those two are different files, and the difference matters. The c19
generator's own `DEFAULT_SPLIT_SIZES` in `c19/generation.py` is
`(88, 132, 132)` — what the generator returns when nobody asks for anything.
The Step 10 study pre-registered a held-out split of **440**, because the
design's minimum detectable effect depends on it (`0.0622` at `tau^2 = 0.05`,
`K_REPEAT = 3`; see `whetstone-study plan`). `PROTOCOL_SPLIT_SIZES` in
`optim/study/spec.py` is the protocol's declaration of the three sizes and is
pinned by a golden test. A study manifest records them in its `splits` block,
and every stage reads them from there rather than from either constant.

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
  --family c19 --optimizer gepa --transport fake --split-sizes 2,2,0 \
  --train-size 1 --val-size 1
uv run --extra optim python scripts/run-optim.py \
  --family c19 --optimizer miprov2 --demo-mode fewshot \
  --transport fake --split-sizes 2,2,0 --train-size 1 --val-size 1
uv run --extra optim python scripts/run-optim.py \
  --family c18 --optimizer copro --transport fake --split-sizes 2,2,0 \
  --n-per-stratum 1
```

`--optimizer codex` is deliberately absent from that list — see below.

`--optimizer null-random` is null-A, the study's selection-on-noise control.
It is not a separate path: it drives COPRO's own search shape with an
uninformative proposer, so it honours `--copro-breadth` and `--copro-depth`
and produces the same evidence a COPRO run does — a result, a store, a
passing audit, priced cost rows. That is what makes it a control *for
selection*: a null that skipped slots or recorded a different shape of
evidence would be controlling for something else. Its perturbations are
seeded, and reproducible for a given `(--seed, --run-id)` pair.

```bash
uv run --extra optim python scripts/run-optim.py \
  --family c19 --optimizer null-random --transport fake --split-sizes 4,2,0
```

null-B (`null-identity`) is deliberately *not* a runner optimizer: it
proposes nothing, so it has no search to drive and no fidelity invariant to
audit. The study harness records it directly.

**The study's two null arms run the same way this section describes.** A
study's null-A arm dispatches `run_optimizer(optimizer="null-random", …)`
like any other arm — same internal split, same proposal budget, same
result, audit, and cost evidence — so its number is the product of a real
selection over real evaluations and the arm controls for
selection-on-noise. `whetstone-study plan` therefore prices it at COPRO's
search shape. Null-B stays the seed measured through the report harness,
and `plan` prices it at one official pass plus one held-out pass.

### Real transport smoke rungs

Before any multi-run spend, one rung per optimizer arm runs end to end on
the real OpenRouter transport over toy splits, so mock-only assumptions
surface before an experiment depends on them. The rungs live in
`tests/real_transport/`, carry the `real_transport` marker, and are
deselected unless `WHETSTONE_ENVS_REAL_TRANSPORT=1` is set. Opting in
without `OPENROUTER_API_KEY` fails loudly rather than skipping — a silent
skip would let a run that called nothing report success.

```bash
mise exec -- bash scripts/check-real-transport.sh
mise exec -- bash scripts/check-real-transport.sh -k rung1
```

The script writes a transcript and a rung table with per-rung ledgered cost
under `~/drotherm/data/whetstone-envs/real-transport/<timestamp>/`. It
spends real money, so it is never part of `pre-check.sh` or the default CI
run; the `Real transport smoke` workflow is `workflow_dispatch`-only.

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
environment variable is unset and clears it, so no ordinary test can opt
in; the one exception is the real-CLI ladder below, which must also set its
own separate `WHETSTONE_ENVS_REAL_CODEX=1`.

A study stage authorizes the same spend the same way, through
`whetstone-study run --allow-real-codex`. The flag reaches the study's
optimizer runner, which forwards it onto the Codex arm's `RunSpec` and onto
no other arm's; the environment variable remains the other half. It is a
**run-time spend authorization, not part of the design**: it is not a field
on `ArmSpec`, it is not recorded in the manifest, and it does not enter the
pre-registration hash, so two runs of one design pre-register identically
whether or not the operator was allowed to bill a session.

Without both halves, a Stage 1 or Stage 2 whose design names the Codex arm
is refused **before any arm runs**. That ordering is the spend-safety
property: the refusal inside `run_optimizer` arrives on the Codex arm's own
turn, by which point every arm ahead of it has already been paid for.

```bash
WHETSTONE_ENVS_ALLOW_REAL_CODEX=1 uv run --extra optim \
  whetstone-study run --study-dir STUDY_DIR --stage stage1 --allow-real-codex
```

A real run therefore requires all of: the opt-in variable, the flag, a live
authenticated Codex session, macOS (the containment profile is
`sandbox-exec`), provider spend for the task-model evaluations, and a go
from Danielle.

CI's claims about the arm rest on the scripted stand-in, which speaks real
MCP to the real evaluation server and so exercises the production
admission, lease, evaluation, and ledger path; only the agent's own
decisions are scripted. The real CLI is covered separately by the ladder
below.

```bash
WHETSTONE_ENVS_ALLOW_REAL_CODEX=1 uv run --extra optim \
  python scripts/run-optim.py \
  --family c19 --optimizer codex --transport fake --split-sizes 2,2,0 \
  --codex-capacity 8 --allow-real-codex
```

### Running the Step 10 study

The study's design is committed, not assembled at the command line.
`whetstone_envs.optim.study.protocols` pins every pre-registered value --
the splits, the models, the arm list, each arm's control shape -- and
`whetstone-study init` mints the pre-Stage-0 `study.json` from it, so two
initialisations of the same protocol produce the same manifest and a
design that disagrees with the module is one somebody edited.

```bash
uv run --extra optim whetstone-study init \
  --study-dir STUDY_DIR --protocol step10-c19
uv run --extra optim whetstone-study plan --study-dir STUDY_DIR
uv run --extra optim whetstone-study run \
  --study-dir STUDY_DIR --stage stage0 --transport openrouter
# read the Stage-0 gate in the output, then:
uv run --extra optim whetstone-study run \
  --study-dir STUDY_DIR --stage stage1 --transport openrouter \
  --allow-real-codex
uv run --extra optim whetstone-study run \
  --study-dir STUDY_DIR --stage stage2 --transport openrouter \
  --allow-real-codex
uv run --extra optim whetstone-study leakage-check --study-dir STUDY_DIR
uv run --extra optim whetstone-study report \
  --study-dir STUDY_DIR --out REPORT_DIR
```

`init` regenerates the pinned population and records each split's task
hashes, the pool manifest's content hash, and the sha256 of the protocol
document itself, so the manifest names the revision of the
pre-registration that was actually in force. **The registered text ships
in the package**, at
`src/whetstone_envs/optim/study/protocol_docs/step10-c19-protocol.md`,
byte-identical to the durable authoring copy and pinned by a golden
digest -- so the digest a manifest records is checkable from any checkout
rather than naming a file on one machine. `--protocol-doc` points at a
different copy of that document; the digest always comes from the file
read.

The manifest records no separate assignment digest: Step 10's authorising
assignment *is* the protocol document, so `assignment_doc_sha256` is
absent and the report says so, rather than carrying the digest of a fixed
marker string that would verify nothing. `init` refuses to overwrite an existing `study.json`, because a
second initialisation over a study that already holds evidence would reset
a design that evidence refers to.

`--toy` authors the sized-down variant of the same protocol. Only the
sized fields differ -- the splits, the per-arm train/val partition, the
population size and seed, the MIPROv2 minibatch, and COPRO's breadth and
depth -- and a golden test asserts every other field matches the real
design, so a toy cannot rehearse a study the real one does not run.

`--study-id` names a rehearsal so its artifacts cannot be mistaken for the
study's. It is refused when it names a *registered protocol* on an
invocation that is not that protocol at full size with no projection in
effect: a toy or a `--without-codex` run initialised as `step10-c19` would
leave every artifact downstream citing the pre-registration by name while
holding a smaller design.

The opt-ins each stage needs, by name:

- **`--transport openrouter`** and a non-blank `OPENROUTER_API_KEY` for
  every stage that spends. `--transport fake` is the default and reaches
  no provider.
- **`--allow-real-codex` plus `WHETSTONE_ENVS_ALLOW_REAL_CODEX=1`** for
  any stage whose design declares the Codex arm. Both halves are required
  and are checked against the design *before any arm runs*, so a
  Codex-bearing stage does not pay for the other arms and then discover it
  cannot finish. This is a run-time spend authorization and never enters
  the pre-registration hash.
- **`--replace-design`** only on `stage0`, to record a deliberate
  amendment over an already-pinned design.
- **`--discard-stale-runs`** to discard a run directory whose own
  artifacts say it belongs to another invocation.

Stage order is enforced rather than documented: `stage1` and `stage2`
refuse without a recorded design, and `stage2` refuses without a `stage1`
whose call-count gate passed. Run `leakage-check` before `report` -- it
writes the L1-L5 verdict the report reads, and a report generated without
it is marked invalid.

`--without-codex` authors the design with the Codex arm dropped. It exists
for fake-transport rehearsals: the Codex guard fires on the *design*
whatever transport the task model is on, so a rehearsal of the rest of the
study drops the arm rather than stubbing it. What it produces is a
strictly smaller design, not the pre-registration, and it says so on every
axis a reader checks: its `study_id` gains a `-without-codex` suffix, its
`models.codex_agent_model` records the omission instead of naming an agent
no arm will reach, the manifest carries
`design_projection: "without-codex"`, and the report's headline is
prefixed with the projection so its numbers cannot be read as the study's.

#### Storage the study needs

Budget roughly **4.6 KB of `runtime.sqlite` per evaluated row**, dominated
by `eval_outputs` and `eval_component_traces`. At the eight-arm design's
row count that is **≈0.4-0.5 GB** for the study's own store, before the
per-run directories the arms leave beside it. This is an operator note,
not a gate: the number that has actually bitten is GEPA's, whose
unpinned `auto` budget produced a 1.73 GB store from a single run --
which is why `gepa_max_metric_calls` is pinned at 200.

### Study transports and the per-stage ledger

A study stage evaluates on one of two transports, named per invocation:

```bash
uv run --extra optim whetstone-study run \
  --study-dir STUDY_DIR --stage stage0 --transport openrouter
uv run --extra optim whetstone-study plan --study-dir STUDY_DIR
uv run --extra optim whetstone-study report --study-dir STUDY_DIR --out OUT_DIR
```

`--transport fake` is the default and reaches no provider: the transport
answers from the experiment's own gold, so the numbers are evidence about
the plumbing and about nothing else. `--transport openrouter` spends, and
takes its key from `OPENROUTER_API_KEY`. A missing or blank key is refused
**before the store is opened, the pool is generated, or any engine is
bound**, so an unauthorized paid run leaves no half-initialized study
directory behind and exits non-zero. The key is checked for presence only
and never reaches an error message.

Models come from the manifest's own `models` block — `task_model` for the
evaluations and `proposer_model` for the optimizers' proposal route — so
selecting a transport does not select a model.

**What the transport bound is recorded too.** `models.provider_calls` holds
one record per transport a stage has bound, naming the route it resolved
and every request control — temperature, top-p, token limit, reasoning,
seed — set or not. A control the study did not set reads `provider
default` rather than being omitted, because "left to the provider" is a
real state with a real bill: it is why the toy Stage 0 spent thousands of
reasoning tokens per call. It is recorded, not hashed, like the transport
itself, and the report prints it in the design section.

**The transport is an invocation property that the manifest records.** Like
`--allow-real-codex` it stays out of the pre-registration hash, so two
studies differing only in it pre-register identically. Unlike it, every
stage writes a `stages` entry naming the transport it ran on, because a
stage run on `fake` and a stage run on `openrouter` are different evidence
for the same claim.

That record is what makes the cross-stage rule checkable: **a study whose
Stage 0 calibrated its anchors on one transport refuses to run Stage 1 or
Stage 2 on the other**, before any arm runs. Every held-out delta is paired
against those anchors, so a cross-transport subtraction is not a comparison
and there is no flag that makes it one — a toy study that wants the other
transport re-runs Stage 0 on it under `--replace-design`.

The refusal checks three things, because a study can hold evidence from two
transports in three ways: Stage 0's transport, **the target stage's own
recorded transport**, and **the transport of every surviving arm run**. A
run records its own transport, because a resumed stage keeps the runs an
earlier invocation paid for — so a stage row can agree while the runs
beneath it do not.

`stage0 --replace-design` onto a *different* transport than the recorded
Stage 0 drops the Stage 1 and Stage 2 records, their arm runs, the
selections over those runs, the held-out claims and rows, the pilot's
call-count gate, and the `leakage_check` verdict computed over those runs:
the design changed and the evidence came from another experiment. Nothing
is deleted silently — the drop is recorded as an `amendments` entry naming
every run id it removed and every run directory it orphaned, and the report
surfaces it. The arms themselves survive with empty run lists, because an
arm is part of the design rather than evidence for it. **Paid evidence is
never discarded automatically**: if any dropped run was measured on a
billed transport the command refuses instead, before the calibration
spends, and names the recovery (archive the study directory and calibrate
the new transport in a fresh one).

The drop clears the manifest; it does not clear the disk. Run directories
are named deterministically from arm and seed — which is what makes a
crashed stage resumable — so the dropped runs' directories remain under
exactly the names the replacement stage will compute. **An arm stage
therefore reuses a run directory only when the directory's own artifacts
say it is the run this invocation would produce**: the transport, family,
model, and run id recorded in its trajectory report. Otherwise the stage
refuses, naming the directory and the recovery, rather than skipping
`run_optimizer` and recording an old free run as this stage's paid one. A
directory that records no readable identity is refused on the same grounds.
`whetstone-study run --discard-stale-runs` authorizes discarding such a
directory instead; it is off by default, because the directory it would
remove may be paid evidence, and a matching directory is still reused
either way.

Each stage record also carries what the stage spent, one entry per provider
role, in the same shape a run reports. The two kinds of stage measure it by
the route they spend by: Stage 0's anchors evaluate through the engine, so
their spend is re-derived from the persisted output rows the evaluations
left behind; an arm stage spends through optimizer runs, so its total is
the fold of the per-run records those runs already carry. Both are
measured, never accumulated while the stage ran. An arm stage's row is
merged rather than replaced when it re-runs: a stage that crashed after its
manifest write executes nothing on resume, so replacing its row would
discard the bill it already paid, and the fold of the existing spend with
this invocation's keeps a stage row from ever shrinking. `whetstone-study plan`
prints that ledger beneath its estimated budget, `run` echoes it, and the
report prints it beside each stage's transport. A role with any unpriced
call reports no USD total at all rather than a partial sum that would look
authoritative.

An empty spend record means one of two opposite things, and the ledger
never conflates them. A **fake-transport** stage reports `no provider
reached (fake transport)`: its rows are real rows and the shared row rule
counts them as billable-and-unpriced, which is right for a provider row and
wrong for a stage that called nobody. A **paid** stage that recorded no
spend reports `UNLEDGERED` — it reached a provider and lost track of what
it bought, so its bill is unknown rather than zero, and that is a defect to
act on rather than a free stage.

**What the ledger covers.** A stage spends by two routes and its row is
the sum of both. Its arms spend through optimizer runs, each of which
projected its own per-role bill, and the stage total is the fold of those
records. Its **reporting pass** — official-selection scoring, the held-out
evaluations, and the anchors' re-measurement — reaches the provider through
the evaluation engine outside any run, so those evaluations are priced one
record per role per evaluation from their own persisted rows and folded
onto the same stage row. Both routes read the numbers back out of evidence
rather than accumulating them, so a stage total and a run total are the
same kind of fact under the same honesty rules.

### Reading the ledger before authorizing a paid stage

**Measure before you authorize.** On the toy Stage 0, `gpt-5-nano` spent
roughly **4,500 completion (reasoning) tokens per call** — about 7× the
planning figure the protocol budgeted — at a measured **$0.00168 per
call**. Carried forward at that rate, Stage 0 costs roughly **$9** and
Stage 2 roughly **$210**, against a protocol estimate built on the smaller
per-call figure.

The rate is a property of the model and the prompts, not a fixed constant,
so it is not a number to plan from twice. Run Stage 0, then read
`whetstone-study plan`'s **measured** per-stage ledger — not the estimated
budget above it — and authorize Stage 1 against what Stage 0 actually
spent. Remember that the measured total excludes official-selection and
held-out scoring, so treat it as a floor.

### The real-CLI ladder

`tests/real_codex/` drives the **real** Codex CLI against a live
subscription session, one rung at a time, through the same
`run_optimizer` entry point the study uses. It exists because the scripted
fake CLI structurally cannot cover four things: it never validates the
agent-facing output schema, never routes through the CLI's tool host, is
*handed* the tool arguments a real agent has to derive, and never proves
that the out-of-process MCP server rebuilds this run's experiment from
`EnvsCodexRuntimeConfig`. A rebuild that landed on a different Eval Config
would have every tool call refused after admission — the agent would burn
its whole capacity and the Step would still terminalize.

The task model stays fake on every rung (`--transport fake`), so a ladder
run spends Codex agent turns and no eval-provider credit.

It is opt-in twice over and never runs in CI: the `real_codex` marker is
deselected by default through `addopts`, and every rung is skipped unless
`WHETSTONE_ENVS_REAL_CODEX=1`. Once opted in, an unmet precondition —
non-macOS host, missing `sandbox-exec`, missing binary, no session, or a
missing spend opt-in — is a loud failure rather than a skip, because pytest
exits 0 on a fully skipped session. The runner script does not trust that
exit status either: it reports `all rungs passed` only when every rung
reached the table with a PASSED verdict and the table holds as many rungs as
the ladder collects, and otherwise reports `ladder not fully observed` and
exits 1.

No rung live-skips. An agent that decides the seed template is best may say
so either by returning no selection or by *selecting* a call whose template
equals the seed; whetstone-ai 0.1.9 ([#138]) records both as
`seed_retained`, so agent taste no longer decides whether a rung is
observed. The pinned 0.1.10 also runs the agent against a per-run scratch
`HOME` and quotes the CLI's own error items when a session fails ([#140]),
so a failure names its cause rather than the first symptom.

[#138]: https://github.com/danielle-rothermel/whetstone-ai/pull/138
[#140]: https://github.com/danielle-rothermel/whetstone-ai/pull/140

```bash
scripts/check-real-codex.sh              # whole ladder, stop at first break
scripts/check-real-codex.sh -k rung3     # one rung
```

The script writes its transcript and a rung table to
`~/drotherm/data/whetstone-envs/real-codex/<timestamp>/`.
`.github/workflows/real-codex.yml` runs the same script on a self-hosted
macOS runner, `workflow_dispatch`-only.

The Codex *agent's* model is not the run's task model: the task model is an
OpenRouter route the Codex CLI cannot run at all, and a subscription
session refuses it outright. `--codex-model` names the agent's; unset, it
is `CODEX_DEFAULT_AGENT_MODEL`.

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
