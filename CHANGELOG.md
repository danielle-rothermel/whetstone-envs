# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.12] - 2026-08-26

### Fixed

- **A single unscoreable row no longer collapses an evaluation's whole
  recomputed aggregate.** `two_stage_task_mean` reduces per task and then
  across tasks to mirror whetstone-ai's `unweighted_task_mean` bit for bit,
  but it required *every* row to carry a score and returned `None`
  otherwise. Upstream does not: it reduces each task through `aggregate()`
  under this package's aggregation config (`reduction=mean`,
  `missing_data=skip`), which means over the **present** rows only, so a
  task whose 3 repeats land 2 scored and 1 invalid contributes the mean of
  those 2 at full weight. Stage 2's first evaluation with partial presence
  -- `miprov2-seed2004`, internal role, `planned=132`, `present=131`,
  `invalid=1` from one provider refusal on task 17 seed 1 -- therefore
  recomputed `None` against a persisted `0.18181818181818182`, and
  `_success_projection` raised "recomputed aggregate disagrees with
  evidence" into a `DurableRunError`. Presence, not completeness, now
  selects the addends at every site that consumes the reduction. A task
  with *zero* present rows is dropped from the across-task mean rather
  than contributing a zero, matching upstream's `NOT_APPLICABLE` (all
  invalid) and `ZERO_DENOMINATOR` (all missing/failed) paths, both of
  which leave the task out of the outer mean while keeping its position;
  the overall result is `None` only when nothing anywhere was scored.
  `StratumSummary` correspondingly requires a score iff some row was
  scored, rather than iff the stratum was complete. #41 closed the
  "no test above `num_seeds=1`" gap and left "no test below full
  presence" open; a `{complete, one-invalid, one-missing,
  zero-present-task} x {K=1, K=3}` matrix now closes both dimensions,
  and the fix is verified to reproduce seed2004's persisted aggregate
  bit for bit from its own rows.

## [0.2.11] - 2026-08-25

### Fixed

- **The Codex arm now gets a wall budget big enough to spend its
  pre-registered eval cap.** D2 admits the Codex agent 8 evaluate-calls, and
  on the c19 design one such call scores 88 internal tasks at `K_REPEAT = 3`
  and costs ~120 s measured -- so the cap needs ~960 s of evaluation before
  the agent's own selection turns. The study forwarded the cap onto every
  Codex `RunSpec` but never a wall, leaving `whetstone-ai`'s
  `CODEX_DEFAULT_WALL_SECONDS` of **600 s** in force: `600 < 8 x 120` made
  the pre-registered cap structurally impossible to spend. The Stage-1 Codex
  arm's real agent session completed 6 of its 8 calls, was SIGKILLed at the
  wall, and terminalized with `codex_wall_budget_exceeded`. A new pinned
  `CODEX_WALL_SECONDS = 1500.0` -- the ~960 s floor plus headroom for
  selection turns and slower-than-mean calls -- is forwarded beside
  `CODEX_EVALUATE_CALL_CAP` from `bound_stage_environment`. Recorded rather
  than hashed, unlike the cap: the wall changes how long the arm may take to
  buy the same 8 calls, not what it buys.
- **A run that terminalizes is now recorded as a failed run instead of
  aborting its whole stage.** §3.9 pre-registers that a run exceeding its
  wall or eval-budget cap "terminalizes with `terminal_failure` and is
  recorded as a failed run, not retried silently", and the recording half was
  never implemented: `ArmRunResult` and `RunRecord` made a failed run
  unrepresentable, and `_result_from` reached for a terminal prompt
  unconditionally, so the `StageError` it raised escaped the stage and
  discarded every sibling run's and every later arm's already-paid evidence
  -- which is what took the Stage-1 nulls down with the Codex arm above.
  `RunRecord.terminal_failure` (schema **v14**) now records the run's own
  failure code and message; such a run contributes no candidate, so selection
  skips it, no held-out claim is issued for it, and its spend and artifacts
  stay on the manifest. The arm degrades to `not validated` -- its arg-max ran
  over fewer runs than the design pre-registered -- and the stage runs its
  remaining seeds and arms. A run with **no** declared failure and no terminal
  prompt still raises: that is an unexplained absence and a real invariant
  violation, not an outcome to file away.

## [0.2.10] - 2026-08-25

### Fixed

- **A COPRO run that honestly kept its own seed no longer fails its audit
  and unclaims the arm.** whetstone-ai 0.1.16 fixes a day-one COPRO bug:
  when the run's seed ties or wins the terminal ranking, COPRO now returns
  `seed_retained=True` with a `retained_candidate_ref` and zero accepted
  candidates -- the mechanism GEPA and MIPROv2 already used -- at both of
  its terminal emission points, the ordinary finalize and the early
  terminal a round takes without a valid proposal. Ties are ordinary rather
  than exotic, because an exact-match reward over `N` internal tasks
  quantizes to `k/N`. Against the previous audit that shape hard-failed
  three invariants, and `audit_passed=False` demotes the whole arm to
  `VERDICT_NOT_VALIDATED` -- so a run reporting the truthful outcome
  "nothing beat the starting point" would have unclaimed its arm.
  `copro_best_so_far` was the blocker: a retaining step accepts nothing, so
  the finalizing-step search came up empty and the invariant failed
  unconditionally. It now judges the retention instead of waving it
  through, requiring the retained ref to be the run's declared seed and the
  seed's own measured reward to equal the maximum the run recorded -- a
  vacuous pass would have retired best-so-far precisely where it does its
  only interesting work. `copro_breadth_per_depth` and `copro_depth_steps`
  gain the same declared-retention exemption alongside their existing
  declared-terminal-failure one; both trip only on the early-terminal
  variant, and neither is loosened for a genuine shape violation -- an
  overfilled round and an undeclared short run still fail. New fixtures
  cover both emission points, and a structurally perfect retention that
  discarded a strictly better measured candidate is the negative control.

### Changed

- **Pinned `whetstone-ai` to 0.1.16** (from 0.1.15), which is what makes
  COPRO's seed retention reachable at all.
- **The protocol document's null-B rationale is corrected, and its digest
  re-pinned** (revision item 22; `PROTOCOL_DOC_SHA256` now
  `0c5c14b4...`, superseding `17ad9c01...`). §3.8 and §5.4 justified
  null-B's shape partly by asserting that COPRO cannot terminalize
  `seed_retained`, since only GEPA and MIPROv2 carried
  `terminal_proposal_count`. whetstone-ai 0.1.16 makes that premise false.
  **Null-B's design is unchanged**: it rests on `diff_check` rejecting a
  no-op mutation, which holds under every optimizer, not on COPRO's
  inability to retain. `optim/nulls.py`'s `NullIdentityTransport` docstring
  carried the same false claim and is rewritten to state the mechanism
  truthfully and to record why the conclusion survives it. No
  pre-registered quantity moves.

## [0.2.9] - 2026-08-25

### Fixed

- **Reporting recomputes an evaluation's score the way whetstone-ai
  computes it: per task, then across tasks.** whetstone-ai persists every
  aggregate through `unweighted_task_mean`, which reduces each task's
  repeats to one value and means those across tasks. The reporting
  projection re-derives that number from the rows as an independent check
  on the persisted evidence, and it summed one flat mean over every row
  instead. The two orders are the same rational number and different
  floats -- IEEE-754 addition is not associative -- so at `k_repeat=3`
  they disagreed by 1 ULP for half the evaluations in a stage, and
  `_success_projection` rejected correct evidence with "recomputed
  aggregate disagrees with evidence", failing the run at publication with
  a `DurableRunError` after the work was paid for. The overall score, the
  per-stratum scores, and both recompute sites in `EvalReport`'s validator
  now apply the two-stage reduction over the persisted matrix order, which
  reproduces whetstone-ai's addend order bit for bit. `numerator` and
  `denominator` are unchanged and still report the row-level pass count.
  `StratumSummary`, which validates in isolation and cannot see rows, now
  checks the completeness rule and the unit interval rather than
  re-deriving the float from its own scalars. The convention was invisible
  at `num_seeds=1`, where each task has one row and the two reductions are
  the same sum, and the whole reporting suite ran at one seed; regression
  coverage now holds `num_seeds=3` on an arrangement pinned to diverge.
- **Two `whetstone-study run` processes can no longer drive one run
  directory.** Run ids are deterministic on arm and seed, so two
  invocations of a stage compute the same run directory, and the only
  interlock was `run_dir.exists()` over a `mkdir(exist_ok=True)`. Existence
  is a fact about the past, not about whether anyone is writing right now:
  two processes launched seconds apart both saw no directory, both
  proceeded, and their effects interleaved -- stranding an intent as
  `leased` and writing off the run's paid work. The sqlite
  `EffectLeaseAuthority` did detect the foreign writer, but only when a
  lease renewal found its row taken, at terminalization and after the
  spend. A run directory is now held under an exclusive `O_EXCL` lockfile
  beside it, carrying the holder's pid, process start time, and hostname,
  for as long as the arm is judged and driven. A conflict is resolved on
  what is true now rather than on the file's mere presence: a **live**
  holder is refused with a `StageError` naming it, before `run_optimizer`
  is reached and so before anything is paid for, while a **dead** holder's
  lockfile is crash residue that is cleared so the run proceeds -- leaving
  the directory's own reusability to `--discard-stale-runs` as before. The
  start time is what distinguishes the two, so a recycled pid neither
  blocks a directory forever nor impersonates a live run; a lock from
  another host, or one too corrupt to read, is treated as live rather than
  guessed at. The lock is released on both the normal and the failing path.
  The `null-identity` control deliberately stays outside the lock: it
  creates no run directory at all -- its record's `artifact_dir` is a
  computed path, and it writes one content-addressed record without
  reaching `run_optimizer` or a provider -- so two invocations of one
  control converge on the same evidence pointer instead of contending. That
  premise is now pinned by a test rather than assumed.

## [0.2.8] - 2026-08-25

### Fixed

- **A run naming a `proposer_model` distinct from its task model no longer
  fails mid-run.** COPRO, GEPA, and MIPROv2 each minted the
  `ProposerConfig`'s `provider_call_config` reference from the *experiment's*
  config while the resolver returned the *proposer's* config, so
  `ProviderProposerTransport.draft` -- which resolves that reference and
  asserts the resolved record matches it -- raised `resolved
  'dr_providers.provider_call_config' record reference does not match
  IdentityRef`, surfacing from inside the durable boundary as a
  `DurableRunError`. The reference and the record are now derived from one
  route object, so they cannot name different configs. Only a run that names
  a distinct proposer was affected: `proposer_model=None` reused the
  experiment's config on both sides already, and its recorded reference hash
  is unchanged. The proposer route remains deliberately unpinned -- reasoning
  effort is a property of the task model a study measures.

## [0.2.7] - 2026-08-25

### Changed

- **The Step 10 c19 task model's reasoning effort is re-pinned from
  `minimal` to `low`** (protocol revision item 21). The `minimal` pin was
  probed and its Stage-0 gate failed on the task model's *capability*
  rather than on the design's power: the ceiling anchor scored **0.1977**
  against the gate's 0.30 floor, leaving **0.1897** of headroom against the
  0.20 minimum, while the two conditions that speak to power both passed --
  naive at **0.008** and the measured MDE at **0.0446**. Attempt 2 of the
  same design at the route's default effort measured a ceiling of
  **0.8068**, which is what identifies the effort rather than the task as
  the cause.

  Everything the original pin established is unchanged: the effort is still
  design rather than an invocation setting, still hashed into
  `pre_registration_design_hash`, still not a sized field, and still
  enforced as a refusal before any paid bind is written. Only the value
  moves. Because it is hashed, the re-pin moves the design hash, so a study
  initialised under `minimal` cannot be continued under this one. The
  protocol document records the probe's numbers as the provenance for the
  change and `PROTOCOL_DOC_SHA256` is re-pinned accordingly, with
  `ec650113...` -- the revision in force for the `minimal` probe -- kept as
  a historical digest.

### Added

- **Stage 0's gate verdict is persisted, as `stage0_gate`.** The manifest
  previously carried only the three numbers the gate *consumed*
  (`mde_measured`, `tau_sq`, `sigma_sq`, all on the design block) and never
  the verdict those numbers produced, nor the two held-out anchor means its
  most consequential conditions are read off. Stage 0 deliberately does not
  abort on a failed gate, so a failed calibration and a passed one left
  manifests distinguishable only by values a reader had to re-derive the
  gate from -- which is the gate's own arithmetic, repeated against the raw
  evidence store by whoever is least placed to do it.

  The record carries `passed`, `naive_mean`, `ceiling_mean`, `headroom`,
  `mde_measured`, and one row per gate condition with its name, verdict,
  observed value, threshold, and the gate's own detail sentence. It is
  written in the same manifest update as `design`, and `passed` is recorded
  whether or not it did. `whetstone-study plan` renders it as a `stage0
  gate` block showing PASS/FAIL per condition with each margin, so the
  verdict is readable without loading the manifest in a Python session. The
  no-abort semantics are unchanged.

  Manifest schema **v13**. Recorded rather than hashed, on the same line
  `mde_measured` already sits on: it is what Stage 0 measured about the
  design, not what the design pre-registered, so two studies of one design
  that calibrate to different anchors still pre-register identically.

## [0.2.6] - 2026-08-25

### Changed

- **Stochastic and infrastructure outcomes degrade a claim instead of
  aborting a stage.** Under the standing ruling that the study cannot
  require perfection from its infrastructure, three gates that demanded
  exactness of provider behaviour are relaxed to record-and-degrade. Every
  deterministic invariant -- pre-registration and design hashes, L2/L3
  once-only selection and held-out claims, split disjointness, ledger
  conservation -- is unchanged and still exact.

  A task that lost every repeat no longer refuses its evaluation. It keeps
  its position in the reported per-task vector with an achieved count of
  zero, so O7's completeness weighting drives its contribution to nothing,
  the vector stays aligned with the naive anchor's and the paired delta
  stays paired, and the loss lowers the row's achieved completeness until
  the report reads it as `VERDICT_INCOMPLETE`. The 90% task-completeness
  floor is unchanged and is now what bounds the loss: a fully-lost task is
  incomplete by construction, so both kinds of shortfall accumulate against
  one bound rather than two rules, and an evaluation below it still refuses
  to report a number. The previous rule fired on the *first* lost task, and
  because it fired inside a reporting stage it discarded every other arm's
  already-paid evidence and left no manifest at all. `measured_per_task`
  (renamed from `_measured_per_task`) and the analysis pass's paired-length
  check follow the same change; `_mean_of`'s fallback averages only the
  tasks that produced a value, so the zero-weight placeholder never reads
  as a task that scored zero.

  Anchor calibration is deliberately *not* relaxed. An anchor is the
  reference every arm's delta is measured against, so Stage 0 still refuses
  a calibration that cannot meet its presence floor.

- **The COPRO breadth audit accepts a realized round.**
  `COPRO_BREADTH_PER_DEPTH` checked every round for exactly
  `control.breadth` occurrences; it now accepts 1..`breadth` and reports the
  realized counts, because a draft can be lost to an infra failure, a
  template that fails validation, or a duplicate. Overfilling and a round
  that measured nothing still fail, and the round count against
  `control.depth` stays exact. At the pinned breadth 6 / depth 3 the old
  equality made a couple of percent of bad drafts a majority chance of
  losing a Stage-2 arm to one unlucky draft.

- **The in-search reward policy sets `missing_data="skip"` explicitly.** It
  previously inherited `RewardPolicy`'s `fail` default, which put the
  stricter rule inside the looser one: a minute-long provider outage
  costing a tenth of one minibatch aborted an optimizer run that the
  aggregation layer -- already at `skip` with a 10% row tolerance -- was
  prepared to tolerate.

- **The GEPA metric-call audit exempts a declared terminal failure.** A run
  that terminalized below its ceiling having declared why is no longer
  flagged, matching the exemption `copro_search_depth` already had. Without
  it a declared failure was reported twice, the second time as a budget
  violation that downgraded the arm for the infrastructure's behaviour. The
  monotonicity and past-the-ceiling checks are not exempted.

### Fixed

- **A held-out evaluation refused after billing no longer wedges the
  study.** The claim is written before the provider call, so a refusal
  *after* the call left it outstanding forever -- indistinguishable, on
  resume, from a process that died mid-call, and therefore neither
  re-issuable nor writable off. Recovery was hand-editing `study.json` or
  discarding a Stage 2 of paid runs. `HeldOutClaimRecord` gains a `refusal`
  field and a `settled` property, the ledger gains `refuse_held_out` and
  `refused_claim_for`, and a refused evaluation settles its claim durably
  with its reason.

  A resumed arm whose claim was settled as refused is **returned** with
  `ArmReport.held_out = None` rather than raised over: it keeps its
  selection and official scores, contributes no held-out row, and the
  report renders it `VERDICT_UNMEASURED`, so the pass finishes and every
  other arm still reports the number it was paid for. `held_out` is
  therefore `HeldOutMeasurement | None`, and `_measurements_by_name` skips
  an arm without one rather than substituting a placeholder.

  Only a **deterministic post-billing judgement** settles a claim. The new
  `HeldOutRefusalError` is raised by `RoleScorer.evidence_for` after pricing,
  and it is the only exception `_evaluate_claimed` writes off. A transient
  failure -- a connection reset, a 503, an OOM -- may never have reached
  the provider and could well succeed on the next attempt, so it
  propagates untouched and leaves the claim outstanding and resumable;
  `KeyboardInterrupt` and other `BaseException`s escape without settling by
  design rather than by inheritance. L3's once-only rule is untouched: a
  settled claim is still spent, and an outstanding one is still refused.

- **The anchors resume from their claims, completed or refused.**
  `completed_claim_for` was consulted only for arms, while the anchor pass
  runs *last* -- after every arm has been scored and measured. A crash
  anywhere in the reporting pass therefore left the anchors with durable L3
  claims that refused a second evaluation and no code path reading them
  back, so the resumed pass re-issued the call, was refused, and wedged a
  study that had already paid for essentially all of its rows. The resume
  now reads both states: a completed claim replays, and a refused one is
  omitted from the returned mapping rather than re-issued into
  `claim_held_out`'s any-claim rejection -- which stays as it is, since
  that rejection is L3 working correctly.

  A refused **ceiling** anchor is narrow: no arm's delta is measured
  against it, so every arm still reports against naive and the study simply
  carries no ceiling row. A refused **naive** anchor is fatal and says so
  -- every delta is measured against it, so there is nothing to degrade to,
  and the refusal names the consequence and the recovery instead of reading
  as an anchor nobody got around to measuring.

- Verified `bootstrap_paired_delta_ci` and `holm_adjust` on all-identical
  (zero-variance) resample sets, an unverified gap in the audit. Both are
  well behaved -- finite, ordered intervals collapsing onto the point
  estimate, `p = 1.0` with no effect -- so no fix was needed, and tests now
  pin it.

- **The reporting CLI's parser no longer reaches the `optim` extra.**
  `build_parser` imported an optional dependency to spell the reasoning
  effort's choices, which broke `whetstone-eval --help` on an extra-free
  install -- exactly what the wheel smoke test runs. The choices are plain
  strings checked against the enum at parse time, and a test exercises
  `--help` under blocked optional imports.

### Added

- **The Step 10 c19 task model is pinned to `minimal` reasoning effort.** A
  reasoning effort changes the task model's capability, and therefore the
  treatment every arm is measured under, so it is design rather than an
  invocation setting: `TASK_REASONING_EFFORT` is a protocol constant and a
  `StudyProtocol` field -- deliberately not a sized field, since the toy and
  the real study run at the same effort or they are not the same protocol --
  and `ModelsRecord.task_reasoning_effort` and
  `PreRegistrationRecord.task_reasoning_effort` record it. The effort enters
  `pre_registration_design_hash`, which is the point of the widening: a
  design that could change the effort while keeping its hash would let the
  effort be chosen after the anchors were visible.
  `openrouter_seeded_call_config` binds it as `GenerationControls(reasoning=)`
  on every task route -- the study environment, the Codex MCP runtime (whose
  cross-process `task_model_identity_hash` guard would otherwise refuse the
  rebuild), the in-search `RunSpec` every `StudyOptimizerRunner` arm builds,
  the standalone runner, and the report projection -- while the proposer
  route stays unpinned, because it writes candidates rather than answering
  tasks and the study makes no claim about it. `--task-reasoning-effort`
  covers the standalone paths.

  Both paid paths report the effort they bound into one witness, and the
  recording path **refuses** a paid bind whose effort disagrees with
  `models.task_reasoning_effort` before the write, so a stage that would
  measure an unpinned task model fails before it bills rather than leaving
  the disagreement for a reader to notice afterwards. The in-search bind was
  previously not recorded at all, so the record described only the reporting
  half of the study. The fake transport is exempt: it binds whetstone's
  reference default and never reaches a provider. Verification is
  request-side -- billed reasoning tokens are not evidence a control was
  honoured, since a provider may spend what it likes at any effort and
  OpenRouter silently ignores `temperature` on nano routes -- so the tests
  pin the built payload instead: a pinned config emits
  `{"reasoning": {"effort": "minimal"}}` and an unpinned route sends no
  `reasoning` key. Protocol document revision item 19 states the pin, its
  cost consequence, and this verification method, replacing text that
  pre-registered the opposite.

- **Pins published whetstone-ai 0.1.15.** Blank generations are now scored
  failing samples: a `SUCCESS` row carrying failure code
  `blank-provider-generation`, rather than a missing row, so a candidate
  that reliably blanks can no longer improve its apparent completeness by
  failing. `ExecutedRowState.INVALID` remains reachable through provider
  refusals, and `Observation.trace_state` already admits it. Anchor
  calibration gains a 0.9 presence floor and deterministic balanced
  subsetting to equal per-task depth -- `run_anchor_calibration` takes a
  `store` to read the row-level outputs balancing needs, threaded through
  `run_stage0` and `calibrate_role`. COPRO gains shortfall tolerance,
  recording `proposal_shortfall`, terminalizing on best-so-far, and adding
  `copro_proposal_round_empty`; `OPTIM_RUN`/`STEP_REQUEST` go schema v3→v4,
  which changes the content hash a record self-addresses by, so the
  committed Codex audit fixtures are regenerated. MIPROv2 replay
  memoization is a behavioural no-op.

- Protocol document revision item 20 (2026-08-25) states the tolerance in
  the pre-registration itself: §3.9 on degrading rather than aborting, on
  blank generations as scored failures, and on anchor calibration's floor
  and balancing; the invariant table on 1..`breadth`. The pre-registered
  quantities -- breadth 6, depth 3, the K values, the 90% floor -- are
  unchanged and remain what the design *requests*, with realized counts
  recorded as measurement. `PROTOCOL_DOC_SHA256` is re-pinned to
  `ec650113...`; `0dfd0c47...` is historical.

## [0.2.5] - 2026-08-24

### Added

- **Pins published whetstone-ai 0.1.14.** Three upstream changes reach the
  evidence this package reads. Evaluation rows gain per-row node-failure
  diagnostics -- `error_type`, `error_message` (bounded to 2000
  characters), `failed_node_id`, and `row_attempts` -- carried through the
  subprocess worker, so a stage that loses rows records what failed rather
  than only that something did. Unattributed node failures are retried at
  row level (`max_row_attempts`, default 3) with a fresh provider call per
  attempt and summed usage, which changes how a transient failure appears
  in a stage's spend: one row, several attempts. Blank or whitespace-only
  generations now score as terminal `invalid` rows under failure code
  `blank-provider-generation` instead of surfacing as
  `node_execution_error`, so the per-task completeness floor sees them as
  lost rather than as a crashed node. `EvalEvidence` stays at schema
  version 6, so the committed Codex audit fixtures validate unchanged and
  are not regenerated.

  One projected field follows the upstream change. `ExecutedRowState` gains
  an `invalid` member -- a row that executed and was billed but produced
  nothing the contract can score -- so `Observation.trace_state` admits it,
  and an invalid observation now reconciles against a trace that says
  `invalid` rather than one that says `failed`. That fold existed only
  because the upstream enum had no way to spell the state; under 0.1.14 it
  would have rejected exactly the rows the blank-generation change makes
  reachable. `trace_state` stays a closed literal rather than the imported
  enum, because it is a persisted report field, and a test pins it against
  `ExecutedRowState` so a future widening is a deliberate change here
  instead of one the projection inherits silently.

- **Each run records the provider concurrency it ran at, and a resume that
  changes the width is refused.** The width was recorded once per *stage*,
  which names what the latest invocation asked for -- not what the runs
  beneath it ran at. A resumed arm stage re-runs only the seeds it has no
  record of and reuses the run directories for the rest, so a resume at a
  different `--provider-concurrency` re-ran none of the survivors at it
  while `_arm_stage_record` overwrote the stage's single
  `provider_concurrency` with the new value. The row then named a width most
  of its runs never ran at, and a stage's wall time and its rate-limit and
  timeout failures are read against nothing else.

  Manifest schema v12 adds `RunRecord.provider_concurrency`, taken from the
  `RunSpec` the stage built rather than read back off the run's artifacts:
  unlike the repeat count, the width is an execution property no optimizer
  reads and nothing persists, which is exactly why a *reused* directory's
  width cannot be recovered by inspection. So an arm stage whose requested
  width differs from its recorded stage width, while any run from the
  earlier invocation survives, is refused before dispatch -- naming both
  widths, the runs that would have been misdescribed, and both recoveries
  (`--provider-concurrency <recorded>`, or a fresh study directory).
  `--allow-width-change` proceeds instead and records the change as an
  appended note on the stage record (`StageRecord.width_change_notes`).
  Widths are invocation properties and never entered the pre-registration
  hash, so an authorized change amends no design and writes no
  `AmendmentRecord`; the study pre-registers exactly as it did before.

  The stage record's `provider_concurrency` is now explicitly *this*
  invocation's width -- the only thing one field can honestly say about a
  stage whose runs span two -- and both renderers name the distinct per-run
  widths when they differ: `whetstone-study run`'s ledger and the report
  packet's stage rows. The ordinary single-width study is not annotated.

### Fixed

- **The width guard no longer reads a crashed stage as a first run.** It
  settled the question entirely from the manifest, so a missing
  `StageRecord` returned early: no recorded width, nothing to disagree
  with. But a stage writes its row *after* its arms finish, so a crash
  between the last run and `write_study_manifest` leaves the run
  directories on disk with no row to say what width produced them -- a
  state the manifest cannot tell apart from a stage that never started.
  The resume then claimed those directories, re-ran nothing, and recorded
  the requested width over paid runs that may never have run at it. It is
  the worse of the two states, because the recorded-width refusal can at
  least say "re-run at the recorded width" and this one cannot: the width
  was never written anywhere, so no inspection recovers it.

  An arm stage now reads the study's `runs/` directory as well as its
  manifest, and a stage with no row that finds run directories the manifest
  cannot account for is refused -- naming the directories, the requested
  width, and three recoveries: a fresh study directory,
  `--allow-width-change` to record the requested width over them
  deliberately (returning a note that says the reused runs' width is
  unrecoverable rather than merely different), or `--discard-stale-runs`,
  under which the directories are discarded and re-run so nothing survives
  to be misdescribed. As with the recorded-width refusal, neither escape
  amends a design: the width does not enter the pre-registration hash.

  *Cannot account for* is the operative condition, because `runs/` is one
  directory for the whole study rather than one per stage. A directory
  behind a recorded `RunRecord` is accounted for -- that is every Stage 2,
  which stands on Stage 1's runs by design, and those runs carry their own
  per-run widths. A directory an amendment deliberately orphaned is
  accounted for too: `--replace-design` records exactly which directories
  it left behind, and they remain the stale-run refusal's subject, which
  reads each directory's own identity and names `--discard-stale-runs`
  against it. What is left is the crash residue. A study with no row and no
  unexplained directories is the ordinary first run and is unaffected.

## [0.2.4] - 2026-08-23

### Added

- **Provider concurrency is an explicit, recorded operator setting.**
  `whetstone-study run --provider-concurrency N` (and the single-run CLI's
  flag of the same name) sets how many task evaluations run against the
  provider at once. It reaches both bounds that matter -- the evaluation
  engine's worker pool and the HTTP client's connection pool, which is
  widened to match so workers are not queued behind sockets -- and is
  written onto the stage record. It is an invocation property like the
  transport: it changes how long a stage takes, never what it measures, so
  it does not enter the pre-registration hash. Previously nothing named
  it and every stage ran at whetstone's default of 5 in-flight calls,
  which at 20-30 s per reasoning-model call is 10-15 rows/minute.
  `plan` and `run` print the width beside the transport. Values below 1
  are refused, and values above 64 are refused unless
  `--force-provider-concurrency` is passed -- OpenRouter publishes no
  per-account concurrency limit for paid models, so that cap is this
  package's own prudence rather than a quoted provider limit.
- **The width and the hardening reach every live engine, not just the
  study's.** Two other surfaces build their own paid engines and were
  left at whetstone's defaults, which meant the concurrency and
  timeout/retry work stopped at the study path:
  - `whetstone-eval run --transport openrouter` (the standalone report
    CLI) built its engine through
    `ReferenceEvalRuntimeConfig.build_engine`, which accepts neither a
    policy nor a concurrency, so it kept spending against the reasoning
    models at the 30 s chat-completion timeout with retries that never
    waited. It now takes `--provider-concurrency` and
    `--force-provider-concurrency` -- declared from the same constants as
    the study CLI -- and binds the widened, hardened policy.
  - The Codex arm's hosted-MCP evaluator rebuilds its engine in a
    *separate process* from `EnvsCodexRuntimeConfig` and nothing else, so
    the study's own hardening could not reach it: it ran at the default
    width, a 30 s timeout, and no waiting retry at all. The config now
    carries `provider_concurrency` as a recorded serialized field
    (forwarded from the `RunSpec`, and pinned by the cross-process field
    golden), applies both transforms to the policy it rebuilds, and binds
    the retrying transport rather than the bare client.

- **Each run records the repeat count its search actually ran at, and the
  study refuses one that disagrees with `K_REPEAT`.** The per-optimizer
  `mipro_repeats_as_recorded` / `gepa_repeats_as_recorded` audits hold a
  run's evaluations to the count that run *itself* recorded, which is a
  within-run check: a run that consistently recorded and searched at one
  repeat passed cleanly under a design registering three, having bought a
  third of the evidence the pre-registration priced. Nothing diffed the two
  numbers, because the run's own count was never on the record to diff.
  Manifest schema v11 adds `RunRecord.search_num_seeds`, read off the run's
  own artifacts rather than taken from the runner that asked for the run --
  MIPROv2's `StudyTranscript.validation_num_seeds` and GEPA's
  `GepaDetailedResult.validation_num_seeds` where those exist, and the
  evaluations' own `EvalEvidence.num_seeds` for COPRO, null-A, and Codex,
  which record no such scalar and inherit the bound engine's sampling
  whole. `None` on null-identity, which runs no optimizer.

- **Pins published whetstone-ai 0.1.13.** `EvalEvidence` reaches schema
  version 6, in which a task that lost every repeat reports `None` for its
  per-task value rather than `0.0`. The per-task completeness floor already
  read presence off the reported vectors rather than inferring it from
  arithmetic, so it reads the new spelling unchanged; the reporting and
  calibration paths now narrow the vector explicitly, which is sound
  because the floor refuses a lost task before either can see one. The
  committed Codex audit fixtures are regenerated at v6: they store real
  evidence records, so a fixture written under v5 fails
  `reported_numbers_resolve` against the new validator.
- **Transport retries are visible as attempts.** `StageRecord` gains
  `provider_attempts` and `provider_transient_outcomes`: a call that
  survived two 429s before succeeding is one `calls` and three attempts,
  reported side by side. `calls` is unchanged and still counts persisted
  rows, which is what the study is billed a completion for. This is the
  one manifest number that cannot be projected from the store, since a
  retried attempt persists no row -- the retrying transport counts its own
  invocations and the stage reads them back at the end. A fake-transport
  stage reports nothing rather than a measured zero.

### Fixed
- **The operator's provider width reaches in-search evaluation, not just
  the report.** `--provider-concurrency N` reached the stage's scoring
  engines and the stage record, but `StudyOptimizerRunner._spec_for` built
  each arm's `RunSpec` without a width -- so every in-search evaluation,
  which is the large majority of a paid Stage 1 or Stage 2's calls, ran at
  the default 5 while the manifest recorded the width the operator asked
  for. One stage ran at two widths and was recorded as one. The runner now
  carries the width and forwards it onto every arm's `RunSpec`,
  unconditionally rather than per-arm: every optimizer evaluates, and the
  Codex arm's separate-process runtime config rebuilds from
  `spec.provider_concurrency` alone, so this is that arm's only route to
  it. Stage 0 was unaffected -- it evaluates through the engines directly
  and never builds a `RunSpec`.
- **A paid arm stage records the retries its reporting pass fought.**
  `_record_report_spend` read the transport's attempt counters back only
  on its `existing is None` branch. Stage 0 takes that branch, so the gap
  was invisible there; a paid Stage 1 or Stage 2 never does -- its run
  pass writes the stage row long before the reporting pass folds a bill
  onto it -- so `provider_attempts` and `provider_transient_outcomes` kept
  whatever the run pass recorded and the official-scoring and held-out
  evaluations' retries were dropped. The counters are now folded on the
  existing-record branch too. Folded rather than assigned: a resumed
  stage's second process starts its transport's tally at zero, so
  overwriting would report the resume's retries as the stage's whole
  history and drop the original run's. Unlike `report_spend` -- re-read
  whole from durable records every time -- a transient attempt persists no
  row, so the in-process tally is the only record it ever had. A pass that
  bound no retrying transport contributes nothing rather than a zero.
- The release workflow's validate jobs allow 30 minutes: the 3.13 and 3.14 legs
  take just over 15 on `ubuntu-latest`, so the v0.2.3 tag's run was cancelled
  before it could publish (v0.2.3 is tagged but not on PyPI).
- **`whetstone-eval` applies the per-task completeness floor.** The floor
  lived only in the study's `RoleScorer`, so the standalone command --
  which publishes the held-out number a claim is finally made from --
  projected and published exactly the biased mean the floor exists to
  refuse. Both paths now apply one shared owner in
  `whetstone_envs.optim.completeness`. Evaluations *inside* a search stay
  exempt: under 0.1.13 a lost task reports `None` per-task and the
  optimizer's reward policy governs what a candidate is worth mid-search,
  so refusing there would make the floor a stopping rule rather than a
  reporting one.
- **The 90% task-completeness floor can actually fire.** It compared
  `(planned - lost) / planned` against 0.90, but the stricter zero-present
  rule above it already refused whenever `lost` was nonzero -- so the
  fraction was always exactly 1.0 at that comparison and the bound could
  never bind on any input. It now counts tasks measuring fewer than
  `k_repeat` present rows as incomplete, which is the population the
  zero-present rule does not cover. That rule remains unconditional.
- **The base install no longer imports the optimizer stack.**
  `whetstone-eval` is a base-install console script, but
  `reporting.cli` imported its concurrency bounds from `optim.provider`,
  whose module scope reaches `dr-providers` and whetstone. Any install
  without the `optim` extra failed the entry point at import. The bounds
  move to a dependency-free `optim.concurrency`, and a regression test
  imports every base-install module with the optional distributions
  blocked.

- **The arm stage refuses on both sides of the dispatch.** Before any arm
  runs it refuses a stage whose bound engine would not sample at the
  design's `K_REPEAT` -- structural, and free, because binding issues no
  evaluation -- and after each run it refuses to record one whose
  `search_num_seeds` disagrees, naming both numbers. The check covers every
  run the arm will carry rather than only the fresh ones, since a run an
  earlier invocation recorded is evidence the stage is about to select and
  report over.
- **Leakage rule L7 checks the recorded repeat counts against the
  pre-registered `K_REPEAT`.** L4 establishes that the *held-out*
  evaluations shared one procedure, which an arm that searched cheaply and
  was then measured like everyone else satisfies perfectly; only a diff
  against the pre-registered number catches it. L7 is appended after L6
  rather than renumbering the rules, which would silently redefine what an
  already-recorded `leakage_check` block claims, and its verdict is
  recorded in the manifest and printed by the report.

- **A single rate-limited row no longer voids a whole paid evaluation.**
  The live Stage 0 evaluated 352 rows, lost exactly one to an unretried
  429, and aborted -- discarding 351 good rows and the money spent on
  them. Two independent causes, both fixed:
  - *Retries never waited.* `ProviderExecutionPolicy` already classified
    rate limits as retryable and already computed a backoff, but the
    delay is applied through an injected `sleep` that the evaluation path
    never supplies, so all three attempts fired within microseconds.
    Paid transports are now wrapped in a `RetryingTransport` that waits:
    5 attempts, 2-32 s exponential backoff with jitter, honouring the
    provider's `Retry-After` (bounded at 120 s, and only in its
    delta-seconds form -- the HTTP-date form is ignored by design, since
    resolving it needs two clocks to agree exactly when that is least
    safe). Transient 5xx and timeouts retry on the same terms; permanent
    rejections still do not.
  - *And then they multiplied.* `RetryingTransport` loops internally and
    then **returns** an exhausted transient failure, which whetstone's
    `run_provider_call` reads as one failed attempt and retries under its
    own `max_attempts` -- so five apiece composed multiplicatively rather
    than additively: 5 x 5 = 25 billed invocations for a single row
    against a rate limit that never clears, sleeping the full backoff
    schedule five times over (up to ~40 minutes of wall clock for one
    row). It also corrupted the record, because the driver appends one
    attempt per *driver* iteration: a row that cost 25 invocations
    persisted 5, so the ledger under-counted billed calls by exactly the
    wrapper's factor. Retries now have exactly one owner. The wrapper
    keeps the budget -- it is the layer that actually waits -- and the
    driver's policy is pinned to `max_attempts=1`, making it a
    pass-through. The manifest records the *effective* count (5) rather
    than the driver's, so the number an operator reconciles spend against
    is the number of calls really made.
  - *One missing row poisoned the aggregate.* Aggregation ran with
    `missing_data="propagate"`, so a single absent row set the mean to
    `None`, which made the reward term missing, which its `FAIL`
    missing-data policy then raised on. Aggregation now runs with
    `missing_data="skip"` and `max_skip_fraction = 0.10`: present rows
    are averaged and the shortfall is reported as reduced completeness,
    which the analysis already weights by per-task achieved counts.
    Beyond that fraction the aggregate returns to `None`, so the
    tolerance is a floor against losing an evaluation to one bad row, not
    permission to average a biased subset. The bound is the complement of
    the 90% completeness backstop §3.9 already pre-registered, so there
    is one threshold rather than two.
  - *But a row bound cannot see a lost task.* `skip` counts rows, and a
    task that loses **every** repeat is dropped from the task mean's
    *denominator* rather than counted as a zero -- whetstone's
    `unweighted_task_mean` gives it `ZERO_DENOMINATOR` and the outer mean
    divides by the tasks that produced a value. At the study's own shape
    the two bounds disagree badly: 76 tasks x 4 repeats is 304 rows, one
    fully-lost task is 1.3% of them, so the 10% row tolerance passes it
    and the evaluation reports `status=ok` with a mean over 75 tasks
    presented as though it covered 76 (measured: 1.0 reported where the
    truth is 0.9868). The bias is upward and systematic rather than
    noise, because a task that loses every repeat is a slow,
    long-generation one -- the task that would have scored low. An
    envs-side validator now refuses the evaluation before it is accepted
    if any task has zero present rows, or if fewer than 90% of planned
    tasks were measured. It lives on the envs side because whetstone's
    `AggregationConfig` has no per-task completeness variable to set:
    its knobs all act on the flat row vector.

  This changes the reward-policy and eval-config hashes. That is
  acceptable only because nothing is pinned yet: the live Stage 0 failed
  before writing `design` or `pre_registration`, and its manifest records
  no stages. Recorded as item 18 of the protocol document's Revision 2
  (2026-08-23), with the digest golden and `PROTOCOL_DOC_SHA256`
  recomputed.
- **Stage 1/2 calibration consistency is fixed upstream, not here.**
  whetstone-ai is making `per_task_score` aggregate over *present* rows
  (reporting `None` for a task with no OK reduction) and `per_task_count`
  count present rows, under `EvalEvidence` schema v6; this package's pin
  will move to the release carrying that change rather than working
  around it locally. The task-completeness floor above is written to
  compose with it: presence is read off the per-task vectors in both
  spellings -- a `None` value or a `0` count -- rather than inferred from
  arithmetic that would assume a missing row scores `0.0`. Refusing at
  that seam also keeps a fully-lost task from reaching calibration, which
  rejects `None` per-task values outright, so the failure surfaces as a
  named completeness refusal rather than further down.
- **Paid task calls are given a reasoning-sized timeout.** The 30 s
  default is a chat-completion bound; the live Stage 0 measured a median
  of 4,466 completion tokens and a maximum of 12,335 per call, which
  routinely outruns it and turns an ordinary slow call into a billed
  timeout. Paid transports now use 300 s. The effective timeout, attempt
  count, and backoff schedule are recorded in the manifest's
  `provider_calls` block, unhashed, beside the controls already there.

## [0.2.3] - 2026-08-23

### Added

- **The Step 10 c19 study is authored from a committed protocol.**
  `whetstone_envs.optim.study.protocols` pins every pre-registered design
  value as a named constant -- splits 88/132/440, the 44/44 per-arm
  train/val partition, `openai/gpt-5-nano` and `openai/gpt-5.4-nano`, the
  `gpt-5.6-sol` Codex agent, GEPA's 200-metric-call budget, MIPROv2's
  trials/candidates/minibatch, the Codex evaluate-call cap, and the arm
  list -- and `whetstone-study init --study-dir DIR --protocol step10-c19`
  writes the pre-Stage-0 `study.json` from it. The task hashes of all three
  splits, the pool manifest hash, and the sha256 of the protocol document
  are recomputed at init rather than declared, so the manifest names the
  population the harness will regenerate and the revision of the
  pre-registration that was in force. `init` refuses to overwrite an
  existing manifest.
- **`--toy` authors the sized-down variant of the same protocol.** Both
  sizes are built by one function from one body of pinned values, so they
  can differ only in the fields `SIZED_FIELDS` names; a golden test asserts
  every other field matches and that every sized field actually differs. A
  toy that could drift on the models, the arm list, or the correction would
  rehearse a study the real one does not run.
- **`--without-codex` authors the design with the Codex arm dropped**, for
  fake-transport rehearsals. The Codex guard fires on the design whatever
  transport the task model is on, so a rehearsal of the rest of the study
  has to drop the arm rather than stub it; the result is a strictly smaller
  design, not the pre-registration.
- **The study harness runs on the real OpenRouter transport.**
  `whetstone-study run --transport openrouter` binds the same provider
  route the single-run path already used — the seeded OpenRouter call
  config, `ProviderKind.OPENROUTER`, and `OPENROUTER_API_KEY` — for every
  role engine and for the arms' optimizer runner, so Stage 0, Stage 1, and
  Stage 2 can spend. One live transport is bound per stage rather than per
  engine binding, because a stage rebinds an engine on every scored
  candidate and a per-binding client would open a connection pool per
  evaluation. Models still come from the manifest's `models` block; the
  transport selects a route, not a model. `--transport fake` remains the
  default and the fake path is unchanged: its prepared experiment carries
  no provider call config, so every Eval Config hash it derives is what it
  was before a paid path existed.
- **A paid stage with no key is refused before anything is opened.**
  `--transport openrouter` without a non-blank `OPENROUTER_API_KEY` fails
  ahead of the store, the pool, and every engine, so an unauthorized run
  leaves no half-initialized study directory and exits non-zero. The key is
  checked for presence only and never reaches an error message; a wrong key
  is the provider's refusal to make.
- **Each stage and each run records the transport it ran on, and a study
  will not mix them.** The manifest gains a `stages` block with one
  `StageRecord` per stage naming its transport and its spend, and every
  `RunRecord` carries its own `transport` (schema `v7`). Like the
  real-Codex authorization the transport is an invocation property and
  stays out of the pre-registration hash, so two studies differing only in
  it pre-register identically — but it is recorded, because a stage run on
  the fake transport and a stage run against a provider are different
  evidence for the same claim. An arm stage is refused **before any arm
  runs** unless three things agree with the requested transport: Stage 0's
  anchors, the stage's own recorded transport, and every surviving arm run.
  A run's transport is its own evidence rather than its stage's — a resumed
  stage keeps runs an earlier invocation paid for, so a stage row can agree
  while the runs beneath it do not, and checking only Stage 0 let a paid
  arg-max run over free runs. A toy re-runs Stage 0 under
  `--replace-design`. A re-run replaces its stage record rather than
  appending a second one, and the manifest refuses two records for one
  stage.
- **`stage0 --replace-design` across transports drops the stale evidence
  and records the drop.** Re-calibrating onto a transport other than the
  recorded Stage 0 invalidates the arm stages twice over — the design they
  ran against is being replaced, and their evidence was measured somewhere
  else — but v6 left the `stage1`/`stage2` records and every arm run in
  place, so a Stage 2 on the paid transport reused fake runs against
  freshly bought anchors. Those records, their runs, the selections over
  them, the held-out claims and rows, and the pilot's call-count gate are
  now dropped, and what was dropped is recorded in a new `amendments`
  block the report surfaces rather than deleted silently. The arms survive
  with empty run lists, because an arm is design rather than evidence.
  **Paid evidence is never discarded automatically**: a drop that would
  remove any run measured on a billed transport is refused instead, before
  the calibration spends, naming the runs and the recovery.
- **Stage 0's anchors produce a spend record, and every stage total is
  printed.** Anchor evaluations reach the provider through the evaluation
  engine rather than through an optimizer run, so they had no
  `OptimResult.cost` and no total. `optim/study/spend.py` re-derives one
  from the persisted `EvalOutputRow` usage fields behind each anchor's
  evidence, aggregated through whetstone's own `aggregate_role_cost` so the
  honesty split — cached, priced, unpriced, missing token breakdown — and
  the rule that an absent `usd` is "not knowable" rather than "zero" are
  the shared aggregator's rather than a restatement. Evidence is
  de-duplicated by reference, matching how a run's aggregation counts one
  evaluation once. `whetstone-study plan` prints the measured ledger
  beneath its estimated budget, `run` echoes the stage's transport and
  ledger, and the report prints each stage's transport and spend in its
  stage history.
- **An arm stage records what its runs spent.** An arm stage spends through
  optimizer runs rather than through the engine, and each run already
  re-derived its own per-role bill — but the stage record ignored them, so
  a fully paid Stage 1 or Stage 2 recorded an empty `spend` and the ledger
  rendered it, under a MEASURED heading, as a stage that reached no
  provider. The stage total is now the fold of the per-role records the
  runs *this invocation executed* reported; a run an earlier stage paid for
  stays billed on that stage's row, so the ledger's rows do not sum to more
  than the study spent. The fold re-applies the honesty rule rather than
  carrying it: one unpriced run withholds the whole role's total.
- **The ledger never calls a paid stage a free one.** An empty spend record
  means one of two opposite things and now reads as two: a fake-transport
  stage reports `no provider reached (fake transport)`, because its rows
  are real rows the shared row rule counts as billable-and-unpriced — right
  for a provider row, and a bill nobody owes for a stage that called
  nobody. A **paid** stage with no records reports a loud `UNLEDGERED`,
  because it reached a provider and lost track of what it bought: its bill
  is unknown, not zero.
- **The ledger states what it covers.** A stage row is the sum of both
  routes it spends by — its arms' optimizer runs and its reporting pass —
  and `plan`, `run`, the report, and the README say so, so a printed total
  reads as the whole bill rather than as the run-side part of it.
- **The fake path's Eval Config hashes are pinned.** A golden test fixes
  the three role config hashes the fake toy study binds, so a change that
  let provider call configuration leak onto the fake path — the way the
  paid path seeds one — would fail the pin rather than silently rebase
  every recorded config and turn L1 into a check that always agrees with
  whatever just ran.

### Changed
- **The registered protocol document is Revision 2 (2026-08-23).** The
  shipped text was written on 2026-08-22 and predates decisions ratified
  the next day, so the pre-registration a reviewer diffs against and the
  design the code executes disagreed on sixteen values — the code being
  correct on all of them. Revision 2 updates the body in place so every
  registered value equals `protocols.py`: held-out 440 with the MDE table
  recomputed at 0.0622/0.0690, the explicit disjoint 44/44 train/val
  partition, the Codex evaluate-call cap lowered from 20 to 8, GEPA pinned
  at 200 metric calls with reflection minibatch 3, null-B moved off the
  runner to the report harness, null-A routed through the ordinary runner,
  MIPROv2 at 3 candidates and a uniform 10 trials across all three demo
  modes, the Codex agent pinned to `gpt-5.6-sol`, MIPROv2 minibatch 35,
  COPRO 6/3, `K_REPEAT` stated to cover in-search evaluations, the
  measured `$0.00168`-per-call cost model with the study priced at
  $152–$176, and the single real Codex run recorded as historical evidence
  only. A dated revision block at the head lists each change against the
  plan note that decided it; the pinned digest moves from `a311de47…`,
  now a historical value, to `1fa2102b…`. MIPROv2's per-mode trial counts
  are retired rather than re-derived: the old 10/9 split came from
  auto-mode at six candidates, the design pins three, and the study sets
  `num_trials` on the control so auto-mode never runs.

### Fixed
- L1 refuses an optimizer evaluation that names no tasks instead of passing
  it as trivially contained in the internal split.

- **L1 is checked as task-set containment, not as one exact Eval Config
  hash.** The registered rule is that an optimizer saw the internal split
  and nothing else, but the implementation demanded that every optimizer
  evaluation resolve the study's *full* internal Eval Config. MIPROv2
  minibatches the internal split and GEPA scores single tasks and Pareto
  subsets, so each such evaluation mints its own derived config over a
  subset -- which the exact-hash predicate reported as a leak. Against the
  full-design dry run that was 499 false offenders and a failing study.
  L1 now requires each evaluation's task hashes to be contained in the
  internal split, under the internal role, against a config that is
  neither the official nor the held-out one; a derived config cannot
  smuggle in a foreign task, because the task hashes are content-addressed
  and checked directly. How many evaluations used the full internal config
  versus a derived subset is recorded as an observation beside the
  verdict rather than as a failure.
- **L1 reads all three evaluation surfaces, so GEPA can no longer pass it
  vacuously.** The evidence walk read only intent resolutions and
  `tool_evidence`. GEPA resolves no intent and records every evaluation as
  `search_evidence`, so a GEPA run contributed nothing -- yet with other
  arms supplying rows, L1 reported itself *checked* having never opened
  the run whose rule it was asserting. `search_evidence` is now read
  alongside the other two, and each observation carries the surface it
  came from so an offender location is unambiguous. On the same dry run
  L1's evidence goes from 679 evaluations to 879, the difference being
  GEPA's 200.
- **The MIPROv2 call estimate models the full-valset passes the schedule
  actually issues.** The band was 1,870-2,458 rows per run, and the
  observed fewshot counts were bimodal at 2,370/2,502 across five
  independent seeds -- deterministically over the ceiling. The two
  non-bootstrap constants had their labels attached to the wrong
  quantities (1,050 is the minibatch volume, `10 x 35 x 3`; 792 was six
  full-valset passes, `6 x 44 x 3`), and the six was the substantive
  error: at the registered control `adjusted_num_trials` is 21 and
  `promotion_due` fires on every even display trial, so **eleven** passes
  are issued -- ten promotions plus the baseline. The gate counts planned
  rows once per distinct evidence record, so a promotion re-evaluating an
  unchanged incumbent collapses onto the earlier record; the surviving
  count ranges from 1 to 11. The band is now **1,210-3,118**, every
  observed value decomposes exactly as `bootstrap + minibatch +
  full-valset`, and a golden pins the decomposition.
- **Pins published whetstone-ai 0.1.12, which unblocks the GEPA arm at the
  protocol's pinned reflection minibatch.** Upstream GEPA's
  `EpochShuffledBatchSampler` pads a shuffled epoch with *duplicate* ids
  whenever `len(trainset) % reflection_minibatch_size != 0`, and the
  protocol pins trainset 44 and minibatch 3 (`44 % 3 = 2`), so every
  reflection minibatch spanning the padding carried one task twice and
  `GepaEvaluationEffectRequest` refused it — `GEPA evaluation positions
  must be unique` — inside the durable run boundary. It was a divisibility
  bug, not a repeats bug: it reproduced at `num_seeds = 1`. 0.1.12 keeps
  the pinned upstream algorithm and reconciles it at the request boundary
  instead: duplicated positions collapse to their distinct instances for
  one evaluation request, and the rows expand back to the upstream batch
  shape, so a repeated instance still carries double weight in the
  `sum(eval_curr.scores)` comparison upstream uses to accept a mutation.
  [#145]

  **The accounting deliberately splits in two, and the Stage-1 gate depends
  on which side it reads.** Logical metric calls stay upstream's — it
  charges the padded batch length, duplicates included — so the pinned
  `max_metric_calls` budget means what it means upstream. Provider rows are
  billed once per *distinct* instance, because a repeated instance under a
  fixed candidate is the same evaluation. So a 200-metric-call GEPA run
  bills marginally *fewer* than 600 rows at `K_REPEAT = 3`, not exactly
  600. `gepa_task_call_ceiling(k_repeat)` is therefore an upper bound and
  is correct as written: it converts the metric-call pin to rows at
  `200 * K_REPEAT`, the run comes in under it, and the Stage-1 gate's
  1.5x tolerance still applies on top. A ceiling derived to equal 600
  exactly would abort the healthy runs it exists to protect.

[#145]: https://github.com/danielle-rothermel/whetstone-ai/pull/145

- **Pins published whetstone-ai 0.1.11, which lets MIPROv2 and GEPA run at
  the study's pre-registered repeat count.** Both optimizers refused a
  multi-repeat evaluation plan outright on 0.1.10 — MIPROv2 with `engine
  sampling repeats (3) do not match the requested num_seeds (1)` and GEPA
  with `GEPA evaluation engine must use a single-repeat plan` — so five of
  the study's eight arms could not run at all under a protocol whose
  `K_REPEAT` covers in-search evaluations. In-search evaluations now run at
  the eval config's repeat count and the score each search consumes is
  unchanged in kind: the existing per-task mean over repeats. [#142]

  Three recorded contracts move with it, and each is a schema bump an audit
  reads: `Miprov2Control` gains `num_seeds` and hashes it into the control's
  identity (control schema **v8**), the persisted MIPROv2 study contract
  records `validation_num_seeds` (study schema **v7**, per-intent context
  **v3**), and `GepaDetailedResult` gains `validation_num_seeds`
  (`whetstone.gepa_detailed_result/v2`). The platform step executor also
  carries the launch's extra pools into the opening Step, without which no
  MIPROv2 run reaches step 0 on the platform path. [#143]

[#142]: https://github.com/danielle-rothermel/whetstone-ai/pull/142
[#143]: https://github.com/danielle-rothermel/whetstone-ai/pull/143

- **The audit holds MIPROv2 and GEPA to the repeat count they recorded.**
  Both now state the repeats every in-search evaluation resolved to, and
  that number is what a manifest diff reads, so it is the number that has
  to be checked rather than trusted. `MIPRO_REPEATS_AS_RECORDED` and
  `GEPA_REPEATS_AS_RECORDED` compare each evaluation's own
  `EvalEvidence.num_seeds` against the recorded count.

  The gap is real on both sides.
  `Miprov2Study._validate_evaluation_binding` already rejects a transcript
  whose `validation_num_seeds` disagrees with its own recorded binding
  *requests*, but it walks no `eval_result_ref`, so what the engine
  actually billed is unchecked — the MIPROv2 negative fixture violates
  exactly that layer. GEPA has no such cross-validator at all, and its
  budget cannot substitute for one: a metric call is one candidate-task
  evaluation at any repeat count, so a run that searched at one repeat
  under a design registering three is indistinguishable in
  `total_metric_calls`. Each invariant ships a negative fixture that FAILs
  it alone.

- **GEPA's row estimate converts its metric-call pin at `K_REPEAT`.** The
  Stage-1 gate compares task-model rows, and GEPA's budget is pinned in
  *metric calls*; the conversion used to be the identity, on the argument
  that every row costs at least one metric call. whetstone-ai 0.1.11 broke
  that: a metric call is one candidate-task evaluation at any repeat count,
  and each repeat bills its own row. `GEPA_TASK_CALL_CEILING` is now
  `gepa_task_call_ceiling(k_repeat)` and returns `200 x K_REPEAT` — **600
  rows** at the design's `K_REPEAT = 3`.

  Left unscaled the gate would have judged a run entitled to 600 rows
  against a limit of `200 x 1.5 = 300` and aborted the healthy GEPA run it
  exists to catch fan-out in. The gate's own docstring now states that
  GEPA's pin is in metric calls while its estimate is in rows, since that
  is the pair most easily confused. The plan's MEASURED rows are scaled the
  same way — every Wave 3 measurement was taken at one repeat, now pinned
  as `MEASUREMENT_NUM_SEEDS` rather than left in a reproduce command — so
  MIPROv2 prints 735 rather than 245 and GEPA 219 rather than 73, and the
  study-wide total moves from 63,326-78,002 to **65,326-80,002**.

- **The storage note is re-derived from measurement.** The README budgeted
  4.6 KB of `runtime.sqlite` per evaluated row and ≈0.4-0.5 GB for the
  study. Measured at the pinned search shapes it is **~20 KB/row**: a COPRO
  run plans 4,752 rows and leaves a 90 MB run directory, and a MIPROv2 run
  leaves 209 MB. The eight-arm design leaves **26 run directories** across
  Stage 1 and Stage 2, totalling **≈2.5 GB** before the study's own store —
  so the per-run directories dominate it rather than the reverse. The note
  now also gives fake-transport wall time per run as rehearsal guidance:
  COPRO ~25 s, MIPROv2 ~10 min.

- **The MIPROv2 effect ceilings are derived from the control's search
  shape.** `miprov2_budget` returned four fixed numbers -- 32 bootstrap
  generations, 32 proposal calls, 32 evaluations, 256 task rows -- chosen
  when the search shape was this module's own small default. The Step 10
  design is far above them: 10 trials on a minibatch of 35 at
  `K_REPEAT = 3` plans roughly 2,900 rows against a 256-row ceiling, and
  under 0.1.11 every bootstrap attempt bills one row per repeat against a
  ceiling of 32. Both were exhausted mid-run, *inside* the durable run
  boundary, before the trial schedule was, and the MIPROv2 arms died with
  "MIPROv2 bootstrap_generations budget exhausted".

  The ceiling now scales with the trainset, valset, batch, trial count,
  candidate count, and repeat count the control pins, times a headroom
  factor. It stays a deliberately loose upper bound: the guard exists to
  catch runaway fan-out, and a ceiling tuned close to the expected cost
  converts an ordinary run into a spurious mid-run failure. A call with no
  control keeps the small-run ceilings unchanged.

- **A MIPROv2 control takes its repeat count from the engine it is bound
  to.** `Miprov2Control.num_seeds` is new in whetstone-ai 0.1.11 and
  defaults to 1, and `build_miprov2_control` did not set it, so a control
  asked for one repeat per in-search evaluation while the engine beneath it
  was bound at `K_REPEAT`. `engine_binding.resolve` refuses exactly that
  disagreement -- "engine sampling repeats (3) do not match the requested
  num_seeds (1)" -- so every MIPROv2 arm of the Step 10 design still died
  inside the durable run boundary on 0.1.11, at the same message 0.1.10
  produced for a different reason. The count is now read off
  `engine.sampling.num_seeds` rather than taken as a parameter: the bound
  engine's seed plan is the authority the resolver checks against, so there
  is no second place for the two to disagree.

- **A fake-transport COPRO arm can fill the breadth the protocol pins.**
  A family scripts exactly two proposal bodies -- the ceiling draft and the
  naive seed -- and the seed fills a slot COPRO never requests, so an
  unaided fake round lands one draft. Pinning the arms to the registered
  6x3 shape therefore made every fake COPRO run die inside the durable run
  boundary with `copro_proposal_cardinality` ("expected 6, actual 2"),
  before it evaluated anything: Stage 1 of a `--without-codex` rehearsal
  aborted on its first arm. The previous rehearsal missed this because the
  arms had not yet been forwarded their shape and ran COPRO at the
  runner's 2x1 default.

  `FamilySpec.rehearsal_proposal_bodies` derives `breadth - 1` further
  distinct drafts from the family's own ceiling template, and the study
  runner hands them to a fake COPRO arm. They are refused on a paid
  transport, where the proposer writes its own bodies, and `null-random`
  is deliberately excluded: it binds its own generative transport and
  would never read them.

- **Fidelity arms no longer produce efficacy verdicts.** MIPROv2's
  `zeroshot` and `ground_only` modes run once each as evidence for two
  audit invariants. They pass their audits and are measured on held-out,
  which was the whole basis the report used to decide a verdict, so both
  were reported as efficacy results — five claims where the design
  pre-registers four, each from a single run and none in the Holm family.
  `ArmKind` gains `FIDELITY` beside `REAL` and `NULL`, `ArmRecord` carries
  the role, and it is hashed into the pre-registration as `kind_by_arm`,
  so an arm cannot be promoted into the family after its interval is
  visible. The analysis writes no held-out row for a fidelity arm rather
  than computing one and declining to print it, and the report lists them
  in their own section with their audit result and no verdict column. A
  golden pins the Holm family to exactly the four `REAL` arms.
- **The pinned search shape reaches the runs it describes** (manifest
  schema v10). Four registered control values never arrived.
  `StudyOptimizerRunner._spec_for` forwarded neither `copro_breadth` nor
  `copro_depth`, so every COPRO and null-A run took `RunSpec`'s smoke-run
  defaults of 2 and 1 where the protocol pins 6 and 3 — and the estimator
  defaulted to the same two values, so the estimate and the run agreed
  with each other while disagreeing with the design both described.
  `ArmRecord` likewise had nowhere to carry COPRO's breadth/depth or
  MIPROv2's trials/candidates, so `spec_from_manifest` rebuilt all four as
  `None` and a manifest-driven MIPROv2 arm ran 2 trials against a
  registered 10. And `build_gepa_control` hardcoded
  `reflection_minibatch_size=1` against the protocol's 3, with no spec
  field able to carry the pin. All four are now pinned in `protocols.py`,
  carried through `ArmDesign`/`ArmSpec`/`ArmRecord`, hashed into the
  pre-registration's new `search_by_arm`, and forwarded to the run; an arm
  record disagreeing with the pinned block is refused as a
  `PreRegistrationViolationError`, as the split and minibatch already were.
- **COPRO's estimate counts evaluating rounds, not its finalizing step.**
  The row estimate multiplied by `depth + 1`, overstating COPRO and null-A
  by a whole round — 6,336 rows per run at the pinned shape against the
  protocol's own 4,752. A run does record `depth + 1` *steps*, but the
  extra one is finalization, which consumes no budget and issues no
  intents. The estimator and the protocol now agree, pinned by a golden.
- **The Codex admission cap reaches the run from the design.**
  `bound_stage_environment` built the runner without `codex_capacity`, so
  the pinned cap arrived only because `RunSpec`'s own default happened to
  equal it.
- **One owner for the GEPA metric-call pin.** `protocols.py` and `gates.py`
  each held their own `200` and the arm forwarded the gates copy; `gates`
  now imports the protocol's, with an equality golden.
- **A `--without-codex` projection can no longer be mistaken for the
  study.** Its manifest was byte-indistinguishable from the
  pre-registration — same `study_id`, same `models` block, nothing
  recording that an arm had been dropped. The projection now takes a
  `-without-codex` study id, records `design_projection`, marks
  `codex_agent_model` as omitted, and the report prefixes its headline;
  an arm stage refuses a projection carrying a registered protocol id.
- **`--study-id` may not claim a design the invocation is not.** A toy or a
  projection could be initialised as `step10-c19`, leaving every artifact
  downstream citing the pre-registration while holding a smaller design.
- **The manifest records no fabricated assignment digest.**
  `assignment_doc_sha256` held the sha256 of a fixed marker string — a
  digest of nothing that read like provenance. Step 10's authorising
  assignment is the protocol document itself, so the field is now absent
  and the report says so.
- **The registered protocol document ships in the package.** The default
  path pointed into one machine's `~/drotherm/data` tree, and `init`
  refuses to author a study without reading it, so `whetstone-study init`
  could not run from any other checkout. The text now lives at
  `optim/study/protocol_docs/step10-c19-protocol.md`, byte-identical to the
  durable copy and pinned by a golden digest.
- **A cross-transport amendment takes the measurements its dropped evidence
  bought.** `stage0 --replace-design` onto another transport dropped the
  arm stages, their runs, the selections, and the held-out claims, but left
  `official_scores` and `report_spend` behind. Run ids are deterministic,
  so the replacement stage recomputed the very names that were dropped and
  `official_score_for` read back a score measured on the transport the
  study had just left — presenting it as this study's selection evidence
  and never re-buying it on the transport now in use. `report_spend` was
  the same error in money: the stage's reporting row is folded from the
  durable per-evaluation records rather than from the row, so entries
  surviving their stage were folded by the next invocation of it, billing a
  paid stage for the fake-transport evaluations an invalidated invocation
  bought. Both are now dropped in the same amendment and counted on it, as
  `dropped_official_scores` and `dropped_report_spend`. Belt and braces on
  each: an `OfficialScoreEntry` and a `ReportSpendEntry` each record the
  transport they were bought on, the score read-back requires it to match
  the stage's, and the reporting fold keys on `(stage, transport)` — so
  neither a stale measurement nor a stale bill can be reused whatever put
  it in the manifest.
- **The reporting pass is durable, so a resume neither loses its spend nor
  bills it twice.** Official-selection scoring, the held-out evaluations,
  and the anchors each reach the provider before the stage's row is
  written, and their cost lived only in an in-memory ledger until that
  write. A crash in that window was wrong in both directions: the spend of
  everything already bought was stranded in a process that was gone, and a
  resume that re-folded the ledger onto a row already carrying it billed
  the same evaluations a second time. Each evaluation now records its own
  spend into the manifest's `report_spend` block the moment it is priced,
  keyed by the evidence's `(schema, content_hash)`, and the stage's row is
  folded from those durable records rather than from what this invocation
  happened to buy — so the fold is a function of what is on disk and is
  safe to repeat. `StageRecord` gains `report_spend` beside `spend`,
  because the two accumulate by opposite rules; `total_spend` derives the
  whole bill so the parts cannot drift from the total.
- **A resumed arm no longer re-buys the official scores it already paid
  for.** Official scoring is a provider call per run and ran
  unconditionally on every invocation, so resuming a stage re-scored every
  run of every already-reported arm purely to rebuild a report the
  manifest could have answered — a second charge that was invisible in the
  result, since the rebuilt report looked identical either way. Each run's
  score is now recorded in `official_scores` the first time it is bought,
  and a fully reported arm rebuilds from the manifest issuing zero scorer
  calls.
- **A study's minibatch design reaches the runs it describes.** The
  pre-registration hashed `minibatch_by_arm`, but `ArmRecord` had nowhere
  to carry it, so `spec_from_manifest` rebuilt every arm unbatched: a
  manifest-driven MIPROv2 study could pin a batch size, validate its own
  design hash, and then evaluate every trial on the whole valset. The arm
  record now carries `minibatch`/`minibatch_size`, they round-trip through
  the rebuilt spec, and an arm record disagreeing with the pinned
  `minibatch_by_arm` is refused as a pre-registration violation — the same
  class of check as the train/val split.
- **The recorded `seed` says what is actually on the wire.** The manifest
  recorded the statically bound seed control, which is unset, so
  `provider_calls[].seed` read `provider default` — telling a reader the
  provider chose the seed. The opposite is true: whetstone's eval contract
  puts a derived seed on every call via `derive_rng_seed(task_hash,
  seed_index)` and refuses a definition that cannot transport it. The
  field now records `derived per call (eval contract)`. `reasoning`,
  `temperature`, and `top_p` remain truthfully `provider default`.
- **null-A no longer flattens the template it controls for.** The
  perturber split the seed on whitespace and rejoined its tokens with
  single spaces, so every draft it produced lost the template's entire
  layout. On the real c19 seed that is six newlines and two blank lines
  holding the grid, the action list, and the question apart — destroyed on
  every seed, in every draft. The control was therefore running a
  structurally degraded prompt no real arm ever ran, which would have made
  a null-A delta partly a measurement of formatting damage rather than of
  selection on noise. The perturber now holds the whitespace runs aside
  and lays them back down unchanged, editing wording only: the output's
  whitespace runs are exactly the input's, and character similarity to the
  real seed rises from 0.854 to 0.910 on average (the collapsed layout put
  a 0.85 ceiling on a template no token had yet moved in).
- **An arm and its optimizer are no longer read as the same name.** The
  study runs one optimizer under more than one arm -- MIPROv2's three demo
  modes are three arms -- and three places read the arm id as an optimizer,
  which held only while every arm was named after one. `k_run_for` gave the
  two MIPROv2 fidelity arms `K_RUN = 5` instead of the protocol's 1,
  buying eight runs the design never registered; `arm_seeds` handed all
  three MIPROv2 arms seeds 2000-2004 of the same range, so three arms the
  report presents as independent shared an RNG stream; and `plan` passed
  the arm id to `estimate_optimizer_calls`, so both fidelity arms printed
  "no estimate" and dropped out of the budget entirely. The seed and
  run-count tables are now keyed by arm with an optimizer fallback, and
  `StudySpec` grows `optimizer_by_arm`. `_arm_seeds_from`, which rebuilds
  an arm's seeds when a manifest is read back, looks up by arm id too, so
  a manifest authored at the disjoint ranges reads back at them. Every arm
  named after its optimizer is unaffected.

- **A report's scores are checked by the family that produced them.** The
  `EvalReport` schema re-derives every scored observation to validate it,
  but re-derived it as normalized exact match — a c19 rule wearing a
  family-agnostic name. C18 scores the terminal verdict it extracts from a
  reasoned reply, so a *correct* c18 answer ending in `True` reported 1.0
  while the schema recomputed 0.0, called the row a lie, and refused the
  whole report: publication failed for the entire run over a check that was
  wrong rather than a score that was. The check now routes through
  `whetstone_envs.scoring.families`, the single owner of how a family's
  generation becomes a score, so it keeps its purpose — a reported score
  must equal what the family's own scorer yields for that row — without
  restating any family's rule. Found by the real-Codex ladder's c18 rung;
  the fake transport replies with bare gold, which both rules score alike.

- **Reading a scored report no longer requires the stack that wrote it.**
  Routing that same check through the *optimizer's* family registry pulled
  in whetstone-ai, which the `optim` extra installs only on Python 3.13+, so
  validating any report with a scored observation raised
  `ModuleNotFoundError: dr_providers` on a base install. The per-family
  scoring rule is pure, so it now lives in `whetstone_envs/scoring/families.py`
  and imports nothing from `optim`; both eval-node runners and the schema's
  check call the same `family_score`, and a golden test pins that they agree
  for representative outputs of both families.

- **The Codex arm's preflight probes the model the arm will actually use.**
  The study stage passed the run's `task_model` to `preflight_codex_session`,
  which is an OpenRouter route the Codex CLI cannot run at all — so the guard
  cleared a session no arm would ever open, and a real study would have failed
  on the Codex arm's turn, after COPRO, MIPROv2, and GEPA had been paid for.
  Both the runner and the preflight now resolve the agent model through one
  shared `resolve_codex_agent_model`.

- **A rejected trajectory row keeps the reason it was rejected.** The
  projection dropped `terminal_failure.message` for a call rejected after
  admission. The structured failure is evidence the schema forbids on such a
  row, but the message is not, and it was the only account of why a paid-for
  call scored nothing that reached the projected trajectory.

- **The real-Codex ladder reports what it observed, not that pytest exited
  0.** `RESULT: all rungs passed` was derived from the exit status, and pytest
  exits 0 on a fully skipped session — which live-skips on rungs 2/6/7/8 make
  the expected path. The footer is now gated on the parsed rung table (every
  row PASSED, and as many rows as the ladder has rungs, counted by collection
  rather than pinned), and a ladder that was not fully observed exits 1. The
  parser feeding that table also stopped misreading `::test_rungN` printed
  inside temp paths and tracebacks, and no longer files a verdict against the
  wrong rung.

- **The ladder's seed-preference skip no longer swallows adapter bugs.**
  `_skip_if_agent_chose_the_seed` matched any `codex_selection_contract`
  failure, including the `base is None` bookkeeping site — a genuine defect
  reported as the known, accepted risk. It now additionally requires no
  accepted candidates, no retained candidate, and the mutation-diff message.
  whetstone-ai 0.1.9 (#138) treats a seed-identical selection as
  `seed_retained`, so this skip became unnecessary when the envs pin moved
  to 0.1.10, and it is deleted below.

- **The ladder's tripwire exception is scoped to ladder-only sessions.** A
  mixed `-m ""` session that collected rungs alongside ordinary tests
  disarmed `WHETSTONE_ENVS_FORBID_REAL_CODEX` for all of them; the claim now
  requires the session to have collected nothing but rungs.

- The Python 3.12 CI job went red on `ty`: `tests/real_codex/**` was missing
  from the unresolved-import overrides. `tests/optim/test_real_codex_preconditions.py`
  also hard-errored at collection there, because it reached the ladder's
  decision function through a conftest that imports the optimizer stack — so
  the coverage that stops an all-skipped ladder being reported as green was
  itself invisible on the job most likely to lose it. The pure decision now
  lives in `tests/real_codex/preconditions.py`.

- `stage0 --replace-design` that records an amendment discards the previous
  Stage-1 call-count verdict: a pilot gate describes the design it was
  computed against, so Stage 2 owes the amended study a fresh pilot.

- **An arm stage refuses to reuse a run directory it cannot claim.** Run
  directories are named deterministically from arm and seed, which is what
  makes a crashed stage resumable — and also what let a cross-transport
  `--replace-design` leak a run across the amendment. The amendment drops
  the stale runs from the manifest, but their directories stay on disk
  under exactly the names the replacement stage computes, so that stage
  found a directory, skipped `run_optimizer`, and recorded the old fake run
  as a paid one: a manifest that reads as a paid study whose numbers came
  from the free transport. A directory is now reusable only when its **own
  artifacts** — the transport, family, model, and run id its trajectory
  report records — say it is the run this invocation would produce.
  Otherwise the stage refuses, naming the directory and both recoveries;
  it never silently re-runs over artifacts that may be paid evidence and
  never silently reuses someone else's run. A directory that records no
  readable identity is refused on the same grounds, because a run that
  cannot vouch for itself is not evidence that it matches. The new
  `whetstone-study run --discard-stale-runs` authorizes discarding such a
  directory instead; it is off by default and, like the other invocation
  authorizations, stays out of the pre-registration hash. The amendment
  record gains `dropped_run_directories`, so an operator resolving a
  refusal can see which directories the drop orphaned without
  reconstructing the naming rule by hand.
- **A cross-transport amendment clears the leakage verdict it invalidated.**
  `manifest.leakage_check` is L6's mechanical pass over the very run
  artifacts the amendment drops, and it was left in place — so a report
  regenerated after the amendment inherited a *passing* leakage result
  established over evidence the study no longer holds, which reads exactly
  like a study whose leakage rules passed over its current runs. It is now
  cleared in the same update that drops the evidence, and the report
  already treats an absent block as not-established. The other verdicts are
  deliberately untouched: `gepa_sizing` and `fanout_check` measure the
  optimizer's own mechanics before Stage 1, `balance` is the key's balance
  at each spend gate, and `c18` carries its own separate run list.
- **A resumed arm stage keeps the spend it already measured.** A stage that
  crashed after its manifest write has already paid for its runs and
  already recorded what they cost; resuming it re-runs nothing, so the
  replacement stage record was built from an empty set of executed runs and
  overwrote the measured bill with silence — the ledger then rendered a
  fully paid stage as `UNLEDGERED`. An arm stage's spend is now merged onto
  whatever its row already carries rather than replacing it, folded per
  role so a partial resume bills both halves and an unknown `usd` on either
  side keeps the total unknown. A stage's row can only grow. A first run of
  a stage has nothing to merge and still records exactly the runs it
  executed, so the ledger's rows continue not to sum to more than the study
  spent.

- **Stage 2 requires a Stage 1 whose call-count gate passed.** The gate
  catches a fan-out bug — an optimizer whose minibatch intents silently
  expanded to the full valset — for the price of one run per arm, which is
  the whole reason the pilot exists. Its verdict was evaluated inside the
  Stage-1 process and recorded nowhere, so a Stage 2 invoked directly after
  Stage 0, or re-invoked after a Stage 1 whose gate had failed, skipped the
  check entirely and paid for the full five-run design behind it. Stage 1
  now records the verdict — passing or failing, with the overrunning runs
  named — in a new `call_count_gate` manifest block, and Stage 2 refuses
  before dispatching any arm when it is missing or failed, naming which of
  the two it is because the actions differ. The manifest schema is `v5`
  accordingly.
- **A held-out delta is only as complete as its thinner side.** The paired
  completeness weighting read the optimizer's achieved row counts alone, so
  an arm that measured every row against a naive anchor whose own held-out
  evaluation had lost most of its repeats reported completeness `1.0`: the
  anchor's thin fallback mean was treated as fully observed, and the delta
  cleared the 90% backstop and was claimed on evidence that was mostly
  missing on one side. Completeness is now the per-task **minimum** of both
  sides' achieved counts — conservative by construction, and the minimum
  rather than the product because the weight is a fraction of a planned
  sample, so two fully observed sides must still weight `1.0`. The held-out
  row records the paired figure, which is what the report's backstop reads,
  and carries the anchor's own completeness beside it as
  `anchor_completeness` so a downgraded arm is not misread as having failed
  to measure itself.
- **A Codex-bearing stage proves the session before it buys anything
  else.** The early guard checked only *authorization*, so a stage with
  both opt-in halves present still discovered an unusable Codex — an
  unsupported platform, a binary absent from the run's PATH, an expired or
  missing session — when the Codex arm's own turn arrived, after COPRO,
  MIPROv2, and GEPA had been paid for. The same preflight `run_optimizer`
  reaches now runs at the guard, before any arm is dispatched, and a
  failure refuses the stage with the preflight's own diagnosis preserved as
  the cause.
- **GEPA study arms run at the pre-registered metric-call budget.** The
  arm's `RunSpec` carried no `gepa_max_metric_calls`, so `build_gepa_control`
  resolved its own `auto` default — roughly `train + val + 1`, about 89 on
  the study's 44/44 split — while the Stage-1 call-count gate and the power
  design are both built on the pinned `GEPA_MAX_METRIC_CALLS_PINNED = 200`.
  A GEPA arm was therefore judged against a ceiling it never ran at. The
  runner now forwards the pin, and on GEPA arms only.
- **A stage refuses arm records that disagree with the pinned split.**
  Stages 1 and 2 rebuild each arm's runnable spec from `ArmRecord`'s
  mutable `train_size`/`val_size` while `pre_registration.split_by_arm` is
  the immutable, hashed truth, and the two were never compared — so an
  edited record could run MIPROv2 or GEPA at a partition the design never
  registered, under a design hash that still validated. Loading a spec now
  raises `PreRegistrationViolationError` when they disagree, and an arm
  stage additionally refuses arms the pinned block does not name at all —
  an arm declared after the design was pinned has no registered partition,
  run count, or place in the correction family. Stage 0 stays permissive
  there, because adding an arm and re-pinning is exactly how
  `stage0 --replace-design` records an amendment.

### Fixed

- **The reporting pass's spend is ledgered, so a stage total is the whole
  bill.** Official-selection scoring, the held-out evaluations, and the
  anchors' re-measurement reach the provider through the evaluation engine
  outside any optimizer run, so neither the run fold nor Stage 0's evidence
  route could see them — and those are the calls every efficacy claim is
  finally made against. `RoleScorer` now prices each evaluation from its
  own persisted rows as it issues it, through `study/spend.py`'s row-derived
  aggregation: one record per role per evaluation, each citing the evidence
  it was derived from, collected by `ReportSpendLedger` and folded onto the
  stage's `StageRecord.spend` once the pass is done. It is merged onto the
  arms' row rather than replacing it — the two are separate bills for one
  stage — and the fold re-applies the honesty rules, so one unpriced
  reporting evaluation withholds the role's whole `usd`. A fake-transport
  stage is skipped, as everywhere else: its rows would total to a bill
  nobody owes.
- **The standalone eval path publishes what it spent.** `run_c19_evaluation`
  wrote `eval-report.json` and `runtime.sqlite` and no cost document at all,
  so a held-out evaluation spent real money that no artifact recorded.
  `EvalReport` gains a `spend` block (schema `v2`) projected from the
  evaluation's own persisted rows, and `whetstone-eval` writes a `cost.json`
  beside the report in the same shape optimizer runs publish. Only the task
  model appears — an evaluation runs no proposer, and an all-zero proposer
  row would claim it measured one and found it free — and an evaluation that
  evidenced no provider call publishes neither block rather than a zero.
  The read path is strict on the current version, so `eval_report/v1`
  reports are not loadable by this release; existing ones remain on disk as
  historical artifacts rather than being migrated or read.

### Fixed

- **The study's null-A arm runs a real optimizer, so the selection control
  is real.** `StudyOptimizerRunner._run_null` handled both nulls and
  bypassed the runner entirely: null-A evaluated nothing, recorded
  `observed_task_calls=0` and `spend=()`, pointed all three of its evidence
  refs at one synthesized record, and its "perturbation" was a
  `(variant N)` suffix on the naive template rather than the protocol's
  placeholder-preserving perturbation. An arm that never evaluated cannot
  control for selection-on-noise, because no selection happened — so a
  study's headline null-A comparison was against a stub. Null-A now
  dispatches `run_optimizer(optimizer="null-random", …)` like every other
  arm: COPRO's search shape with an uninformative proposer, evaluating on
  the internal split, spending the same proposal budget, and leaving a run
  directory, a result, a passing audit, and priced cost rows. Its arm-stage
  spend folds into `StageRecord.spend` the way every other arm's does, and
  a recorded null-A resumes by re-reading its run rather than by
  re-synthesizing a template no evaluation ranked. `whetstone-study plan`
  already priced it at COPRO's shape; that estimate is now what the arm
  costs rather than a mis-estimate of a stub. Null-B is unchanged and still
  runs no optimizer — it proposes nothing, so there is no search to drive —
  and `null_random_template` is deleted with the path that used it.

### Changed

- **`--miprov2-minibatch` requires `--miprov2-minibatch-size`.** Left
  unset the batch resolved to the whole validation split, so minibatching
  was on in name only — and the run's `mipro_minibatch_sizing` invariant
  then FAILed the audit of a run that had already spent. The combination is
  refused at pure spec validation, in a message naming both flags, so the
  same finding is free.
- **The Codex agent model is pre-registered design, not a runner default.**
  `manifest.models.codex_agent_model` had no production writer and nothing
  read it: the stage guard resolved the agent through the *runner's*
  `CODEX_DEFAULT_AGENT_MODEL`, so whatever that constant said silently
  became the study's proposer. A study declaring the Codex arm now names
  its agent in the hand-authored `models` block, `StudySpec` carries it,
  and `require_pinned_codex_agent_model` refuses a stage whose resolved
  control disagrees with a `PreRegistrationViolationError` — the same class
  of refusal an unregistered split gets, because running an unregistered
  proposer is the same error. `CODEX_DEFAULT_AGENT_MODEL` keeps its job:
  the default for a single run nobody pre-registered. A study that declares
  no Codex arm pins nothing, and a pin without an arm is refused too.

### Added

- **The manifest records the effective provider call config** (schema `v8`).
  `models` named which model a study meant to run and `stages` named which
  transport it ran on, but nothing recorded what the transport actually
  bound — the resolved route and the request controls — so neither the
  spend model nor the claim that two stages ran "the same experiment" was
  auditable from the manifest. `models.provider_calls` now carries one
  `ProviderCallRecord` per transport a stage has bound, read off the
  prepared experiment rather than off the argument (on the fake transport
  that argument is `None` and the effective config is the reference default
  the experiment builds for itself), and the report's design section prints
  it. Recorded, not hashed: it is a property of the invocation like the
  transport, so two studies of one design still pre-register identically.
  Every control appears whether or not the study set one —
  `"provider default"` is a real and consequential state, and it is why the
  toy Stage 0 billed thousands of reasoning tokens per call. **No
  reasoning-effort knob is added**: whether the design pins one is an open
  decision, and a settable field would answer it by accident.
- **A MIPROv2 arm's minibatch size is a pre-registered design field.**
  `ArmSpec` gains `miprov2_minibatch`/`miprov2_minibatch_size`, which
  travel together — on without a size is refused at the design level for
  the reason the runner refuses it — and the size enters the
  pre-registration's hashed payload as `minibatch_by_arm`, alongside
  `split_by_arm`. An arm that scored every trial on the whole valset and
  one that scored on a sampled batch bought different evidence for the same
  claim, so two designs differing only in the batch size now pin
  differently.

### Added

- **A Stage-1 MIPROv2 arm cannot silently take the shape that crashed.**
  `num_candidates < 3` with minibatching is refused at spec validation,
  pointing at whetstone-ai #137, **unless** the installed whetstone-ai is
  at least 0.1.9 — the release whose `select_promotion` degrades to DSPy's
  own behaviour instead of raising `No valid program found in
  param_score_dict` inside the durable run boundary. The gate is a floor,
  not the pin: this repo pins 0.1.10, which is above it, so the refusal only
  re-arms on a downgrade. An absent or unrankable version reads as unfixed
  and keeps the refusal, because the uncertain case should cost a validation
  error rather than a run that dies mid-flight.

### Changed

- **Pins published whetstone-ai 0.1.10.** Three upstream fixes matter here.
  [#138] (0.1.9) records a seed-identical selection as `seed_retained`
  rather than a `codex_selection_contract` violation, so the real-Codex
  ladder's seed-preference live-skip is gone: an agent that decides the seed
  wins no longer decides whether a rung is observed. [#137] (0.1.9) gives
  MIPROv2 a spent-combination fallback, so `num_candidates=2` with
  minibatching no longer raises `No valid program found in
  param_score_dict` inside the durable run boundary. [#140] (0.1.10)
  redirects the Codex agent's `HOME` to a per-run scratch directory and
  quotes the CLI's own stdout error items in failure messages — the envs
  rung-9 "skills" failure had hidden a 401 behind a message that named
  neither.

[#137]: https://github.com/danielle-rothermel/whetstone-ai/pull/137
[#138]: https://github.com/danielle-rothermel/whetstone-ai/pull/138
[#140]: https://github.com/danielle-rothermel/whetstone-ai/pull/140

### Security

- **The test suite cannot reach a real Codex session, even under
  `monkeypatch`.** The two-part opt-in is process state, and setting the
  environment half is the ordinary way to test that a gate lifts — so an
  authorization test that supplied no scripted seam could reach the real
  CLI by way of a session probe that runs *after* the opt-in is satisfied.
  `refuse_unauthorized_real_codex` now honours a new
  `WHETSTONE_ENVS_FORBID_REAL_CODEX` above every other input, and the
  suite's session fixture arms it for the whole run: a test may
  monkeypatch the allow variable and still cannot reach a real session,
  while scripted runs through a `CodexTestSeam` are unaffected because they
  reach no real CLI to forbid. Every production path to the real preflight
  or adapter — including the study harness's early stage guard — routes
  through that one gate, so the tripwire cannot be bypassed by reaching a
  preflight another way.

### Changed
- Pins published whetstone-ai 0.1.8, whose Codex-direct optimizer is the
  first to have produced evaluations against the real `codex` CLI: the
  output schema, the MCP tool host and approval mode, and the
  `model_route`/`base_ref` the agent must supply were all fixed there.

### Added

- **Every optimizer with a train/val concept runs on an explicit disjoint
  split.** `--train-size` and `--val-size` are required for
  `--optimizer miprov2` and `--optimizer gepa` and refused on the
  optimizers that have no such concept. They partition the internal split
  deterministically — trainset first, valset next — so the two sets are
  disjoint and reproducible from the spec alone, and both are recorded on
  the control the run persists. MIPROv2 bootstraps its demonstrations from
  the trainset and GEPA writes its reflections from it, while both score
  on the valset: an overlapping split would let demonstration or
  instruction memorization read as an in-search improvement, which is not
  a distinction the study can afford to lose while the setup is being
  debugged. There is no default, because a run that did not state its
  partition could not be audited for this. `ArmSpec` carries the two sizes
  as design fields; `ArmRecord` records them and the pre-registration
  hashes them per arm as `split_by_arm`, so an arm rerun at a different
  partition is a different pinned design rather than the same one, which
  the manifest schema records from `v4` on. `default_arms` pins the protocol's
  44/44 of the internal 88 — an even split keeps one full valset pass
  affordable while leaving the bootstrap a 44-task trainset, and the two
  cover the internal split exactly, which GEPA requires. Two new audit
  invariants, `mipro_train_val_disjoint` and `gepa_train_val_disjoint`,
  check the persisted control's two sets are disjoint and that every
  evaluation the run paid for touched only those tasks.
- **GEPA's partition must cover the internal split exactly.** whetstone's
  GEPA factory builds its data registry from the whole internal split and
  then requires the control's trainset and valset to cover it, so
  `train_size + val_size < internal` is not a legal GEPA partition — it was
  rejected inside the durable run boundary, after the run directory
  existed. `run_optimizer` now refuses it at spec validation with a message
  naming the coverage requirement, so a partial GEPA partition leaves no
  run directory behind. MIPROv2 is unaffected and keeps the `<=` rule: it
  bootstraps and scores without building a registry over the whole split,
  so a partition that leaves tasks unused is legal for it.
- **`whetstone-study run --allow-real-codex` authorizes a Codex stage.**
  The study's optimizer runner forwards the flag onto the Codex arm's
  `RunSpec` and onto no other arm's; `WHETSTONE_ENVS_ALLOW_REAL_CODEX=1`
  remains the other half of the gate. Without both, a Stage 1 or Stage 2
  whose design names the Codex arm is refused **before any arm runs**,
  which is the spend-safety property: the refusal inside `run_optimizer`
  arrives on the Codex arm's own turn, by which point every arm ordered
  ahead of it has already been paid for. The authorization is a run-time
  permission rather than a design choice — it is not an `ArmSpec` field, it
  is not recorded in the manifest, and it does not enter the
  pre-registration hash, so two runs of one design pre-register identically
  whether or not the operator was allowed to bill a session.
- **`whetstone-study plan` states the pre-registered MDE.** One row per
  pinned `tau^2` (0.05 and 0.10), computed from `power.py` at the
  manifest's own held-out size and `K_REPEAT` and at the worst-case binary
  `sigma^2` of 0.25 — `0.0622` and `0.0690` at the study's 440 and 3.
  Labelled as pre-registered rather than measured, because Stage 0 measures
  both variances and records the MDE that follows. `plan` is the command
  read before authorizing spend, and it previously said nothing about what
  effect the design could resolve.
- **`PROTOCOL_SPLIT_SIZES` in `optim/study/spec.py`** declares the study's
  `(88, 132, 440)`, pinned by a golden test. It is deliberately not the c19
  generator's `DEFAULT_SPLIT_SIZES` of `(88, 132, 132)`: the protocol
  pre-registered a held-out split of 440 because the design's MDE depends
  on it.
- **The Codex arm runs.** `--optimizer codex` drives the study's
  foreign-agent arm through the same `run_optimizer` every other arm uses:
  `whetstone_envs.optim.codex` builds a `CodexControl` and a `CodexAdapter`
  from the family's own render contract and mutation field, and
  `prepare_codex_run` binds the one Tool the agent may use. The agent
  searches out of process under dr-exec containment and reaches exactly one
  MCP tool, which evaluates a candidate on the internal split. It is the
  first `TOOL_USING` run here, so it is the first to get a durable
  admission authority and a tool executor: its capacity is enforced across
  processes because the evaluation server that admits its calls is a
  different process. `--codex-capacity` is that cap and defaults to the
  pre-registered 8; `--codex-binary`, `--codex-model`,
  `--codex-reasoning-effort`, and `--codex-wall-seconds` configure the
  agent. An authentication preflight runs before any capacity is
  committed, and no flag or `RunSpec` field can substitute it — the
  scripted stand-in is reachable only through a keyword-only test seam.
  **Known platform limitation:** the Codex containment profile is macOS
  `sandbox-exec` only, so the spawning tests are Darwin-gated. Everything
  above the sandbox — the control, the runtime config, the capacity
  arithmetic, and the audit — runs everywhere.
- **A real Codex run is refused unless it is deliberately opted in.** The
  authentication preflight is not a spend guard: it proves a session by
  *spawning* the Codex CLI, and on a machine with `~/.codex/auth.json` that
  spawn succeeds and is billed. So "the caller passed no test seam" used to
  mean "the run reaches the real, paid CLI" — which a study arm, a
  parametrization over the optimizer list, or a plain `--optimizer codex`
  could all trigger without meaning to buy anything. `run_optimizer` now
  refuses a Codex run outright, raising `RealCodexRefusedError` before any
  preflight, adapter, admission authority, or subprocess exists and before
  the durable run boundary creates a directory, unless either a
  `CodexTestSeam` is supplied or *both* halves of the opt-in are present:
  `WHETSTONE_ENVS_ALLOW_REAL_CODEX=1` in the environment and
  `--allow-real-codex` (`RunSpec.allow_real_codex`). Requiring both means
  neither a serialized spec nor an exported variable authorizes spend on its
  own. A session-scoped autouse fixture asserts the variable is unset and
  clears it, so no test can opt in.
- **The pre-spawn identity assertion covers the model route and the task
  set, not just the Eval Config.** `eval_config_ref` does not pin the task
  model on the openrouter transport — the route is carried by the provider
  call config — so a runtime config naming a different model rebuilt to the
  *same* Eval Config and the run would have completed, reported a coherent
  trajectory, and measured a model the study never asked for.
  `build_codex_adapter` now proves `task_model_identity_hash()` and
  `sampling.task_hashes` equal across the in-process engine and the rebuilt
  runtime-config engine as well, refusing loudly on either mismatch.
- **`whetstone_envs.optim.codex_runtime` carries the launch across the
  process boundary.** The Codex MCP evaluation server rebuilds its engine
  from one serialized config, and whetstone-ai's
  `ReferenceEvalRuntimeConfig` always rebuilds the *toy* experiment — a
  limitation whetstone-ai names itself. A study run wired to it comes up
  fine and then refuses every single tool call as "not bound to the
  engine's exact Eval Config", leaving the agent nothing to select.
  `EnvsCodexRuntimeConfig` carries the family's generation parameters
  instead and rebuilds the identical experiment, and the adapter builder
  proves the rebuilt Eval Config matches the run's before anything is
  spawned.
- **The Codex fidelity audit runs on every Codex run.** The six invariants
  and the shared one now execute against a real fake-CLI run rather than
  only against committed fixtures, and `reported_numbers_resolve` counts
  tool-mediated evaluations: a `TOOL_USING` run resolves no intent by
  design, so an invariant reading only the intent paths failed every honest
  Codex run for reporting nothing.
- **The trajectory report renders a Codex run's evaluations.** They are
  projected from tool evidence, each attributed to the candidate its Tool
  Call actually built, with the evaluation's own reward. Reading only the
  intent path showed a terminal candidate with no measurement behind it.
- `whetstone_envs.optim.audit`: offline fidelity audits over one optimizer
  run's durable evidence. `audit_run(run_dir)` reads `result.json` and
  `runtime.sqlite` through whetstone-ai's public API and returns an
  `AuditReport`; `python -m whetstone_envs.optim.audit <run_dir>` writes it
  to `audit.json` beside the result. An audit performs no network access, no
  re-execution, and no re-scoring, so it is equally valid in CI against
  fake-transport artifacts and against a paid run later. Exit codes separate
  a fidelity failure (`1`, an invariant was violated) from unreadable
  evidence (`2`, nothing was judged).
- The audit result contract (`AuditReport`, `AuditFinding`, `InvariantId`,
  `AuditStatus`, `EvidenceRef`) is a persisted format owned by
  `optim/audit/schema.py`, with its wire literals pinned by golden tests.
  `AuditReport.passed` is derived from the findings rather than stored, so a
  report cannot disagree with its own evidence.
- `registry.py` is the single place an invariant is enumerated, mapping each
  optimizer to its invariant tuple. It ships one worked invariant,
  `REPORTED_NUMBERS_RESOLVE` (every reported evaluation resolves to eval
  evidence in the run's own store), wired end to end with a negative fixture;
  the per-optimizer invariant sets follow.
- `optim/audit/gepa.py`: GEPA's eight fidelity invariants, each shipping a
  negative fixture that makes exactly it FAIL. They check that the terminal
  step persisted a result artifact bound to the run, that the candidate
  front is the per-instance argmax over internal validation scores and that
  selection came from it, that every mutated candidate traces to a
  reflection over an earlier evaluation of its recorded parent, that the
  metric-call counter advanced monotonically and terminalized at the
  configured ceiling, that every step persisted its rejected reflections,
  that every step which spent metric calls carries evidence of the spend,
  that the terminal candidate is a live search draft or an honest seed
  retention, and that a platform-dispatched run replayed each deferral
  episode identically.
- The GEPA audit reads the search result where it is actually persisted:
  the terminal step's history carries `GEPA_TERMINAL_ARTIFACT_KEY`, which
  resolves to a `GepaRunResultArtifact` and from there to the
  `GepaDetailedResult` and `GepaEffectTranscript`. There is no Pareto front,
  no per-instance score, and no reflection record under `GEPA_STATE_KEY`,
  which holds only `metric_calls_consumed` and `terminal`. No whetstone-ai
  change was needed: every record on that path is reachable through a public
  module.
- `GEPA_REFLECTION_MINIBATCH` is deliberately not shipped. Nothing persisted
  witnesses how many instance traces one reflection consumed, so it could
  never have a failing fixture. `GEPA_TERMINAL_ARTIFACT_PRESENT` replaces
  it, keeping GEPA at eight invariants.
- `_mutate.py` builds a negative fixture by copying a real fake-transport run
  and violating one named evidence field. It re-seals `OptimResult`'s
  self-verifying step wrapper refs and request chain, so a fixture stays
  schema-valid and fails only the semantic invariant under test, and it
  refuses a no-op rewrite that would let a negative test pass for the wrong
  reason. `reseal_run_binding` extends this to a mutation of the `OptimRun`
  record itself -- its optimizer control or seed candidate -- which is
  embedded in every step request and so needs re-deriving in all of them.
  `reseal_request_refs`, `reseal_candidate_refs`, and
  `rethread_snapshot_refs` cover the three further integrity refs a mutation
  can invalidate -- a step's own request ref, a candidate wrapper's
  self-verifying ref, and the next request's `prior_state_ref` /
  `prior_history_ref` -- and `reseal_all` runs all four passes in dependency
  order, which is what `mutate_run` now does. Resealing therefore has one
  owner: an optimizer whose mutations reach the request, a candidate
  payload, or a snapshot no longer re-derives its own copies.
- `optim/audit/copro.py`: COPRO's seven fidelity invariants, each a pure
  function over the run's evidence returning a finding with an evidence-ref
  citation for every judgment, and each shipping a negative fixture built
  from a real fake-transport run on which that invariant alone fails.
  `COPRO_BREADTH_PER_DEPTH` counts each round's measured occurrences against
  the persisted control's `breadth`; `COPRO_DEPTH_STEPS` checks the step
  count is `depth + 1` unless a terminal failure is recorded;
  `COPRO_INTERNAL_ONLY` checks every completed evaluation bound the
  control's internal Eval Config and recorded the internal role;
  `COPRO_BEST_SO_FAR` checks the finalizing step selected a candidate
  holding the maximum measured internal reward; `COPRO_DISTINCT_BASES`
  checks proposals within a round carry distinct bases;
  `COPRO_NO_SEARCH_EVALS` checks no step recorded search evidence; and
  `COPRO_TERMINAL_PROVENANCE` walks the terminal candidate's base chain back
  to the run's declared `initial_candidate_ref`. Missing or unreadable
  evidence reports FAIL rather than raising, so a defective run is judged
  rather than left unaudited.
- `RunEvidence` now resolves the run's optimizer control record, so an audit
  reads the configured search from the content-addressed control the run
  binds itself to rather than from a state-delta echo written by the code
  path under audit.
- The eight MIPROv2 fidelity invariants (`optim/audit/miprov2.py`), each a
  pure function over one run's durable evidence with cited evidence refs:
  demonstrations are bootstrapped before any instruction is proposed;
  `zeroshot` still runs DSPy's 3/0 grounding bootstrap and ships no demo
  set; `ground_only` is flagged `demo_mode:ground_only` rather than claiming
  frozen DSPy faithfulness; the recorded trials replay from a fresh seeded
  Optuna TPE sampler; every trial evaluation drew its scheduled batch from
  the validation split; the incumbent is re-evaluated on the full split on
  the configured cadence; bootstrap generations are paid through the
  evaluation engine rather than the proposer transport; and the observed
  trial count matches the configured budget unless a terminal failure
  truncated the run. Each reports FAIL rather than raising when its evidence
  is absent, so a run that persisted nothing is judged rather than skipped.
- The minibatch-sizing invariant doubles as the F16 fan-out assertion: a
  trial intent whose task set covers the whole validation split is the
  deferral row-expansion defect F16 names, and it fails there.
- Every MIPROv2 invariant ships a negative fixture built from a real
  fake-transport run in all three demo modes, plus one run with minibatching
  enabled so the periodic-full-evaluation invariant is exercised rather than
  permanently `NOT_APPLICABLE`.
- `COPRO_DISTINCT_BASES` now checks that proposals within a round are
  pairwise distinct *candidates* rather than that they carry distinct
  *bases*. Whetstone's COPRO adapter binds one base per round for every
  draft in it -- `base = initial`, taken before the round's drafts are read,
  with the ranked history reaching the proposer as prompt context rather
  than as per-draft bases -- so every proposal in every round carries the
  initial candidate's base by construction. The old rule could only have
  failed an honest run at a `breadth` above 2 and passed where it had
  nothing to compare. The invariant keeps its wire value and its
  "explicitly vacuous" reporting, and now ships a negative fixture: a round
  whose second proposal is a copy of its first.
- MIPROv2's trial schedule is a runner setting. `RunSpec.miprov2_minibatch`,
  `miprov2_minibatch_size`, and `miprov2_minibatch_full_eval_steps` -- and the
  matching `--miprov2-minibatch{,-size,-full-eval-steps}` flags -- reach
  `configure_miprov2` through the ordinary path, defaulting to the
  non-minibatched schedule the runner always produced. `optim/miprov2.py` used
  to pin `minibatch=False`, so the protocol's auto-light configuration was
  unreachable and `MIPRO_PERIODIC_FULL_EVAL` could only be exercised by
  patching the symbol that module imports; the audit fixture now asks for the
  schedule through the spec, so the setting under test is the one a study
  would set. A non-default value on another optimizer is refused at spec
  validation rather than silently ignored.
- `RunSpec.extra_proposal_bodies` supplies further scripted proposer bodies
  for a fake-transport run, and `FamilySpec.proposal_bodies` takes them.
  They are ordered *before* the family's naive body, because the naive body
  is the seed: a draft filling its slot is rejected as a no-op mutation, so
  bodies after it occupy slots the optimizer never requests. Without this a
  fake COPRO run above the smallest `breadth` underfills its round and ends
  in a proposal-cardinality failure, which is why no fake run had a
  genuinely multi-draft round to audit. Refused on a real transport, where
  the proposer writes the bodies.
- The Step 10 study report generator at
  `whetstone_envs.reporting.study_report`. `generate_study_report(*,
  manifest, out_dir)` writes a report packet -- `report.md`, `report.html`,
  a copy of `study.json`, and packet-local `doc.css` and `favicon.svg` --
  and is the study CLI's default `report` generator, so `whetstone-study
  report` no longer reports a wiring gap. The generator reads only the
  manifest and the evidence the manifest names, and recomputes no
  statistic.
- `python -m whetstone_envs.optim.study`. The study is documented under
  both a module invocation and the `whetstone-study` console script, and
  only the console script existed; the new `__main__` delegates to the same
  `main`, so a subcommand cannot exist under one entry point and not the
  other.
- Evidence provenance is a value type rather than a formatting convention.
  Every number the report renders is a `Figure` bound to the manifest field
  it came from and the `(schema, content hash)` pointer the manifest cites
  for it, and both emitters render a figure the same way. `figures_in`
  walks the built report, which makes "every rendered number resolves to a
  manifest pointer" a mechanical test over the report object rather than a
  regex over its output.
- The report distinguishes three verdicts, with fidelity gating efficacy:
  an arm whose audit failed is *not validated* and its held-out number is
  descriptive only, whatever its interval says. The h1 keeps that
  distinction too -- a fidelity failure never reads as a measured null
  result. Absent facts are named rather than invented: an unpriced role
  renders as `unpriced (n/total)` and never as zero, wall time is reported
  unrecorded because no manifest field or `cost.json` carries a duration,
  and an audit document that does not resolve shows the recorded verdict
  and says the finding table was not resolved.
- The report's HTML follows the `html-doc-polish` kit and renders with no
  network access: the stylesheet and favicon live inside the packet, and
  the document fetches no font, script, or highlighter. The kit's Google
  Fonts `@import` is dropped for that reason, falling back to the system
  faces the kit already names.
- **Stages 1 and 2 are operational end to end on the fake transport.**
  `whetstone_envs.optim.study.arms` supplies the three collaborators the
  stage harness takes as callables and refuses to import.
  `StudyOptimizerRunner` drives one arm at one seed through `optim/run.py`'s
  `RunSpec`, audits the run it produced, projects its cost, and copies the
  three records the manifest cites into the study's own store, so
  `whetstone-study manifest check` resolves every pointer the report prints.
  `RoleScorer` evaluates a candidate on one role's split through the *same*
  engine binder Stage 0 calibrated its anchors with, which makes L4's
  identical-procedure rule a construction rather than an assertion. The two
  controls run no optimizer and record that fact in their own run record
  instead of borrowing an optimizer's evidence. `whetstone-study run --stage
  stage0|stage1|stage2` over a toy c19 spec now produces a manifest with arm
  runs, a persisted selection, held-out rows with their statistics, and a
  report generated from it — with zero provider calls. Real transports stay
  refused until a study spec selects one and a stage gate authorizes it.
- **The statistics reach the manifest.**
  `whetstone_envs.optim.study.analysis` is the second pass the numbers need:
  a Holm-corrected p-value is a whole-study computation that cannot exist
  until every arm is measured, so the anchors are measured through the
  identical once-only ledger and one `held_out` row per reported candidate
  is written afterwards. Each row carries its interval, its uncorrected
  p-value, its row-completeness weighting, and — for the four real
  optimizers only — its Holm-corrected p-value, and each cites the per-task
  vector it was computed from as evidence in the study's own store. Nulls
  and anchors are controls rather than hypotheses, so their Holm column is
  empty by design. `null_triggers_downgrade` and the D5 held-out nesting
  check run over what the stage recorded.
- **Stage 2 continues from Stage 1 rather than refusing.** Previously it ran
  the new seeds and then raised, leaving paid runs with no selection and no
  recovery. A run an earlier stage recorded is now re-read from its own
  artifacts, and selection runs over the union in seed order. Selections and
  held-out claims carry the stage that made them, so L2 and L3 are "once per
  arm per stage" — which is what the design describes, since a pilot's
  arg-max over two runs and the full design's over five are different
  decisions, both recorded rather than one overwriting the other.
- **The pre-registration is immutable once pinned.** `study.json` gains a
  `pre_registration` block holding the design fields fixed before any spend
  plus the content hash over them (schema v3). Stage 0 pins it; every later
  write must carry it back byte for byte or is refused with
  `PreRegistrationViolationError`, so no stage that has seen results can
  restate the power arithmetic it is judged against. A second `stage0` is
  refused outright; `--replace-design` records the replacement as an
  `amended` block naming the design hash it replaced, and a re-calibration
  landing on the same design is written back unchanged rather than
  relabelled.
- **`leakage-check` really checks L1.** It extracts each run's completed
  intent resolutions from the run stores the manifest names, so the rule
  that an optimizer saw the internal split and nothing else is observed
  rather than reported unchecked forever. That exposed why it could never
  have passed: the repeat count is part of an Eval Config's identity, and
  Stage 0 recorded the calibration's config (`K_CAL`) where every later
  evaluation resolves the design's (`K_REPEAT`). Stage 0 now records the
  config the runs actually use, and a clean study passes all six rules.
- An arm stage is resumable, which matters because it is the path that
  spends. A seed whose run is already recorded is not re-executed — that is
  what makes "Stage 1's runs count toward Stage 2" a checkable property of
  the seeds rather than an assertion — and the merged arm record keeps every
  previously recorded run instead of replacing the list, so a stage that
  crashed after paying for some runs does not discard them. A run an earlier
  stage recorded is re-read from its own artifacts so Stage 2's arg-max
  covers the arm's whole `K_RUN`; selecting over a subset is refused loudly
  rather than quietly turning a `K_RUN = 5` arg-max into a `K_RUN = 3` one.
  Reporting resumes too: an arm whose selection and completed held-out claim
  are both durable is rebuilt from them rather than re-reported, so a crash
  partway through a stage no longer leaves every later resume raising
  "already selected" with the study's paid runs stranded behind it. An arm
  that selected but never claimed continues from its selection, and one
  whose evaluation was in flight is refused with the recovery named rather
  than silently re-billed. Held-out claims now carry the per-task vector the
  rebuild needs.
- **The Stage-1 budget gate measures planned rows.** Its numerator read an
  attribute no evidence type defines, so every evaluation counted as one and
  a COPRO pilot measured 2 against 48 real rows — a 24x undercount in the
  gate built to catch fan-out. It now dereferences
  `EvalEvidence.row_accounting.planned` through the same measurement the F16
  fan-out check uses, deduplicated by content hash so GEPA's replayed prefix
  cannot inflate the count with the step number and false-abort a long run.
- The study report surfaces an amended pre-registration. The design hash,
  its provenance, and the hash an amendment replaced all render in the
  design section, and an amended design raises a warning — a design changed
  after Stage 0 is not the one first registered, and a reader must not have
  to infer that from the manifest.
- **`leakage-check` really checks L1.** It extracts each run's completed
  intent resolutions from the run stores the manifest names, so the rule
  that an optimizer saw the internal split and nothing else is observed
  rather than reported unchecked forever. That exposed why it could never
  have passed: the repeat count is part of an Eval Config's identity, and
  Stage 0 recorded the calibration's config (`K_CAL`) where every later
  evaluation resolves the design's (`K_REPEAT`). Stage 0 now records the
  config the runs actually use, and a clean study passes all six rules. The
  verdict is also *recorded* into `manifest.leakage_check` rather than only
  printed: the report treats an absent block exactly as it treats a failed
  one, so a study whose rules passed on the terminal but were never written
  back could never clear the downgrade.
- `whetstone_envs.optim.study.fanout`: the F16 measurement. `measure_fanout`
  reads a completed run's evidence and reports, per evaluation, the task
  subset the optimizer requested beside the rows the platform actually
  planned. It reads both evidence surfaces -- MIPROv2 resolves intents, GEPA
  emits search evidence -- and counts each eval-evidence record once, keyed
  by its content hash, because GEPA re-emits its whole replayed prefix on
  every step.
- Wave 3's fake-transport measurements are pinned in
  `optim/study/gates.py`, each with the run and split sizes it came from,
  and golden-tested as literals. `tests/optim/study/test_fanout.py`
  re-measures mechanically on a small fake run and asserts the result
  against the formula the code implements, so a regression fails a test
  rather than a provider bill.
- `whetstone-study plan` now prints Wave 3's measured per-run call counts
  beside the control-derived estimates, labelling each row `MEASURED` or
  `ESTIMATE`. Arms Wave 3 did not measure print only their estimate.
- **MIPROv2's search shape and split are settings, not literals.**
  `num_trials` and `num_candidates` were hardcoded at 2 and 3 in
  `optim/miprov2.py` while the protocol's auto-light configuration assumes
  10 and 6 — so Wave 3's measured 245 task calls are the cost of *this*
  runner's shape, not the protocol's, and there was no way to ask for the
  protocol's without editing the module. Both are now `RunSpec` fields with
  `--miprov2-num-trials` / `--miprov2-num-candidates` CLI flags, validated
  as positive and refused on any other optimizer, and `ArmSpec` carries
  them so a study arm can request the protocol's shape. Defaults are
  unchanged, which keeps the fake-transport end-to-end runs fast.

### Changed

- **`whetstone-study` is an installed console script.** The study CLI
  named itself `whetstone-study` and its `__main__` documented the console
  script, but nothing registered one, so only
  `python -m whetstone_envs.optim.study` actually worked. Both entry points
  now resolve to the same `main`.
- **The pre-registered held-out split is 440, not 220.** The c19 protocol
  splits are now internal 88 / official 132 / held-out 440, using 660 of
  the 704 available instances. Doubling the reporting split halves the
  variance term the Stage-0 gate inverts, so the pre-registered MDE row
  the study is judged against moves with it. `MEASUREMENT_SPLIT_SIZES`
  stays `(88, 132, 220)`: that is what Wave 3 actually measured, and
  rewriting it would misstate the provenance of every measured number
  derived from those runs. Because the split is deterministic, held-220
  remains a prefix of held-440, so `check_held_out_nesting` keeps its
  meaning — but it is now an invariant the construction guarantees rather
  than a growth decision taken at the Stage-0 gate, and its docstring says
  so.

### Fixed

- **The Stage-1 budget gate compared two different units, and a real GEPA
  run would have tripped it.** `estimate_optimizer_calls` returned GEPA's
  732 *metric calls* while `call_count_within_estimate` receives
  `observed_task_calls`, which is a count of *task-model rows* — the unit
  `cost.json` reports as `task_model.calls`. Every estimate is now
  denominated in task rows, with the relation between the two units derived
  rather than fitted: the Wave 3 `w3-gepa-full` run's 91 distinct
  evaluations and 265 rows have exactly one decomposition into the two
  shapes this control can produce (`2` full 88-task valset passes plus `89`
  one-task reflection minibatches, since `build_gepa_control` sets
  `reflection_minibatch_size=1` and `valset_task_hashes=None`). Because
  upstream charges one metric call per example, a run's distinct rows can
  never exceed its metric-call budget, so the budget is a sound and
  deliberately loose upper bound on rows — the 732 charged against 265
  executed is the per-step prefix-replay factor, which the durable effect
  cache serves without re-executing a row. GEPA's gated estimate is now
  sourced from `GEPA_MAX_METRIC_CALLS_PINNED = 200`, the Wave 3 D3
  decision and the budget Stage 1 and Stage 2 actually run, rather than
  from the retired 732.
- **`null-identity` was priced as a COPRO search it never runs.** Null-B's
  estimate was `(depth + 1) × breadth × internal × K_REPEAT`, but
  `StudyOptimizerRunner._run_null` never calls `run_optimizer`: it emits the
  naive anchor unchanged and reports `observed_task_calls=0`. Null-B is the
  seed carried through the *report harness*, so its per-run estimate is now
  the official and held-out passes every arm pays — one of each, at
  `K_REPEAT` — via the new `null_identity_report_rows`. `null-random` keeps
  COPRO's shape, which it genuinely shares. The corrected per-arm estimates
  are golden-pinned with their derivations, and `whetstone-study plan`
  prints them; its GEPA row now reports the measurement scaled to the pinned
  budget rather than to the retired one.

### Known limitations

- **No real Codex run has been performed.** Every claim about the Codex arm
  in this release rests on the scripted fake CLI. That stand-in is a real
  subprocess speaking real MCP over HTTP to the real whetstone-hosted
  evaluation server, so the production admission, lease, evaluation, and
  ledger path *is* exercised end to end — only the agent's own decisions
  are scripted. What remains unproven is the part only a live agent can
  demonstrate: that a real Codex session reads its prompt, chooses its own
  candidates, spends its capacity sensibly, and returns a `selected_call_id`
  that resolves. The §6 "one real run" is therefore still open.

  Performing it requires all of: `WHETSTONE_ENVS_ALLOW_REAL_CODEX=1` and
  `--allow-real-codex` (see the spend guard above), a live authenticated
  Codex session, macOS, provider spend for the task-model evaluations it
  buys, and an explicit go from Danielle. Until then, treat the arm as
  mechanically complete and behaviorally unvalidated.

### Notes

- `COPRO_TERMINAL_PROVENANCE` ships narrower than the Step 10 assignment's
  Section 3.4 text, in two respects, both recorded in its docstring. Its
  "the terminal candidate is a proposal minted in this run, or the seed
  under `seed_retained`" clauses are already schema invariants of
  `OptimResult` and `OptimStepResult` -- proposals must equal the final
  step's accepted candidates, every proposal must bind an exact request
  candidate and differ from it on the mutation field alone, and a
  seed-retaining step must accept nothing and name the exact run seed -- so
  no schema-valid artifact can violate them, and an invariant with no
  failing fixture does not ship. The invariant instead covers the one
  provenance fact nothing structural ties together: that the terminal
  candidate's base chain reaches the run's declared
  `initial_candidate_ref`. Its "never `PROBES.ceiling_template`" clause is
  omitted as a task-family literal, which Section 3.5 makes a design defect
  in an audit; it is also unusable in CI, because the fake transport
  proposes the ceiling template by construction.
- `COPRO_DISTINCT_BASES` passes without comparing anything on a run at the
  smallest admissible `breadth` of 2, where the seed round proposes one
  candidate and re-measures the initial one. The finding says so explicitly
  rather than reporting a bare PASS. A run with a genuinely multi-draft
  round is not producible on the fake transport today: the scripted proposer
  supplies two bodies, one of which is the seed itself and is rejected as a
  no-op mutation.
- `optim/audit/codex.py`: the six Codex-direct fidelity invariants, carrying
  the F6 retarget. A Codex run's search is a foreign agent's, so these check
  containment rather than recorded decisions: that every evaluation the run
  paid for went through the one granted Tool and stayed reachable as Tool
  Evidence (both directions); that the returned candidate's `selected_call_id`
  resolves to a completed, *scored* admission entry; that no capacity debit
  ordinal exceeds the run's configured cap and no refusal consumed capacity;
  that the output artifact carries this run's own lease binding; that exactly
  one tool is granted and used, over the pinned `CODEX_EVAL_INPUT_FIELDS`;
  and that a failed run's recorded spend still matches its admission ledger.
- The Codex audit reads the admission ledger through `admitted_entries`, never
  by raw SQL over `whetstone_tool_admission_entry`, and scopes it from
  `OptimRun.tool_configs` rather than from the run's own reported evidence --
  scoping by what a run reported would let an under-reporting run read an
  empty ledger and pass totality vacuously. Per F6 there is no `read-scores`
  tool and no `tool_name` column: the name is dereferenced from
  `tool_config.record.definition.record`. Per OQ1 there is no `codex_agent`
  cost role, so task-model spend is read at role granularity.
- `_evidence.py` resolves `state_delta` refs named in `STATE_RECORD_REF_KEYS`
  at load time, exposed as `RunEvidence.stored_record`. Invariants stay pure
  functions over already-resolved evidence rather than reopening the store.
- `_mutate.py` gains `mutate_ledger_entry` and `delete_ledger_entry` for
  building negatives against the durable admission ledger. A mutation of the
  run record leaves a loadable artifact through `reseal_run_binding`, which
  re-seals the `OptimRun` wrapper and re-threads it through every step
  request.
- Two committed Codex run fixtures under `tests/optim/audit/fixtures/`, with
  the generator that produced them. They are committed rather than built at
  test time because the Codex-direct adapter, the one-tool MCP surface, and
  `ToolAdmissionAuthority.admitted_entries` are all whetstone-ai 0.1.7
  surface; the Codex tests skip until an install can read them, and turn
  themselves on when 0.1.7 lands.

### Changed

- The C19 optimizer runner is now a family-agnostic runner. `C19RunSpec`
  becomes `RunSpec` and `run_c19_optimizer` becomes `run_optimizer`; both
  read every family-specific decision -- pool generator, experiment builder,
  probes, mutation field, render contract, and scripted proposer bodies --
  from a new `whetstone_envs.optim.families` registry rather than from
  literals in the runner. `run_optimizer` now names no task family, which is
  what lets a second family reach the optimizers through the identical path.
  `C19_OPTIMIZERS` and `C19_TRANSPORTS` become `OPTIMIZERS` and
  `TRANSPORTS`, and `C19_PROPOSAL_BODIES` becomes `FamilySpec
  .proposal_bodies()`.
- Task-pool generation is no longer hardcoded. The runner used to call
  `generate_pool(n_per_stratum=2, seed_start=765_432)` with both values
  written into its body; they are now `RunSpec` fields defaulting to the
  family's own values, so an unparameterised run generates the pool it
  always generated.
- COPRO's breadth and depth are no longer hardcoded to `2` and `1`. Both are
  `RunSpec` fields with those values as defaults. A breadth below 2 is
  refused at spec validation rather than inside the durable run boundary,
  matching `CoproControl`'s own rule.
- **The optimizer adapters are family-generic.** `gepa.py`, `miprov2.py`, and
  `provider.py` read the family's contract instead of C19's constants, so
  `build_c19_gepa_adapter` becomes `build_gepa_adapter`,
  `build_c19_miprov2_{control,adapter,state}` become
  `build_miprov2_{control,adapter,state}`, `c19_fake_transport_factory` and
  `c19_fake_gold_by_prompt` become `fake_transport_factory` and
  `fake_gold_by_prompt` (taking the family's render contract and ceiling
  template), and `C19_DEMO_MODES` becomes `DEMO_MODES`. Wave 1 generalized
  `run.py`; these three modules still carried C19's mutation field, prompt
  fields, render contract, and probes, so C18 could not have reached the
  optimizers without a C19 branch somewhere. Each family now namespaces the
  identities GEPA, MIPROv2, and COPRO mint for their inline proposal executors
  and GEPA component schema, so two families' runs never share a policy
  identity.
- `FamilySpec` carries a `contract` rather than repeating its namespace,
  mutation field, prompt fields, and render contract; those are now properties
  reading the one contract. It gains `eval_runner`, `rendering_rules`, and
  `example_execution`, the last two being the proposer-facing prose MIPROv2's
  opening state used to hardcode as C19's.
- The persisted evaluation report no longer assumes C19. `EvalRun.family` is
  the pinned `FamilyName` literal `"c19" | "c18"`, read from the prepared
  experiment's own namespace rather than written in; `EvalRun.dataset_revision`
  comes from the evaluated split's task set; and `TaskRecord` requires nonempty
  nonblank prompt-input names instead of exactly `{grid, command, question}`.
  Before this, a C18 run reached `project_eval_report` and was rejected by the
  report schema — the one place C19 vocabulary had leaked into a shared
  contract.
- `project_trajectory_report` and `project_eval_report` annotate `prepared` as
  the `PreparedExperiment` protocol they actually read, removing the
  `cast("PreparedC19Experiment", ...)` the runner carried. The concrete
  dataclass both families return is now `PreparedSplitExperiment`.
- MIPROv2 runs complete on whetstone-ai 0.1.6. The upstream bootstrap teacher
  no longer hardcodes the `user_prompt_template` mutation field, so
  `test_miprov2_fake_transport_completes` runs as an ordinary passing test in
  every demonstration mode; its `xfail(strict=True)` marker and the
  `MIPROV2_UPSTREAM_BLOCKER` reason it carried are gone.
- **Leakage gates the report's headline and every verdict.** A study whose
  leakage rules failed — or were never run, which makes the same claim to a
  reader — reports every arm as `invalid (leakage)` and headlines the
  downgrade. An interval measured through a procedure the study could not
  establish is not a result, whatever its width, so the gate runs before the
  per-arm fidelity check rather than beside it.
- **Holm's `m` is the pre-registered family size, not the number of arms in
  hand.** A pilot or a partial resume corrects at `m = 4` instead of
  under-correcting exactly when multiplicity risk is unchanged; the
  unanalysed members enter as `p = 1`, which is Holm over the declared
  family with the missing arms unrejected. Analysing more arms than the
  family declares is refused rather than silently widening it after the
  fact.
- **`reported_numbers_resolve` no longer passes vacuously.** A run whose
  steps completed no evaluation intent now FAILs, and a run with no steps at
  all is `NOT_APPLICABLE`. "All 0 of 0 resolve" read as audited fidelity on
  a run whose numbers nobody had verified.
- **The rendered-number guard covers prose, not only table figures.**
  `rendered_text_in` walks every non-figure string the report renders —
  paragraphs, prose cells, captions, headings, checklist items, code-block
  labels — and the test refuses any digit that is not either a `Figure` or
  one of an explicitly named set of structural identifiers (rule ids, run
  ids, schema names, content hashes, the MDE formula). Several real numbers
  it found — the stage-history counts, the fan-out and GEPA-sizing details,
  the trajectory step count, the measured MDE and held-out size in the
  threats table — are now figures citing their manifest paths.
- **R6 is retired.** The feared 2.51x minibatch fan-out does not happen:
  across four fake-transport runs at the protocol's own `(88, 132, 220)`
  splits, every evaluation executed exactly the task subset it requested,
  for a measured fan-out ratio of 1.0. The platform's deferral row expansion
  honours per-intent task sets. Two independent records agree on the count
  -- summed `EvalEvidence.row_accounting.planned` and `cost.json`'s
  `task_model.calls`.
- **F10's 28-616 bootstrap bound does not apply to this runner.**
  `build_miprov2_control` slices `trainset=task_hashes[:1]`, giving MIPROv2
  a one-task trainset at every split size, so bootstrapping costs 1 row per
  run (2 for `zeroshot`, which emits no `LABELS_ONLY` plan). The derivation
  was also wrong in two further ways: the plan count is `num_candidates - 2`,
  not a fixed 7, and there is no `/p_accept` inflation because `max_rounds`
  is 1, so a rejected attempt still advances the cursor. The loose ceiling
  is retained as the Stage-1 gate's denominator, since a loose upper bound
  cannot false-abort a run.
- **D3 resolves to the pre-registered fallback: GEPA runs at
  `max_metric_calls = 200`, not 732.** One measured 732-call run took 556
  whetstone steps and 22 minutes -- inside the wall-clock bar -- but produced
  a 1.73 GB `runtime.sqlite` and a 766 MB `result.json`, against a ~1 GB
  bar. Each step restarts `optimize()` and replays the whole prefix, and
  persists that prefix as its own search evidence, so both artifacts grow
  superlinearly: 155,956 search-evidence entries addressing 91 distinct
  evaluations.

### Added

- A task-family registry at `whetstone_envs.optim.families`. `FamilySpec`
  carries one family's generator, experiment builder, probes, mutation
  field, prompt fields, render contract, and pool defaults;
  `register_family` admits it and `family_spec` resolves it. `c19` is
  registered; `c18` is a known identifier the registry already admits, so
  registering it later is a registration rather than a change to the
  registry's vocabulary. A known-but-unregistered family reports a wiring
  gap distinctly from an unrecognised name.
- `RunSpec.seed` plumbs one explicit algorithmic seed per run. GEPA and
  MIPROv2 carry it onto their controls as an explicit field; `CoproControl`
  has no seed, so a COPRO run's stochasticity remains the proposer LM and
  the provider `SEED` control. `seed_disposition` names that difference for
  the study manifest rather than faking a seed the control never reads. An
  omitted seed keeps each optimizer's own default, so an unseeded run keeps
  the control identity hash it always had.
- `RunSpec.proposer_model` separates the proposal route from the task route,
  so a study can run a cheap task model against a stronger proposer.
  `None` reuses the experiment's own route.
- `RunSpec.gepa_max_metric_calls` pins GEPA's paid metric-call ceiling, and
  `RunSpec.codex_capacity` carries the Codex arm's evaluate-call cap. Each
  is refused on an optimizer that cannot honour it, so a flag nothing reads
  cannot look respected in the study manifest.
- New `whetstone_envs.optim.cli` flags: `--proposer-model`, `--seed`,
  `--n-per-stratum`, `--pool-seed-start`, `--copro-breadth`,
  `--copro-depth`, `--gepa-max-metric-calls`, and `--codex-capacity`.
  `--family` now offers exactly the registered families.

- The C19 evaluation CLI and report accept `--role held_out`, evaluating the
  held-out split through the same path as `internal` and `official`. The
  persisted `EvalRun.role` records which of the three roles a report covers,
  and every role reports against its own tasks and its own
  `eval_config_hash`. An experiment prepared without a held-out split refuses
  the role by name before creating a run directory rather than falling back to
  another split.


- The study's two null optimizers, in `whetstone_envs.optim.nulls`, as
  `ProposerTransport` implementations so they reach the optimizer through the
  same surface a real proposer does. `NullRandomTransport` (null-A) is the
  selection-on-noise control: it perturbs the base candidate's template with a
  seeded RNG — swapping, deleting, and duplicating whitespace-delimited tokens
  at `NULL_PERTURBATION_RATE` — and returns the results as ordinary drafts, so
  best-on-internal selection runs over candidates carrying no information. The
  perturber treats every `{field}` placeholder as atomic and immovable and
  re-validates against the run's `TemplateRenderContract` before returning, so
  a required field is never dropped; a draw the contract rejects, that repeats
  the base, or that collides with an earlier slot in the same batch is retried
  from the same seeded stream, bounded, then falls back to the base unchanged
  with `identity_fallback` recorded on the draft. Perturbation is deterministic
  in the run seed and the proposal request, so a null-A run replays exactly.
  `NullIdentityTransport` (null-B) is the pipeline-overhead control: it
  proposes the base unchanged. Neither null makes a provider call, so every
  draft reports `proposer_calls: 0` with no call id and no price, and run cost
  records no proposer spend for a null arm rather than a zero-dollar phantom
  call.

  Two upstream contracts shape null-B and are pinned by its tests. A
  *successful* draft repeating its base is not representable — `diff_check`
  rejects a mutation equal to its base, because a no-op is not a proposal — so
  null-B reports an unfilled slot, which is how a real proposer that returned
  nothing reports it. What an optimizer does with that is the optimizer's
  contract: GEPA and MIPROv2 set `terminal_proposal_count` on their step
  contracts and so may terminalize `seed_retained` when their search accepts
  nothing over the seed, while COPRO does not set it, so under COPRO's control
  shape an unfilled round is a `copro_proposal_cardinality` terminal failure
  whose result still names the seed as the run's outcome.
- The Step 10 study harness at `whetstone_envs.optim.study`: the design
  record, the Stage-0 gate arithmetic, the stage runner, the
  selection-and-reporting harness, the leakage checks, the study manifest,
  and the `whetstone-study` CLI they are reached through. `StudySpec`
  pre-registers the population, splits, arms, and repeat counts before any
  provider spend, and keeps `k_cal` and `k_repeat` separate fields — the
  Stage-0 calibration count is a measurement input, and borrowing the design
  repeat count for it biases the gate optimistically through `tau^2`'s
  `2 sigma^2 / k_cal` correction. `run_stage0` calibrates the naive and
  ceiling anchors on internal, official, and held-out through one procedure
  at `K_CAL = 4`, with the doubling rule capped at 16 and refusing an odd
  count that no split-half check could read.
- `minimum_detectable_effect` is the study's single pre-registered MDE,
  `(z_{1-alpha/2} + z_power) * sqrt((tau^2 + 2 sigma^2 / K) / T)`, computed
  in the study's own code as one auditable line rather than read off
  `analyze_power`'s surface, whose grid resolution would otherwise be able to
  move the gate's design point. The power analysis is still the source of
  `tau^2` and `sigma^2`; `within_variance_divergence` reports the pooled
  within-variance estimate beside the shipped naive-arm-only one and flags a
  divergence above 20%, so that caveat is a measured number rather than a
  footnote. `evaluate_stage0_gate` reports all four gate conditions whether
  or not they held, since the one permitted design adjustment is priced
  against the measured MDE.
- `report_arm` is the single entry point through which a held-out number can
  enter the study, and it makes leakage rules L2 and L3 structural rather
  than conventional. It scores every run on official, takes the arg-max,
  persists the selection, *reads the persisted record back*, and only then
  issues exactly one held-out evaluation against a ledger that refuses a
  second one. Two ledgers satisfy that contract: `SelectionLog` holds it in
  memory for a dry run, and `ManifestSelectionLog` writes each selection into
  the study's own `study.json` through `write_study_manifest(replace=True)`
  and reads it back off disk, so a paid stage's ordering is a filesystem fact
  rather than a variable still in scope and a crash between selecting and
  measuring leaves the selection recorded. A second selection for an arm and
  a second held-out evaluation for a candidate are both refused before the
  provider call, so a violation costs nothing rather than being detected
  after it was paid for. Anchors and nulls reach held-out through
  `report_reference_candidate`, which keeps the identical procedure and the
  identical once-only ledger without an arg-max they have nothing to apply.
- `held_out_claims` closes L3's real window. A held-out *row* carries a
  Holm-corrected p-value, which is a whole-study computation that cannot
  exist until every arm is measured — so waiting for the row would leave the
  gap between paying for an evaluation and recording that it happened
  completely unguarded. A claim is instead written *before* the evaluation is
  issued and completed with its result when it returns, so a stage that
  crashes mid-evaluation resumes knowing the candidate already spent its one
  shot, and an outstanding claim reads as a crashed evaluation rather than as
  one that never happened. `StudyManifest` refuses two claims for one
  candidate, refuses a half-recorded measurement, and refuses a held-out row
  with no completed claim behind it — a reported number whose evaluation was
  never claimed came from outside `report_arm`, which is exactly the leak L3
  exists to catch. L3 and L4 read the claims rather than the rows for the
  same reason: each claim carries the Eval Config and repeat count its own
  evaluation used, while the rows are written from one study-wide config and
  would agree by construction whether or not the evaluations did. The two
  rules read different slices of the same claims — L3 counts every claim,
  because what it limits is evaluations *issued* and a crashed one still
  spent the candidate's shot, while L4 compares only completed ones, because
  an outstanding claim has no procedure to compare and substituting the
  study's own values for it would rebuild the tautology.
- `LeakageFinding` distinguishes "this rule held" from "this rule had no
  evidence to hold against". Every one of these predicates is vacuously true
  over an empty observation set, and a vacuous truth reported as a pass is
  how a study comes to claim a leakage rule nobody verified — so an
  unchecked finding never counts as passed, `LeakageReport.passed` requires
  every rule to have been checked *and* held, and L6's detail names the
  rules that actually ran rather than claiming all five did.
- `study_leakage_check` runs L1-L5 mechanically over recorded evidence and
  fails the study loudly before report generation (L6), reached from the CLI
  as `whetstone-study leakage-check --study-dir`. L1 matches both the
  evaluation role and the resolved Eval Config, since either alone would pass
  a leak the other catches; L5 reads content-addressed task hashes rather
  than task ids; and `check_held_out_nesting` encodes D5's growth rule, that
  held-220 must be contained in held-440 for the anchors measured on the
  first to describe the second. Run from a manifest, L1 is reported
  `NOT CHECKED` and **fails the command**: its evidence is each run's own
  intent resolutions, which live in the run stores, and an empty observation
  set is vacuously true rather than checked.
- `analyze_arms` computes each arm's paired bootstrap interval and applies
  Holm over the four real optimizers only — nulls are controls, not
  hypotheses, and correcting them would make a null harder to trip exactly
  when tripping it matters. `null_triggers_downgrade` requires both magnitude
  above the measured MDE and an interval excluding zero, so a small
  significant delta does not void the study. `weighted_per_task_delta`
  weights each task's delta by its achieved row count and keeps a
  fully-missing task in the vector at zero weight rather than dropping it,
  which would shrink `T` and tighten the interval dishonestly; an arm below
  the 90% completeness backstop is reported but never claimed.
- The stage runner at `whetstone_envs.optim.study.stages` turns a stage into
  a manifest. Stage 0 calibrates and writes the `design` block every later
  stage reads; Stages 1 and 2 run each arm's optimizer, file its runs, and
  route every held-out number through `report_arm`. An arm stage refuses to
  start without a recorded design, and Stage 0 refuses a study that has not
  declared its arms — `k_run_by_arm` is a pre-registration, so a placeholder
  would read as a design. `spec_from_manifest` reads those run counts from
  the design table via `k_run_for`, never from `len(arm.runs)`: how many
  runs an arm has executed is progress, not design, and reading it as design
  would make an unstarted study look like a one-run-per-arm study and
  under-report the whole budget by a factor of `K_RUN`. An arm naming an
  optimizer with no assigned seed range is refused rather than seeded from
  zero, matching every other read of that table.
- An arm stage is resumable, which matters because it is the path that
  spends. A seed whose run is already recorded is not re-executed — that is
  what makes "Stage 1's runs count toward Stage 2" a checkable property of
  the seeds rather than an assertion — and the merged arm record keeps every
  previously recorded run instead of replacing the list, so a stage that
  crashed after paying for some runs does not discard them. Selecting over a
  subset of an arm's runs is refused loudly rather than quietly turning a
  `K_RUN = 5` arg-max into a `K_RUN = 3` one.
- Stage 1 applies the call-count gate rather than merely defining it. Each
  run's measured task calls are compared against its pre-spend estimate at
  the `1.5x` tolerance, and an overrun refuses the stage *before* any
  held-out evaluation is issued, so the study does not pay for a number it
  is about to refuse to trust. The gate runs at Stage 1 only: it exists to
  catch a fan-out bug before Stage 2 pays five times over for the same
  defect, and re-running it at Stage 2 would restate a fact the pilot
  already settled. Every provider-touching collaborator arrives as a
  callable on `StageEnvironment`, so the whole harness runs end to end on a
  fake transport with zero provider calls;
  `whetstone_envs.optim.study.environment` is where the real ones come from,
  regenerating the pre-registered pool deterministically from the manifest's
  own `n_per_stratum` and `pool_seed_start` rather than drawing a fresh
  sample of the same size — and then *checking* it, by comparing every
  regenerated split's content-addressed task hashes against the ones the
  manifest recorded. A mismatch refuses the bind, because the alternative is
  a study whose Stage-2 numbers describe different tasks than its Stage-0
  anchors did, and only a hash comparison catches a generator whose output
  changed while its shape did not.
- The Step 10 study manifest at `whetstone_envs.optim.study.manifest`.
  `study.json` is the study's single accounting surface: every number the
  report prints is a field of `StudyManifest` or a deterministic function of
  evidence the manifest names, and it names that evidence as an
  `EvidencePointer` — a `(schema_name, content_hash)` pair the run's object
  store resolves — so the report generator never recomputes a score from a
  loose file. The manifest carries the population and its three
  content-addressed splits, the models and what the study could not control,
  the design's `K_CAL`/`K_REPEAT`/`K_RUN` and power arithmetic, the GEPA
  sizing and fan-out preconditions, the arms and their runs with per-run
  audit verdicts, cost pointers, and spend, the selection and held-out
  records, the balance at each spend gate, the leakage verdict, and the
  second family's runs and adapter-swap result.
- The manifest schema lives in the repository and its instances do not.
  `STUDY_MANIFEST_SCHEMA` and the `ManifestKey` wire-key enum own every
  persisted literal, and `tests/optim/study/test_manifest.py` pins each one
  as a written-out string rather than deriving it from a field name, so a
  rename that changed stored identity fails the golden test. A study
  instance is a durable work document, so `write_study_manifest` refuses to
  write inside any detected repository — the same rule run artifacts follow.
- Two of the study's pre-registered rules are structural rather than
  conventional. `StudyManifest` refuses a second selection for one arm (L2)
  and a second held-out row for one candidate (L3), and `SplitsRecord`
  refuses overlapping splits at construction (L5), so those leakages are
  impossible to record rather than merely detectable afterwards.
- `write_study_manifest` never overwrites an existing manifest silently:
  replacement is an explicit `replace=True`, and the default raises
  `ManifestExistsError` with the untouched original left in place. It
  validates by round-tripping the exact bytes that will land, so a manifest
  is written only if what lands re-validates.
- `check_manifest_pointers` resolves every evidence pointer a manifest cites
  against a named store, walking the model's own fields rather than a
  hand-kept list, and reports one verdict per pointer. Distinct pointers
  resolve once each; a pointer whose record is absent or whose bytes drifted
  fails the check rather than reaching the report as a number nobody can
  reproduce.
- Every optimizer run now writes `cost.json` beside `result.json` and
  content-addresses the same bytes into its own `runtime.sqlite`, so the
  manifest's per-run `cost_ref` is a pointer `manifest check` resolves rather
  than a number typed in by hand. It is a pinned-key *projection* —
  `result.json` stays the authority — and it carries the full honesty split
  per role: billable calls, cached calls, the priced/unpriced split, calls
  the provider priced without a token breakdown, and a USD total only when
  every contributing call carried a price. A run whose result records no cost
  report writes no `cost.json`, because an all-zero document would claim the
  run was free rather than unmeasured.
- The `whetstone-study` console script at
  `whetstone_envs.optim.study.cli`. `plan` prints the run matrix and two
  budgets; `run --stage stage0|stage1|stage2` runs a stage through the
  harness; `leakage-check` runs L1-L6 over a study directory; `report` hands
  the manifest to the report generator; `manifest check <path>` resolves
  every evidence pointer against the run store beside the manifest or one
  named by `--store`. The spec loader and the stage runner default to the
  real implementations, so the CLI is the study's actual entry point; the
  report generator stays injected until its wave lands and `report` says so
  rather than failing obscurely. Every collaborator is still named as a
  protocol and overridable, which is how the ordering is tested
  independently of the wiring.
- `plan` prints its two budgets separately because they are known to
  different degrees. The selection-and-reporting budget is *derived* from
  the matrix, so a spec cannot assert a budget its matrix does not imply.
  The optimizer-side budget is an *estimate* from the control defaults in a
  new `whetstone_envs.optim.study.gates`, labelled as one, and it dominates
  the total. Each constant carries its derivation: MIPROv2's `fewshot` range
  is the **F10-corrected** 1,870-2,458 task calls per run — seven
  bootstrapping plans over a cursor walk, 28 rows best case and 616 worst,
  never the protocol's original 72 — and the Stage-1 "within 1.5x" gate
  divides by the ceiling, so a low naive anchor, which causes *more*
  bootstrap rows, does not read as a budget overrun. GEPA's estimate is its
  resolved 732 metric calls, COPRO's and both nulls' are derived from the
  configured breadth and depth, and Codex reports its cap of 8 admitted
  evaluate-calls as a cap rather than an estimate: per OQ3 it is excluded
  from the call-count comparison and gated on capacity respect instead,
  because applying a fan-out detector to a non-deterministic agent invites a
  false abort. An arm the estimator does not recognize reports "no estimate"
  rather than a number derived from a guess.
- `FamilySpec.build_candidate` mints one validated candidate from a
  template, reading the family's own `ExperimentContract` the way
  `render_contract` does, so how a candidate is built is family knowledge
  like its pool generator. The study's anchors reach the engine through it
  rather than through a c19 import, which is the C3 boundary working as
  intended.
- The C19 optimizer CLI and runner accept `--optimizer gepa` and
  `--optimizer miprov2` through the same shared runner path as COPRO.
- `--num-seeds` on the CLI and `RunSpec.num_seeds` make repeats per task
  (`K_REPEAT`) a runner parameter instead of a hardcoded 1.
- **C18 PrOntoQA is the study's second optimizer family (C3).** It reaches
  every optimizer through the identical `run_optimizer`, with only the family
  adapter swapped: `whetstone_envs.optim.c18_experiment` carries C18's
  `ExperimentContract` (its own namespace, dataset revision, root candidate
  schema, reward policy, and the `{question}`/`{query}` placeholders both C18
  probes use), a pool generator matching the registry's calling convention, a
  `prepare_c18_experiment` that is `prepare_experiment` under that contract,
  and `C18VerdictEvalProcedureRunner`. C18 is registered in
  `whetstone_envs.optim.families`; `--family c18` now parses. Splits follow
  C18's own `SplitPlan`, which at `n_per_stratum=30` over four depth strata
  gives `C18_PROTOCOL_SPLIT_SIZES = (24, 48, 48)` — pinned as a literal and
  checked against `default_split_sizes` so a generator change cannot resize
  the study's second family unnoticed.
- C18 scores by terminal verdict rather than by whole-reply exact match.
  `C18VerdictEvalProcedureRunner` delegates to `whetstone_envs.c18.score_gold`,
  which extracts the final `True`/`False` line. C18's ceiling probe asks for
  step-by-step reasoning ending in that line, so an exact-match runner would
  have scored every reasoned answer zero and flattened the ceiling anchor into
  the floor. Which runner a run uses is now a `FamilySpec` field, so this is a
  family fact rather than a branch in the runner.
- `whetstone_envs.optim.experiment.ExperimentContract` and
  `prepare_experiment`: one family-generic experiment builder both families
  call. A family's persisted identity, mutation field, placeholders, and probe
  templates have exactly one owner, and `prepare_c19_experiment` is that
  builder bound to `C19_CONTRACT`. `PreparedExperiment` — the protocol the
  runner and the reporting projection read — now lives beside it.
- The adapter-swap assertion, `tests/optim/test_c18_adapter_swap.py`. It
  traces every function entered under `whetstone_envs/optim/` during a C19 run
  and a C18 run of the same optimizer and asserts the two differ only inside
  the family-adapter file set, so a C19 branch in the runner or a C18 special
  case in the fake transport fails the test rather than passing quietly. The
  source is checked too — no shared module imports a family's package, and the
  only family literal outside the adapters is the CLI's default `--family`.
  Every optimizer, and every MIPROv2 demo mode, drives C18 end to end on the
  fake transport, and no private whetstone-ai import was needed to add it.

## [0.2.2] - 2026-08-22

### Changed

- C19 optimizer runs record effect leases in a SQLite authority instead of an
  in-memory one. The authority shares the run directory's `runtime.sqlite`
  with the object store, mirroring whetstone-ai's platform CLI; the two
  components own disjoint tables. Leases now outlive the process, so a re-run
  against a completed run directory replays its recorded effects rather than
  re-executing them. The runtime is closed on every exit path, releasing the
  eval engine and the authority's connection.

### Added

- The C19 optimizer CLI and runner accept `--optimizer gepa` and
  `--optimizer miprov2` through the same `run_c19_optimizer` path as COPRO.
- MIPROv2 support for C19: `build_c19_miprov2_control`,
  `build_c19_miprov2_adapter`, and `build_c19_miprov2_state` assemble the
  control, adapter, and opening durable state on whetstone-ai's public
  MIPROv2 surface. `--demo-mode fewshot|zeroshot|ground_only` selects the
  demonstration regime. Demonstrations reach the candidate through MIPROv2's
  own composed `### Demonstrations` section rather than a C19 placeholder:
  `fewshot` searches over demo sets and renders the selected set, while
  `zeroshot` and `ground_only` bootstrap demos only to ground instruction
  proposals and leave the section empty. All three modes compose a template
  that still satisfies the C19 `{grid}`/`{command}`/`{question}` contract.
  Labeled demonstrations carry the task's oracle gold as the component's
  `response` output, and the proposer's prompt model binds the experiment's
  `ProviderCallConfig` reference, as COPRO and GEPA already do.
- `--num-seeds` on the CLI and `C19RunSpec.num_seeds` make repeats per task
  (`K_REPEAT`) a runner parameter instead of a hardcoded 1.
- Trajectory reports carry a `spend` block projected from
  `OptimResult.cost`: per-role billable calls, cached calls, token totals,
  the priced/unpriced split, and a USD total only when every contributing
  call carried a provider-reported price. Terminal and HTML trajectory views
  render an unpriced role as `unpriced (n/total)` rather than as a zero. The
  block's `schema_version` is pinned to the one whetstone-ai cost-report
  version this package projects, so an embedded report at any other version
  is rejected rather than reinterpreted.
- A C19 evaluation CLI for task-family information, standalone fake or
  OpenRouter execution, strict local report publication, summary/failure/task
  inspection, and paired candidate comparison, exposed as the installed
  `whetstone-eval` command and a thin source-checkout launcher.
- Strict bounded `eval-report.json` and `trajectory-report.json` contracts,
  including safe typed provider-failure projection, exact binary C19 score
  reconciliation, COPRO intent and GEPA effect-transcript trajectory
  publication, explicit
  delta/cumulative budgets, and exact full-text candidate views.
- Deterministic, portable, offline `eval-report.html` and
  `trajectory-report.html` renderings with strict embedded-data escaping,
  hashed inline CSP, complete C19 debugging surfaces, exact-ref trajectory
  lineage, exact per-resolution navigation and diagnosis, responsive layouts,
  and pinned Chromium interaction and screenshot-comparison checks.

### Changed

- Pin the `optim` extra to published `whetstone-ai==0.1.6` and `dr-store`
  to 0.2.6, which that release requires. 0.1.6 drives MIPROv2 through
  `state.control.mutation_field`, so C19 runs MIPROv2 in every demonstration
  mode, and it resolves the provider/`eval.drivers` import cycle, so the
  optimizer CLI imports whetstone-ai's modules in any order.
- C19 reaches whetstone-ai through its current production entry point:
  `build_runtime` with an explicit adapter registry and effect authority,
  replacing the removed `register_runtime`. Registry membership is part of
  controller identity, so each run registers exactly the adapter it drives.
- C19's GEPA adapter is built by `build_gepa_harness_adapter`, whetstone-ai's
  canonical GEPA constructor, rather than by a hand-assembled authority
  chain. `build_c19_gepa_adapter` is now the C19 prompt services and control
  plus that one call, and the trainset/valset partition follows the control
  instead of handing the search the union of both splits.
- `prepare_c19_experiment` derives held-out rows into a full `EvalSplit`
  under the `held_out` split role and passes it as `EvalConfigs.held_out`,
  replacing the removed `held_out_task_hashes=` constructor argument.
  `EvalConfigs` enforces split disjointness at construction, and
  `held_out_task_hashes` remains readable as a derived property. An
  experiment with no held-out rows leaves the split absent.
- A GEPA run that finds no improvement completes by reporting the
  retained seed instead of substituting a candidate, and its steps carry
  resolvable eval and reward evidence for the evaluations its search
  drove. The GEPA trainset remains the internal eval split, and fake GEPA
  still accepts the ceiling draft because the fake task model emits gold
  for ceiling-rendered prompts; exact-match scoring is unchanged.
- The public-surface guard runs with an empty allowlist and also rejects
  `getattr(obj, "_name")` reads of whetstone privates alongside the
  existing `cast("Any", …)._name` check.
- Simplify HTML reports to a stable white scientific layout with
  colorblind-safe blue navigation and green success emphasis.
- Preserve the authoritative C19 `PoolSplit` beside each prepared Whetstone
  experiment so report projection joins evidence to source instances exactly.
- `whetstone_envs.optim.experiment` owns the C19 provider-call-config
  reference and the task-hash-to-gold mapping, so COPRO, GEPA, and MIPROv2
  bind one derivation instead of per-optimizer copies.

## [0.2.1] - 2026-08-21

### Changed

- Pin `dr-store` to 0.2.5 and `dr-graph` to 0.1.3 so this package can share
  an environment with whetstone-ai.
- Pin the `optim` extra to published `whetstone-ai==0.1.1` from PyPI.

### Added

- An optional `optim` extra that maps C19 pools, probes, and exact-match
  scoring onto whetstone-ai experiments through the public optimizer surface.

## [0.2.0] - 2026-08-06

### Added

- Deterministic C11 RFC 8785 task generation across five adversarial strata.
- An exactly pinned independent canonicalization oracle, naive and known-good
  probes, and a canonical persisted pool manifest.
- The C22 instruction-following environment with fixed default and hard
  presets, canonical manifests, naive and ceiling probes, and strict all-pass
  scoring.
- A namespaced Google Research IFEval runtime pinned with provenance,
  reproducible patch verification, and hand-built fixtures for every supported
  constraint.
- The C23 single-rule subregular induction environment with four balanced
  ISL/OSL strata, determinate six-demonstration tasks, ceiling and naive
  probes, exact scoring, default split sizing, and a committed pool manifest.
- A private-RNG adaptation of the pinned InductionBench generation and
  reference-transducer path with packaged Apache-2.0 attribution.
- The C18 PrOntoQA task family with frozen default and hard generation
  configurations, independent surface-text entailment, two public probes, and
  checked-in pool manifests.
- A pinned vendored generator boundary, an optional `c18` dependency extra,
  and an explicit script for regenerating either canonical C18 manifest.
- Deterministic C19 MiniGrid state-prediction tasks across navigation, object
  manipulation, and door-interaction scenarios at two grid sizes.
- A supported answer-relevant physical-state oracle independently
  cross-checked against live MiniGrid transitions, naive and known-good probes,
  bounded regeneration, and a canonical persisted pool manifest protected from
  custom generation inputs.
- Distribution validation that checks artifact metadata, package contents, and
  isolated installed-wheel imports before publication.

### Changed

- Restrict Depot cache writes to trusted `main` pushes while allowing pull
  requests to restore the rotated cache namespace.
- Run repository safety hooks in CI and release validation in addition to the
  canonical format, lint, type, definition, test, and build gate.
- Validate binding-contract structure and require `.defs` mappings for every
  symbol exported by an owning public package.
- Require a finalized dated changelog entry before a version tag can publish
  distributions.
- Model C22 gold as a closed composition of concrete constraint variants and
  derive checker descriptions and arguments through one vendor adapter.
- Cut one package release from the combined `main` tip after all task families
  assigned to that version have merged and passed the release gate.
- Extend the owning-subpackage API model to the higher-layer
  `whetstone_envs.c23` environment.
- Package and test C18 through the published instances, pools, probes,
  scoring, and manifests capability boundaries.
- Expand the package scope to include task families and exactly pin C19's
  MiniGrid, Gymnasium, and NumPy transition semantics.

## [0.1.1] - 2026-08-06

### Added

- Immutable task instances and canonical public prompt identity.
- Validated task pools with deterministic, disjoint, destination-balanced
  splits.
- Probe rendering, idempotent prediction normalization, explicit evaluation
  observations, exact-match scoring, and complete-matrix aggregation.
- Frozen persisted pool manifests with versioned `dr-serialize` identities,
  `dr-store` atomic canonical publication and bounded descriptor-pinned reads,
  and explicit retained-pool matching.
- The authoritative `.defs` vocabulary and contracts reference, published by
  GitHub Pages from the current TOML sources.
- Locked, multi-version Depot CI, a canonical local pre-check hook, and
  tag-triggered PyPI trusted publishing.

### Changed

- Organize the package and test suite by the `instances`, `manifests`, `pools`,
  `probes`, and `scoring` capability boundaries.
- Express pool-split coverage and balance as explicit marginal-cost policy
  solved through `dr-graph`'s exact separable transport primitive.
- Pin foundational runtime dependencies and validate workflow syntax in the
  canonical local and CI gate.
- Update package metadata and repository guidance for the public harness and
  its release process.

### Removed

- Remove the former `whetstone_envs.core` package and its import paths without
  compatibility shims.
- Remove completed implementation-planning documentation from the live tree.

## [0.1.0] - 2026-07-22

### Added

- Initial Python package, test, lint, type-checking, and CI scaffold.
