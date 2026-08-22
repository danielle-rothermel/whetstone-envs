# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
