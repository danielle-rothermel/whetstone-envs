"""GEPA's nine fidelity invariants over one run's durable evidence.

GEPA claims a specific search: an evolutionary loop that keeps a Pareto
front of candidates over per-instance validation scores, mutates a selected
candidate by reflecting over that candidate's own execution traces, records
every rejected reflection rather than silently narrowing the search, and
stops at a declared metric-call ceiling, over a train/val partition it
declared up front. These nine invariants check each
of those claims against what the run persisted, and nothing else.

Where the evidence lives (whetstone-ai ``miprofix-ai`` @ ``716976f2``)
-------------------------------------------------------------------

The Pareto data is **not** under ``GEPA_STATE_KEY``.
``step_engine.py:22-38`` defines the whole persisted step state as
``GepaStepCheckpoint{schema_version, metric_calls_consumed, terminal}``, and
``step_contract.py:70-76`` restores exactly two pool keys. The search data
is persisted once, on the terminal step, and reached like this:

- terminal ``step.history_ref`` carries ``GEPA_TERMINAL_ARTIFACT_KEY``
  (``harness_adapter.py:46``, written at ``harness_adapter.py:331``), whose
  value is a serialized ``TypedRef`` to a ``GepaRunResultArtifact``
  (``result_artifact.py:21-52``, persisted by
  ``result_artifact.py:70-120``).
- ``GepaRunResultArtifact.detailed_result_ref`` reaches
  ``GepaDetailedResult`` (``engine.py:108-164``), carrying ``candidates``,
  ``parents``, ``val_aggregate_scores``, ``val_subscores``,
  ``per_val_instance_best_candidates``, ``discovery_eval_counts``,
  ``total_metric_calls``, ``best_idx`` and ``seed``.
- ``GepaRunResultArtifact.effect_transcript_ref`` reaches
  ``GepaEffectTranscript`` (``contracts.py:666-691``), whose ``entries`` are
  ``GepaEffectTranscriptEntry`` (``contracts.py:640-663``) with
  ``effect_kind in {"evaluate", "propose"}`` and a
  ``semantic_candidate_identity_hash``.
- Per-step ``skipped_mutations`` are durable on **every** step's state delta
  (``harness_adapter.py:66-81`` writes the key unconditionally, and
  ``harness_adapter.py:405-412`` writes it on the reflection-failure path
  too).

:mod:`.._evidence` resolves all of that; the functions here are pure over
the resolved ``RunEvidence`` and never open a store.

``GEPA_REFLECTION_MINIBATCH`` is deliberately absent
----------------------------------------------------

Nothing persisted records how many instance traces one reflection consumed.
``GepaEffectTranscriptEntry.data_ids`` is per-effect, not
per-reflection-batch, and ``control.reflection_minibatch_size`` is a
*configured* value the evidence does not independently witness. An audit
with no failing fixture is not an audit, so it ships not at all rather than
as a permanent ``NOT_APPLICABLE``. :data:`GEPA_INVARIANTS` keeps nine by
adding :func:`gepa_terminal_artifact_present`, the precondition every other
invariant reads through.

Three restatements the code forced
----------------------------------

These are named here because each replaces a check that reads plausible but
fails against real, correct evidence -- an audit that fires on a healthy run
is worse than no audit.

1. **The metric-call ceiling is a request bound, not an outcome bound.**
   ``run_one_gepa_iteration`` (``step_engine.py:79-82``) requests
   ``min(resolved_max_metric_calls, consumed + 1)`` and hands it to upstream
   ``optimize()``; upstream then finishes the iteration it started, so
   ``total_metric_calls`` legitimately *overshoots* whenever the ceiling is
   not a multiple of a full valset pass. A measured smoke run with
   ``resolved_max_metric_calls == 3`` recorded ``total_metric_calls == 6``.
   :func:`gepa_metric_call_budget` therefore checks what the harness
   controls: the per-step requested budget never exceeds the ceiling, the
   counter never moves backwards, and the run terminalizes exactly when the
   counter reaches the ceiling.
2. **A dominated candidate is retained on purpose.**
   ``GepaDetailedResult.candidates`` is the full discovery history, not the
   front: candidate 0 is always the seed and stays in the tuple after it is
   dominated. The front is
   ``per_val_instance_best_candidates``, so :func:`gepa_pareto_front`
   checks that mapping *is* the per-instance argmax over ``val_subscores``
   and that the selected candidate sits on it -- not that the pool was
   pruned.
3. **The terminal candidate is judged by provenance, not by text.** On the
   fake transport the scripted reflection body happens to be the family's
   ceiling probe, so comparing the terminal text against
   ``PROBES.ceiling_template`` would fail every healthy CI run.
   :func:`gepa_no_forged_terminal` instead requires the terminal candidate
   to be ``candidates[best_idx]`` reached through a recorded parent, or an
   honest ``seed_retained`` whose retained ref is the run's own
   ``initial_candidate_ref``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import compute_identity_hash
from whetstone.optim.contracts import StepStatus
from whetstone.optim.gepa.contracts import GepaCandidateComponent
from whetstone.optim.gepa.control import GepaControl

from whetstone_envs.optim.audit._evidence import evidence_ref
from whetstone_envs.optim.audit.schema import (
    AuditFinding,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from whetstone.optim.contracts import OptimRun, OptimStepResult
    from whetstone.optim.gepa.contracts import GepaEffectTranscriptEntry
    from whetstone.optim.gepa.engine import GepaDetailedResult

    from whetstone_envs.optim.audit._evidence import (
        GepaTerminalEvidence,
        RunEvidence,
        StepEvidence,
    )

#: The identity ``GepaEffectRecorder`` hashes a candidate under when it
#: stamps ``semantic_candidate_identity_hash`` onto a transcript entry
#: (``whetstone/optim/gepa/contracts.py:874-879``). Recomputing it here is
#: what lets a pure function tie ``GepaDetailedResult.candidates`` to the
#: transcript without dereferencing either effect record. Pinned by
#: ``tests/optim/audit/test_gepa.py``: these are whetstone's persisted
#: spellings, and deriving them from anything local would drift silently.
SEMANTIC_CANDIDATE_SCHEMA = "whetstone.gepa.semantic_candidate"
SEMANTIC_CANDIDATE_SCHEMA_VERSION = 1

#: ``GepaControl.step_hyperparameters`` publishes the resolved ceiling under
#: this key on every step request (``whetstone/optim/gepa/control.py:362``),
#: so a pure invariant reads it inline from ``result.json`` rather than
#: dereferencing the optimizer config out of the store.
GEPA_MAX_METRIC_CALLS_HYPERPARAMETER = "max_metric_calls"
#: The same mapping publishes the run's algorithmic seed.
GEPA_SEED_HYPERPARAMETER = "seed"

#: The effect kinds a transcript entry may carry
#: (``whetstone/optim/gepa/contracts.py:644``).
EVALUATE_EFFECT = "evaluate"
PROPOSE_EFFECT = "propose"

#: The state-delta key GEPA writes its per-step rejections under
#: (``whetstone/optim/gepa/harness_adapter.py:50``). Spelled out here rather
#: than imported so the audit's expectation of the persisted format is
#: explicit and pinned by a golden test; the reader in :mod:`.._evidence`
#: imports the real constant, so a drift between the two fails that test.
SKIPPED_MUTATION_KEY_NAME = "skipped_mutations"

#: ``GepaSkippedMutation.exhausted`` marks the rejection that consumed the
#: last permitted attempt, so only those dropped a mutation
#: (``whetstone/optim/gepa/contracts.py:601-627``).
SKIPPED_MUTATION_EXHAUSTED_FIELD = "exhausted"
SKIPPED_MUTATION_COMPONENT_FIELD = "component_name"
SKIPPED_MUTATION_ATTEMPT_FIELD = "attempt_ordinal"
SKIPPED_MUTATION_DETAIL_FIELD = "rejection_detail"

#: Every field a durable skipped-mutation record must carry for the run's
#: skip record to be readable at all.
REQUIRED_SKIPPED_MUTATION_FIELDS = (
    SKIPPED_MUTATION_COMPONENT_FIELD,
    SKIPPED_MUTATION_ATTEMPT_FIELD,
    SKIPPED_MUTATION_DETAIL_FIELD,
    SKIPPED_MUTATION_EXHAUSTED_FIELD,
)


def _missing_terminal_artifact(
    invariant_id: InvariantId,
) -> AuditFinding:
    """The uniform FAIL when the precondition invariant already failed.

    Every invariant below the precondition needs the terminal artifact. If
    it is absent they FAIL rather than raise: a crash would make the defect
    unreportable, and ``NOT_APPLICABLE`` would let a run with no search
    evidence read as validated.
    """
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.FAIL,
        detail=(
            "the run persisted no GEPA terminal artifact, so this "
            "invariant has no evidence to judge; see "
            f"{InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT.value}"
        ),
    )


def _semantic_candidate_hash(candidate: Mapping[str, str]) -> str:
    """The transcript's identity for one ``GepaDetailedResult`` candidate.

    Mirrors ``GepaEffectRecorder``'s stamping exactly, including the
    component ordering, so a mismatch here means the recorded candidate and
    the recorded effect genuinely disagree.
    """
    components = tuple(
        GepaCandidateComponent(name=name, text=text)
        for name, text in candidate.items()
    )
    return compute_identity_hash(
        schema=SEMANTIC_CANDIDATE_SCHEMA,
        schema_version=SEMANTIC_CANDIDATE_SCHEMA_VERSION,
        payload=[
            component.model_dump(mode="json") for component in components
        ],
    )


def _artifact_refs(terminal: GepaTerminalEvidence) -> tuple[EvidenceRef, ...]:
    """The three records every GEPA invariant reads through."""
    return (
        evidence_ref(terminal.artifact_ref),
        evidence_ref(terminal.artifact.detailed_result_ref),
        evidence_ref(terminal.artifact.effect_transcript_ref),
    )


def _resolved_ceiling(entry: StepEvidence) -> int | None:
    """This step's requested metric-call ceiling, or None when unstated."""
    raw = dict(entry.step.request.record.hyperparameters).get(
        GEPA_MAX_METRIC_CALLS_HYPERPARAMETER
    )
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


# --- 1 · the precondition --------------------------------------------------


def gepa_terminal_artifact_present(evidence: RunEvidence) -> AuditFinding:
    """The terminal step persisted a GEPA result artifact bound to the run.

    This is the precondition every other GEPA invariant reads through, so it
    checks the whole chain resolves *and* that the artifact belongs to this
    run: its effect context's ``run_id`` must be the run's own, and the
    detailed result's ``control_identity_hash`` must equal the artifact's.
    An artifact from a neighbouring run in a shared store would otherwise
    let every downstream invariant pass against another run's search.
    """
    invariant_id = InvariantId.GEPA_TERMINAL_ARTIFACT_PRESENT
    terminal = evidence.gepa_terminal
    if terminal is None:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"no step's history carries a GEPA terminal artifact across "
                f"{len(evidence.steps)} steps, so the run persisted no "
                f"search result"
            ),
        )
    refs = _artifact_refs(terminal)
    artifact = terminal.artifact
    problems: list[str] = []
    if artifact.context.run_id != evidence.run_id:
        problems.append(
            f"the artifact's effect context names run "
            f"{artifact.context.run_id!r}, not {evidence.run_id!r}"
        )
    if (
        terminal.detailed_result.control_identity_hash
        != artifact.control_identity_hash
    ):
        problems.append(
            f"the detailed result's control hash "
            f"{terminal.detailed_result.control_identity_hash[:12]} does not "
            f"equal the artifact's {artifact.control_identity_hash[:12]}"
        )
    if terminal.effect_transcript.context != artifact.context:
        problems.append(
            "the effect transcript's context does not equal the artifact's"
        )
    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"the GEPA terminal artifact does not bind this run: "
                f"{'; '.join(problems)}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"the terminal step's history resolves to a GEPA result "
            f"artifact for run {evidence.run_id} carrying "
            f"{len(terminal.detailed_result.candidates)} candidates and "
            f"{len(terminal.effect_transcript.entries)} recorded effects"
        ),
        evidence_refs=refs,
    )


# --- 2 · the Pareto front --------------------------------------------------


def gepa_pareto_front(evidence: RunEvidence) -> AuditFinding:
    """The candidate front is the per-instance argmax over internal scores.

    GEPA's claim is that its pool is a Pareto front over per-instance
    validation scores rather than an aggregate leaderboard. The durable form
    of that claim is ``per_val_instance_best_candidates``: for each
    validation instance it names the candidate indices achieving the maximum
    ``val_subscores`` on that instance. This recomputes that mapping from
    ``val_subscores`` and requires it to match exactly, then requires the
    selected ``best_idx`` to sit on the resulting front.

    It deliberately does not require the pool to be pruned of dominated
    candidates: ``candidates`` is the discovery history and always retains
    the seed at index 0. Demanding a pruned pool would fail every healthy
    run. What must hold is that the *front* is honestly derived and that
    selection came from it.
    """
    invariant_id = InvariantId.GEPA_PARETO_FRONT
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)
    detailed = terminal.detailed_result
    subscores = detailed.val_subscores
    recorded = detailed.per_val_instance_best_candidates

    instances = {key for scores in subscores for key in scores}
    problems: list[str] = []
    if set(recorded) != instances:
        missing = sorted(instances - set(recorded))
        extra = sorted(set(recorded) - instances)
        problems.append(
            f"the front covers {len(recorded)} instances but the subscores "
            f"cover {len(instances)}"
            + (f"; unscored on the front: {extra[:2]}" if extra else "")
            + (f"; scored but off the front: {missing[:2]}" if missing else "")
        )
    for instance in sorted(instances & set(recorded)):
        scored = {
            index: scores[instance]
            for index, scores in enumerate(subscores)
            if instance in scores
        }
        if not scored:  # pragma: no cover - excluded by the set intersection
            continue
        best = max(scored.values())
        expected = tuple(
            sorted(index for index, score in scored.items() if score == best)
        )
        if tuple(recorded[instance]) != expected:
            problems.append(
                f"instance {instance[:12]} records front "
                f"{tuple(recorded[instance])} but its argmax at score "
                f"{best} is {expected}"
            )
    front = {index for indices in recorded.values() for index in indices}
    if front and detailed.best_idx not in front:
        problems.append(
            f"the selected candidate {detailed.best_idx} is dominated on "
            f"every instance; the front is {sorted(front)}"
        )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"the GEPA candidate front is not the per-instance argmax "
                f"over internal scores: {'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"the front over {len(recorded)} validation instances is exactly "
            f"the per-instance argmax across {len(detailed.candidates)} "
            f"candidates, and the selected candidate "
            f"{detailed.best_idx} sits on it"
        ),
        evidence_refs=refs,
    )


# --- 3 · mutation provenance ----------------------------------------------


def _propose_entries_by_base(
    entries: Sequence[GepaEffectTranscriptEntry],
) -> dict[str, list[GepaEffectTranscriptEntry]]:
    """Propose entries grouped by the candidate they reflected over.

    A propose effect's ``semantic_candidate_identity_hash`` is the hash of
    the candidate *handed to* the reflection -- the base -- because the
    recorder stamps ``request.candidate``
    (``whetstone/optim/gepa/contracts.py:874-879``, over
    ``GepaProposalEffectRequest.candidate``). The mutated output is not
    hashed into the entry, so a mutation is traced base-first.
    """
    grouped: dict[str, list[GepaEffectTranscriptEntry]] = {}
    for entry in entries:
        if entry.effect_kind == PROPOSE_EFFECT:
            grouped.setdefault(
                entry.semantic_candidate_identity_hash, []
            ).append(entry)
    return grouped


def gepa_mutation_traces_to_reflection(
    evidence: RunEvidence,
) -> AuditFinding:
    """Every accepted mutation traces to a reflection over execution traces.

    For each candidate that has a recorded parent -- that is, every
    candidate GEPA created rather than seeded -- this requires:

    1. a ``propose`` transcript entry over the *parent's* semantic identity,
       so the mutation came from a recorded reflection call rather than
       appearing in the pool unexplained; and
    2. at least one ``evaluate`` entry over that same parent identity at an
       earlier invocation ordinal, so the reflection had execution traces of
       its base candidate to reflect over rather than reflecting on nothing.

    Ordinals are contiguous from zero and monotone by construction
    (``GepaEffectTranscript._validate``), so comparing them orders the
    effects without a clock.

    A run that accepted no mutation -- ``seed_retained``, or a search that
    found nothing better -- has nothing to trace and passes with that stated.
    """
    invariant_id = InvariantId.GEPA_MUTATION_TRACES_TO_REFLECTION
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)
    detailed = terminal.detailed_result
    entries = terminal.effect_transcript.entries
    proposals = _propose_entries_by_base(entries)
    hashes = tuple(
        _semantic_candidate_hash(candidate)
        for candidate in detailed.candidates
    )

    problems: list[str] = []
    traced = 0
    for index, parents in enumerate(detailed.parents):
        named = [parent for parent in parents if parent is not None]
        if not named:
            continue
        traced += 1
        for parent in named:
            if not 0 <= parent < len(hashes):
                problems.append(
                    f"candidate {index} names parent {parent}, which is not "
                    f"one of the {len(hashes)} recorded candidates"
                )
                continue
            parent_hash = hashes[parent]
            matching = proposals.get(parent_hash, [])
            if not matching:
                problems.append(
                    f"candidate {index} descends from candidate {parent} but "
                    f"no propose effect reflected over it"
                )
                continue
            earliest = min(entry.invocation_ordinal for entry in matching)
            grounded = any(
                entry.effect_kind == EVALUATE_EFFECT
                and entry.semantic_candidate_identity_hash == parent_hash
                and entry.invocation_ordinal < earliest
                for entry in entries
            )
            if not grounded:
                problems.append(
                    f"the reflection producing candidate {index} at ordinal "
                    f"{earliest} had no earlier evaluation of candidate "
                    f"{parent} to reflect over"
                )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"{len(problems)} of {traced} mutated candidates do not "
                f"trace to a grounded reflection: {'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    if traced == 0:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.PASS,
            detail=(
                "the run accepted no mutated candidate, so there is no "
                "mutation whose reflection provenance could be missing"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"all {traced} mutated candidates trace to a propose effect over "
            f"their recorded parent, each preceded by an evaluation of that "
            f"parent across {len(entries)} recorded effects"
        ),
        evidence_refs=refs,
    )


# --- 4 · the metric-call ceiling ------------------------------------------


def gepa_metric_call_budget(evidence: RunEvidence) -> AuditFinding:
    """The run respected the metric-call ceiling it was configured with.

    The ceiling bounds what the harness *requests*, not what upstream
    ultimately spends: ``run_one_gepa_iteration`` requests
    ``min(ceiling, consumed + 1)`` and upstream finishes the iteration it
    started, so ``total_metric_calls`` overshoots whenever the ceiling is
    not a multiple of a full valset pass. Checking the outcome against the
    ceiling would fail correct runs, so this checks the three things the
    harness genuinely controls:

    - every step advertises the same ceiling;
    - ``metric_calls_consumed`` never decreases across steps, so a resumed
      run cannot re-spend a prefix; and
    - the run terminalizes exactly when the counter reaches the ceiling --
      no step continues past it, and no step terminalizes below it.

    It also requires ``total_metric_calls`` on the detailed result to equal
    the terminal checkpoint's counter, so the number the report prints is
    the number the harness accounted for.
    """
    invariant_id = InvariantId.GEPA_METRIC_CALL_BUDGET
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)

    ceilings = {
        entry.index: _resolved_ceiling(entry) for entry in evidence.steps
    }
    stated = {value for value in ceilings.values() if value is not None}
    if not stated:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"no step advertises a "
                f"{GEPA_MAX_METRIC_CALLS_HYPERPARAMETER} hyperparameter, so "
                f"the run states no metric-call ceiling to respect"
            ),
            evidence_refs=refs,
        )
    problems: list[str] = []
    if len(stated) > 1:
        problems.append(
            f"the ceiling changed mid-run across steps: {sorted(stated)}"
        )
    ceiling = min(stated)

    previous = 0
    last: int | None = None
    for entry in evidence.steps:
        checkpoint = entry.gepa_checkpoint()
        if checkpoint is None:
            problems.append(f"step {entry.index} persisted no GEPA checkpoint")
            continue
        consumed = checkpoint.metric_calls_consumed
        if consumed < previous:
            problems.append(
                f"step {entry.index} reports {consumed} metric calls "
                f"consumed, below step {entry.index - 1}'s {previous}"
            )
        if checkpoint.terminal and consumed < ceiling and consumed != 0:
            problems.append(
                f"step {entry.index} terminalized at {consumed} metric "
                f"calls, below the ceiling of {ceiling}"
            )
        if not checkpoint.terminal and consumed >= ceiling:
            problems.append(
                f"step {entry.index} continued at {consumed} metric calls, "
                f"at or past the ceiling of {ceiling}"
            )
        previous = consumed
        last = consumed

    accounted = terminal.detailed_result.total_metric_calls
    if last is not None and accounted is not None and accounted != last:
        problems.append(
            f"the detailed result accounts for {accounted} metric calls but "
            f"the terminal checkpoint records {last}"
        )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"the run did not respect its metric-call ceiling of "
                f"{ceiling}: {'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"metric calls advanced monotonically to {last} against a "
            f"ceiling of {ceiling} across {len(evidence.steps)} steps, and "
            f"the detailed result accounts for the same total"
        ),
        evidence_refs=refs,
    )


# --- 5 · skipped mutations -------------------------------------------------


def _as_skip_record(value: object) -> dict[str, object] | None:
    """A persisted skipped-mutation entry as a string-keyed record.

    ``gepa_skipped_mutations`` yields the decoded list verbatim, so each
    entry is untyped and a malformed one is a finding rather than a crash.
    Returns None when the entry is not a JSON object, which the caller
    reports as "is not a record".
    """
    if not isinstance(value, dict):
        return None
    record: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        record[key] = item
    return record


def gepa_skipped_mutations_recorded(evidence: RunEvidence) -> AuditFinding:
    """Every rejected reflection is durable on the step that produced it.

    A rejected reflection silently narrows the search, so GEPA records one
    entry per rejected *attempt* under ``GEPA_SKIPPED_MUTATIONS_KEY``. The
    key is written unconditionally on every step's state delta, including
    the reflection-failure path, so this requires:

    - every step that persisted GEPA state also persisted the key, even
      when empty -- an absent key is indistinguishable from "nothing was
      rejected" only if it is never absent;
    - every recorded entry carries the fields that make a rejection legible
      (``component_name``, ``attempt_ordinal``, ``rejection_detail``,
      ``exhausted``); and
    - the run-level record is the union over every step, and the terminal
      transcript's own skips are contained in it.

    That last clause is the caveat the ``GepaSkippedMutation`` docstring
    states outright: the terminal transcript carries only the terminal
    step's skips, so a run-level count taken from the transcript alone would
    under-report every rejection on a continuing step.
    """
    invariant_id = InvariantId.GEPA_SKIPPED_MUTATIONS_RECORDED
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)

    problems: list[str] = []
    union: list[Mapping[str, object]] = []
    exhausted = 0
    for entry in evidence.steps:
        if entry.gepa_checkpoint() is None:
            continue
        state = entry.state or {}
        if SKIPPED_MUTATION_KEY_NAME not in state:
            problems.append(
                f"step {entry.index} persisted GEPA state without a "
                f"{SKIPPED_MUTATION_KEY_NAME!r} key, so a rejection on that "
                f"step would not be durable"
            )
            continue
        for position, raw_record in enumerate(entry.gepa_skipped_mutations()):
            record = _as_skip_record(raw_record)
            if record is None:
                problems.append(
                    f"step {entry.index} skip {position} is not a record"
                )
                continue
            missing = [
                field
                for field in REQUIRED_SKIPPED_MUTATION_FIELDS
                if field not in record
            ]
            if missing:
                problems.append(
                    f"step {entry.index} skip {position} omits "
                    f"{', '.join(missing)}"
                )
                continue
            union.append(record)
            if record[SKIPPED_MUTATION_EXHAUSTED_FIELD] is True:
                exhausted += 1

    def _identity(record: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(
            record.get(field) for field in REQUIRED_SKIPPED_MUTATION_FIELDS
        )

    recorded = {_identity(record) for record in union}
    for skipped in terminal.effect_transcript.skipped_mutations:
        identity = _identity(skipped.model_dump(mode="json"))
        if identity not in recorded:
            problems.append(
                f"the terminal transcript records a rejection of "
                f"{skipped.component_name!r} at attempt "
                f"{skipped.attempt_ordinal} that no step's state carries"
            )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"rejected reflections are not durably recorded per step: "
                f"{'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"every GEPA step persisted a skipped-mutation record; the "
            f"run-level union over all steps holds {len(union)} rejected "
            f"attempts, {exhausted} of which dropped a mutation, and it "
            f"contains every rejection the terminal transcript reports"
        ),
        evidence_refs=refs,
    )


# --- 6 · step evidence -----------------------------------------------------


def gepa_step_evidence_present(evidence: RunEvidence) -> AuditFinding:
    """Every step that spent metric calls carries evidence of the spend.

    A GEPA step re-runs upstream ``optimize()`` from the seed and replays
    the already-paid prefix through the durable effect cache, so a step can
    legitimately produce no *new* evidence while the cache serves it. The
    discriminator is the checkpoint: a step whose ``metric_calls_consumed``
    advanced past the prior step's did buy something, and must carry at
    least one ``resolved_intents`` or ``search_evidence`` entry to show for
    it. A step whose counter did not advance bought nothing and is exempt.

    Deriving the predicate from ``budget_delta`` instead would exempt
    nothing: ``GepaStepCheckpoint.budget_delta`` reports
    ``metric_calls: 1`` for every non-degenerate step regardless of what
    upstream spent, so the two consecutive checkpoints are the only honest
    source.
    """
    invariant_id = InvariantId.GEPA_STEP_EVIDENCE_PRESENT
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)

    problems: list[str] = []
    paying = 0
    exempt = 0
    previous = 0
    for entry in evidence.steps:
        checkpoint = entry.gepa_checkpoint()
        if checkpoint is None:
            problems.append(f"step {entry.index} persisted no GEPA checkpoint")
            continue
        consumed = checkpoint.metric_calls_consumed
        advanced = consumed > previous
        previous = consumed
        if not advanced:
            exempt += 1
            continue
        paying += 1
        if not entry.resolved_intents and not entry.search_evidence:
            problems.append(
                f"step {entry.index} advanced to {consumed} metric calls but "
                f"carries neither a resolved intent nor search evidence"
            )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"{len(problems)} of {paying} paying steps carry no "
                f"evaluation evidence: {'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"all {paying} steps whose metric-call counter advanced carry "
            f"evaluation evidence; {exempt} pure-replay steps consumed "
            f"nothing and are exempt"
        ),
        evidence_refs=refs,
    )


# --- 7 · terminal provenance ----------------------------------------------


def gepa_no_forged_terminal(evidence: RunEvidence) -> AuditFinding:
    """The terminal candidate came from this run's own search.

    Two honest outcomes, and nothing else:

    - **A live draft.** The step accepted a candidate whose
      ``payload[mutation_field]`` equals ``candidates[best_idx]`` on the
      detailed result, and that candidate has a recorded parent -- so it was
      produced by the search rather than injected.
    - **An honest retention.** The step set ``seed_retained``, its
      ``retained_candidate_ref`` is the run's own ``initial_candidate_ref``,
      and it accepted no candidate. GEPA takes this path when the search
      found nothing better than the seed
      (``harness_adapter.py:333-346``), which is a clean completion.

    The retention clauses are re-checked here but are **already structurally
    guaranteed upstream**: ``OptimStepResult._validate``
    (``optim/contracts.py:1166-1193``) refuses a ``seed_retained`` step that
    is not COMPLETE, that accepts any candidate, whose run names no
    ``initial_candidate_ref``, or whose ``retained_candidate_ref`` is not
    that exact seed. A dishonest retention is therefore unrepresentable in a
    schema-valid artifact, so those clauses have no negative fixture and
    cannot get one. They are kept as defence in depth against an upstream
    weakening; the clauses this invariant genuinely adds -- and which its
    fixtures exercise -- are the two live-draft ones below.

    Judging by provenance rather than by comparing the terminal text against
    the family's ceiling probe is deliberate: the fake transport's scripted
    reflection body *is* the ceiling probe, so a textual check would fail
    every healthy CI run while still missing a forged terminal whose text
    happened to differ.
    """
    invariant_id = InvariantId.GEPA_NO_FORGED_TERMINAL
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)

    completed = [
        entry
        for entry in evidence.steps
        if entry.step.status is StepStatus.COMPLETE
    ]
    if not completed:
        return _forged(
            refs,
            f"no step completed across {len(evidence.steps)} steps, so the "
            f"run produced no terminal candidate to account for",
        )
    final = completed[-1].step
    run = evidence.result.run.record
    if final.seed_retained:
        return _honest_retention(final, run=run, refs=refs)
    return _live_draft(
        final, run=run, detailed=terminal.detailed_result, refs=refs
    )


def _forged(refs: tuple[EvidenceRef, ...], detail: str) -> AuditFinding:
    return AuditFinding(
        invariant_id=InvariantId.GEPA_NO_FORGED_TERMINAL,
        status=AuditStatus.FAIL,
        detail=detail,
        evidence_refs=refs,
    )


def _accounted(refs: tuple[EvidenceRef, ...], detail: str) -> AuditFinding:
    return AuditFinding(
        invariant_id=InvariantId.GEPA_NO_FORGED_TERMINAL,
        status=AuditStatus.PASS,
        detail=detail,
        evidence_refs=refs,
    )


def _honest_retention(
    final: OptimStepResult,
    *,
    run: OptimRun,
    refs: tuple[EvidenceRef, ...],
) -> AuditFinding:
    """The seed-retention outcome, re-checked as defence in depth.

    Every clause here is already enforced by
    ``OptimStepResult._validate`` (``optim/contracts.py:1166-1193``), so no
    schema-valid artifact can reach a FAIL below. They are kept so an
    upstream weakening surfaces as an audit failure rather than silently.
    """
    retained = final.retained_candidate_ref
    if retained is None:
        return _forged(
            refs,
            "the terminal step claims seed_retained but cites no retained "
            "candidate, so the retention is unverifiable",
        )
    initial = run.initial_candidate_ref
    seed = "none" if initial is None else initial.record_ref.content_hash[:12]
    kept = retained.record_ref.content_hash[:12]
    if seed != kept:
        return _forged(
            refs,
            f"the terminal step retained candidate {kept}, which is not the "
            f"run's initial candidate {seed}",
        )
    if final.accepted_candidates:
        return _forged(
            refs,
            f"the terminal step claims seed_retained while also accepting "
            f"{len(final.accepted_candidates)} candidates",
        )
    return _accounted(
        refs,
        f"the run honestly retained its seed: the terminal step accepted "
        f"nothing and cites initial candidate {kept}",
    )


def _live_draft(
    final: OptimStepResult,
    *,
    run: OptimRun,
    detailed: GepaDetailedResult,
    refs: tuple[EvidenceRef, ...],
) -> AuditFinding:
    """The accepted-candidate outcome: it must be the search's own draft."""
    if len(final.accepted_candidates) != 1:
        return _forged(
            refs,
            f"the terminal step accepted {len(final.accepted_candidates)} "
            f"candidates without claiming seed_retained; GEPA returns "
            f"exactly one",
        )
    accepted = final.accepted_candidates[0].record
    text = accepted.payload.get(run.mutation_field)
    best = detailed.candidates[detailed.best_idx]
    if text not in set(best.values()):
        return _forged(
            refs,
            f"the terminal candidate's {run.mutation_field!r} payload is not "
            f"the text of selected candidate {detailed.best_idx} in the "
            f"run's own search result",
        )
    parents = detailed.parents[detailed.best_idx]
    named = [parent for parent in parents if parent is not None]
    if not named:
        return _forged(
            refs,
            f"the terminal candidate is search candidate "
            f"{detailed.best_idx}, which records no parent, so it was never "
            f"produced by a reflection in this run",
        )
    return _accounted(
        refs,
        f"the terminal candidate is search candidate {detailed.best_idx}, "
        f"descended from {named} in this run's own recorded search",
    )


# --- 8 · platform resume identity -----------------------------------------


def gepa_platform_resume_identity(evidence: RunEvidence) -> AuditFinding:
    """A deferred-and-resumed run replayed its paid prefix, never re-bought.

    Platform dispatch splits one GEPA run into deferral episodes. Each
    episode re-runs upstream ``optimize()`` from the seed and must be served
    the already-paid prefix from the durable effect cache; a re-execution
    would mint a conflicting effect and raise ``GepaEffectConflictError``.

    **Deferral is read from the intents, not from a status.** ``StepStatus``
    has no deferred member; a deferral returns ``CONTINUE`` carrying the
    persisted intent (``harness_adapter.py:277-285``), and the harness turns
    ``AdapterOutput.optim_eval_requests`` into the step's
    ``resolved_intents`` (``optim/harness.py:339,390``). In-process GEPA
    drives every evaluation through the effect cache and emits **no**
    intents at all -- a measured in-process run recorded zero
    ``resolved_intents`` and all of its evaluations under
    ``search_evidence``. So a GEPA step carrying a resolved intent is a
    deferral episode, and a run with none was dispatched in process.

    The durable witness of a healthy resume is that the run reached its
    terminal artifact with no ``terminal_failure`` while its effect ordinals
    stayed contiguous and no replayed effect recorded a score mismatch --
    upstream re-derived the same effects and the cache matched them.

    In-process runs never defer, so there is no resume to judge and this
    reports ``NOT_APPLICABLE`` with that reason. That is legitimate here
    precisely because it is conditional on dispatch rather than always true
    for GEPA: a platform-dispatched run reaches the checks below.
    """
    invariant_id = InvariantId.GEPA_PLATFORM_RESUME_IDENTITY
    terminal = evidence.gepa_terminal
    if terminal is None:
        return _missing_terminal_artifact(invariant_id)
    refs = _artifact_refs(terminal)

    deferred = [entry for entry in evidence.steps if entry.resolved_intents]
    if not deferred:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.NOT_APPLICABLE,
            detail=(
                f"no step deferred across {len(evidence.steps)} steps, so "
                f"this run was dispatched in process and has no resume "
                f"episode whose replay identity could drift"
            ),
            evidence_refs=refs,
        )

    entries = terminal.effect_transcript.entries
    ordinals = [entry.invocation_ordinal for entry in entries]
    problems: list[str] = []
    if ordinals != list(range(len(entries))):
        problems.append(
            f"the terminal transcript's effect ordinals are not contiguous "
            f"from zero across {len(entries)} effects"
        )
    if evidence.result.terminal_failure is not None:
        problems.append(
            f"the run ended in terminal failure "
            f"{evidence.result.terminal_failure.code!r}"
        )
    if terminal.effect_transcript.score_mismatch_evidence:
        problems.append(
            f"{len(terminal.effect_transcript.score_mismatch_evidence)} "
            f"replayed effects recorded a score mismatch"
        )

    if problems:
        return AuditFinding(
            invariant_id=invariant_id,
            status=AuditStatus.FAIL,
            detail=(
                f"the run's {len(deferred)} deferral episodes did not replay "
                f"identically: {'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.PASS,
        detail=(
            f"the run resumed across {len(deferred)} deferral episodes and "
            f"reached its terminal artifact with {len(entries)} contiguous "
            f"effects, no conflict and no score mismatch"
        ),
        evidence_refs=refs,
    )


# --- 9 · train/val disjointness -------------------------------------------


def gepa_train_val_disjoint(evidence: RunEvidence) -> AuditFinding:
    """Reflection and Pareto selection read disjoint task sets.

    GEPA writes each mutation by reflecting over trainset failures, then
    selects its Pareto frontier by scoring candidates on the valset. If the
    two sets overlap, a candidate is selected on the very tasks whose
    failures were quoted into the instruction that produced it, so the
    frontier rewards fitting the reflection examples rather than
    generalizing past them -- and the study reads that as search efficacy.

    Read from the persisted control at the ref the run binds itself to,
    not from a step echo written by the code path under audit. Every
    evaluation the run issued must also fall inside the union of the two
    sets: an intent reaching outside them scored a task the declared
    partition never admitted.
    """
    invariant = InvariantId.GEPA_TRAIN_VAL_DISJOINT
    refs = (evidence_ref(evidence.control_ref),)
    if evidence.control_record is None:
        return AuditFinding(
            invariant_id=invariant,
            status=AuditStatus.FAIL,
            detail=(
                "the run's optimizer config ref resolves to no control "
                "record, so the declared train/val split is unreadable"
            ),
            evidence_refs=refs,
        )
    try:
        control = GepaControl.model_validate(evidence.control_record)
    except ValueError:
        return AuditFinding(
            invariant_id=invariant,
            status=AuditStatus.FAIL,
            detail=(
                "the run's persisted control does not validate as a "
                "GepaControl, so the declared train/val split is unreadable"
            ),
            evidence_refs=refs,
        )

    trainset = tuple(control.trainset_task_hashes)
    valset = tuple(control.valset_task_hashes)
    overlap = set(trainset) & set(valset)
    if overlap:
        return AuditFinding(
            invariant_id=invariant,
            status=AuditStatus.FAIL,
            detail=(
                f"the control's trainset ({len(trainset)}) and valset "
                f"({len(valset)}) share {len(overlap)} task(s), so the "
                f"Pareto frontier is scored on reflected-over tasks"
            ),
            evidence_refs=refs,
        )

    admitted = set(trainset) | set(valset)
    problems: list[str] = []
    checked = 0
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            tasks = resolution.optim_eval_request.task_hashes
            if tasks is None:
                continue
            checked += 1
            outside = set(tasks) - admitted
            if outside:
                problems.append(
                    f"step {entry.index} intent {position} evaluates "
                    f"{len(outside)} task(s) outside the declared "
                    f"train/val partition"
                )

    if problems:
        return AuditFinding(
            invariant_id=invariant,
            status=AuditStatus.FAIL,
            detail="; ".join(problems),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=invariant,
        status=AuditStatus.PASS,
        detail=(
            f"the control's {len(trainset)}-task trainset and "
            f"{len(valset)}-task valset are disjoint, and all {checked} "
            f"task-scoped evaluation intent(s) drew only from them"
        ),
        evidence_refs=refs,
    )


#: GEPA's nine invariants, in the order an ``audit.json`` reports them.
#: The precondition comes first so a reader triaging a failing run sees
#: immediately whether the search result was readable at all.
GEPA_INVARIANTS = (
    gepa_terminal_artifact_present,
    gepa_pareto_front,
    gepa_mutation_traces_to_reflection,
    gepa_metric_call_budget,
    gepa_skipped_mutations_recorded,
    gepa_step_evidence_present,
    gepa_no_forged_terminal,
    gepa_platform_resume_identity,
    gepa_train_val_disjoint,
)


__all__ = [
    "EVALUATE_EFFECT",
    "GEPA_INVARIANTS",
    "GEPA_MAX_METRIC_CALLS_HYPERPARAMETER",
    "GEPA_SEED_HYPERPARAMETER",
    "PROPOSE_EFFECT",
    "REQUIRED_SKIPPED_MUTATION_FIELDS",
    "SEMANTIC_CANDIDATE_SCHEMA",
    "SEMANTIC_CANDIDATE_SCHEMA_VERSION",
    "SKIPPED_MUTATION_KEY_NAME",
    "gepa_metric_call_budget",
    "gepa_mutation_traces_to_reflection",
    "gepa_no_forged_terminal",
    "gepa_pareto_front",
    "gepa_platform_resume_identity",
    "gepa_skipped_mutations_recorded",
    "gepa_step_evidence_present",
    "gepa_terminal_artifact_present",
    "gepa_train_val_disjoint",
]
