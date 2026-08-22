"""The six fidelity invariants of one Codex-direct optimizer run.

Codex is the only optimizer in the study whose search is not whetstone's.
A foreign agent runs out of process, decides for itself what to try, and
returns one selection. So the other optimizers' invariants -- which read
whetstone's own recorded decisions -- have nothing to check here. What
these six check instead is the *containment*: whether every evaluation
that agent bought went through the one Tool it was granted, and whether
the candidate it returned is one those purchases actually measured.

That reframing is why the ledger is the authority. The Codex adapter
reconciles what the agent *reported* against what the run *durably
admitted*, and fails the Step when the two disagree
(``codex_unreported_evaluation``). An audit cannot re-run that
reconciliation, but it can check its output survived into the artifact:
the admission ledger says what was paid for, ``tool_evidence`` says what
is reachable from the Step Result, and totality is the two agreeing.

Read paths, verified against whetstone-ai ``08-22-codex`` @ ``0516f5dd``:

- **Ledger.** ``ToolAdmissionAuthority.sqlite(runtime.sqlite)`` then
  ``admitted_entries(...)``, per F6 -- never raw SQL over
  ``whetstone_tool_admission_entry``. ``optim/tools/facade.py:139`` is
  the authority-level reader and ``:524`` the ``ToolCallStore`` one; the
  Codex adapter itself reads through the latter
  (``optim/codex/adapter.py:522``). The run's own ``runtime.sqlite``
  carries the admission tables beside the object store, so the audit
  reaches the ledger without reconstructing a live tool store.
- **Reported evidence.** ``OptimStepResult.tool_evidence`` -- one
  ``ToolEvidence`` per admitted call the Step put on its Issued Tool Call
  ledger, each binding a ``ToolCallStoreEntry`` to its ``ToolResultRef``
  (``optim/contracts.py:833``).
- **Selection.** ``CodexOutputArtifact.selected_call_id`` and
  ``lease_token_hash``, reached through the terminal step's
  ``state_delta`` key ``codex_output_artifact_ref``
  (``optim/codex/adapter.py:331``).
- **Tool surface.** ``entry.tool_config.record.definition.record``. There
  is no ``tool_name`` column: the name is nested and must be
  dereferenced, and there is exactly one tool
  (``optim/codex/mcp_bridge.py:22,147``), so the F6 restatement drops the
  ``read-scores`` tool the protocol assumed.
- **Spend.** ``OptimResult.cost`` as a ``RunCostReport``: per OQ1 there
  is no ``codex_agent`` cost role, so task-model spend is read at role
  granularity (``task_model.calls``) rather than attributed per call.

**One guard this module deliberately does not lean on.**
``ToolEvidence._validate`` already requires the entry to cite the exact
Tool Call and Tool Result, and requires ``capacity_debit_ordinal ==
provenance_ordinal``, so a well-formed ``ToolEvidence`` cannot lie about
that pairing. But an audit that trusted the upstream validator would go
silent the moment the validator changed, which is the drift this package
exists to catch -- so ``codex_no_eval_outside_tools`` re-derives the
citation itself.

**Why totality is checked in both directions.** The obvious direction is
ledger-to-evidence: paid work that no longer appears on the Step Result.
The reverse matters because ``admitted_entries`` selects on
``(namespace, tool_config_hash, capacity scope)`` -- so an entry whose
Tool Config was altered does not read as *wrong*, it silently stops
matching the scope and vanishes from the ledger read. Checking only one
direction would let exactly that tampering pass.

**What §3.4's sixth row could not be.** Two readings of "wall/interrupted
failures carry their paid evidence" turned out to be already guaranteed,
and an invariant with no reachable failure is not an invariant: evidence
reachability is the first invariant's business on every run, and a Step's
verdict accounting for its nested evaluation failures is enforced by
``OptimStepResult`` itself (``_validate_shared_terminal_failure``), so no
artifact violating it can even load. The spend record is what remains
unguarded -- see ``codex_failures_carry_paid_evidence``.

**Import posture.** ``CodexOutputArtifact`` and ``CODEX_EVAL_INPUT_FIELDS``
are 0.1.7 surface. This module imports them lazily inside the functions
that need them, so registering the Codex invariants never imports the
Codex module -- an envs install pinned to 0.1.6 still loads the registry,
and only a Codex run's own invariants report that the surface is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from whetstone.core.identity import TypedRef
from whetstone.optim.cost import CostRole
from whetstone.optim.tools.admission import ToolCallState
from whetstone.optim.tools.contracts import (
    GLOBAL_CAPACITY_SCOPE_ID,
    ToolCapacityScope,
)

from whetstone_envs.optim.audit._evidence import (
    RUNTIME_STORE_FILENAME,
    evidence_ref,
)
from whetstone_envs.optim.audit.schema import (
    AuditFinding,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

if TYPE_CHECKING:
    from whetstone.optim.contracts import ToolEvidence
    from whetstone.optim.tools.admission import ToolCallStoreEntry

    from whetstone_envs.optim.audit._evidence import RunEvidence

#: The ``state_delta`` key the Codex adapter writes its output artifact's
#: ref under (``optim/codex/adapter.py:331``). It is a persisted key in
#: whetstone's state snapshot, so it is named here rather than spelled at
#: a call site, and pinned by a golden test. ``_evidence`` resolves the
#: ref it addresses at load time, so the same spelling appears in
#: ``STATE_RECORD_REF_KEYS`` and a golden test pins the two together.
CODEX_OUTPUT_ARTIFACT_REF_KEY = "codex_output_artifact_ref"

#: The ``state_delta`` key carrying the run's durable accepted-call count
#: (``optim/codex/adapter.py:334``).
CODEX_ACCEPTED_CALL_COUNT_KEY = "harness_store_accepted_call_count"

#: The ``RunCostReport`` field naming task-model spend. Per OQ1 there is
#: no ``codex_agent`` role, so the Codex agent's own tokens are not
#: whetstone-attributable and this is the only role a Codex audit reads.
#: It is a persisted wire key, so it is named from the owning enum rather
#: than spelled as a literal.
TASK_MODEL_COST_ROLE = CostRole.TASK_MODEL.value

#: Length of a hex-encoded SHA-256 digest. The run lease token's hash is
#: one (``adapter.py:897``), and an artifact carrying anything else is
#: not carrying a run-lease binding at all.
SHA256_HEX_LENGTH = 64

#: The admission cap this study buys per Codex run (D2). The audit reads
#: the run's own configured cap from its Tool Config and additionally
#: holds it to this pre-registered number, so a run that quietly bought a
#: larger budget is a finding rather than a silently different arm.
CODEX_CAPACITY_CAP = 8


class _CodexSurfaceMissingError(RuntimeError):
    """The installed whetstone-ai carries no Codex-direct surface.

    Raised by the lazy importers below. Every invariant turns it into a
    FAIL rather than propagating: an audit that crashed on an old install
    would be indistinguishable from one that never ran.
    """


def _codex_artifact_model() -> Any:
    try:
        from whetstone.optim.codex.adapter import (  # noqa: PLC0415
            CodexOutputArtifact,
        )
    except ImportError as error:  # pragma: no cover - 0.1.6 installs only
        raise _CodexSurfaceMissingError(str(error)) from error
    if "selected_call_id" not in CodexOutputArtifact.model_fields:
        raise _CodexSurfaceMissingError(
            "the installed CodexOutputArtifact predates the Codex-direct "
            "selection surface; upgrade whetstone-ai to 0.1.7"
        )
    return CodexOutputArtifact


def _admission_authority() -> Any:
    """The ledger reader, or a refusal when the install predates it.

    ``ToolAdmissionAuthority.admitted_entries`` is 0.1.7 surface: 0.1.6
    exposes ``accepted_count`` only, which says how many evaluations were
    paid for but not which ones. F6 requires reading the ledger rows
    themselves, so an install without this method cannot audit a Codex
    run at all -- and says so rather than auditing a subset silently.
    """
    from whetstone.optim.tools.facade import (  # noqa: PLC0415
        ToolAdmissionAuthority,
    )

    if not hasattr(ToolAdmissionAuthority, "admitted_entries"):
        raise _CodexSurfaceMissingError(
            "the installed ToolAdmissionAuthority exposes no "
            "admitted_entries; upgrade whetstone-ai to 0.1.7"
        )
    return ToolAdmissionAuthority


def _codex_eval_input_fields() -> frozenset[str]:
    try:
        from whetstone.optim.codex.mcp_bridge import (  # noqa: PLC0415
            CODEX_EVAL_INPUT_FIELDS,
        )
    except ImportError as error:  # pragma: no cover - 0.1.6 installs only
        raise _CodexSurfaceMissingError(str(error)) from error
    return frozenset(CODEX_EVAL_INPUT_FIELDS)


def _missing_surface(
    invariant_id: InvariantId, error: _CodexSurfaceMissingError
) -> AuditFinding:
    """A Codex invariant that could not be judged at all.

    This is a FAIL, not ``NOT_APPLICABLE``: the run *is* a Codex run, so
    its invariant genuinely applies, and reporting it inapplicable would
    let an un-auditable Codex arm read as validated.
    """
    return AuditFinding(
        invariant_id=invariant_id,
        status=AuditStatus.FAIL,
        detail=(
            f"this Codex run cannot be audited against the installed "
            f"whetstone-ai: {error}"
        ),
    )


def _ledger_entries(
    evidence: RunEvidence,
) -> tuple[ToolCallStoreEntry, ...]:
    """Every durable admission entry this run debited capacity for.

    Read through ``admitted_entries``, per F6. The run's Tool Config and
    capacity binding come from the Step's own reported Tool Evidence:
    a Codex Step is granted exactly one Tool, so any reported entry names
    the scope, and a run with no reported evidence has nothing to scope a
    ledger read by. That is not a gap -- ``codex_no_eval_outside_tools``
    is precisely the invariant that reads the count another way and
    reports the disagreement.
    """
    scope = _tool_scope(evidence)
    if scope is None:
        return ()
    namespace, config_hash, capacity_scope, scope_id = scope
    authority = _admission_authority().sqlite(
        evidence.run_dir / RUNTIME_STORE_FILENAME
    )
    try:
        return authority.admitted_entries(
            store_namespace_key=namespace,
            tool_config_hash=config_hash,
            capacity_scope=capacity_scope,
            capacity_scope_id=scope_id,
        )
    finally:
        authority.close()


def _tool_scope(
    evidence: RunEvidence,
) -> tuple[str, str, ToolCapacityScope, str] | None:
    """The one capacity scope this Codex run admitted evaluations in.

    Read from ``OptimRun.tool_configs``, not from the Step's reported
    evidence. Scoping the ledger by what the run *reported* would be
    circular in the worst direction: a run that reported nothing would
    scope to nothing, read an empty ledger, and pass ledger totality
    vacuously -- which is precisely the under-reporting these invariants
    exist to catch. The run record names its one granted Tool
    independently of anything the agent did.

    A Codex run is granted exactly one Tool (D12), so a run carrying a
    different number is not a scope ambiguity to resolve here; it is a
    finding, and ``codex_tool_surface_minimal`` reports it.
    """
    configs = evidence.result.run.record.tool_configs
    if len(configs) != 1:
        return None
    config = configs[0].record
    scope = config.capacity.scope
    if scope is ToolCapacityScope.GLOBAL:
        scope_id = GLOBAL_CAPACITY_SCOPE_ID
    elif scope is ToolCapacityScope.RUN:
        scope_id = evidence.result.run.record_ref.content_hash
    else:
        # A STEP-scoped capacity binds one Step Request, so it has no
        # single run-level scope id. Codex runs one Step and binds at
        # RUN scope; anything else is out of contract rather than a
        # shape to guess at.
        return None
    return (
        str(config.store_namespace_key),
        str(configs[0].config_hash),
        scope,
        str(scope_id),
    )


def _reported_evidence(evidence: RunEvidence) -> tuple[ToolEvidence, ...]:
    """Every ``ToolEvidence`` entry across the run's steps, in order."""
    return tuple(
        item for step in evidence.steps for item in step.tool_evidence
    )


def _reported_entries(
    evidence: RunEvidence,
) -> tuple[ToolCallStoreEntry, ...]:
    return tuple(item.store_entry for item in _reported_evidence(evidence))


def _artifact(evidence: RunEvidence) -> Any | None:
    """The Codex output artifact, read from the step that recorded it.

    Returns None when no step recorded one. That is itself evidence: a
    Codex Step that failed before producing a usable artifact records no
    ref, and the invariants that care say so in their own detail rather
    than this reader raising.
    """
    model = _codex_artifact_model()
    for step in reversed(evidence.steps):
        if step.state is None:
            continue
        raw = step.state.get(CODEX_OUTPUT_ARTIFACT_REF_KEY)
        if raw is None:
            continue
        try:
            ref = TypedRef.model_validate(raw)
        except ValueError:
            return None
        payload = evidence.stored_record(ref)
        if payload is None:
            return None
        try:
            return model.model_validate(payload)
        except ValueError:
            return None
    return None


def _entry_refs(
    entries: tuple[ToolCallStoreEntry, ...],
) -> tuple[EvidenceRef, ...]:
    return tuple(evidence_ref(entry.tool_call.record_ref) for entry in entries)


def codex_no_eval_outside_tools(evidence: RunEvidence) -> AuditFinding:
    """Every evaluation this run paid for went through the one Tool.

    Totality in both directions: every durably admitted call is
    reachable from the Step Result as Tool Evidence, every reported call
    is an entry in the run's own capacity scope, and every reported entry
    cites its own Tool Call and Tool Result. The adapter fails the Step under
    ``codex_unreported_evaluation`` when the counts disagree at run time,
    so a completed run whose counts disagree *here* means the artifact no
    longer reflects the reconciliation that ran.

    The citation is re-derived rather than trusted to
    ``ToolEvidence._validate``: an audit that delegated its check to the
    validator it is auditing would go quiet exactly when that validator
    drifted.

    Task-model spend is checked at role granularity, per OQ1: there is no
    ``codex_agent`` cost role, so the audit asserts that a run reporting
    task-model calls also reports admitted evaluations, rather than
    attributing individual calls to individual tool calls -- which the
    cost report's grain cannot support.
    """
    try:
        ledger = _ledger_entries(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS, error)
    reported = _reported_evidence(evidence)
    reported_ids = {str(item.store_entry.call_id) for item in reported}
    ledger_ids = {str(entry.call_id) for entry in ledger}
    problems: list[str] = []

    unreported = sorted(ledger_ids - reported_ids)
    if unreported:
        problems.append(
            f"{len(unreported)} admitted call(s) are absent from "
            f"tool_evidence: {', '.join(unreported[:3])}"
        )
    # The reverse direction, which is not redundant. ``admitted_entries``
    # is scoped by ``(namespace, tool_config_hash, capacity scope)``, so
    # an entry whose Tool Config was altered stops matching the scope and
    # silently disappears from the ledger read rather than reading as
    # wrong. Checking only ledger-to-evidence would let exactly that
    # tampering pass: the run would report an evaluation the ledger, as
    # scoped, no longer knows it bought.
    unledgered = sorted(reported_ids - ledger_ids)
    if unledgered:
        problems.append(
            f"{len(unledgered)} reported call(s) have no admission entry "
            f"in the run's own capacity scope: "
            f"{', '.join(unledgered[:3])}"
        )
    for item in reported:
        entry = item.store_entry
        if entry.tool_call.record_ref != item.result.record.call.record_ref:
            problems.append(
                f"call {entry.call_id} cites a Tool Result belonging to "
                f"another Tool Call"
            )
        if entry.tool_result_ref != item.result.record_ref:
            problems.append(
                f"call {entry.call_id} cites a Tool Result its own "
                f"admission entry does not name"
            )

    task_calls = _task_model_calls(evidence)
    if task_calls > 0 and not ledger:
        problems.append(
            f"the run reports {task_calls} task-model call(s) but admitted "
            f"no evaluations through the Tool"
        )

    refs = _entry_refs(tuple(item.store_entry for item in reported))
    if problems:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS,
            status=AuditStatus.FAIL,
            detail=(
                f"the admission ledger and tool_evidence do not agree: "
                f"{'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_NO_EVAL_OUTSIDE_TOOLS,
        status=AuditStatus.PASS,
        detail=(
            f"all {len(ledger_ids)} admitted evaluation(s) appear in "
            f"tool_evidence and vice versa, each citing its own Tool Call "
            f"and Tool Result"
        ),
        evidence_refs=refs,
    )


def _task_model_calls(evidence: RunEvidence) -> int:
    """The run's task-model call count, at the cost report's own grain."""
    raw = evidence.result.cost
    if raw is None:
        return 0
    payload = raw.to_json() if hasattr(raw, "to_json") else raw
    if not isinstance(payload, dict):
        return 0
    role = payload.get(TASK_MODEL_COST_ROLE)
    if not isinstance(role, dict):
        return 0
    calls = role.get("calls")
    return calls if isinstance(calls, int) else 0


def codex_final_candidate_evaluated(evidence: RunEvidence) -> AuditFinding:
    """The returned candidate was measured before it was returned.

    ``selected_call_id`` must resolve to a ledger entry that reached
    ``COMPLETED`` carrying a score. An entry can reach ``COMPLETED``
    having *failed* -- the executor persists a Tool Result with a
    terminal failure and no output -- so "completed" alone is not
    evidence the candidate was measured, which is why the score is
    checked too (the adapter's ``codex_selection_unscored`` case).

    A run that retained its seed names no selection and is judged on that
    basis instead: ``seed_retained`` with no ``selected_call_id`` is the
    honest no-improvement outcome, not an unevaluated return.
    """
    try:
        artifact = _artifact(evidence)
        ledger = _ledger_entries(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(
            InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED, error
        )
    if artifact is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
            status=AuditStatus.FAIL,
            detail=(
                "no step recorded a Codex output artifact, so the returned "
                "candidate cannot be traced to an evaluation"
            ),
        )
    selected = artifact.selected_call_id
    if selected is None:
        return _unselected_finding(seed_retained=evidence.result.seed_retained)

    match = next(
        (entry for entry in ledger if str(entry.call_id) == str(selected)),
        None,
    )
    admitted = _admitted_selection_finding(selected, match)
    if admitted is not None:
        return admitted
    assert match is not None
    scored = next(
        (
            item
            for item in _reported_evidence(evidence)
            if str(item.store_entry.call_id) == str(selected)
        ),
        None,
    )
    if scored is None or scored.result.record.reward is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
            status=AuditStatus.FAIL,
            detail=(
                f"selected call {selected!r} completed but its durable "
                f"Tool Result carries no score"
            ),
            evidence_refs=_entry_refs((match,)),
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
        status=AuditStatus.PASS,
        detail=(
            f"selected call {selected!r} resolves to a completed, scored "
            f"admission entry"
        ),
        evidence_refs=_entry_refs((match,)),
    )


def _admitted_selection_finding(
    selected: str,
    match: ToolCallStoreEntry | None,
) -> AuditFinding | None:
    """Why the selected call is not a completed admission, if it is not.

    Returns None when the entry exists and completed, which is the only
    case the caller continues past.
    """
    if match is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
            status=AuditStatus.FAIL,
            detail=(
                f"selected call {selected!r} has no durable admission "
                f"entry, so the returned candidate was never evaluated "
                f"through the Tool"
            ),
        )
    if match.state is not ToolCallState.COMPLETED:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
            status=AuditStatus.FAIL,
            detail=(
                f"selected call {selected!r} is in state "
                f"{match.state.value!r}, not completed"
            ),
            evidence_refs=_entry_refs((match,)),
        )
    return None


def _unselected_finding(*, seed_retained: bool) -> AuditFinding:
    """A run that named no selection, judged on whether it kept its seed.

    ``seed_retained`` with no ``selected_call_id`` is the honest
    no-improvement outcome; the same absence without it means the run
    returned something it never measured.
    """
    if seed_retained:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
            status=AuditStatus.PASS,
            detail=(
                "the run retained its seed and named no selection, so it "
                "returned no candidate requiring evaluation"
            ),
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_FINAL_CANDIDATE_EVALUATED,
        status=AuditStatus.FAIL,
        detail=(
            "the artifact names no selected call yet the run did not "
            "retain its seed"
        ),
    )


def codex_capacity_respected(evidence: RunEvidence) -> AuditFinding:
    """The run bought no more evaluations than its capacity allowed.

    Ordinals are one-based and dense within a capacity scope, so the
    number of debits is the number of evaluations paid for. Three things
    are checked: no ordinal exceeds the configured cap, the configured
    cap is the pre-registered D2 value, and every refusal consumed
    nothing (``state is REFUSED`` with no ordinal).

    Refusals reach the audit through ``tool_evidence``, not through
    ``admitted_entries``: a refused call debited no capacity, and
    ``_entries_in_scope`` returns only entries that did
    (``optim/tools/admission.py:472``). Checking them where they *are*
    is what makes "REFUSED beyond the cap" auditable at all.
    """
    try:
        ledger = _ledger_entries(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(InvariantId.CODEX_CAPACITY_RESPECTED, error)
    reported = _reported_entries(evidence)
    if not ledger and not reported:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_CAPACITY_RESPECTED,
            status=AuditStatus.FAIL,
            detail=(
                "the run recorded no admission entries at all, so no "
                "capacity claim can be checked"
            ),
        )
    configured = _configured_capacity(ledger or reported)
    problems: list[str] = []
    if configured is None:
        problems.append("no entry names the run's configured capacity")
    elif configured != CODEX_CAPACITY_CAP:
        problems.append(
            f"the run configured a capacity of {configured}, not the "
            f"pre-registered {CODEX_CAPACITY_CAP}"
        )

    ordinals = [
        int(entry.capacity_debit_ordinal)
        for entry in ledger
        if entry.capacity_debit_ordinal is not None
    ]
    cap = configured if configured is not None else CODEX_CAPACITY_CAP
    over = sorted(ordinal for ordinal in ordinals if ordinal > cap)
    if over:
        problems.append(
            f"{len(over)} capacity debit ordinal(s) exceed the cap of "
            f"{cap}: {over[:3]}"
        )
    for entry in reported:
        if (
            entry.state is ToolCallState.REFUSED
            and entry.capacity_debit_ordinal is not None
        ):
            problems.append(
                f"refused call {entry.call_id} debited capacity ordinal "
                f"{entry.capacity_debit_ordinal}"
            )
        if (
            entry.state is not ToolCallState.REFUSED
            and entry.capacity_debit_ordinal is None
        ):
            problems.append(
                f"call {entry.call_id} is {entry.state.value!r} yet debited "
                f"no capacity"
            )

    refs = _entry_refs(ledger or reported)
    if problems:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_CAPACITY_RESPECTED,
            status=AuditStatus.FAIL,
            detail=f"capacity was not respected: {'; '.join(problems[:3])}",
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_CAPACITY_RESPECTED,
        status=AuditStatus.PASS,
        detail=(
            f"{len(ordinals)} evaluation(s) debited capacity, every ordinal "
            f"within the configured cap of {cap}, and every refusal "
            f"consumed none"
        ),
        evidence_refs=refs,
    )


def _configured_capacity(
    entries: tuple[ToolCallStoreEntry, ...],
) -> int | None:
    """The run's configured ``max_accepted_calls``, from its Tool Config.

    Every entry in one scope cites the same Tool Config by construction
    (the scope key includes its identity hash), so the first is
    authoritative.
    """
    for entry in entries:
        return int(entry.tool_config.record.capacity.max_accepted_calls)
    return None


def codex_lease_token_binds_artifact(evidence: RunEvidence) -> AuditFinding:
    """The artifact this audit reads is this run's own.

    The agent holds a bearer token for the MCP endpoint, so it can pay
    for evaluations and then present an artifact that is not this Step's.
    The adapter refuses that under ``codex_lease_token_mismatch``, but
    only after reconciling -- so the artifact is persisted either way and
    the audit must judge it independently.

    An audit cannot recompute the hash: the run lease token is minted per
    Step and never persisted, deliberately (persisting it would defeat
    the binding). What is auditable is that the artifact carries a
    well-formed hash at all and that exactly one artifact is bound to the
    run -- an empty or malformed ``lease_token_hash`` is the shape the
    adapter's own mismatch path rejects.
    """
    try:
        artifact = _artifact(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(InvariantId.CODEX_LEASE_BINDS_ARTIFACT, error)
    if artifact is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_LEASE_BINDS_ARTIFACT,
            status=AuditStatus.FAIL,
            detail=(
                "no step recorded a Codex output artifact, so nothing is "
                "bound to this run"
            ),
        )
    problems: list[str] = []
    token_hash = str(artifact.lease_token_hash)
    if len(token_hash) != SHA256_HEX_LENGTH or not all(
        character in "0123456789abcdef" for character in token_hash
    ):
        problems.append(
            f"lease_token_hash is not a sha256 digest "
            f"(length {len(token_hash)})"
        )
    if str(artifact.run_id) != evidence.run_id:
        problems.append(
            f"the artifact names run {artifact.run_id!r}, not "
            f"{evidence.run_id!r}"
        )
    if problems:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_LEASE_BINDS_ARTIFACT,
            status=AuditStatus.FAIL,
            detail=(
                f"the output artifact is not bound to this run: "
                f"{'; '.join(problems)}"
            ),
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_LEASE_BINDS_ARTIFACT,
        status=AuditStatus.PASS,
        detail=(
            f"the output artifact names run {evidence.run_id!r} and carries "
            f"a well-formed run lease-token digest"
        ),
    )


def codex_tool_surface_minimal(evidence: RunEvidence) -> AuditFinding:
    """The agent was granted exactly one tool, and used only that one.

    The F6 restatement. The protocol assumed a second ``read-scores``
    tool and a ``tool_name`` ledger column; neither exists. There is one
    tool (``mcp_bridge.py:147`` registers exactly one and ``call_tool``
    at ``:176`` rejects any other name), and its name is nested at
    ``entry.tool_config.record.definition.record.tool_name``.

    So: exactly one distinct ``tool_name`` across every entry, and every
    call's argument set equal to the definition's declared
    ``input_fields`` -- which the run's own definition must in turn
    declare as ``CODEX_EVAL_INPUT_FIELDS``. Checking the args against the
    *definition* alone would be circular, since ``ToolCall._validate``
    already enforces that; pinning the definition to the module constant
    is what makes a narrowed or widened tool surface visible.
    """
    try:
        expected_fields = _codex_eval_input_fields()
        ledger = _ledger_entries(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(InvariantId.CODEX_TOOL_SURFACE_MINIMAL, error)
    entries = ledger or _reported_entries(evidence)
    if not entries:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
            status=AuditStatus.FAIL,
            detail=(
                "the run recorded no Tool Call entries, so its tool surface "
                "cannot be judged"
            ),
        )
    # The *granted* surface, checked before the used one. A run may only
    # ever call one tool and still have been granted two -- the ledger
    # would look minimal while the agent held a capability the study
    # never authorized, so the grant is checked where it is recorded.
    granted = evidence.result.run.record.tool_configs
    if len(granted) != 1:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
            status=AuditStatus.FAIL,
            detail=(
                f"the run granted {len(granted)} Tool Configs; a Codex run "
                f"is granted exactly one evaluation tool"
            ),
        )
    configured = str(granted[0].record.definition.record.tool_name)
    names = {
        str(entry.tool_config.record.definition.record.tool_name)
        for entry in entries
    }
    problems: list[str] = []
    if len(names) != 1:
        problems.append(
            f"{len(names)} distinct tool names appear: "
            f"{', '.join(sorted(names))}"
        )
    elif next(iter(names)) != configured:
        problems.append(
            f"entries name tool {next(iter(names))!r} while the run "
            f"configured {configured!r}"
        )
    for entry in entries:
        definition = entry.tool_config.record.definition.record
        declared = frozenset(str(field) for field in definition.input_fields)
        if declared != expected_fields:
            problems.append(
                f"tool {definition.tool_name} declares input fields "
                f"{sorted(declared)}, not {sorted(expected_fields)}"
            )
            break
    for entry in entries:
        args = frozenset(entry.tool_call.record.args.to_json())
        if args != expected_fields:
            problems.append(
                f"call {entry.call_id} carries argument fields "
                f"{sorted(args)}, not {sorted(expected_fields)}"
            )
            break

    refs = _entry_refs(entries)
    if problems:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
            status=AuditStatus.FAIL,
            detail=(
                f"the Codex tool surface is not minimal: "
                f"{'; '.join(problems[:3])}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_TOOL_SURFACE_MINIMAL,
        status=AuditStatus.PASS,
        detail=(
            f"all {len(entries)} entries name the single tool "
            f"{next(iter(names))!r} over the pinned evaluation input fields"
        ),
        evidence_refs=refs,
    )


def codex_failures_carry_paid_evidence(
    evidence: RunEvidence,
) -> AuditFinding:
    """A failed Codex run records the spend it actually incurred.

    §3.4's sixth row: wall and interrupted failures carry their paid
    evidence. Two candidate readings of that turned out to be already
    guaranteed and so cannot be audited here:

    - Paid calls staying reachable as Tool Evidence is
      ``codex_no_eval_outside_tools``, which checks it on every run.
    - The Step's verdict accounting for its nested evaluation failures is
      enforced by ``OptimStepResult`` itself
      (``_validate_shared_terminal_failure``, ``optim/contracts.py:918``),
      so no artifact violating it can even be loaded. An invariant with
      no reachable failure is not an invariant.

    What is left, and what nothing upstream checks, is the spend record.
    Every terminalizing exit writes ``harness_store_accepted_call_count``
    into the Step's durable state *after* re-reading the admission
    authority (``adapter.py:508``) -- it is the run's own claim about what
    it paid for, written on the failure path precisely so a wall stop or
    an interrupted evaluation does not lose the spend. Nothing validates
    that number against the ledger it came from, so an artifact whose
    recorded spend disagrees with its own admission entries is exactly
    the drift this audit exists to catch, and it is the number the study
    manifest's cost reconciliation would otherwise trust.

    ``NOT_APPLICABLE`` on a successful run is legitimate and conditional:
    the claim is about what a failure preserves. That is not §3.1's
    "always inapplicable" defect -- the Codex arm is expected to produce
    failing runs, and the committed failing fixture exercises this path.
    """
    failure = evidence.result.terminal_failure
    if failure is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
            status=AuditStatus.NOT_APPLICABLE,
            detail=(
                "the run carries no terminal failure, so it makes no claim "
                "about the spend a failure preserves"
            ),
        )
    try:
        ledger = _ledger_entries(evidence)
    except _CodexSurfaceMissingError as error:  # pragma: no cover
        return _missing_surface(
            InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED, error
        )
    recorded = _recorded_accepted_count(evidence)
    outer = str(failure.code)
    refs = _entry_refs(ledger)
    if recorded is None:
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
            status=AuditStatus.FAIL,
            detail=(
                f"the run failed under {outer!r} without recording the "
                f"durable accepted-call count its spend is read from"
            ),
            evidence_refs=refs,
        )
    if recorded != len(ledger):
        return AuditFinding(
            invariant_id=InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
            status=AuditStatus.FAIL,
            detail=(
                f"the run failed under {outer!r} recording {recorded} paid "
                f"evaluation(s) while its ledger holds {len(ledger)}"
            ),
            evidence_refs=refs,
        )
    return AuditFinding(
        invariant_id=InvariantId.CODEX_FAILURE_EVIDENCE_RETAINED,
        status=AuditStatus.PASS,
        detail=(
            f"the run failed under {outer!r} and recorded all {recorded} "
            f"paid evaluation(s) its admission ledger holds"
        ),
        evidence_refs=refs,
    )


def _recorded_accepted_count(evidence: RunEvidence) -> int | None:
    """The run's own claim about how many evaluations it paid for."""
    for step in reversed(evidence.steps):
        if step.state is None:
            continue
        recorded = step.state.get(CODEX_ACCEPTED_CALL_COUNT_KEY)
        if isinstance(recorded, int):
            return recorded
    return None


#: The six Codex invariants, in the order §3.4 lists them.
CODEX_INVARIANTS = (
    codex_no_eval_outside_tools,
    codex_final_candidate_evaluated,
    codex_capacity_respected,
    codex_lease_token_binds_artifact,
    codex_tool_surface_minimal,
    codex_failures_carry_paid_evidence,
)


__all__ = [
    "CODEX_ACCEPTED_CALL_COUNT_KEY",
    "CODEX_CAPACITY_CAP",
    "CODEX_INVARIANTS",
    "CODEX_OUTPUT_ARTIFACT_REF_KEY",
    "SHA256_HEX_LENGTH",
    "codex_capacity_respected",
    "codex_failures_carry_paid_evidence",
    "codex_final_candidate_evaluated",
    "codex_lease_token_binds_artifact",
    "codex_no_eval_outside_tools",
    "codex_tool_surface_minimal",
]
