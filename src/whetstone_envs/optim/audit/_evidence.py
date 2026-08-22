"""Read one run's durable evidence through whetstone's public surface.

``RunEvidence`` is the single place the audit package knows how a run
directory is laid out and how a ``TypedRef`` becomes a record. Invariants
receive it already-resolved and stay pure functions over data, so an
invariant never opens a store, never re-executes, and never re-scores.

Every read path here is a public whetstone import. The package is walked by
``tests/optim/test_public_imports.py`` with an empty allowlist, so a private
module path or a ``cast("Any", x)._name`` reach-through fails that test
rather than shipping.

Verified read paths (whetstone-ai ``miprofix-ai`` @ ``716976f2``):

- ``OptimResult`` / ``OPTIM_RESULT_SCHEMA`` -- ``whetstone.optim.contracts``.
  ``result.run`` is an ``OptimRunRef`` wrapper, so the run record is
  ``result.run.record``; ``result.step_results`` are ``OptimStepResultRef``
  wrappers, so a step is ``ref.record``.
- ``IntentResolution`` / ``SearchEvidence`` / ``ToolEvidence`` are inline on
  the step result -- no store round trip.
- ``EvalEvidence`` -- deref ``resolution.eval_result_ref``. Dispatch on
  ``eval_result_ref.schema_name``: whetstone writes ``EVAL_EVIDENCE_SCHEMA``
  for a completed intent and ``EVAL_FAILURE_SCHEMA`` for a failed one, so
  branching on ``outcome`` alone would mis-parse a failure.
- GEPA terminal artifact -- the terminal step's ``history_ref`` resolves to
  a dict carrying ``GEPA_TERMINAL_ARTIFACT_KEY``; that value is a serialized
  ``TypedRef`` addressing a ``GepaRunResultArtifact``, whose
  ``detailed_result_ref`` and ``effect_transcript_ref`` reach the
  ``GepaDetailedResult`` and ``GepaEffectTranscript``.
- GEPA per-step state -- ``step.state_ref`` resolves to a state snapshot dict
  keyed by pool name; GEPA's checkpoint sits under ``GEPA_STATE_KEY`` and its
  skipped mutations under ``GEPA_SKIPPED_MUTATIONS_KEY``.
- MIPROv2 state -- the same snapshot dict under ``MIPROV2_STATE_KEY``
  validates as ``Miprov2State``, whose ``study_transcript`` is inline.
- The optimizer control -- ``result.run.record.optimizer_config`` is an
  ``IdentityRef``, so the control record itself lives in the store at its
  ``record_ref``. It is loaded raw here and each optimizer's audit validates
  it as its own control type; the ``record_hash`` is that control's identity
  hash, so an audit can check the persisted control is the one the run
  claims rather than trusting a state-delta echo of it.

``store.get`` returns a decoded Python object, so ``model_validate`` is the
correct call. The ``model_validate_json(json.dumps(...))`` idiom seen
elsewhere is specific to ``EvalTraces``, whose trace payloads validate under
pydantic's JSON input mode only; nothing this reader touches needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dr_store.sync import open_sqlite
from whetstone.core.identity import TypedRef
from whetstone.eval.schema import EvalEvidence
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.optim.contracts import (
    OptimResult,
    OptimStepResult,
)
from whetstone.optim.gepa.contracts import GepaEffectTranscript
from whetstone.optim.gepa.engine import GepaDetailedResult
from whetstone.optim.gepa.harness_adapter import (
    GEPA_SKIPPED_MUTATIONS_KEY,
    GEPA_TERMINAL_ARTIFACT_KEY,
)
from whetstone.optim.gepa.result_artifact import GepaRunResultArtifact
from whetstone.optim.gepa.step_engine import (
    GEPA_STATE_KEY,
    GepaStepCheckpoint,
)
from whetstone.optim.miprov2.adapter import MIPROV2_STATE_KEY
from whetstone.optim.miprov2.runtime import Miprov2State

from whetstone_envs.optim.audit.schema import EvidenceRef

if TYPE_CHECKING:
    from collections.abc import Iterator

    from whetstone.optim.contracts import (
        IntentResolution,
        SearchEvidence,
        ToolEvidence,
    )

#: The run artifacts an audit reads. Both must exist; an audit never
#: re-executes a run to reconstruct a missing one.
RESULT_FILENAME = "result.json"
RUNTIME_STORE_FILENAME = "runtime.sqlite"

#: Adapter keys, as they appear on ``OptimRun.adapter_key``, mapped to the
#: optimizer name the registry dispatches on. Pinned by a golden test: these
#: are whetstone's persisted spellings, not names we may rename freely.
COPRO_OPTIMIZER = "copro"
GEPA_OPTIMIZER = "gepa"
MIPROV2_OPTIMIZER = "miprov2"
CODEX_OPTIMIZER = "codex"


class AuditEvidenceError(RuntimeError):
    """The run directory could not be read as auditable evidence.

    This is distinct from an invariant failing: a failing invariant is a
    finding about the run, while this means the audit could not be performed
    at all.
    """


def evidence_ref(ref: TypedRef) -> EvidenceRef:
    """Project a whetstone ``TypedRef`` onto the audit report's wire shape."""
    return EvidenceRef(
        schema_name=ref.schema_name,
        content_hash=ref.content_hash,
    )


@dataclass(frozen=True, slots=True)
class GepaTerminalEvidence:
    """The GEPA terminal artifact and the two records it addresses."""

    artifact_ref: TypedRef
    artifact: GepaRunResultArtifact
    detailed_result: GepaDetailedResult
    effect_transcript: GepaEffectTranscript


@dataclass(frozen=True, slots=True)
class StepEvidence:
    """One step result plus the durable state it points at.

    ``state`` is the raw state-snapshot mapping. The optimizer-specific
    accessors below decode the slice each optimizer owns, so an invariant
    never indexes a pool key by hand.
    """

    index: int
    step: OptimStepResult
    state: dict[str, object] | None
    history: dict[str, object] | None

    @property
    def resolved_intents(self) -> tuple[IntentResolution, ...]:
        return self.step.resolved_intents

    @property
    def search_evidence(self) -> tuple[SearchEvidence, ...]:
        return self.step.search_evidence

    @property
    def tool_evidence(self) -> tuple[ToolEvidence, ...]:
        return self.step.tool_evidence

    def gepa_checkpoint(self) -> GepaStepCheckpoint | None:
        """This step's GEPA checkpoint, or None when it carries no state."""
        payload = self._state_value(GEPA_STATE_KEY)
        if payload is None:
            return None
        return GepaStepCheckpoint.model_validate(payload)

    def gepa_skipped_mutations(self) -> tuple[object, ...]:
        """This step's skipped-mutation records.

        The key is written unconditionally on every step, so an empty tuple
        here means "this step skipped nothing", while a missing key means the
        step carried no GEPA state at all.
        """
        payload = self._state_value(GEPA_SKIPPED_MUTATIONS_KEY)
        if payload is None:
            return ()
        if not isinstance(payload, list):
            raise AuditEvidenceError(
                f"step {self.index} skipped mutations are not a list"
            )
        return tuple(payload)

    def miprov2_state(self) -> Miprov2State | None:
        """This step's MIPROv2 durable state, or None when absent."""
        payload = self._state_value(MIPROV2_STATE_KEY)
        if payload is None:
            return None
        return Miprov2State.model_validate(payload)

    def _state_value(self, key: str) -> object | None:
        if self.state is None:
            return None
        return self.state.get(key)


@dataclass(frozen=True, slots=True)
class RunEvidence:
    """Everything an invariant may read about one completed run.

    Constructed once per audit by :func:`load_run_evidence`; invariants
    receive it and return a finding without touching the filesystem.
    """

    run_dir: Path
    result: OptimResult
    steps: tuple[StepEvidence, ...]
    eval_evidence_by_ref: dict[TypedRef, EvalEvidence]
    gepa_terminal: GepaTerminalEvidence | None
    #: The optimizer control record as persisted, still raw. Each
    #: optimizer's audit validates it as its own control type. None when the
    #: run's ``optimizer_config`` ref resolves to nothing -- itself an
    #: auditable defect, so loading returns rather than raising.
    control_record: dict[str, object] | None
    #: Records addressed by an optimizer's own ``state_delta`` refs,
    #: resolved at load time while the store is still open. An invariant
    #: is a pure function over already-resolved evidence, so it cannot
    #: reopen the store to chase a ref itself.
    state_records: dict[TypedRef, object]

    @property
    def run_id(self) -> str:
        return str(self.result.run_id)

    @property
    def optimizer(self) -> str:
        """The optimizer that produced this run, read from the result.

        The audit takes only a directory, so the optimizer is evidence, not
        a caller-supplied claim that could disagree with the artifact.
        """
        return str(self.result.run.record.adapter_key)

    @property
    def control_ref(self) -> TypedRef:
        """Where the run says its optimizer control is stored."""
        return self.result.run.record.optimizer_config.record_ref

    @property
    def control_identity_hash(self) -> str:
        """The control identity hash the run binds itself to."""
        return str(self.result.run.record.optimizer_config.record_hash)

    def eval_evidence(self, ref: TypedRef) -> EvalEvidence | None:
        """The ``EvalEvidence`` at ``ref``, or None if it is a failure record.

        A failed intent addresses an ``EVAL_FAILURE_SCHEMA`` record, which is
        not eval evidence; returning None keeps the caller from mis-parsing
        one as the other.
        """
        return self.eval_evidence_by_ref.get(ref)

    def stored_record(self, ref: TypedRef) -> object | None:
        """The decoded record at ``ref``, or None when it resolved to none.

        Only refs an optimizer's state delta names are resolved, and a
        dangling one returns None rather than raising: a ref that
        addresses nothing is itself a finding, and refusing to load would
        crash the audit instead of letting an invariant report it.
        """
        return self.state_records.get(ref)

    def all_eval_evidence(self) -> Iterator[tuple[TypedRef, EvalEvidence]]:
        """Every resolvable eval-evidence record this run produced."""
        yield from self.eval_evidence_by_ref.items()


def _require_dict(value: object, what: str) -> dict[str, object]:
    """Narrow a decoded store payload to a string-keyed JSON object.

    ``store.get`` is typed as returning ``object``, and an ``isinstance``
    check against bare ``dict`` leaves the key and value types unknown, so
    the mapping is rebuilt here with its JSON key type made explicit. A
    non-string key means the payload was never a JSON object.

    Evidence is read-only once loaded, so returning a shallow copy costs
    nothing and keeps the narrowing honest rather than asserted.
    """
    if not isinstance(value, dict):
        raise AuditEvidenceError(f"{what} is not a JSON object")
    narrowed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AuditEvidenceError(f"{what} is not a JSON object")
        narrowed[key] = item
    return narrowed


def _typed_ref(value: object, what: str) -> TypedRef:
    try:
        return TypedRef.model_validate(value)
    except ValueError as error:
        raise AuditEvidenceError(f"{what} is not a TypedRef") from error


def _load_gepa_terminal(
    store: object,
    steps: tuple[StepEvidence, ...],
) -> GepaTerminalEvidence | None:
    """Resolve the GEPA terminal artifact chain, when this run has one.

    Absence is not an error here -- ``GEPA_TERMINAL_ARTIFACT_PRESENT`` is the
    invariant that decides whether a missing artifact is a defect, and it
    cannot report that if loading refuses to return.
    """
    for entry in reversed(steps):
        if entry.history is None:
            continue
        raw = entry.history.get(GEPA_TERMINAL_ARTIFACT_KEY)
        if raw is None:
            continue
        artifact_ref = _typed_ref(
            raw, f"step {entry.index} terminal artifact ref"
        )
        artifact = GepaRunResultArtifact.model_validate(
            _get(store, artifact_ref, "GEPA terminal artifact")
        )
        detailed = GepaDetailedResult.model_validate(
            _get(
                store,
                artifact.detailed_result_ref,
                "GEPA detailed result",
            )
        )
        transcript = GepaEffectTranscript.model_validate(
            _get(
                store,
                artifact.effect_transcript_ref,
                "GEPA effect transcript",
            )
        )
        return GepaTerminalEvidence(
            artifact_ref=artifact_ref,
            artifact=artifact,
            detailed_result=detailed,
            effect_transcript=transcript,
        )
    return None


def _get_optional(store: object, ref: TypedRef) -> object | None:
    """Dereference ``ref``, returning None when nothing is stored at it.

    Used for refs whose absence is itself auditable evidence.
    """
    getter = getattr(store, "get", None)
    if getter is None:  # pragma: no cover - guarded by the caller's type
        raise AuditEvidenceError("store does not support get")
    try:
        return getter(ref.reference)
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        return None


def _get(store: object, ref: TypedRef, what: str) -> object:
    """Dereference one ``TypedRef`` against the run's own object store."""
    getter = getattr(store, "get", None)
    if getter is None:  # pragma: no cover - guarded by the caller's type
        raise AuditEvidenceError("store does not support get")
    try:
        return getter(ref.reference)
    except Exception as error:
        raise AuditEvidenceError(
            f"{what} at {ref.schema_name}:{ref.content_hash[:12]} "
            f"is not in the run's store"
        ) from error


def _collect_eval_evidence(
    store: object,
    steps: tuple[StepEvidence, ...],
) -> dict[TypedRef, EvalEvidence]:
    """Deref every eval-evidence ref the run's evidence cites.

    Three paths cite one: an intent resolution, a search-evidence entry,
    and -- for the one ``TOOL_USING`` optimizer, Codex -- a Tool Result's
    ``evaluation_evidence_refs``. A Codex run resolves no intent at all,
    so omitting the tool path would leave its every paid evaluation
    unreachable from this map and make an honest run look like one that
    reported nothing.

    Two kinds of ref are deliberately *not* errors here:

    - A failure record (``EVAL_FAILURE_SCHEMA``) is skipped: a run that
      recorded an evaluation failure is still auditable, and the invariants
      that care read the intent's own outcome.
    - A ref that resolves to nothing is skipped rather than raising. A
      dangling ref is exactly the defect ``REPORTED_NUMBERS_RESOLVE``
      exists to report, so refusing to load would make the violation
      unreportable -- the audit would crash instead of returning FAIL.

    Either way the ref is simply absent from the returned mapping, and the
    invariant reports what that absence means.
    """
    collected: dict[TypedRef, EvalEvidence] = {}
    for entry in steps:
        refs = [
            resolution.eval_result_ref
            for resolution in entry.resolved_intents
            if resolution.eval_result_ref is not None
        ]
        refs.extend(
            evidence.eval_result_ref
            for evidence in entry.search_evidence
            if evidence.eval_result_ref is not None
        )
        for tool in entry.tool_evidence:
            refs.extend(tool.result.record.evaluation_evidence_refs)
        for ref in refs:
            if ref in collected or ref.schema_name != EVAL_EVIDENCE_SCHEMA:
                continue
            raw = _get_optional(store, ref)
            if raw is None:
                continue
            try:
                collected[ref] = EvalEvidence.model_validate(raw)
            except ValueError:
                # A record that is present but not parseable as eval
                # evidence is likewise a finding, not a load failure.
                continue
    return collected


#: State-snapshot keys whose value is a serialized ``TypedRef`` to a
#: record an invariant needs. Resolved at load time so an invariant never
#: reopens the store. Codex writes its output artifact's ref under the
#: first of these (``optim/codex/adapter.py:331``).
STATE_RECORD_REF_KEYS = ("codex_output_artifact_ref",)


def _collect_state_records(
    store: object,
    steps: tuple[StepEvidence, ...],
) -> dict[TypedRef, object]:
    """Deref every state-delta ref an invariant may need to read.

    A malformed or dangling ref is skipped rather than raising, for the
    same reason ``_collect_eval_evidence`` skips one: the absence is the
    evidence, and an invariant reports it.
    """
    collected: dict[TypedRef, object] = {}
    for entry in steps:
        if entry.state is None:
            continue
        for key in STATE_RECORD_REF_KEYS:
            raw = entry.state.get(key)
            if raw is None:
                continue
            try:
                ref = TypedRef.model_validate(raw)
            except ValueError:
                continue
            if ref in collected:
                continue
            found = _get_optional(store, ref)
            if found is not None:
                collected[ref] = found
    return collected


def load_run_evidence(run_dir: Path) -> RunEvidence:
    """Read ``run_dir`` into the evidence every invariant reads.

    ``run_dir`` needs only ``result.json`` and ``runtime.sqlite``; nothing
    here reaches the network, re-runs the optimizer, or recomputes a score.
    """
    result_path = run_dir / RESULT_FILENAME
    store_path = run_dir / RUNTIME_STORE_FILENAME
    if not result_path.is_file():
        raise AuditEvidenceError(f"{result_path} is missing")
    if not store_path.is_file():
        raise AuditEvidenceError(f"{store_path} is missing")
    try:
        result = OptimResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    except ValueError as error:
        raise AuditEvidenceError(
            f"{result_path} is not a valid OptimResult"
        ) from error

    with open_sqlite(str(store_path)) as store:
        steps: list[StepEvidence] = []
        for index, step_ref in enumerate(result.step_results):
            step = step_ref.record
            state = None
            if step.state_ref is not None:
                state = _require_dict(
                    _get(store, step.state_ref, f"step {index} state"),
                    f"step {index} state snapshot",
                )
            history = None
            if step.history_ref is not None:
                history = _require_dict(
                    _get(store, step.history_ref, f"step {index} history"),
                    f"step {index} history",
                )
            steps.append(
                StepEvidence(
                    index=index,
                    step=step,
                    state=state,
                    history=history,
                )
            )
        frozen_steps = tuple(steps)
        eval_evidence = _collect_eval_evidence(store, frozen_steps)
        gepa_terminal = _load_gepa_terminal(store, frozen_steps)
        raw_control = _get_optional(
            store, result.run.record.optimizer_config.record_ref
        )
        control_record = (
            _require_dict(raw_control, "optimizer control record")
            if isinstance(raw_control, dict)
            else None
        )
        state_records = _collect_state_records(store, frozen_steps)

    return RunEvidence(
        run_dir=run_dir,
        result=result,
        steps=frozen_steps,
        eval_evidence_by_ref=eval_evidence,
        gepa_terminal=gepa_terminal,
        control_record=control_record,
        state_records=state_records,
    )


__all__ = [
    "CODEX_OPTIMIZER",
    "COPRO_OPTIMIZER",
    "GEPA_OPTIMIZER",
    "MIPROV2_OPTIMIZER",
    "RESULT_FILENAME",
    "RUNTIME_STORE_FILENAME",
    "STATE_RECORD_REF_KEYS",
    "AuditEvidenceError",
    "GepaTerminalEvidence",
    "RunEvidence",
    "StepEvidence",
    "evidence_ref",
    "load_run_evidence",
]
