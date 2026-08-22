"""The study manifest: ``study.json``, its schema, writer, and reader.

The manifest is the study's single accounting surface. Every number the
report prints is a field here or a deterministic function of evidence this
file names, and it names that evidence as an :class:`EvidencePointer` --
a ``(schema_name, content_hash)`` pair the run's object store resolves. The
report generator therefore never recomputes a score from a loose file; it
reads the manifest and dereferences pointers.

**Schema in the repo, instances outside it.** The schema is a
persisted-format contract, so it lives here with its literals pinned by a
golden test and never derived from field names. A study *instance* is a
durable work document, not a versioned deliverable, so
:func:`write_study_manifest` refuses to write inside a repository -- the
same rule run artifacts follow -- and refuses to overwrite an existing
manifest unless the caller says so explicitly.

**Wire keys are owned here.** :class:`ManifestKey` names every top-level
key of the persisted document. Pydantic field names happen to agree with it
today, and the golden test asserts that agreement, but the enum is the
owner: a field rename that silently changed stored identity fails the
golden test rather than the report.
"""

from __future__ import annotations

import json
from enum import UNIQUE, StrEnum, verify
from typing import TYPE_CHECKING, Protocol

from dr_store import CanonicalJsonFile, ObjectReference
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone_envs.reporting.publication import validate_output_root

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

# --------------------------------------------------------------------------
# Persisted literals. Owned here, pinned by tests/optim/study/test_manifest.py.
# --------------------------------------------------------------------------

#: The manifest's schema name. Bump the version on any persisted-format
#: change; the golden test is what makes that bump deliberate.
STUDY_MANIFEST_SCHEMA_NAME = "whetstone_envs.step10_study"
#: v2 makes three deliberate persisted-format changes, each because v1 could
#: not state a fact the study has to state.
#:
#: * ``RunSpendRecord`` gains the per-role honesty split -- cached,
#:   priced/unpriced, and missing-token-breakdown counts. An absent ``usd``
#:   is the report's most consequential number and a reader cannot act on it
#:   without knowing how much of the run a total would have covered.
#: * ``RunRecord`` gains ``cost_ref``, the pointer each run's ``cost.json``
#:   is stored under, so ``manifest check`` proves the printed spend was not
#:   typed in by hand.
#: * ``held_out_claims`` records each held-out evaluation at the moment it is
#:   *issued*, which a held-out row cannot: a row carries a Holm-corrected
#:   p-value that does not exist until every arm is measured, so v1 left the
#:   window between paying for an evaluation and recording it unguarded.
STUDY_MANIFEST_SCHEMA_VERSION = 2
STUDY_MANIFEST_SCHEMA = (
    f"{STUDY_MANIFEST_SCHEMA_NAME}/v{STUDY_MANIFEST_SCHEMA_VERSION}"
)

#: The manifest's filename inside a study directory.
STUDY_MANIFEST_NAME = "study.json"

#: A study manifest retains per-split task-hash vectors and one entry per
#: run, not per row, so it stays far smaller than an evaluation report.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024

#: The selection rule the study pre-registered. It is persisted verbatim so
#: a reader can tell that this study selected by argmax on official rather
#: than by some later rule.
SELECTION_RULE_ARGMAX_OFFICIAL = "argmax-official"

#: The multiplicity correction applied across the four real optimizers.
CORRECTION_HOLM_BONFERRONI = "holm-bonferroni"

#: The number of hypotheses Holm corrects over: the four real optimizers.
#: Nulls are controls, not hypotheses, and stay uncorrected.
CORRECTION_FAMILY_SIZE = 4

#: The completeness backstop below which an arm is reported incomplete and
#: its CI is not claimed.
COMPLETENESS_BACKSTOP = 0.90


@verify(UNIQUE)
class ManifestKey(StrEnum):
    """Every top-level wire key of the persisted manifest.

    Explicit values, not ``auto()``: these are stored identity, and the
    golden test pins them against this enum rather than against whatever a
    field happens to be called.
    """

    SCHEMA = "schema"
    STUDY_ID = "study_id"
    CREATED_AT = "created_at"
    PROTOCOL_DOC_PATH = "protocol_doc_path"
    PROTOCOL_DOC_SHA256 = "protocol_doc_sha256"
    ASSIGNMENT_DOC_SHA256 = "assignment_doc_sha256"
    POPULATION = "population"
    SPLITS = "splits"
    MODELS = "models"
    DESIGN = "design"
    GEPA_SIZING = "gepa_sizing"
    FANOUT_CHECK = "fanout_check"
    ARMS = "arms"
    SELECTION = "selection"
    HELD_OUT_CLAIMS = "held_out_claims"
    HELD_OUT = "held_out"
    BALANCE = "balance"
    LEAKAGE_CHECK = "leakage_check"
    C18 = "c18"


@verify(UNIQUE)
class SplitName(StrEnum):
    """The three evaluation splits the study keeps disjoint."""

    INTERNAL = "internal"
    OFFICIAL = "official"
    HELD_OUT = "held_out"


@verify(UNIQUE)
class StageId(StrEnum):
    """The study's three stages, named where a stage is recorded."""

    STAGE0 = "stage0"
    STAGE1 = "stage1"
    STAGE2 = "stage2"


#: The stage names a CLI accepts, in run order.
STAGE_IDS: tuple[str, ...] = tuple(member.value for member in StageId)


class _StrictModel(BaseModel):
    """Frozen, strict, extra-forbidding, no NaN or infinity."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


# --------------------------------------------------------------------------
# Evidence pointers
# --------------------------------------------------------------------------

_SHA256_HEX_LENGTH = 64


class EvidencePointer(_StrictModel):
    """One ``(schema_name, content_hash)`` pair resolvable in a store.

    This is the manifest's only way to cite evidence. It deliberately
    mirrors ``dr_store.ObjectReference`` rather than embedding it, because
    the manifest is a persisted document with its own strictness rules and
    must not inherit a dependency's constructor validation as its wire
    contract.
    """

    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate_pointer(self) -> EvidencePointer:
        if not self.schema_name.strip():
            raise ValueError("evidence pointers require a nonblank schema")
        if len(self.content_hash) != _SHA256_HEX_LENGTH:
            raise ValueError(
                "evidence pointer content hashes are full SHA-256 hex"
            )
        if any(char not in "0123456789abcdef" for char in self.content_hash):
            raise ValueError(
                "evidence pointer content hashes are lowercase hex"
            )
        return self

    def as_object_reference(self) -> ObjectReference:
        """The store reference this pointer names."""
        return ObjectReference(
            schema=self.schema_name,
            content_hash=self.content_hash,
        )


class EvidenceStore(Protocol):
    """The read surface :func:`check_manifest_pointers` needs.

    ``dr_store.sync.BlockingObjectStore`` satisfies this structurally, so
    the checker takes an open store rather than a path and stays testable
    against an in-memory double.
    """

    def get(self, reference: ObjectReference) -> JsonValue: ...


# --------------------------------------------------------------------------
# Population, splits, models
# --------------------------------------------------------------------------


class PopulationRecord(_StrictModel):
    """The task pool the study drew every split from."""

    family: StrictStr
    generator_version: StrictStr
    n_per_stratum: StrictInt
    pool_seed_start: StrictInt
    pool_manifest_content_hash: StrictStr
    stratum_counts: dict[StrictStr, StrictInt]

    @model_validator(mode="after")
    def _validate_population(self) -> PopulationRecord:
        if not self.family.strip():
            raise ValueError("a population names its task family")
        if not self.generator_version.strip():
            raise ValueError("a population names its generator version")
        if self.n_per_stratum < 1:
            raise ValueError("n_per_stratum must be at least 1")
        if not self.pool_manifest_content_hash.strip():
            raise ValueError("a population cites its pool manifest hash")
        if not self.stratum_counts:
            raise ValueError("a population records its stratum counts")
        if any(count < 0 for count in self.stratum_counts.values()):
            raise ValueError("stratum counts must be non-negative")
        return self


class SplitRecord(_StrictModel):
    """One evaluation split, content-addressed by its task hashes.

    The hash vector is the split's identity: L5's disjointness check is a
    set intersection over exactly these, so they are stored rather than
    recomputed.
    """

    size: StrictInt
    task_hashes: tuple[StrictStr, ...]
    eval_config_hash: StrictStr

    @model_validator(mode="after")
    def _validate_split(self) -> SplitRecord:
        if self.size != len(self.task_hashes):
            raise ValueError("a split's size is its task-hash count")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("a split's task hashes are distinct")
        if any(not value.strip() for value in self.task_hashes):
            raise ValueError("task hashes must be nonblank")
        if not self.eval_config_hash.strip():
            raise ValueError("a split names its eval config hash")
        return self


class SplitsRecord(_StrictModel):
    """The three splits, proven pairwise disjoint at construction."""

    internal: SplitRecord
    official: SplitRecord
    held_out: SplitRecord

    @model_validator(mode="after")
    def _validate_disjoint(self) -> SplitsRecord:
        pairs = (
            (
                SplitName.INTERNAL,
                self.internal,
                SplitName.OFFICIAL,
                self.official,
            ),
            (
                SplitName.INTERNAL,
                self.internal,
                SplitName.HELD_OUT,
                self.held_out,
            ),
            (
                SplitName.OFFICIAL,
                self.official,
                SplitName.HELD_OUT,
                self.held_out,
            ),
        )
        for left_name, left, right_name, right in pairs:
            overlap = set(left.task_hashes) & set(right.task_hashes)
            if overlap:
                raise ValueError(
                    f"{left_name.value} and {right_name.value} splits share "
                    f"{len(overlap)} task hashes"
                )
        return self


class ModelsRecord(_StrictModel):
    """Which models ran, and what the study could and could not control.

    ``temperature``, ``seed_control``, and ``codex_agent_model`` are honest
    strings rather than numbers because the study does not control them:
    the provider ignores temperature on the nano models, the seed control is
    advertised rather than guaranteed, and the Codex agent's own model runs
    off the study's key entirely.
    """

    task_model: StrictStr
    proposer_model: StrictStr
    temperature: StrictStr
    provider: StrictStr
    seed_control: StrictStr
    codex_agent_model: StrictStr

    @model_validator(mode="after")
    def _validate_models(self) -> ModelsRecord:
        values = (
            self.task_model,
            self.proposer_model,
            self.temperature,
            self.provider,
            self.seed_control,
            self.codex_agent_model,
        )
        if any(not value.strip() for value in values):
            raise ValueError("every model field is a nonblank statement")
        return self


# --------------------------------------------------------------------------
# Design, sizing, fan-out
# --------------------------------------------------------------------------


class DesignRecord(_StrictModel):
    """The pre-registered design, including its power arithmetic.

    ``k_cal`` and ``k_repeat`` are distinct on purpose: ``k_cal`` is a
    Stage-0 measurement input and ``k_repeat`` is the design's per-task
    repeat count. Conflating them would let a calibration choice silently
    restate the design.
    """

    k_cal: StrictInt
    k_repeat: StrictInt
    k_run_by_arm: dict[StrictStr, StrictInt]
    ci_level: StrictFloat
    resamples: StrictInt
    bootstrap_seed: StrictInt
    correction: StrictStr
    m: StrictInt
    mde_formula: StrictStr
    mde_measured: StrictFloat
    tau_sq: StrictFloat
    sigma_sq: StrictFloat
    completeness_rule: StrictStr
    completeness_backstop: StrictFloat

    @model_validator(mode="after")
    def _validate_design(self) -> DesignRecord:
        if self.k_cal < 1 or self.k_repeat < 1:
            raise ValueError("K_CAL and K_REPEAT are at least 1")
        if not self.k_run_by_arm:
            raise ValueError("a design records K_RUN for at least one arm")
        if any(value < 1 for value in self.k_run_by_arm.values()):
            raise ValueError("every arm's K_RUN is at least 1")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("the CI level is a proportion in (0, 1)")
        if self.resamples < 1:
            raise ValueError("bootstrap resamples must be positive")
        if not self.mde_formula.strip():
            raise ValueError("the design states its MDE formula")
        if self.mde_measured < 0.0:
            raise ValueError("a measured MDE is non-negative")
        if self.tau_sq < 0.0 or self.sigma_sq < 0.0:
            raise ValueError("variance components are non-negative")
        if not 0.0 < self.completeness_backstop <= 1.0:
            raise ValueError("the completeness backstop is in (0, 1]")
        if not self.completeness_rule.strip():
            raise ValueError("the design states its completeness rule")
        return self


class GepaSizingRecord(_StrictModel):
    """The F9 measurement, recorded before any Stage-1 spend.

    ``max_metric_calls_pinned`` is ``None`` when the measurement did not
    force a pin, which is a different fact from pinning the default.
    """

    steps_per_run: StrictInt
    wall_seconds: StrictFloat
    sqlite_bytes: StrictInt
    max_metric_calls_pinned: StrictInt | None

    @model_validator(mode="after")
    def _validate_sizing(self) -> GepaSizingRecord:
        if self.steps_per_run < 0:
            raise ValueError("step counts are non-negative")
        if self.wall_seconds < 0.0:
            raise ValueError("wall time is non-negative")
        if self.sqlite_bytes < 0:
            raise ValueError("store sizes are non-negative")
        if (
            self.max_metric_calls_pinned is not None
            and self.max_metric_calls_pinned < 1
        ):
            raise ValueError("a pinned metric-call ceiling is at least 1")
        return self


class FanoutCheckRecord(_StrictModel):
    """The F16 precondition: minibatch intents did not fan out to the
    full valset."""

    passed: StrictBool
    minibatch_intents: StrictInt
    full_valset_intents: StrictInt

    @model_validator(mode="after")
    def _validate_fanout(self) -> FanoutCheckRecord:
        if self.minibatch_intents < 0 or self.full_valset_intents < 0:
            raise ValueError("intent counts are non-negative")
        return self


# --------------------------------------------------------------------------
# Arms and runs
# --------------------------------------------------------------------------


class RunSpendRecord(_StrictModel):
    """One provider role's spend on one run.

    ``usd`` is ``None`` when any contributing call was unpriced. The rule is
    applied per role, so the Codex arm reports real USD for its task-model
    evaluations even though its agent's own calls are off-key entirely.

    **The honesty split is part of the record, not a derived nicety.** An
    absent ``usd`` is the report's most consequential number, and a reader
    cannot act on it without knowing *how much* of the run it would have
    covered. ``priced_calls``/``unpriced_calls`` say that; ``cached_calls``
    distinguishes a cheap run from a small one, since a prompt-cache hit
    carries the original call's tokens and is deliberately not billed again;
    and ``rows_missing_token_breakdown`` marks billable calls the provider
    priced without splitting tokens by direction, which is the one case
    where the token totals understate real usage while ``usd`` stays exact.
    These mirror ``whetstone.optim.cost.RoleCost`` field for field, so the
    projection from ``OptimResult.cost`` is a rename-free copy.
    """

    role: StrictStr
    calls: StrictInt
    cached_calls: StrictInt
    input_tokens: StrictInt
    output_tokens: StrictInt
    priced_calls: StrictInt
    unpriced_calls: StrictInt
    rows_missing_token_breakdown: StrictInt
    usd: StrictFloat | None

    @model_validator(mode="after")
    def _validate_spend(self) -> RunSpendRecord:
        if not self.role.strip():
            raise ValueError("spend names its provider role")
        counts = (
            self.calls,
            self.cached_calls,
            self.input_tokens,
            self.output_tokens,
            self.priced_calls,
            self.unpriced_calls,
            self.rows_missing_token_breakdown,
        )
        if any(value < 0 for value in counts):
            raise ValueError("spend counters are non-negative")
        if self.priced_calls + self.unpriced_calls != self.calls:
            raise ValueError(
                "priced and unpriced calls exhaust the billable calls"
            )
        if self.rows_missing_token_breakdown > self.calls:
            raise ValueError(
                "calls missing a token breakdown cannot exceed all calls"
            )
        if self.usd is not None:
            if self.usd < 0.0:
                raise ValueError("reported spend is non-negative")
            if self.unpriced_calls:
                # An absent total is "not knowable", never "zero"; a total
                # reported beside unpriced calls would understate spend
                # while looking authoritative.
                raise ValueError(
                    "a role with unpriced calls reports no total spend"
                )
        return self


class RunRecord(_StrictModel):
    """One optimizer run: where it lives, what it cost, whether it is valid.

    ``audit_passed`` gates the arm's efficacy claim mechanically: a failed
    audit makes the number descriptive, never a claim, and it is recorded
    beside the audit's own pointer so a reader can check the verdict rather
    than trust it.

    ``spend`` and ``cost_ref`` are the same facts twice on purpose: the
    inline records are what the report prints, and the pointer is the run's
    own ``cost.json`` in the store, so ``manifest check`` proves the printed
    numbers were not typed in by hand. ``result.json`` remains the
    authority; ``cost.json`` is a pinned-key projection of it.
    """

    run_id: StrictStr
    seed: StrictInt | None
    artifact_dir: StrictStr
    result_ref: EvidencePointer
    audit_ref: EvidencePointer
    cost_ref: EvidencePointer
    audit_passed: StrictBool
    spend: tuple[RunSpendRecord, ...]

    @model_validator(mode="after")
    def _validate_run(self) -> RunRecord:
        if not self.run_id.strip():
            raise ValueError("a run has a nonblank id")
        if not self.artifact_dir.strip():
            raise ValueError("a run names its artifact directory")
        roles = [entry.role for entry in self.spend]
        if len(set(roles)) != len(roles):
            raise ValueError("a run reports each provider role once")
        return self


class ArmRecord(_StrictModel):
    """One arm: an optimizer, its configuration, and its ``K_RUN`` runs.

    ``seed_note`` carries how the arm honoured its requested seed. COPRO has
    no control seed field, so recording the difference beats a manifest that
    implies every arm was seeded the same way.
    """

    arm_id: StrictStr
    optimizer: StrictStr
    demo_mode: StrictStr | None
    control_identity_hash: StrictStr
    seed_note: StrictStr
    runs: tuple[RunRecord, ...]

    @model_validator(mode="after")
    def _validate_arm(self) -> ArmRecord:
        if not self.arm_id.strip() or not self.optimizer.strip():
            raise ValueError("an arm names itself and its optimizer")
        if not self.control_identity_hash.strip():
            raise ValueError("an arm cites its control identity hash")
        if not self.seed_note.strip():
            raise ValueError("an arm records how its seed was honoured")
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("an arm's run ids are distinct")
        return self


# --------------------------------------------------------------------------
# Selection, held-out, balance, leakage, c18
# --------------------------------------------------------------------------


class SelectionRecord(_StrictModel):
    """The arg-max on official, persisted before any held-out call.

    This record is what makes an L3 violation impossible rather than merely
    detectable: ``report_arm`` reads it back before issuing the held-out
    evaluation and raises if it is absent.
    """

    arm_id: StrictStr
    selected_run_id: StrictStr
    official_score: StrictFloat
    rule: StrictStr

    @model_validator(mode="after")
    def _validate_selection(self) -> SelectionRecord:
        if not self.arm_id.strip() or not self.selected_run_id.strip():
            raise ValueError("a selection names its arm and chosen run")
        if self.rule != SELECTION_RULE_ARGMAX_OFFICIAL:
            raise ValueError(
                "this study pre-registered "
                f"{SELECTION_RULE_ARGMAX_OFFICIAL!r} as its selection rule"
            )
        return self


class HeldOutClaimRecord(_StrictModel):
    """One candidate's consumption of its single held-out evaluation.

    This exists because L3's guard and L3's *result* become durable at
    different moments. A held-out row carries the candidate's interval and
    its Holm-corrected p-value, and those are a whole-study computation --
    Holm runs across all four real arms at once -- so a row cannot be
    written until every arm has been measured. The claim can be, and must
    be: it is written the moment the evaluation is issued, so a stage that
    crashes after paying for a held-out evaluation resumes knowing the
    candidate already spent its one shot.

    The measurement's own numbers land on the claim once the evaluation
    returns, which is why they are optional: a claim written and never
    completed is a *crashed* evaluation, and that is a different fact from
    an evaluation that never happened. Recording them here rather than only
    on the later row lets the analysis pass reconstruct that row without
    re-issuing the evaluation.
    """

    candidate_name: StrictStr
    eval_config_hash: StrictStr | None = None
    repeats: StrictInt | None = None
    mean: StrictFloat | None = None
    completeness: StrictFloat | None = None

    @model_validator(mode="after")
    def _validate_claim(self) -> HeldOutClaimRecord:
        if not self.candidate_name.strip():
            raise ValueError("a held-out claim names its candidate")
        if self.eval_config_hash is not None and (
            not self.eval_config_hash.strip()
        ):
            raise ValueError("a completed claim names its Eval Config")
        if self.repeats is not None and self.repeats < 1:
            raise ValueError("a held-out evaluation runs at least once")
        if self.completeness is not None and not (
            0.0 <= self.completeness <= 1.0
        ):
            raise ValueError("completeness is a proportion in [0, 1]")
        completed = (
            self.eval_config_hash,
            self.repeats,
            self.mean,
            self.completeness,
        )
        if any(value is not None for value in completed) and any(
            value is None for value in completed
        ):
            # Half a measurement is not a measurement; either the claim is
            # outstanding or it carries the whole result.
            raise ValueError(
                "a held-out claim is either outstanding or fully measured"
            )
        return self

    @property
    def completed(self) -> bool:
        """Whether the claimed evaluation returned."""
        return self.mean is not None


class HeldOutRecord(_StrictModel):
    """One reported candidate's single held-out evaluation.

    ``p_holm`` is ``None`` for the nulls, which are controls rather than
    hypotheses and are therefore uncorrected.
    """

    candidate_name: StrictStr
    eval_evidence_ref: EvidencePointer
    per_task_scores_ref: EvidencePointer
    mean: StrictFloat
    ci_low: StrictFloat
    ci_high: StrictFloat
    delta_vs_naive: StrictFloat
    p_bootstrap: StrictFloat
    p_holm: StrictFloat | None
    completeness: StrictFloat

    @model_validator(mode="after")
    def _validate_held_out(self) -> HeldOutRecord:
        if not self.candidate_name.strip():
            raise ValueError("a held-out row names its candidate")
        if self.ci_low > self.ci_high:
            raise ValueError("a confidence interval is ordered")
        for label, value in (
            ("p_bootstrap", self.p_bootstrap),
            ("p_holm", self.p_holm),
            ("completeness", self.completeness),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} is a proportion in [0, 1]")
        return self


class BalanceRecord(_StrictModel):
    """The key's balance at each spend gate, in USD."""

    before_stage0_usd: StrictFloat
    before_stage1_usd: StrictFloat
    before_stage2_usd: StrictFloat
    after_usd: StrictFloat

    @model_validator(mode="after")
    def _validate_balance(self) -> BalanceRecord:
        values = (
            self.before_stage0_usd,
            self.before_stage1_usd,
            self.before_stage2_usd,
            self.after_usd,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("a balance is non-negative")
        return self


class LeakageCheckEntry(_StrictModel):
    """One of L1-L5, with its verdict and one sentence of detail."""

    check_id: StrictStr
    passed: StrictBool
    detail: StrictStr

    @model_validator(mode="after")
    def _validate_entry(self) -> LeakageCheckEntry:
        if not self.check_id.strip() or not self.detail.strip():
            raise ValueError("a leakage check names itself and its finding")
        return self


class LeakageCheckRecord(_StrictModel):
    """L6: the mechanical run of L1-L5 over every run artifact."""

    passed: StrictBool
    checks: tuple[LeakageCheckEntry, ...]

    @model_validator(mode="after")
    def _validate_leakage(self) -> LeakageCheckRecord:
        if not self.checks:
            raise ValueError("a leakage report runs at least one check")
        ids = [entry.check_id for entry in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("each leakage check is reported once")
        if self.passed != all(entry.passed for entry in self.checks):
            raise ValueError(
                "the leakage verdict is the conjunction of its checks"
            )
        return self


class AdapterSwapRecord(_StrictModel):
    """The C3 assertion: which modules differed between the two families.

    A non-empty ``differing_modules`` outside the two permitted entries is
    the study's generality finding, so the list is persisted rather than
    reduced to a boolean.
    """

    passed: StrictBool
    differing_modules: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _validate_swap(self) -> AdapterSwapRecord:
        if any(not name.strip() for name in self.differing_modules):
            raise ValueError("differing module names are nonblank")
        if len(set(self.differing_modules)) != len(self.differing_modules):
            raise ValueError("differing modules are listed once each")
        return self


class C18Record(_StrictModel):
    """The second family's runs and its adapter-swap result."""

    runs: tuple[RunRecord, ...]
    adapter_swap: AdapterSwapRecord

    @model_validator(mode="after")
    def _validate_c18(self) -> C18Record:
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("c18 run ids are distinct")
        return self


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


class StudyManifest(_StrictModel):
    """``study.json``: everything the report is allowed to print.

    ``schema`` is stored so a reader identifies the document without
    guessing from its shape, and it is validated against this module's
    pinned constant rather than merely being nonblank -- a manifest written
    by a different schema version is a different document.

    Sections that a stage has not reached yet are ``None`` rather than
    empty: a study that has not run Stage 0 has no design measurement, and
    an empty ``DesignRecord`` would claim otherwise.
    """

    schema_: StrictStr = Field(
        default=STUDY_MANIFEST_SCHEMA,
        alias=ManifestKey.SCHEMA.value,
    )
    study_id: StrictStr
    created_at: StrictStr
    protocol_doc_path: StrictStr
    protocol_doc_sha256: StrictStr
    assignment_doc_sha256: StrictStr
    population: PopulationRecord
    splits: SplitsRecord
    models: ModelsRecord
    design: DesignRecord | None = None
    gepa_sizing: GepaSizingRecord | None = None
    fanout_check: FanoutCheckRecord | None = None
    arms: tuple[ArmRecord, ...] = ()
    selection: tuple[SelectionRecord, ...] = ()
    held_out_claims: tuple[HeldOutClaimRecord, ...] = ()
    held_out: tuple[HeldOutRecord, ...] = ()
    balance: BalanceRecord | None = None
    leakage_check: LeakageCheckRecord | None = None
    c18: C18Record | None = None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> StudyManifest:
        if self.schema_ != STUDY_MANIFEST_SCHEMA:
            raise ValueError(
                f"expected schema {STUDY_MANIFEST_SCHEMA!r}, "
                f"got {self.schema_!r}"
            )
        identifiers = (
            self.study_id,
            self.created_at,
            self.protocol_doc_path,
            self.protocol_doc_sha256,
            self.assignment_doc_sha256,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("a manifest's provenance fields are nonblank")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm ids are distinct")
        selected = [entry.arm_id for entry in self.selection]
        if len(set(selected)) != len(selected):
            # L2 as a structural rule: one selection per arm, never two.
            raise ValueError("each arm is selected at most once")
        unknown = set(selected) - set(arm_ids)
        if unknown:
            raise ValueError(
                f"selection names unknown arms: {sorted(unknown)}"
            )
        for entry in self.selection:
            arm = next(arm for arm in self.arms if arm.arm_id == entry.arm_id)
            if entry.selected_run_id not in {run.run_id for run in arm.runs}:
                raise ValueError(
                    f"arm {entry.arm_id!r} selected run "
                    f"{entry.selected_run_id!r}, which it did not run"
                )
        claimed = [entry.candidate_name for entry in self.held_out_claims]
        if len(set(claimed)) != len(claimed):
            # L3 as a structural rule, at the moment the evaluation is
            # issued rather than at the moment its statistics are known.
            raise ValueError("each candidate claims held-out at most once")
        names = [entry.candidate_name for entry in self.held_out]
        if len(set(names)) != len(names):
            raise ValueError("each candidate is evaluated on held-out once")
        completed = {
            entry.candidate_name
            for entry in self.held_out_claims
            if entry.completed
        }
        unbacked = set(names) - completed
        if unbacked:
            # A reported held-out number whose evaluation was never claimed
            # is a number from outside ``report_arm``, which is exactly the
            # leak L3 exists to catch.
            raise ValueError(
                "held-out rows without a completed claimed evaluation: "
                f"{sorted(unbacked)}"
            )
        return self

    def evidence_pointers(self) -> tuple[EvidencePointer, ...]:
        """Every evidence pointer this manifest cites, in document order.

        ``manifest check`` resolves exactly this tuple, so a pointer added
        to the schema without being reachable here would go unchecked. The
        walk is over the model's own fields rather than a hand-kept list.
        """
        return tuple(_walk_pointers(self))


def _walk_pointers(model: BaseModel) -> Iterator[EvidencePointer]:
    """Yield every :class:`EvidencePointer` nested anywhere in ``model``."""
    for name in type(model).model_fields:
        yield from _walk_value(getattr(model, name))


def _walk_value(value: object) -> Iterator[EvidencePointer]:
    if isinstance(value, EvidencePointer):
        yield value
    elif isinstance(value, BaseModel):
        yield from _walk_pointers(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_value(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_value(item)


# --------------------------------------------------------------------------
# Writer and reader
# --------------------------------------------------------------------------


class ManifestExistsError(FileExistsError):
    """A manifest is already present and the caller did not ask to replace
    it.

    Silently overwriting would destroy the record of a paid study, so the
    default is refusal and replacement is an explicit argument.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(f"a study manifest already exists at {path}")
        self.path = path


def _manifest_document(directory: Path) -> CanonicalJsonFile:
    return CanonicalJsonFile(
        directory,
        STUDY_MANIFEST_NAME,
        max_bytes=MAX_MANIFEST_BYTES,
    )


def study_manifest_path(study_dir: Path) -> Path:
    """Where a study directory keeps its manifest."""
    return study_dir / STUDY_MANIFEST_NAME


def write_study_manifest(
    study_dir: Path,
    manifest: StudyManifest,
    *,
    replace: bool = False,
) -> Path:
    """Validate ``manifest``, then write it into ``study_dir``.

    Validation is a full round-trip through the persisted JSON, not a trust
    of the in-memory object: a manifest is only written if the exact bytes
    that will land re-validate. The directory must be outside every
    detected repository, because a study instance is a durable work
    document rather than a versioned deliverable.
    """
    resolved = validate_output_root(study_dir)
    validated = StudyManifest.model_validate_json(
        manifest.model_dump_json(by_alias=True)
    )
    path = study_manifest_path(resolved)
    if path.exists() and not replace:
        raise ManifestExistsError(path)
    resolved.mkdir(parents=True, exist_ok=True)
    document = _manifest_document(resolved)
    document.publish(validated.model_dump(mode="json", by_alias=True))
    return document.path


def read_study_manifest(study_dir_or_file: Path) -> StudyManifest:
    """Load and validate the manifest at ``study_dir_or_file``."""
    path = study_dir_or_file.resolve()
    directory, filename = (
        (path.parent, path.name)
        if path.is_file()
        else (path, STUDY_MANIFEST_NAME)
    )
    raw = CanonicalJsonFile(
        directory, filename, max_bytes=MAX_MANIFEST_BYTES
    ).read()
    return StudyManifest.model_validate_json(json.dumps(raw))


# --------------------------------------------------------------------------
# Pointer checking
# --------------------------------------------------------------------------


class PointerCheck(_StrictModel):
    """Whether one cited pointer resolved, and why not when it did not."""

    pointer: EvidencePointer
    resolved: StrictBool
    detail: StrictStr

    @model_validator(mode="after")
    def _validate_check(self) -> PointerCheck:
        if not self.detail.strip():
            raise ValueError("a pointer check states its finding")
        return self


class PointerCheckReport(_StrictModel):
    """The verdict of resolving every pointer a manifest cites."""

    passed: StrictBool
    checks: tuple[PointerCheck, ...]

    @model_validator(mode="after")
    def _validate_report(self) -> PointerCheckReport:
        if self.passed != all(check.resolved for check in self.checks):
            raise ValueError(
                "the pointer verdict is the conjunction of its checks"
            )
        return self

    def unresolved(self) -> tuple[PointerCheck, ...]:
        """Only the pointers that did not resolve."""
        return tuple(check for check in self.checks if not check.resolved)


_RESOLVED_DETAIL = "resolved in the named store"


def check_manifest_pointers(
    manifest: StudyManifest,
    store: EvidenceStore,
) -> PointerCheckReport:
    """Resolve every pointer ``manifest`` cites against ``store``.

    A pointer resolves when the store returns a record for its exact
    ``(schema_name, content_hash)`` pair. The store verifies the content
    hash on read, so a record whose bytes drifted fails here rather than
    reaching the report as a number nobody can reproduce.

    Distinct pointers are resolved once each; the report keeps them in the
    manifest's document order so a reader can find the cited field.
    """
    checks: list[PointerCheck] = []
    seen: dict[tuple[str, str], PointerCheck] = {}
    for pointer in manifest.evidence_pointers():
        key = (pointer.schema_name, pointer.content_hash)
        cached = seen.get(key)
        if cached is None:
            cached = _check_pointer(pointer, store)
            seen[key] = cached
        checks.append(cached)
    return PointerCheckReport(
        passed=all(check.resolved for check in checks),
        checks=tuple(checks),
    )


def _check_pointer(
    pointer: EvidencePointer,
    store: EvidenceStore,
) -> PointerCheck:
    try:
        reference = pointer.as_object_reference()
    except (ValueError, TypeError) as error:
        # A pointer the store's own reference type refuses is unresolvable
        # for a reason worth reporting, not an internal failure.
        return PointerCheck(pointer=pointer, resolved=False, detail=str(error))
    try:
        store.get(reference)
    except Exception as error:  # noqa: BLE001
        # Every backend reports a missing or corrupt record with its own
        # exception type, and a checker that let one through would report a
        # crash where the answer is "this pointer does not resolve".
        return PointerCheck(
            pointer=pointer,
            resolved=False,
            detail=f"{type(error).__name__}: {error}",
        )
    return PointerCheck(
        pointer=pointer, resolved=True, detail=_RESOLVED_DETAIL
    )


def format_pointer_report(report: PointerCheckReport) -> Iterable[str]:
    """One line per pointer, for the CLI to print."""
    for check in report.checks:
        mark = "ok" if check.resolved else "MISSING"
        yield (
            f"{mark} {check.pointer.schema_name} "
            f"{check.pointer.content_hash} :: {check.detail}"
        )


__all__ = [
    "COMPLETENESS_BACKSTOP",
    "CORRECTION_FAMILY_SIZE",
    "CORRECTION_HOLM_BONFERRONI",
    "MAX_MANIFEST_BYTES",
    "SELECTION_RULE_ARGMAX_OFFICIAL",
    "STAGE_IDS",
    "STUDY_MANIFEST_NAME",
    "STUDY_MANIFEST_SCHEMA",
    "STUDY_MANIFEST_SCHEMA_NAME",
    "STUDY_MANIFEST_SCHEMA_VERSION",
    "AdapterSwapRecord",
    "ArmRecord",
    "BalanceRecord",
    "C18Record",
    "DesignRecord",
    "EvidencePointer",
    "EvidenceStore",
    "FanoutCheckRecord",
    "GepaSizingRecord",
    "HeldOutClaimRecord",
    "HeldOutRecord",
    "LeakageCheckEntry",
    "LeakageCheckRecord",
    "ManifestExistsError",
    "ManifestKey",
    "ModelsRecord",
    "PointerCheck",
    "PointerCheckReport",
    "PopulationRecord",
    "RunRecord",
    "RunSpendRecord",
    "SelectionRecord",
    "SplitName",
    "SplitRecord",
    "SplitsRecord",
    "StageId",
    "StudyManifest",
    "check_manifest_pointers",
    "format_pointer_report",
    "read_study_manifest",
    "study_manifest_path",
    "write_study_manifest",
]
