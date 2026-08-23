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

from dr_store import (
    CanonicalJsonFile,
    ObjectReference,
    compute_content_hash,
)
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
    from collections.abc import Iterable, Iterator, Mapping
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
#: v3 adds ``pre_registration``: the design fields that were fixed before
#: any spend, plus the hash over them. v2 could record a design and then let
#: a later write silently restate it, which would make every downstream
#: "pre-registered" claim unfalsifiable.
#: v4 adds each arm's train/val partition, on ``ArmRecord`` and inside the
#: pre-registration's hashed payload. v3 pinned every other field that
#: decides what an arm's number means but left the partition unhashed, so a
#: MIPROv2 or GEPA arm could be rerun at a different train/val split under
#: an unchanged design hash -- the one post-hoc adjustment the block was
#: supposed to forbid and did not.
#: v5 adds ``call_count_gate``: Stage 1's call-count verdict, recorded where
#: Stage 2 can read it. v4 evaluated that gate inside the Stage-1 process and
#: kept the verdict nowhere, so a Stage 2 invoked straight after Stage 0 --
#: or after a Stage 1 whose gate failed -- ran the full five-run design
#: without the pilot ever having cleared the fan-out check the pilot exists
#: to perform. v5 also carries ``anchor_completeness`` on each held-out row,
#: so a downgraded paired delta can be read as the anchor's loss rather than
#: the arm's.
#: v6 adds ``stages``: one record per stage that ran, naming the transport
#: it ran on and the spend it produced. A stage run on the fake transport
#: and a stage run against a provider are different evidence for the same
#: claim, and v5 could not tell them apart -- a study could calibrate its
#: anchors for free and then report paid optimizer runs against them
#: without anything in the manifest saying so. The transport is a property
#: of the invocation rather than of the design, so it is recorded here and
#: deliberately kept out of the pre-registration hash.
#: v7 pushes the transport down onto each run and adds ``amendments``.
#:
#: * ``RunRecord`` gains ``transport``. v6 recorded a transport per *stage*,
#:   which names what the latest invocation of that stage bound -- not what
#:   the runs beneath it were measured on. A resumed stage keeps the runs it
#:   already paid for, so a stage row and its runs can disagree, and the
#:   cross-transport refusal had nothing but the stage row to check.
#: * ``amendments`` records evidence a re-calibration invalidated and
#:   dropped. ``stage0 --replace-design`` onto a different transport
#:   invalidates the arm stages twice over -- the design changed and the
#:   evidence came from elsewhere -- and v6 left both in place, so a Stage 2
#:   could reuse fake runs against freshly bought anchors. Dropping them
#:   silently would leave a manifest indistinguishable from one whose arm
#:   stages simply never ran, so what was dropped is recorded.
#: v8 adds ``models.provider_calls``: the task model's *effective* provider
#: call config, one record per transport a stage has bound. v7 recorded
#: which model a study meant to run and which transport it ran on, but not
#: what the transport actually bound -- the resolved route and the request
#: controls -- so neither the spend model nor the claim that two stages ran
#: "the same experiment" was auditable from the manifest. Recorded rather
#: than hashed, like the transport itself: it is a property of the
#: invocation, so two studies of one design still pre-register identically.
#: v9 makes the reporting pass durable and completes the arm record. It adds
#: ``report_spend`` and ``official_scores`` -- each reporting evaluation's
#: spend and each run's official score, written as they are bought rather
#: than folded from memory at the end of the pass, so a crash mid-pass
#: neither strands the spend of an evaluation already paid for nor lets a
#: resume charge for it a second time. It also adds ``minibatch``/
#: ``minibatch_size`` to ``ArmRecord``: v8 hashed ``minibatch_by_arm`` into
#: the pre-registration but gave the arm record nowhere to carry it, so a
#: manifest-driven MIPROv2 study silently ran unbatched under a design hash
#: that said otherwise.
STUDY_MANIFEST_SCHEMA_VERSION = 9
STUDY_MANIFEST_SCHEMA = (
    f"{STUDY_MANIFEST_SCHEMA_NAME}/v{STUDY_MANIFEST_SCHEMA_VERSION}"
)

#: The manifest's filename inside a study directory.
STUDY_MANIFEST_NAME = "study.json"

#: The evidence store a study directory keeps beside its manifest. Owned
#: here because it is part of a study directory's layout, and every module
#: that resolves a pointer needs it without importing the binder.
STUDY_STORE_NAME = "runtime.sqlite"

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
    PRE_REGISTRATION = "pre_registration"
    AMENDMENTS = "amendments"
    DESIGN = "design"
    STAGES = "stages"
    REPORT_SPEND = "report_spend"
    OFFICIAL_SCORES = "official_scores"
    GEPA_SIZING = "gepa_sizing"
    FANOUT_CHECK = "fanout_check"
    CALL_COUNT_GATE = "call_count_gate"
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


@verify(UNIQUE)
class TransportName(StrEnum):
    """The transports a stage may run on.

    Persisted on every stage record, so these are stored identity rather
    than a local vocabulary: ``fake`` names the offline transport that
    answers from the experiment's own gold, and ``openrouter`` names the
    billed provider route. The distinction is the whole point of recording
    it -- a number measured on ``fake`` is plumbing evidence and a number
    measured on ``openrouter`` is a study result, and nothing downstream
    can tell them apart without this field.
    """

    FAKE = "fake"
    OPENROUTER = "openrouter"


#: The transport names the CLI accepts, in the order they are offered.
TRANSPORT_NAMES: tuple[str, ...] = tuple(
    member.value for member in TransportName
)


#: The operator's opt-in to discarding run directories a stage cannot
#: claim, spelled once.
#:
#: A run directory is named deterministically from its arm and seed, so a
#: cross-transport ``stage0 --replace-design`` -- which drops the stale runs
#: from the manifest but leaves their directories on disk -- leaves behind
#: directories the replacement stage would otherwise silently reuse. The
#: stage refuses instead, and names this flag as the recovery, so the
#: refusal message and the CLI declaration cannot drift apart into advice
#: for a flag that does not exist.
#:
#: It lives here rather than beside the runner because the CLI declares it
#: and the runner quotes it, and the CLI deliberately does not import the
#: optimizer stack at module scope.
DISCARD_STALE_RUNS_FLAG = "--discard-stale-runs"


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


#: What a recorded provider control says when the study set none.
#:
#: A persisted-format literal: "the provider's own default" is a real
#: state with real consequences, and spelling it out is what stops it
#: from reading as an omission or, worse, as a zero.
PROVIDER_CONTROL_UNSET = "provider default"

#: What the recorded ``seed`` says, because no static value is the truth.
#:
#: The seed is the one control the study never leaves to the provider and
#: never binds to a constant. Every eval call carries a seed derived from
#: the evaluation's own identity -- whetstone's
#: ``provider_call_config_with_parameters`` sets it unconditionally from
#: ``derive_rng_seed(task_hash, seed_index)``, and refuses any definition
#: that cannot transport ``RequestControl.SEED`` rather than running
#: unseeded. Recording the statically bound control here would print
#: :data:`PROVIDER_CONTROL_UNSET` and tell a reader the provider chose the
#: seed, which is the opposite of what the eval contract guarantees.
#:
#: A persisted-format literal, like the constant above it.
PROVIDER_SEED_DERIVED_PER_CALL = "derived per call (eval contract)"


class ProviderCallRecord(_StrictModel):
    """The task model's *effective* provider call config, as bound.

    The manifest's ``models`` block names which model a study meant to run.
    This block names what the transport actually bound: the route it
    resolved, and the request controls the study did or did not set. The
    two are different facts, and only the second explains what a call
    cost and what "the same experiment" means -- a route the study never
    named, or a reasoning control it never asked for, changes both the
    bill and the treatment while leaving ``task_model`` unchanged.

    Recorded, not hashed. The config is a property of the transport the
    invocation bound, like the transport itself, so two studies of one
    design pre-register identically and this block tells them apart.

    Every field is a string, and an unset control is the literal
    :data:`PROVIDER_CONTROL_UNSET` rather than an omission or a zero.
    "Left to the provider's default" is a real and consequential state --
    it is why the toy Stage 0 billed thousands of reasoning tokens per
    call -- so it is stated rather than left to be inferred from a missing
    key.
    """

    transport: StrictStr
    provider: StrictStr
    protocol: StrictStr
    model_route: StrictStr
    temperature: StrictStr
    top_p: StrictStr
    token_limit: StrictStr
    #: Whatever reasoning control the bound config carries, verbatim.
    #:
    #: Recorded because it is the single largest term in this study's
    #: per-call bill, and **not** settable from here: whether the design
    #: pins a task-model reasoning effort is an open decision, and a
    #: manifest field that looked like a knob would answer it by accident.
    reasoning: StrictStr
    seed: StrictStr
    extensions: StrictStr

    @model_validator(mode="after")
    def _validate_provider_call(self) -> ProviderCallRecord:
        values = (
            self.transport,
            self.provider,
            self.protocol,
            self.model_route,
            self.temperature,
            self.top_p,
            self.token_limit,
            self.reasoning,
            self.seed,
            self.extensions,
        )
        if any(not value.strip() for value in values):
            raise ValueError(
                "every provider call field is a nonblank statement; an "
                f"unset control is {PROVIDER_CONTROL_UNSET!r}"
            )
        return self


class ModelsRecord(_StrictModel):
    """Which models ran, and what the study could and could not control.

    ``temperature`` and ``seed_control`` are honest strings rather than
    numbers because the study does not control them: the provider ignores
    temperature on the nano models, and the seed control is advertised
    rather than guaranteed.

    ``codex_agent_model`` is different, and is a string for a different
    reason. It **is** pre-registered -- the Codex arm refuses to run an
    agent that disagrees with it -- but the agent's own calls run off the
    study's key entirely, so what the study controls is *which* agent ran,
    never what it cost.
    """

    task_model: StrictStr
    proposer_model: StrictStr
    temperature: StrictStr
    provider: StrictStr
    seed_control: StrictStr
    codex_agent_model: StrictStr
    #: The task model's effective provider call config, one record per
    #: transport this study has bound.
    #:
    #: Empty until a stage runs. Keyed by transport in the record itself
    #: rather than by a dict key, so the block stays a tuple like every
    #: other repeated record here and a reader sees the transport beside
    #: the config it produced. A stage re-run on a transport already
    #: recorded replaces its entry rather than appending a second.
    provider_calls: tuple[ProviderCallRecord, ...] = ()

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
        transports = [entry.transport for entry in self.provider_calls]
        if len(set(transports)) != len(transports):
            # One effective config per transport. Two would leave the
            # study unable to say which one its numbers came from, which
            # is the whole reason this block exists.
            raise ValueError(
                "each transport records its provider call config once"
            )
        return self


# --------------------------------------------------------------------------
# Design, sizing, fan-out
# --------------------------------------------------------------------------


#: The provenance a pre-registration carries. ``original`` is what Stage 0
#: writes; ``amended`` is the only way a second Stage 0 may replace it, and
#: it is recorded so an amended design can never read as the first one.
PROVENANCE_ORIGINAL = "original"
PROVENANCE_AMENDED = "amended"

#: Every provenance value the manifest accepts.
PROVENANCE_VALUES: tuple[str, ...] = (PROVENANCE_ORIGINAL, PROVENANCE_AMENDED)


class PreRegistrationRecord(_StrictModel):
    """The design fields fixed before any spend, and the hash over them.

    This block exists because "pre-registered" is a claim about *when* a
    value was chosen, and a document that can be rewritten records no such
    thing. Stage 0 writes it once; :func:`write_study_manifest` then refuses
    any later write that does not carry it back byte for byte, so every
    number the report calls pre-registered is one no later stage could have
    chosen after seeing a result.

    The fields here are exactly the ones whose post-hoc adjustment would
    change what the study is allowed to claim: the repeat counts and run
    matrix that set the power, the interval settings and bootstrap seed that
    set the intervals, the multiplicity family, and the completeness
    backstop that decides which arms may claim at all. ``design_hash`` is
    the content hash over the rest of this block, so a reader checks the
    pinning arithmetically rather than trusting the writer.

    ``provenance`` distinguishes the first pre-registration from a deliberate
    amendment. An amendment is a different study design and says so; there
    is no path that quietly replaces one with the other.
    """

    k_repeat: StrictInt
    k_run_by_arm: dict[StrictStr, StrictInt]
    #: Each arm's pre-registered train/val partition of the internal split,
    #: as ``{arm_id: [train_size, val_size]}``, with ``None`` for an arm
    #: whose optimizer has no train/val concept.
    #:
    #: Hashed with the rest of the design because it *is* design: MIPROv2
    #: and GEPA measure search efficacy against a declared partition, and a
    #: partition chosen after a result is the same post-hoc adjustment the
    #: rest of this block exists to forbid.
    split_by_arm: dict[StrictStr, tuple[StrictInt, StrictInt] | None]
    #: Each arm's pre-registered MIPROv2 minibatch size, as
    #: ``{arm_id: size}``, with ``None`` for an arm that does not
    #: minibatch.
    #:
    #: Hashed with the rest of the design for ``split_by_arm``'s reason:
    #: an arm that evaluated every trial on the whole valset and an arm
    #: that evaluated on a sampled batch of it bought different evidence
    #: for the same claim, so a batch size chosen after a result is the
    #: post-hoc adjustment this block exists to forbid.
    minibatch_by_arm: dict[StrictStr, StrictInt | None]
    ci_level: StrictFloat
    resamples: StrictInt
    bootstrap_seed: StrictInt
    correction: StrictStr
    m: StrictInt
    completeness_backstop: StrictFloat
    design_hash: StrictStr
    provenance: StrictStr = PROVENANCE_ORIGINAL
    amended_from: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_pre_registration(self) -> PreRegistrationRecord:
        if self.k_repeat < 1:
            raise ValueError("K_REPEAT is at least 1")
        if not self.k_run_by_arm:
            raise ValueError("a pre-registration records at least one arm")
        if any(value < 1 for value in self.k_run_by_arm.values()):
            raise ValueError("every arm's K_RUN is at least 1")
        _require_valid_split_by_arm(
            split_by_arm=self.split_by_arm, k_run_by_arm=self.k_run_by_arm
        )
        _require_valid_minibatch_by_arm(
            minibatch_by_arm=self.minibatch_by_arm,
            k_run_by_arm=self.k_run_by_arm,
        )
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("the CI level is a proportion in (0, 1)")
        if self.resamples < 1:
            raise ValueError("bootstrap resamples must be positive")
        if not self.correction.strip():
            raise ValueError("a pre-registration names its correction")
        if self.m < 1:
            raise ValueError("the correction family holds at least one test")
        if not 0.0 < self.completeness_backstop <= 1.0:
            raise ValueError("the completeness backstop is in (0, 1]")
        if self.provenance not in PROVENANCE_VALUES:
            raise ValueError(
                f"provenance is one of {PROVENANCE_VALUES}; "
                f"got {self.provenance!r}"
            )
        if (self.provenance == PROVENANCE_AMENDED) != (
            self.amended_from is not None
        ):
            # An amendment that does not name what it replaced erases the
            # design it amended, which is the same as not recording one.
            raise ValueError(
                "an amended pre-registration names the design hash it "
                "replaced, and an original one names none"
            )
        expected = pre_registration_design_hash(
            k_repeat=self.k_repeat,
            k_run_by_arm=self.k_run_by_arm,
            split_by_arm=self.split_by_arm,
            minibatch_by_arm=self.minibatch_by_arm,
            ci_level=self.ci_level,
            resamples=self.resamples,
            bootstrap_seed=self.bootstrap_seed,
            correction=self.correction,
            m=self.m,
            completeness_backstop=self.completeness_backstop,
        )
        if self.design_hash != expected:
            raise ValueError(
                "the pre-registration's design hash does not cover its own "
                f"fields: expected {expected}, got {self.design_hash}"
            )
        return self

    def pinned_fields(self) -> dict[str, JsonValue]:
        """The hashed fields, as the hash sees them."""
        return _pre_registration_payload(
            k_repeat=self.k_repeat,
            k_run_by_arm=self.k_run_by_arm,
            split_by_arm=self.split_by_arm,
            minibatch_by_arm=self.minibatch_by_arm,
            ci_level=self.ci_level,
            resamples=self.resamples,
            bootstrap_seed=self.bootstrap_seed,
            correction=self.correction,
            m=self.m,
            completeness_backstop=self.completeness_backstop,
        )


def _require_valid_split_by_arm(
    *,
    split_by_arm: Mapping[str, tuple[int, int] | None],
    k_run_by_arm: Mapping[str, int],
) -> None:
    """The per-arm split names every arm, at positive sizes.

    Split out of the record's validator so that validator stays readable;
    the rule is the record's own, not a general one about splits.
    """
    if set(split_by_arm) != set(k_run_by_arm):
        # A pre-registration naming a split for some arms and not others
        # would leave the rest's partition unpinned, which is exactly the
        # drift this block exists to prevent.
        raise ValueError(
            "the pre-registered split names exactly the arms K_RUN does"
        )
    if any(
        size < 1
        for split in split_by_arm.values()
        if split is not None
        for size in split
    ):
        raise ValueError("a pre-registered split size is at least 1")


def _require_valid_minibatch_by_arm(
    *,
    minibatch_by_arm: Mapping[str, int | None],
    k_run_by_arm: Mapping[str, int],
) -> None:
    """The per-arm minibatch names every arm, at positive sizes.

    Named for every arm rather than only the minibatching ones, for
    ``split_by_arm``'s reason: a block that named some arms and not others
    would leave the rest's shape unpinned, and "this arm does not
    minibatch" is itself a pre-registered fact.
    """
    if set(minibatch_by_arm) != set(k_run_by_arm):
        raise ValueError(
            "the pre-registered minibatch names exactly the arms K_RUN does"
        )
    if any(
        size is not None and size < 1 for size in minibatch_by_arm.values()
    ):
        raise ValueError("a pre-registered minibatch size is at least 1")


def _pre_registration_payload(  # noqa: PLR0913
    *,
    k_repeat: int,
    k_run_by_arm: dict[str, int],
    split_by_arm: Mapping[str, tuple[int, int] | None],
    minibatch_by_arm: Mapping[str, int | None],
    ci_level: float,
    resamples: int,
    bootstrap_seed: int,
    correction: str,
    m: int,
    completeness_backstop: float,
) -> dict[str, JsonValue]:
    """The exact document the design hash is taken over.

    Spelled as an explicit dict rather than derived from the model's fields:
    the hash is stored identity, and deriving it from field names would let
    a rename silently change every recorded design hash.

    ``split_by_arm`` is written as a two-element list per arm, sorted by arm
    id like ``k_run_by_arm``, so the document is canonical whatever order
    the arms were declared in.

    Deliberately **not** in here: whether the invocation was authorized to
    spend on a real Codex session. That is a run-time permission rather than
    a design choice, and hashing it would make two runs of one design
    pre-register differently.
    """
    return {
        "k_repeat": k_repeat,
        "k_run_by_arm": dict(sorted(k_run_by_arm.items())),
        "split_by_arm": {
            arm_id: (None if split is None else [split[0], split[1]])
            for arm_id, split in sorted(split_by_arm.items())
        },
        "minibatch_by_arm": dict(sorted(minibatch_by_arm.items())),
        "ci_level": ci_level,
        "resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "correction": correction,
        "m": m,
        "completeness_backstop": completeness_backstop,
    }


def pre_registration_design_hash(  # noqa: PLR0913
    *,
    k_repeat: int,
    k_run_by_arm: dict[str, int],
    split_by_arm: Mapping[str, tuple[int, int] | None],
    minibatch_by_arm: Mapping[str, int | None],
    ci_level: float,
    resamples: int,
    bootstrap_seed: int,
    correction: str,
    m: int,
    completeness_backstop: float,
) -> str:
    """The content hash pinning one pre-registered design."""
    return compute_content_hash(
        _pre_registration_payload(
            k_repeat=k_repeat,
            k_run_by_arm=k_run_by_arm,
            split_by_arm=split_by_arm,
            minibatch_by_arm=minibatch_by_arm,
            ci_level=ci_level,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
            correction=correction,
            m=m,
            completeness_backstop=completeness_backstop,
        )
    )


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


class CallCountGateRecord(_StrictModel):
    """Stage 1's call-count verdict, durable so Stage 2 can require it.

    The pilot exists to catch a fan-out bug -- an optimizer whose minibatch
    intents silently expanded to the full valset -- before the full design
    pays five times over for the same defect. A verdict that lived only
    inside the Stage-1 process bought none of that for a Stage 2 started in
    a fresh process: nothing downstream could tell a cleared pilot from a
    pilot that never ran.

    ``passed`` is recorded whether or not it did. A failed gate is a
    finding the study reports rather than an absence, and recording it is
    what lets Stage 2 name *which* prerequisite is missing instead of
    reporting the same "run stage1 first" for both cases.

    ``overruns`` names each run that exceeded its pre-spend estimate, in the
    same ``arm/run_id: N calls`` form the refusal prints, so the manifest
    carries the evidence and not merely the verdict.
    """

    stage: StrictStr
    passed: StrictBool
    tolerance: StrictFloat
    overruns: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _validate_call_count_gate(self) -> CallCountGateRecord:
        if self.stage != StageId.STAGE1.value:
            # Only the pilot runs this gate; a record naming another stage
            # would assert a check that stage never performed.
            raise ValueError(
                f"the call-count gate is Stage 1's; got {self.stage!r}"
            )
        if self.tolerance <= 0.0:
            raise ValueError("the call-count tolerance is positive")
        if self.passed and self.overruns:
            raise ValueError("a passed call-count gate names no overruns")
        if not self.passed and not self.overruns:
            raise ValueError("a failed call-count gate names its overruns")
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
    #: The transport this run executed on.
    #:
    #: **A run's transport is its own evidence, not its stage's.** A stage
    #: record says what the *latest* invocation of that stage bound; the
    #: runs beneath it may have been produced by an earlier invocation on
    #: another transport, because a resumed stage keeps the runs it already
    #: paid for. Without this field, a run measured against the experiment's
    #: own gold and a run measured against a provider are indistinguishable
    #: once recorded, and the cross-transport refusal can only ever check
    #: the stage rows rather than the evidence itself.
    transport: StrictStr

    @model_validator(mode="after")
    def _validate_run(self) -> RunRecord:
        if not self.run_id.strip():
            raise ValueError("a run has a nonblank id")
        if not self.artifact_dir.strip():
            raise ValueError("a run names its artifact directory")
        if self.transport not in TRANSPORT_NAMES:
            raise ValueError(
                f"a run records one of {list(TRANSPORT_NAMES)}, "
                f"got {self.transport!r}"
            )
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
    #: The arm's train/val partition of the internal split, or ``None`` on
    #: an arm whose optimizer has no train/val concept. Recorded because it
    #: is part of the design the study pre-registers -- a GEPA arm rerun at
    #: a different partition is a different arm -- so it is hashed into the
    #: pre-registration rather than left implicit in the run's own control.
    train_size: StrictInt | None
    val_size: StrictInt | None
    #: Whether this arm minibatched, and at what size. Design rather than a
    #: runtime knob, exactly like the train/val split: the two enter the
    #: pre-registration's ``minibatch_by_arm``, and an arm that evaluated
    #: each trial on a sampled batch bought different evidence for the same
    #: claim than one that evaluated on the whole valset. Recorded here
    #: because the spec every stage after Stage 0 runs is rebuilt from the
    #: arm record; without it a manifest-driven MIPROv2 arm silently fell
    #: back to unbatched under a design hash that said it batched.
    #:
    #: The two travel together, as they do on ``ArmSpec``: minibatching on
    #: without a size resolves the batch to the whole valset, which is
    #: minibatching in name only.
    minibatch: StrictBool = False
    minibatch_size: StrictInt | None = None
    control_identity_hash: StrictStr
    seed_note: StrictStr
    runs: tuple[RunRecord, ...]

    @model_validator(mode="after")
    def _validate_arm(self) -> ArmRecord:
        if not self.arm_id.strip() or not self.optimizer.strip():
            raise ValueError("an arm names itself and its optimizer")
        if (self.train_size is None) != (self.val_size is None):
            raise ValueError(
                "an arm records both halves of its train/val split or neither"
            )
        if self.minibatch and self.minibatch_size is None:
            raise ValueError(
                "an arm that minibatches records the size it batched at"
            )
        if not self.minibatch and self.minibatch_size is not None:
            raise ValueError(
                "an arm that records a minibatch size minibatches"
            )
        if self.minibatch_size is not None and self.minibatch_size < 1:
            raise ValueError("a recorded minibatch_size is at least 1")
        if self.train_size is not None and self.train_size < 1:
            raise ValueError("a recorded train_size is at least 1")
        if self.val_size is not None and self.val_size < 1:
            raise ValueError("a recorded val_size is at least 1")
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

    ``stage`` is part of the record because the study selects twice for a
    reason: Stage 1's pilot arg-max is over two runs and Stage 2's is over
    all five, and the second is the study's reported selection. L2 is "once
    per arm **per stage**", which is what the pre-registration actually
    describes -- a pilot that could not select would have no preliminary
    delta to gate on, and a Stage 2 that could not select would have paid
    for three more runs it was forbidden to use.
    """

    arm_id: StrictStr
    selected_run_id: StrictStr
    official_score: StrictFloat
    rule: StrictStr
    stage: StrictStr = StageId.STAGE2.value

    @model_validator(mode="after")
    def _validate_selection(self) -> SelectionRecord:
        if not self.arm_id.strip() or not self.selected_run_id.strip():
            raise ValueError("a selection names its arm and chosen run")
        if self.stage not in set(STAGE_IDS):
            raise ValueError(
                f"a selection names one of {STAGE_IDS}; got {self.stage!r}"
            )
        if self.stage == StageId.STAGE0.value:
            raise ValueError("stage0 selects nothing; it runs no optimizers")
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
    stage: StrictStr = StageId.STAGE2.value
    eval_config_hash: StrictStr | None = None
    repeats: StrictInt | None = None
    mean: StrictFloat | None = None
    completeness: StrictFloat | None = None
    #: The per-task vector the evaluation returned. The paired delta is
    #: computed from this vector rather than from ``mean``, so recording it
    #: is what lets a resumed stage rebuild an arm's report instead of
    #: re-issuing an evaluation it already paid for. Always empty on an
    #: outstanding claim; a completed claim written before this field
    #: existed may also be empty, and a resume treats that as unrebuildable
    #: rather than as a measurement it can fabricate.
    per_task: tuple[StrictFloat, ...] = ()
    #: Rows achieved per task, aligned with ``per_task``. Optional even on
    #: a completed claim, because the measurement itself treats it as
    #: optional and falls back to spreading completeness evenly.
    per_task_counts: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def _validate_claim(self) -> HeldOutClaimRecord:
        if not self.candidate_name.strip():
            raise ValueError("a held-out claim names its candidate")
        if self.stage not in set(STAGE_IDS):
            raise ValueError(
                f"a held-out claim names one of {STAGE_IDS}; "
                f"got {self.stage!r}"
            )
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
        if self.per_task_counts and len(self.per_task_counts) != len(
            self.per_task
        ):
            raise ValueError(
                "per_task_counts aligns with per_task when it is recorded"
            )
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
        if self.mean is None and self.per_task:
            raise ValueError(
                "an outstanding held-out claim has no per-task vector"
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
    #: The naive anchor's own completeness, carried on every row whose
    #: delta is measured against it.
    #:
    #: The delta is paired, so a row's ``completeness`` -- the paired
    #: minimum this candidate's comparison achieved -- can be low for two
    #: quite different reasons: this candidate lost rows, or the anchor
    #: did. Recording the anchor's side means a reader can tell which,
    #: rather than reading a downgraded arm as the arm's own failure. It
    #: is ``None`` on the anchor's own row, which has no anchor to compare
    #: against, and on a row written before an anchor was measured.
    anchor_completeness: StrictFloat | None = None

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
            ("anchor_completeness", self.anchor_completeness),
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
# Stages
# --------------------------------------------------------------------------


class StageRecord(_StrictModel):
    """One stage that ran: on which transport, and what it spent.

    **The transport is evidence, not design.** It is a property of the
    invocation -- like the real-Codex authorization -- so it never enters
    the pre-registration hash, and two studies that differ only in it
    pre-register identically. It is recorded all the same, because a stage
    calibrated on the fake transport and a stage calibrated against a
    provider are different evidence for the same claim, and every number
    the report prints downstream of a stage inherits which one it was.

    ``spend`` is what the stage's *runs* cost, one entry per role, in the
    same record shape a run reports. Stage 0's anchors spend through the
    evaluation engine rather than through an optimizer run, so without this
    the study's most expensive calibration would be the one part of the
    accounting with no total. A stage that spent nothing measurable -- a
    fake-transport stage, whose rows carry no provider telemetry -- records
    an empty tuple rather than a fabricated zero-cost role.

    ``report_spend`` is what the *reporting pass* cost, kept separate
    because the two accumulate by opposite rules. Run spend is carried by
    the runs a given invocation executed, so a resume adds what it newly
    paid to what earlier invocations paid. Reporting spend is folded from
    the manifest's own durable per-evaluation records, so it is already the
    whole pass every time it is computed -- adding it to what was there
    would bill the same evaluations once per resume. Summing them into one
    field made those two rules indistinguishable; :attr:`total_spend` is
    the number a reader wants, and it is derived rather than stored so the
    parts cannot drift from the total.
    """

    stage: StrictStr
    transport: StrictStr
    spend: tuple[RunSpendRecord, ...] = ()
    report_spend: tuple[RunSpendRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_stage(self) -> StageRecord:
        if self.stage not in STAGE_IDS:
            raise ValueError(
                f"a stage record names one of {list(STAGE_IDS)}, "
                f"got {self.stage!r}"
            )
        if self.transport not in TRANSPORT_NAMES:
            raise ValueError(
                f"a stage record names one of {list(TRANSPORT_NAMES)}, "
                f"got {self.transport!r}"
            )
        for field in (self.spend, self.report_spend):
            roles = [entry.role for entry in field]
            if len(set(roles)) != len(roles):
                raise ValueError(
                    "each provider role is reported once per stage"
                )
        return self

    @property
    def total_spend(self) -> tuple[RunSpendRecord, ...]:
        """The stage's whole bill: its runs plus its reporting pass.

        Derived rather than stored, so the total can never disagree with
        the parts it is made of. The fold re-applies the honesty rules, so
        an unknown ``usd`` on either side keeps the total unknown rather
        than letting the priced half stand in for the whole.

        The fold is imported at call time because it lives in the module
        that imports *this* one: spend projects into these records, so the
        dependency runs that way and a module-level import would close the
        cycle.
        """
        from whetstone_envs.optim.study.spend import (  # noqa: PLC0415
            run_spend_records,
        )

        return run_spend_records((*self.spend, *self.report_spend))


class ReportSpendEntry(_StrictModel):
    """One reporting evaluation's spend, made durable as it is bought.

    The reporting pass buys evaluations one at a time -- an official score
    per run, then a held-out measurement per arm, then the anchors -- and
    each one reaches the provider before the stage's row is written. Held
    only in memory, those numbers are wrong in both directions across a
    crash: a resume that re-folded the in-memory ledger onto a row that
    already carried them would *double* the pass's bill, and a resume that
    rebuilt its claims without re-evaluating would drop the spend of every
    evaluation the crashed invocation had already paid for.

    So each evaluation records its own spend here the moment it completes,
    and the stage's row is folded from these records rather than from what
    accumulated in this process. That makes the fold idempotent: re-folding
    the same evaluations yields the same total, because the total is a
    function of what is on disk rather than of what this invocation
    happened to buy.

    ``evidence_key`` is the evaluation's own ``(schema, content_hash)``,
    which is what makes an entry both checkable and de-duplicable: the same
    evaluation recorded twice is one purchase, and the number is
    re-derivable from the evidence rather than taken on the ledger's word.
    """

    #: The evidence's schema name and content hash, in that order. A pair
    #: rather than a nested record: it is an identity, and the manifest
    #: cites identities as the two strings that form them.
    evidence_schema: StrictStr
    evidence_content_hash: StrictStr
    #: Which pass bought it -- official selection, held-out, an anchor --
    #: and which candidate it measured, so a reader can attribute any part
    #: of the total to the evaluation that produced it.
    purpose: StrictStr
    candidate_name: StrictStr
    stage: StrictStr
    #: The transport the evaluation was bought on. Recorded because the
    #: stage's reporting row is folded from these entries, and a fold that
    #: keyed on the stage alone would bill a paid stage for a
    #: fake-transport invocation's evaluations -- a total nobody owes.
    transport: StrictStr
    spend: tuple[RunSpendRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_report_spend(self) -> ReportSpendEntry:
        if not self.evidence_schema.strip() or not (
            self.evidence_content_hash.strip()
        ):
            raise ValueError("a report spend entry cites its evidence")
        if not self.purpose.strip() or not self.candidate_name.strip():
            raise ValueError(
                "a report spend entry names its purpose and candidate"
            )
        if self.stage not in STAGE_IDS:
            raise ValueError(
                f"a report spend entry names one of {list(STAGE_IDS)}, "
                f"got {self.stage!r}"
            )
        if self.transport not in TRANSPORT_NAMES:
            raise ValueError(
                f"a report spend entry names one of {list(TRANSPORT_NAMES)}, "
                f"got {self.transport!r}"
            )
        roles = [entry.role for entry in self.spend]
        if len(set(roles)) != len(roles):
            raise ValueError(
                "each provider role is reported once per evaluation"
            )
        return self

    @property
    def evidence_key(self) -> tuple[str, str]:
        """The evidence identity this entry was derived from."""
        return (self.evidence_schema, self.evidence_content_hash)


class OfficialScoreEntry(_StrictModel):
    """One run's official score, made durable the first time it is bought.

    Official-selection scoring is a provider call per run, and it happens
    on every reporting invocation -- including a resume, which re-scored
    every run of every already-reported arm purely to rebuild a report the
    manifest could already have answered. That is a second charge for a
    number the study had already bought.

    Recording the score here is what lets a resume rebuild an arm's report
    without re-issuing anything. The spend for these calls is recorded
    separately, as a :class:`ReportSpendEntry`, so the two facts stay
    independently checkable: what the run scored, and what learning that
    cost.
    """

    run_id: StrictStr
    arm_id: StrictStr
    stage: StrictStr
    #: The transport the score was measured on. Run ids are deterministic,
    #: so a cross-transport re-calibration recomputes the same names; the
    #: read-back checks this so one transport's measurement can never be
    #: reused as another's selection evidence.
    transport: StrictStr
    score: StrictFloat
    eval_config_hash: StrictStr
    #: Rows achieved over rows requested, carried because a mean without
    #: its completeness cannot be judged against the backstop.
    completeness: StrictFloat
    #: The per-task vector the scoring returned. Recorded because the
    #: rebuild must reproduce the score object exactly rather than an
    #: aggregate that merely agrees with it on the mean.
    per_task: tuple[StrictFloat, ...]

    @model_validator(mode="after")
    def _validate_official_score(self) -> OfficialScoreEntry:
        if not self.run_id.strip() or not self.arm_id.strip():
            raise ValueError("an official score names its run and arm")
        if not self.per_task:
            raise ValueError("an official score carries its per-task vector")
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("completeness is a fraction in [0, 1]")
        if not self.eval_config_hash.strip():
            raise ValueError("an official score names its Eval Config")
        if self.stage not in STAGE_IDS:
            raise ValueError(
                f"an official score names one of {list(STAGE_IDS)}, "
                f"got {self.stage!r}"
            )
        if self.stage == StageId.STAGE0.value:
            raise ValueError("stage0 selects nothing; it scores no runs")
        if self.transport not in TRANSPORT_NAMES:
            raise ValueError(
                f"an official score names one of {list(TRANSPORT_NAMES)}, "
                f"got {self.transport!r}"
            )
        return self


#: Why an amendment dropped evidence: the design was replaced onto a
#: transport the dropped evidence was not measured on. Pinned because the
#: report reads it back and an operator greps for it.
AMENDMENT_REASON_TRANSPORT_CHANGE = "replace-design-across-transports"

#: Every amendment reason the manifest accepts.
AMENDMENT_REASONS: tuple[str, ...] = (AMENDMENT_REASON_TRANSPORT_CHANGE,)


class AmendmentRecord(_StrictModel):
    """Evidence a re-calibration invalidated, recorded rather than erased.

    ``stage0 --replace-design`` onto a different transport invalidates the
    arm stages twice over: the design they were run against no longer
    exists, and their evidence was measured somewhere else. Dropping them
    silently would leave a manifest that reads like a study which simply
    never ran those stages, and keeping them would let Stage 2 reuse runs
    from another experiment against freshly bought anchors.

    So they are dropped *and* the drop is recorded. What this names is
    what the study once held and no longer does, which is the one fact a
    reader cannot recover from the manifest's current contents.
    """

    #: When this amendment was recorded, ISO-8601.
    at: StrictStr
    #: Which stage's re-run caused it. Always ``stage0`` today; recorded
    #: rather than assumed, because a later amendment path would be a
    #: different fact under the same key.
    amended_stage: StrictStr
    reason: StrictStr
    #: The transport the dropped evidence was measured on, and the one the
    #: re-calibration bound. Both named, because "this study changed
    #: transport" is not actionable without knowing in which direction.
    from_transport: StrictStr
    to_transport: StrictStr
    #: The stage records dropped, by stage id.
    dropped_stages: tuple[StrictStr, ...]
    #: Every arm run dropped, by run id. Named individually because a count
    #: cannot be checked against the artifacts still on disk.
    dropped_run_ids: tuple[StrictStr, ...]
    #: The directories those runs left behind, which the drop does *not*
    #: remove. A run id identifies the evidence in the manifest; this
    #: identifies it on disk, and the two are different facts precisely
    #: because the amendment separates them. Recorded because the stage
    #: that re-runs these arms refuses to reuse a directory it cannot
    #: claim, and an operator resolving that refusal needs to know which
    #: directories the amendment orphaned without reconstructing the
    #: deterministic naming rule by hand.
    dropped_run_directories: tuple[StrictStr, ...] = ()
    dropped_selections: StrictInt
    dropped_held_out_claims: StrictInt
    dropped_held_out_rows: StrictInt
    #: Whether the pilot's call-count verdict went with them.
    dropped_call_count_gate: StrictBool
    #: The official scores dropped with their runs. Recorded rather than
    #: inferred from ``dropped_run_ids``: a run is dropped whether or not
    #: it had been scored yet, so how many *measurements* the study lost
    #: is a separate fact from how many runs it lost.
    dropped_official_scores: StrictInt = 0
    #: The reporting purchases dropped with their stage rows. Recorded for
    #: the same reason and, unlike the rest, it is money: an operator
    #: reconciling the study's bill needs to see what the amendment
    #: removed from it.
    dropped_report_spend: StrictInt = 0

    @model_validator(mode="after")
    def _validate_amendment(self) -> AmendmentRecord:
        if not self.at.strip():
            raise ValueError("an amendment records when it happened")
        if self.amended_stage not in STAGE_IDS:
            raise ValueError(
                f"an amendment names one of {list(STAGE_IDS)}, "
                f"got {self.amended_stage!r}"
            )
        if self.reason not in AMENDMENT_REASONS:
            raise ValueError(
                f"an amendment reason is one of {list(AMENDMENT_REASONS)}; "
                f"got {self.reason!r}"
            )
        for name, value in (
            ("from_transport", self.from_transport),
            ("to_transport", self.to_transport),
        ):
            if value not in TRANSPORT_NAMES:
                raise ValueError(
                    f"an amendment's {name} is one of "
                    f"{list(TRANSPORT_NAMES)}, got {value!r}"
                )
        if self.from_transport == self.to_transport:
            raise ValueError(
                "a transport-change amendment names two different transports"
            )
        for stage in self.dropped_stages:
            if stage not in STAGE_IDS:
                raise ValueError(
                    f"a dropped stage is one of {list(STAGE_IDS)}, "
                    f"got {stage!r}"
                )
        if len(set(self.dropped_stages)) != len(self.dropped_stages):
            raise ValueError("each dropped stage is named once")
        if len(set(self.dropped_run_ids)) != len(self.dropped_run_ids):
            raise ValueError("each dropped run is named once")
        counts = (
            self.dropped_selections,
            self.dropped_held_out_claims,
            self.dropped_held_out_rows,
            self.dropped_official_scores,
            self.dropped_report_spend,
        )
        if any(value < 0 for value in counts):
            raise ValueError("an amendment's drop counts are non-negative")
        return self


def recorded_transport(
    stages: tuple[StageRecord, ...], stage: StageId | str
) -> str | None:
    """The transport ``stage`` ran on, or ``None`` if it has not run.

    Owned here beside the record rather than at each caller, because "which
    transport did this study's Stage 0 run on" is the question the
    cross-stage refusal is built from and it must have one answer.
    """
    wanted = stage.value if isinstance(stage, StageId) else stage
    for entry in stages:
        if entry.stage == wanted:
            return entry.transport
    return None


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def _validate_design_matches_pre_registration(
    *,
    pre_registration: PreRegistrationRecord | None,
    design: DesignRecord | None,
) -> None:
    """Refuse a design that contradicts the pinned pre-registration.

    The two blocks overlap deliberately: ``design`` is what Stage 0
    measured, and ``pre_registration`` is the subset of it that was fixed
    before any spend. Letting them disagree would make the pinning
    decorative -- a later stage could restate the design and leave the
    frozen block untouched -- so the overlap is checked rather than
    assumed.
    """
    if pre_registration is None or design is None:
        return
    mismatches = [
        f"{name}: design {design_value!r} vs pre-registration {pinned!r}"
        for name, design_value, pinned in (
            ("k_repeat", design.k_repeat, pre_registration.k_repeat),
            (
                "k_run_by_arm",
                dict(sorted(design.k_run_by_arm.items())),
                dict(sorted(pre_registration.k_run_by_arm.items())),
            ),
            ("ci_level", design.ci_level, pre_registration.ci_level),
            ("resamples", design.resamples, pre_registration.resamples),
            (
                "bootstrap_seed",
                design.bootstrap_seed,
                pre_registration.bootstrap_seed,
            ),
            ("correction", design.correction, pre_registration.correction),
            ("m", design.m, pre_registration.m),
            (
                "completeness_backstop",
                design.completeness_backstop,
                pre_registration.completeness_backstop,
            ),
        )
        if design_value != pinned
    ]
    if mismatches:
        raise ValueError(
            "the design contradicts the pinned pre-registration: "
            + "; ".join(mismatches)
        )


def _validate_reporting_purchases(
    *,
    report_spend: tuple[ReportSpendEntry, ...],
    official_scores: tuple[OfficialScoreEntry, ...],
) -> None:
    """Each reporting purchase is recorded at most once per stage.

    Structural rather than checked where the numbers are folded, because
    the fold's whole correctness rests on it: the stage's reporting row is
    computed from these records on every invocation, so a duplicate entry
    would bill one evaluation once per resume -- which is the failure the
    durable records exist to prevent.
    """
    bought = [(entry.evidence_key, entry.stage) for entry in report_spend]
    if len(set(bought)) != len(bought):
        # One evaluation cited twice was paid for once.
        raise ValueError(
            "each reporting evaluation is recorded at most once per stage"
        )
    scored = [(entry.run_id, entry.stage) for entry in official_scores]
    if len(set(scored)) != len(scored):
        # One official score per run per stage: the pilot and the full
        # design each score, over different run sets.
        raise ValueError(
            "each run is officially scored at most once per stage"
        )


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
    pre_registration: PreRegistrationRecord | None = None
    #: Evidence a re-calibration invalidated and dropped, in the order it
    #: was dropped. Append-only: an amendment records what the study once
    #: held, so removing one would erase the very fact it exists to keep.
    amendments: tuple[AmendmentRecord, ...] = ()
    design: DesignRecord | None = None
    stages: tuple[StageRecord, ...] = ()
    #: Every reporting evaluation this study has paid for, in issue order.
    #: Durable as it is bought rather than folded at the end of the pass,
    #: so a crash mid-pass neither loses the spend of what was already
    #: bought nor lets a resume charge for it twice.
    report_spend: tuple[ReportSpendEntry, ...] = ()
    #: Every run's official-selection score, durable the first time it is
    #: bought, so a resume rebuilds an arm's report from the manifest
    #: instead of re-scoring runs the study already paid to score.
    official_scores: tuple[OfficialScoreEntry, ...] = ()
    gepa_sizing: GepaSizingRecord | None = None
    fanout_check: FanoutCheckRecord | None = None
    call_count_gate: CallCountGateRecord | None = None
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
        _validate_design_matches_pre_registration(
            pre_registration=self.pre_registration, design=self.design
        )
        stage_ids = [entry.stage for entry in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            # A stage records what it ran on. Two records for one stage
            # would mean the study could not say which transport its
            # numbers came from, which is the whole reason the block
            # exists; a re-run replaces its record rather than appending.
            raise ValueError("each stage is recorded at most once")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm ids are distinct")
        _validate_reporting_purchases(
            report_spend=self.report_spend,
            official_scores=self.official_scores,
        )
        selected = [(entry.arm_id, entry.stage) for entry in self.selection]
        if len(set(selected)) != len(selected):
            # L2 as a structural rule: one selection per arm per stage,
            # never two. The stage is part of the key because the pilot and
            # the full design each select once, over different run sets.
            raise ValueError("each arm is selected at most once per stage")
        unknown = {arm_id for arm_id, _stage in selected} - set(arm_ids)
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
        claimed = [
            (entry.candidate_name, entry.stage)
            for entry in self.held_out_claims
        ]
        if len(set(claimed)) != len(claimed):
            # L3 as a structural rule, at the moment the evaluation is
            # issued rather than at the moment its statistics are known.
            # Scoped by stage for the same reason selection is: the pilot's
            # representative candidate and the full design's are different
            # candidates, each measured exactly once.
            raise ValueError(
                "each candidate claims held-out at most once per stage"
            )
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


class PreRegistrationViolationError(ValueError):
    """A write would have altered a pinned pre-registration.

    A ``ValueError`` because every caller of :func:`write_study_manifest`
    already treats one as a refusal to write, and this is exactly that: the
    manifest on disk pre-registered a design and the document offered does
    not carry it back unchanged.
    """


def write_study_manifest(
    study_dir: Path,
    manifest: StudyManifest,
    *,
    replace: bool = False,
    amend_pre_registration: bool = False,
) -> Path:
    """Validate ``manifest``, then write it into ``study_dir``.

    Validation is a full round-trip through the persisted JSON, not a trust
    of the in-memory object: a manifest is only written if the exact bytes
    that will land re-validate. The directory must be outside every
    detected repository, because a study instance is a durable work
    document rather than a versioned deliverable.

    **A pinned pre-registration survives every later write.** Once a
    manifest on disk carries a ``pre_registration`` block, any write that
    drops it or changes any of its bytes is refused with
    :class:`PreRegistrationViolationError`. That refusal is what makes the
    block a pre-registration rather than a field: Stage 1 and Stage 2 write
    the manifest constantly, and without it a design could be restated after
    its own results were known. ``amend_pre_registration`` is the single
    deliberate exception, and the amended block must record its
    ``amended`` provenance and the hash it replaced -- so an amendment is
    always legible as one.
    """
    resolved = validate_output_root(study_dir)
    validated = StudyManifest.model_validate_json(
        manifest.model_dump_json(by_alias=True)
    )
    path = study_manifest_path(resolved)
    if path.exists() and not replace:
        raise ManifestExistsError(path)
    if path.exists():
        _require_preserved_pre_registration(
            existing=read_study_manifest(path),
            offered=validated,
            amend=amend_pre_registration,
        )
    resolved.mkdir(parents=True, exist_ok=True)
    document = _manifest_document(resolved)
    document.publish(validated.model_dump(mode="json", by_alias=True))
    return document.path


def _require_preserved_pre_registration(
    *,
    existing: StudyManifest,
    offered: StudyManifest,
    amend: bool,
) -> None:
    """Refuse a write that would alter what a study pre-registered.

    Compared as persisted documents rather than as objects, because what is
    pinned is the bytes a reader will resolve the design hash against.
    """
    pinned = existing.pre_registration
    if pinned is None:
        # Nothing pinned yet: this write may be the Stage-0 write that pins
        # it. Immutability starts once a block exists, not before.
        return
    offered_block = offered.pre_registration
    if amend:
        _require_valid_amendment(pinned=pinned, offered=offered_block)
        return
    if offered_block is None:
        raise PreRegistrationViolationError(
            "this study pre-registered its design at "
            f"{pinned.design_hash[:12]} and this write drops the block "
            "entirely; a pre-registration is preserved by every later write"
        )
    before = pinned.model_dump(mode="json", by_alias=True)
    after = offered_block.model_dump(mode="json", by_alias=True)
    if before != after:
        changed = sorted(
            key
            for key in {*before, *after}
            if before.get(key) != after.get(key)
        )
        raise PreRegistrationViolationError(
            "this write would change the pinned pre-registration "
            f"({', '.join(changed)}); pass amend_pre_registration to record "
            "a deliberate amendment instead"
        )


def _require_valid_amendment(
    *,
    pinned: PreRegistrationRecord,
    offered: PreRegistrationRecord | None,
) -> None:
    """An amendment names itself and the design it replaced."""
    if offered is None:
        raise PreRegistrationViolationError(
            "an amendment replaces a pre-registration with another one; "
            "this write carries no pre_registration block"
        )
    if offered.provenance != PROVENANCE_AMENDED:
        raise PreRegistrationViolationError(
            "an amended pre-registration records provenance "
            f"{PROVENANCE_AMENDED!r}; got {offered.provenance!r}"
        )
    if offered.amended_from != pinned.design_hash:
        raise PreRegistrationViolationError(
            "an amendment names the design hash it replaced: expected "
            f"{pinned.design_hash}, got {offered.amended_from}"
        )


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
    "AMENDMENT_REASONS",
    "AMENDMENT_REASON_TRANSPORT_CHANGE",
    "COMPLETENESS_BACKSTOP",
    "CORRECTION_FAMILY_SIZE",
    "CORRECTION_HOLM_BONFERRONI",
    "DISCARD_STALE_RUNS_FLAG",
    "MAX_MANIFEST_BYTES",
    "PROVENANCE_AMENDED",
    "PROVENANCE_ORIGINAL",
    "PROVENANCE_VALUES",
    "PROVIDER_CONTROL_UNSET",
    "PROVIDER_SEED_DERIVED_PER_CALL",
    "SELECTION_RULE_ARGMAX_OFFICIAL",
    "STAGE_IDS",
    "STUDY_MANIFEST_NAME",
    "STUDY_MANIFEST_SCHEMA",
    "STUDY_MANIFEST_SCHEMA_NAME",
    "STUDY_MANIFEST_SCHEMA_VERSION",
    "STUDY_STORE_NAME",
    "TRANSPORT_NAMES",
    "AdapterSwapRecord",
    "AmendmentRecord",
    "ArmRecord",
    "BalanceRecord",
    "C18Record",
    "CallCountGateRecord",
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
    "OfficialScoreEntry",
    "PointerCheck",
    "PointerCheckReport",
    "PopulationRecord",
    "PreRegistrationRecord",
    "PreRegistrationViolationError",
    "ProviderCallRecord",
    "ReportSpendEntry",
    "RunRecord",
    "RunSpendRecord",
    "SelectionRecord",
    "SplitName",
    "SplitRecord",
    "SplitsRecord",
    "StageId",
    "StageRecord",
    "StudyManifest",
    "TransportName",
    "check_manifest_pointers",
    "format_pointer_report",
    "pre_registration_design_hash",
    "read_study_manifest",
    "recorded_transport",
    "study_manifest_path",
    "write_study_manifest",
]
