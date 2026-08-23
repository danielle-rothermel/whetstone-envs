"""The study manifest's persisted format, writer, reader, and checker.

The golden test at the top is the persisted-format contract: it pins every
wire literal as a written-out string rather than deriving one from a field
name or iterating an enum. A field rename that changed stored identity
fails here, which is the whole point -- only a pinned test catches silent
drift of stored identity.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import rfc8785
from dr_store import ObjectNotFoundError, ObjectReference
from dr_store.document_file.errors import DocumentReadError
from pydantic import ValidationError

from whetstone_envs.optim.study.manifest import (
    AMENDMENT_REASON_TRANSPORT_CHANGE,
    COMPLETENESS_BACKSTOP,
    CORRECTION_FAMILY_SIZE,
    CORRECTION_HOLM_BONFERRONI,
    PROVENANCE_AMENDED,
    PROVENANCE_ORIGINAL,
    PROVIDER_CONTROL_UNSET,
    SELECTION_RULE_ARGMAX_OFFICIAL,
    STAGE_IDS,
    STUDY_MANIFEST_NAME,
    STUDY_MANIFEST_SCHEMA,
    STUDY_MANIFEST_SCHEMA_NAME,
    STUDY_MANIFEST_SCHEMA_VERSION,
    TRANSPORT_NAMES,
    AdapterSwapRecord,
    AmendmentRecord,
    ArmRecord,
    BalanceRecord,
    C18Record,
    CallCountGateRecord,
    DesignRecord,
    EvidencePointer,
    FanoutCheckRecord,
    GepaSizingRecord,
    HeldOutClaimRecord,
    HeldOutRecord,
    LeakageCheckEntry,
    LeakageCheckRecord,
    ManifestExistsError,
    ManifestKey,
    ModelsRecord,
    OfficialScoreEntry,
    PopulationRecord,
    PreRegistrationRecord,
    PreRegistrationViolationError,
    ProviderCallRecord,
    ReportSpendEntry,
    RunRecord,
    RunSpendRecord,
    SelectionRecord,
    SplitName,
    SplitRecord,
    SplitsRecord,
    StageId,
    StudyManifest,
    TransportName,
    check_manifest_pointers,
    format_pointer_report,
    pre_registration_design_hash,
    read_study_manifest,
    study_manifest_path,
    write_study_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue


def _hash(char: str) -> str:
    return char * 64


class _FakeStore:
    """An in-memory stand-in for the blocking object store's read side."""

    def __init__(self, records: dict[tuple[str, str], JsonValue]) -> None:
        self._records = records

    def get(self, reference: ObjectReference) -> JsonValue:
        key = (reference.schema, reference.content_hash)
        if key not in self._records:
            raise ObjectNotFoundError(reference=reference)
        return self._records[key]


RESULT_SCHEMA = "whetstone.optim_result"
AUDIT_SCHEMA = "whetstone_envs.audit_report"
EVIDENCE_SCHEMA = "whetstone.eval_evidence"
SCORES_SCHEMA = "whetstone_envs.per_task_scores"
COST_SCHEMA = "whetstone_envs.run_cost"


def _pointer(schema: str, char: str) -> EvidencePointer:
    return EvidencePointer(schema_name=schema, content_hash=_hash(char))


def _run(
    run_id: str, *, result: str, audit: str, cost: str = "e"
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        seed=1000,
        artifact_dir=f"/tmp/runs/{run_id}",  # noqa: S108
        result_ref=_pointer(RESULT_SCHEMA, result),
        audit_ref=_pointer(AUDIT_SCHEMA, audit),
        cost_ref=_pointer(COST_SCHEMA, cost),
        audit_passed=True,
        transport=TransportName.FAKE.value,
        spend=(
            RunSpendRecord(
                role="task_model",
                calls=10,
                cached_calls=2,
                input_tokens=100,
                output_tokens=20,
                priced_calls=10,
                unpriced_calls=0,
                rows_missing_token_breakdown=0,
                usd=0.5,
            ),
        ),
    )


def _minimal_manifest() -> StudyManifest:
    return StudyManifest(
        study_id="step10-2026-08-22",
        created_at="2026-08-22T12:00:00+00:00",
        protocol_doc_path="~/drotherm/data/.claude/protocol.md",
        protocol_doc_sha256=_hash("a"),
        assignment_doc_sha256=_hash("b"),
        population=PopulationRecord(
            family="c19",
            generator_version="whetstone-envs 0.2.1",
            n_per_stratum=32,
            pool_seed_start=1_000_000,
            pool_manifest_content_hash=_hash("c"),
            stratum_counts={"navigation/small": 8},
        ),
        splits=SplitsRecord(
            internal=SplitRecord(
                size=2,
                task_hashes=(_hash("1"), _hash("2")),
                eval_config_hash="internal-config",
            ),
            official=SplitRecord(
                size=1,
                task_hashes=(_hash("3"),),
                eval_config_hash="official-config",
            ),
            held_out=SplitRecord(
                size=1,
                task_hashes=(_hash("4"),),
                eval_config_hash="held-out-config",
            ),
        ),
        models=ModelsRecord(
            task_model="openai/gpt-5-nano",
            proposer_model="openai/gpt-5.4-nano",
            temperature="unset",
            provider="openrouter",
            seed_control="advertised",
            codex_agent_model="uncontrolled",
        ),
    )


def _full_manifest() -> StudyManifest:
    base = _minimal_manifest()
    return base.model_copy(
        update={
            "design": DesignRecord(
                k_cal=4,
                k_repeat=3,
                k_run_by_arm={"copro": 5, "null-identity": 1},
                ci_level=0.95,
                resamples=10_000,
                bootstrap_seed=7,
                correction=CORRECTION_HOLM_BONFERRONI,
                m=CORRECTION_FAMILY_SIZE,
                mde_formula=(
                    "MDE(T, K) = 2.8016 * sqrt((tau_sq + 2*sigma_sq/K) / T)"
                ),
                mde_measured=0.08,
                tau_sq=0.02,
                sigma_sq=0.05,
                completeness_rule="achieved-count weighted per task",
                completeness_backstop=COMPLETENESS_BACKSTOP,
            ),
            "amendments": (
                AmendmentRecord(
                    at="2026-08-22T12:00:00+00:00",
                    amended_stage="stage0",
                    reason=AMENDMENT_REASON_TRANSPORT_CHANGE,
                    from_transport="fake",
                    to_transport="openrouter",
                    dropped_stages=("stage1",),
                    dropped_run_ids=("copro-seed1000",),
                    dropped_selections=1,
                    dropped_held_out_claims=2,
                    dropped_held_out_rows=1,
                    dropped_call_count_gate=True,
                    dropped_official_scores=1,
                    dropped_report_spend=1,
                ),
            ),
            "gepa_sizing": GepaSizingRecord(
                steps_per_run=732,
                wall_seconds=1200.0,
                sqlite_bytes=4096,
                max_metric_calls_pinned=200,
            ),
            "fanout_check": FanoutCheckRecord(
                passed=True, minibatch_intents=12, full_valset_intents=3
            ),
            "call_count_gate": CallCountGateRecord(
                stage="stage1", passed=True, tolerance=1.5
            ),
            "arms": (
                ArmRecord(
                    arm_id="copro",
                    optimizer="copro",
                    demo_mode=None,
                    train_size=None,
                    val_size=None,
                    control_identity_hash=_hash("d"),
                    seed_note="provider-seed-control-only",
                    runs=(_run("copro-1", result="5", audit="6", cost="e"),),
                ),
            ),
            "report_spend": (
                ReportSpendEntry(
                    evidence_schema=EVIDENCE_SCHEMA,
                    evidence_content_hash=_hash("f"),
                    purpose="official",
                    candidate_name="copro-1",
                    stage="stage1",
                    transport="openrouter",
                    spend=(
                        RunSpendRecord(
                            role="task_model",
                            calls=4,
                            cached_calls=0,
                            input_tokens=100,
                            output_tokens=20,
                            priced_calls=4,
                            unpriced_calls=0,
                            rows_missing_token_breakdown=0,
                            usd=0.25,
                        ),
                    ),
                ),
            ),
            "official_scores": (
                OfficialScoreEntry(
                    run_id="copro-1",
                    arm_id="copro",
                    stage="stage1",
                    transport="openrouter",
                    score=0.42,
                    eval_config_hash="official-config",
                    completeness=1.0,
                    per_task=(0.42,),
                ),
            ),
            "selection": (
                SelectionRecord(
                    arm_id="copro",
                    selected_run_id="copro-1",
                    official_score=0.42,
                    rule=SELECTION_RULE_ARGMAX_OFFICIAL,
                ),
            ),
            "held_out_claims": (
                HeldOutClaimRecord(
                    candidate_name="copro-representative",
                    eval_config_hash="held-out-config",
                    repeats=3,
                    mean=0.45,
                    completeness=1.0,
                ),
            ),
            "held_out": (
                HeldOutRecord(
                    candidate_name="copro-representative",
                    eval_evidence_ref=_pointer(EVIDENCE_SCHEMA, "7"),
                    per_task_scores_ref=_pointer(SCORES_SCHEMA, "8"),
                    mean=0.45,
                    ci_low=0.40,
                    ci_high=0.50,
                    delta_vs_naive=0.05,
                    p_bootstrap=0.03,
                    p_holm=0.12,
                    completeness=1.0,
                ),
            ),
            "balance": BalanceRecord(
                before_stage0_usd=40.0,
                before_stage1_usd=35.0,
                before_stage2_usd=30.0,
                after_usd=10.0,
            ),
            "leakage_check": LeakageCheckRecord(
                passed=True,
                checks=(
                    LeakageCheckEntry(
                        check_id="L1",
                        passed=True,
                        detail="every intent resolved the internal config",
                    ),
                ),
            ),
            "c18": C18Record(
                runs=(_run("c18-copro-1", result="9", audit="a", cost="b"),),
                adapter_swap=AdapterSwapRecord(
                    passed=True,
                    differing_modules=("c18_experiment.py", "families.py"),
                ),
            ),
        }
    )


# --------------------------------------------------------------------------
# Golden literals
# --------------------------------------------------------------------------


def test_persisted_schema_literals_are_pinned() -> None:
    assert STUDY_MANIFEST_SCHEMA_NAME == "whetstone_envs.step10_study"
    assert STUDY_MANIFEST_SCHEMA_VERSION == 9
    assert STUDY_MANIFEST_SCHEMA == "whetstone_envs.step10_study/v9"
    assert STUDY_MANIFEST_NAME == "study.json"


def test_persisted_vocabulary_literals_are_pinned() -> None:
    assert SELECTION_RULE_ARGMAX_OFFICIAL == "argmax-official"
    assert CORRECTION_HOLM_BONFERRONI == "holm-bonferroni"
    assert CORRECTION_FAMILY_SIZE == 4
    assert COMPLETENESS_BACKSTOP == 0.90


def test_manifest_wire_keys_are_pinned() -> None:
    assert [member.value for member in ManifestKey] == [
        "schema",
        "study_id",
        "created_at",
        "protocol_doc_path",
        "protocol_doc_sha256",
        "assignment_doc_sha256",
        "population",
        "splits",
        "models",
        "pre_registration",
        "amendments",
        "design",
        "stages",
        "report_spend",
        "official_scores",
        "gepa_sizing",
        "fanout_check",
        "call_count_gate",
        "arms",
        "selection",
        "held_out_claims",
        "held_out",
        "balance",
        "leakage_check",
        "c18",
    ]


def test_split_and_stage_names_are_pinned() -> None:
    assert [member.value for member in SplitName] == [
        "internal",
        "official",
        "held_out",
    ]
    assert [member.value for member in StageId] == [
        "stage0",
        "stage1",
        "stage2",
    ]
    assert STAGE_IDS == ("stage0", "stage1", "stage2")
    assert [member.value for member in TransportName] == [
        "fake",
        "openrouter",
    ]
    assert TRANSPORT_NAMES == ("fake", "openrouter")


def test_serialized_document_keys_match_the_owned_wire_keys() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    assert list(payload) == [member.value for member in ManifestKey]
    assert payload["schema"] == STUDY_MANIFEST_SCHEMA


def test_nested_record_wire_keys_are_pinned() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    assert list(payload["population"]) == [
        "family",
        "generator_version",
        "n_per_stratum",
        "pool_seed_start",
        "pool_manifest_content_hash",
        "stratum_counts",
    ]
    assert list(payload["splits"]) == ["internal", "official", "held_out"]
    assert list(payload["splits"]["internal"]) == [
        "size",
        "task_hashes",
        "eval_config_hash",
    ]
    assert list(payload["models"]) == [
        "task_model",
        "proposer_model",
        "temperature",
        "provider",
        "seed_control",
        "codex_agent_model",
        "provider_calls",
    ]
    assert list(payload["design"]) == [
        "k_cal",
        "k_repeat",
        "k_run_by_arm",
        "ci_level",
        "resamples",
        "bootstrap_seed",
        "correction",
        "m",
        "mde_formula",
        "mde_measured",
        "tau_sq",
        "sigma_sq",
        "completeness_rule",
        "completeness_backstop",
    ]
    assert list(payload["gepa_sizing"]) == [
        "steps_per_run",
        "wall_seconds",
        "sqlite_bytes",
        "max_metric_calls_pinned",
    ]
    assert list(payload["fanout_check"]) == [
        "passed",
        "minibatch_intents",
        "full_valset_intents",
    ]
    assert list(payload["amendments"][0]) == [
        "at",
        "amended_stage",
        "reason",
        "from_transport",
        "to_transport",
        "dropped_stages",
        "dropped_run_ids",
        "dropped_run_directories",
        "dropped_selections",
        "dropped_held_out_claims",
        "dropped_held_out_rows",
        "dropped_call_count_gate",
        "dropped_official_scores",
        "dropped_report_spend",
    ]
    assert list(payload["call_count_gate"]) == [
        "stage",
        "passed",
        "tolerance",
        "overruns",
    ]
    assert list(payload["arms"][0]) == [
        "arm_id",
        "optimizer",
        "demo_mode",
        "train_size",
        "val_size",
        "minibatch",
        "minibatch_size",
        "control_identity_hash",
        "seed_note",
        "runs",
    ]
    assert list(payload["arms"][0]["runs"][0]) == [
        "run_id",
        "seed",
        "artifact_dir",
        "result_ref",
        "audit_ref",
        "cost_ref",
        "audit_passed",
        "spend",
        "transport",
    ]
    assert list(payload["arms"][0]["runs"][0]["spend"][0]) == [
        "role",
        "calls",
        "cached_calls",
        "input_tokens",
        "output_tokens",
        "priced_calls",
        "unpriced_calls",
        "rows_missing_token_breakdown",
        "usd",
    ]
    assert list(payload["report_spend"][0]) == [
        "evidence_schema",
        "evidence_content_hash",
        "purpose",
        "candidate_name",
        "stage",
        "transport",
        "spend",
    ]
    assert list(payload["official_scores"][0]) == [
        "run_id",
        "arm_id",
        "stage",
        "transport",
        "score",
        "eval_config_hash",
        "completeness",
        "per_task",
    ]
    assert list(payload["selection"][0]) == [
        "arm_id",
        "selected_run_id",
        "official_score",
        "rule",
        "stage",
    ]
    assert list(payload["held_out_claims"][0]) == [
        "candidate_name",
        "stage",
        "eval_config_hash",
        "repeats",
        "mean",
        "completeness",
        "per_task",
        "per_task_counts",
    ]
    assert list(payload["held_out"][0]) == [
        "candidate_name",
        "eval_evidence_ref",
        "per_task_scores_ref",
        "mean",
        "ci_low",
        "ci_high",
        "delta_vs_naive",
        "p_bootstrap",
        "p_holm",
        "completeness",
        "anchor_completeness",
    ]
    assert list(payload["balance"]) == [
        "before_stage0_usd",
        "before_stage1_usd",
        "before_stage2_usd",
        "after_usd",
    ]
    assert list(payload["leakage_check"]) == ["passed", "checks"]
    assert list(payload["c18"]) == ["runs", "adapter_swap"]


def test_evidence_pointer_wire_keys_are_pinned() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    pointer = payload["arms"][0]["runs"][0]["result_ref"]
    assert list(pointer) == ["schema_name", "content_hash"]


# --------------------------------------------------------------------------
# Schema behaviour
# --------------------------------------------------------------------------


def test_manifest_forbids_unknown_fields() -> None:
    payload = _minimal_manifest().model_dump(mode="json", by_alias=True)
    payload["unknown"] = True
    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_manifest_rejects_a_foreign_schema() -> None:
    payload = _minimal_manifest().model_dump(mode="json", by_alias=True)
    payload["schema"] = "whetstone_envs.step10_study/v10"
    with pytest.raises(ValidationError, match="expected schema"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_manifest_rejects_overlapping_splits() -> None:
    shared = _hash("1")
    with pytest.raises(ValidationError, match="share 1 task hashes"):
        SplitsRecord(
            internal=SplitRecord(
                size=1, task_hashes=(shared,), eval_config_hash="a"
            ),
            official=SplitRecord(
                size=1, task_hashes=(shared,), eval_config_hash="b"
            ),
            held_out=SplitRecord(
                size=1, task_hashes=(_hash("2"),), eval_config_hash="c"
            ),
        )


def test_split_size_must_equal_its_task_hash_count() -> None:
    with pytest.raises(ValidationError, match="size is its task-hash count"):
        SplitRecord(size=2, task_hashes=(_hash("1"),), eval_config_hash="a")


def test_selection_is_at_most_once_per_arm() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    payload["selection"].append(dict(payload["selection"][0]))
    with pytest.raises(ValidationError, match="selected at most once"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_selection_must_name_a_run_the_arm_actually_ran() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    payload["selection"][0]["selected_run_id"] = "copro-99"
    with pytest.raises(ValidationError, match="which it did not run"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_selection_rule_is_the_pre_registered_one() -> None:
    with pytest.raises(ValidationError, match="pre-registered"):
        SelectionRecord(
            arm_id="copro",
            selected_run_id="copro-1",
            official_score=0.4,
            rule="best-on-held-out",
        )


def test_a_candidate_reaches_held_out_only_once() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    payload["held_out"].append(dict(payload["held_out"][0]))
    with pytest.raises(ValidationError, match="held-out once"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_leakage_verdict_is_the_conjunction_of_its_checks() -> None:
    with pytest.raises(ValidationError, match="conjunction"):
        LeakageCheckRecord(
            passed=True,
            checks=(
                LeakageCheckEntry(
                    check_id="L1", passed=False, detail="a leak"
                ),
            ),
        )


def test_evidence_pointer_requires_a_full_sha256_hash() -> None:
    with pytest.raises(ValidationError, match="full SHA-256 hex"):
        EvidencePointer(schema_name="s", content_hash="abc")


def test_evidence_pointer_requires_lowercase_hex() -> None:
    with pytest.raises(ValidationError, match="lowercase hex"):
        EvidencePointer(schema_name="s", content_hash="A" * 64)


def test_manifest_rejects_nonfinite_numbers() -> None:
    payload = _full_manifest().model_dump(mode="json", by_alias=True)
    payload["design"]["mde_measured"] = float("nan")
    with pytest.raises(ValidationError):
        StudyManifest.model_validate(payload)


# --------------------------------------------------------------------------
# Writer and reader
# --------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    manifest = _full_manifest()
    path = write_study_manifest(tmp_path, manifest)
    assert path == study_manifest_path(tmp_path)
    assert read_study_manifest(tmp_path) == manifest
    assert read_study_manifest(path) == manifest


def test_written_document_carries_the_pinned_schema(tmp_path: Path) -> None:
    write_study_manifest(tmp_path, _minimal_manifest())
    payload = json.loads(
        study_manifest_path(tmp_path).read_text(encoding="utf-8")
    )
    assert payload["schema"] == STUDY_MANIFEST_SCHEMA


def test_writing_twice_refuses_to_overwrite(tmp_path: Path) -> None:
    write_study_manifest(tmp_path, _minimal_manifest())
    with pytest.raises(ManifestExistsError) as caught:
        write_study_manifest(tmp_path, _full_manifest())
    assert caught.value.path == study_manifest_path(tmp_path)
    # The refusal left the first manifest exactly as written.
    assert read_study_manifest(tmp_path) == _minimal_manifest()


def test_replace_is_explicit(tmp_path: Path) -> None:
    write_study_manifest(tmp_path, _minimal_manifest())
    write_study_manifest(tmp_path, _full_manifest(), replace=True)
    assert read_study_manifest(tmp_path) == _full_manifest()


def test_writing_inside_the_repository_is_refused() -> None:
    from pathlib import Path as _Path

    inside = _Path(__file__).parent
    with pytest.raises(ValueError, match="must not be written inside"):
        write_study_manifest(inside, _minimal_manifest())


def test_reading_a_noncanonical_manifest_fails(tmp_path: Path) -> None:
    """A hand-edited manifest fails before its schema is even consulted."""
    write_study_manifest(tmp_path, _full_manifest())
    path = study_manifest_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"][0]["rule"] = "best-on-held-out"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DocumentReadError):
        read_study_manifest(tmp_path)


def test_reading_a_canonically_rewritten_manifest_fails_validation(
    tmp_path: Path,
) -> None:
    """Restoring canonical form does not restore schema validity."""
    write_study_manifest(tmp_path, _full_manifest())
    path = study_manifest_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"][0]["rule"] = "best-on-held-out"
    path.write_bytes(rfc8785.dumps(payload))
    with pytest.raises(ValidationError, match="pre-registered"):
        read_study_manifest(tmp_path)


# --------------------------------------------------------------------------
# Pointer checking
# --------------------------------------------------------------------------


def _store_for(manifest: StudyManifest) -> _FakeStore:
    return _FakeStore(
        {
            (pointer.schema_name, pointer.content_hash): {"ok": True}
            for pointer in manifest.evidence_pointers()
        }
    )


def test_evidence_pointers_are_walked_in_document_order() -> None:
    manifest = _full_manifest()
    assert [
        (pointer.schema_name, pointer.content_hash[0])
        for pointer in manifest.evidence_pointers()
    ] == [
        (RESULT_SCHEMA, "5"),
        (AUDIT_SCHEMA, "6"),
        (COST_SCHEMA, "e"),
        (EVIDENCE_SCHEMA, "7"),
        (SCORES_SCHEMA, "8"),
        (RESULT_SCHEMA, "9"),
        (AUDIT_SCHEMA, "a"),
        (COST_SCHEMA, "b"),
    ]


def test_every_pointer_resolves_against_a_complete_store() -> None:
    manifest = _full_manifest()
    report = check_manifest_pointers(manifest, _store_for(manifest))
    assert report.passed
    assert len(report.checks) == len(manifest.evidence_pointers())
    assert report.unresolved() == ()


def test_a_mutated_pointer_does_not_resolve() -> None:
    manifest = _full_manifest()
    store = _store_for(manifest)
    mutated = manifest.model_copy(
        update={
            "held_out": (
                manifest.held_out[0].model_copy(
                    update={
                        "eval_evidence_ref": _pointer(EVIDENCE_SCHEMA, "f")
                    }
                ),
            )
        }
    )
    report = check_manifest_pointers(mutated, store)
    assert not report.passed
    unresolved = report.unresolved()
    assert len(unresolved) == 1
    assert unresolved[0].pointer.content_hash == _hash("f")
    assert "ObjectNotFoundError" in unresolved[0].detail


def test_a_pointer_to_a_foreign_schema_does_not_resolve() -> None:
    manifest = _full_manifest()
    store = _store_for(manifest)
    mutated = manifest.model_copy(
        update={
            "held_out": (
                manifest.held_out[0].model_copy(
                    update={
                        "per_task_scores_ref": EvidencePointer(
                            schema_name="whetstone_envs.not_a_schema",
                            content_hash=_hash("8"),
                        )
                    }
                ),
            )
        }
    )
    report = check_manifest_pointers(mutated, store)
    assert not report.passed
    assert len(report.unresolved()) == 1


def test_repeated_pointers_are_resolved_once(tmp_path: Path) -> None:  # noqa: ARG001
    manifest = _full_manifest()
    shared = _pointer(RESULT_SCHEMA, "5")
    arm = manifest.arms[0]
    duplicated = arm.model_copy(
        update={
            "runs": (
                arm.runs[0],
                arm.runs[0].model_copy(
                    update={"run_id": "copro-2", "result_ref": shared}
                ),
            )
        }
    )
    mutated = manifest.model_copy(update={"arms": (duplicated,)})
    calls: list[str] = []

    class _CountingStore(_FakeStore):
        def get(self, reference: ObjectReference) -> JsonValue:
            calls.append(reference.content_hash)
            return super().get(reference)

    store = _CountingStore(
        {
            (pointer.schema_name, pointer.content_hash): {"ok": True}
            for pointer in mutated.evidence_pointers()
        }
    )
    report = check_manifest_pointers(mutated, store)
    assert report.passed
    assert calls.count(_hash("5")) == 1


def test_pointer_report_formats_one_line_per_pointer() -> None:
    manifest = _full_manifest()
    report = check_manifest_pointers(manifest, _store_for(manifest))
    lines = list(format_pointer_report(report))
    assert len(lines) == len(report.checks)
    assert all(line.startswith("ok ") for line in lines)


# --------------------------------------------------------------------------
# Held-out claims: L3's guard, durable before the evaluation
# --------------------------------------------------------------------------


def test_a_candidate_claims_held_out_at_most_once() -> None:
    """L3's guard, at the moment the evaluation is issued."""
    payload = _minimal_manifest().model_dump(mode="json", by_alias=True)
    payload["held_out_claims"] = [
        {"candidate_name": "naive"},
        {"candidate_name": "naive"},
    ]
    with pytest.raises(ValidationError, match="claims held-out at most once"):
        StudyManifest.model_validate_json(json.dumps(payload))


def test_an_outstanding_claim_is_a_crashed_evaluation_not_a_missing_one(
    tmp_path: Path,
) -> None:
    """The claim is written before the call, so the two states differ."""
    outstanding = HeldOutClaimRecord(candidate_name="naive")
    assert not outstanding.completed
    manifest = _minimal_manifest().model_copy(
        update={"held_out_claims": (outstanding,)}
    )
    write_study_manifest(tmp_path, manifest)
    assert read_study_manifest(tmp_path) == manifest


def test_half_a_measurement_is_refused() -> None:
    with pytest.raises(ValidationError, match="outstanding or fully"):
        HeldOutClaimRecord(candidate_name="naive", mean=0.4)


def test_a_held_out_row_needs_a_completed_claim_behind_it() -> None:
    """A reported number whose evaluation was never claimed came from
    outside ``report_arm``, which is exactly the leak L3 catches."""
    base = _full_manifest()
    with pytest.raises(ValidationError, match="without a completed claimed"):
        StudyManifest.model_validate_json(
            base.model_copy(update={"held_out_claims": ()}).model_dump_json(
                by_alias=True
            )
        )


def test_a_row_backed_only_by_an_outstanding_claim_is_refused() -> None:
    """A crashed evaluation cannot be reported as a measured one."""
    base = _full_manifest()
    with pytest.raises(ValidationError, match="without a completed claimed"):
        StudyManifest.model_validate_json(
            base.model_copy(
                update={
                    "held_out_claims": (
                        HeldOutClaimRecord(
                            candidate_name="copro-representative"
                        ),
                    )
                }
            ).model_dump_json(by_alias=True)
        )


# --------------------------------------------------------------------------
# The pre-registration is immutable once pinned
# --------------------------------------------------------------------------

_PINNED_K_RUN_BY_ARM = {"copro": 5, "gepa": 5}

#: COPRO has no train/val concept and GEPA does, so the pinned block below
#: exercises both shapes of the per-arm split at once.
_PINNED_SPLIT_BY_ARM: dict[str, tuple[int, int] | None] = {
    "copro": None,
    "gepa": (44, 44),
}


def _pre_registration(
    *,
    k_repeat: int = 3,
    split_by_arm: dict[str, tuple[int, int] | None] | None = None,
    provenance: str = PROVENANCE_ORIGINAL,
    amended_from: str | None = None,
) -> PreRegistrationRecord:
    """A pinned block whose hash actually covers its own fields."""
    splits = dict(
        _PINNED_SPLIT_BY_ARM if split_by_arm is None else split_by_arm
    )
    # No arm in this fixture minibatches; the block still names every arm,
    # because "this arm does not minibatch" is itself pre-registered.
    minibatch: dict[str, int | None] = dict.fromkeys(_PINNED_K_RUN_BY_ARM)
    return PreRegistrationRecord(
        k_repeat=k_repeat,
        k_run_by_arm=dict(_PINNED_K_RUN_BY_ARM),
        split_by_arm=splits,
        minibatch_by_arm=minibatch,
        ci_level=0.95,
        resamples=10_000,
        bootstrap_seed=0,
        correction=CORRECTION_HOLM_BONFERRONI,
        m=CORRECTION_FAMILY_SIZE,
        completeness_backstop=COMPLETENESS_BACKSTOP,
        design_hash=pre_registration_design_hash(
            k_repeat=k_repeat,
            k_run_by_arm=dict(_PINNED_K_RUN_BY_ARM),
            split_by_arm=splits,
            minibatch_by_arm=minibatch,
            ci_level=0.95,
            resamples=10_000,
            bootstrap_seed=0,
            correction=CORRECTION_HOLM_BONFERRONI,
            m=CORRECTION_FAMILY_SIZE,
            completeness_backstop=COMPLETENESS_BACKSTOP,
        ),
        provenance=provenance,
        amended_from=amended_from,
    )


def _pinned_manifest() -> StudyManifest:
    return _minimal_manifest().model_copy(
        update={"pre_registration": _pre_registration()}
    )


def test_a_pre_registration_hash_covers_its_own_fields() -> None:
    record = _pre_registration()
    assert record.provenance == PROVENANCE_ORIGINAL
    assert record.pinned_fields()["k_repeat"] == 3
    assert record.pinned_fields()["m"] == CORRECTION_FAMILY_SIZE


def test_the_hashed_payload_keys_are_pinned() -> None:
    """The hashed document is stored identity, so its keys are literals.

    ``split_by_arm`` and ``minibatch_by_arm`` are in the list and an
    authorization to spend is not: the partition an arm was measured at
    and the batch each trial was scored on are design, and whether the
    operator was allowed to bill a Codex session for this invocation is
    not.
    """
    assert list(_pre_registration().pinned_fields()) == [
        "k_repeat",
        "k_run_by_arm",
        "split_by_arm",
        "minibatch_by_arm",
        "ci_level",
        "resamples",
        "bootstrap_seed",
        "correction",
        "m",
        "completeness_backstop",
    ]


def test_the_hashed_payload_writes_each_split_as_a_pair() -> None:
    """The per-arm split's wire shape, pinned as a literal."""
    assert _pre_registration().pinned_fields()["split_by_arm"] == {
        "copro": None,
        "gepa": [44, 44],
    }


def test_changing_a_split_size_changes_the_design_hash() -> None:
    """Fails-before evidence that the partition is actually pinned.

    Before the split entered the payload, an arm rerun at a different
    train/val partition produced a byte-identical design hash, so the
    pre-registration certified a design the study had not run.
    """
    assert (
        _pre_registration(
            split_by_arm={"copro": None, "gepa": (40, 48)}
        ).design_hash
        != _pre_registration().design_hash
    )


def test_a_pre_registration_whose_hash_does_not_cover_it_is_refused() -> None:
    """The hash is the pinning; a block whose hash drifted pins nothing."""
    payload = _pre_registration().model_dump(mode="json")
    payload["design_hash"] = _hash("f")
    with pytest.raises(ValidationError, match="does not cover its own"):
        PreRegistrationRecord.model_validate_json(json.dumps(payload))


def test_changing_a_pinned_field_changes_the_design_hash() -> None:
    assert (
        _pre_registration(k_repeat=5).design_hash
        != _pre_registration().design_hash
    )


def test_a_later_write_may_not_change_the_pre_registration(
    tmp_path: Path,
) -> None:
    """The load-bearing refusal: a design fixed before spend stays fixed."""
    write_study_manifest(tmp_path, _pinned_manifest())
    restated = _minimal_manifest().model_copy(
        update={"pre_registration": _pre_registration(k_repeat=9)}
    )
    with pytest.raises(PreRegistrationViolationError, match="k_repeat"):
        write_study_manifest(tmp_path, restated, replace=True)
    # The document on disk is untouched by the refused write.
    assert read_study_manifest(tmp_path).pre_registration == (
        _pre_registration()
    )


def test_a_later_write_may_not_change_a_pinned_split_size(
    tmp_path: Path,
) -> None:
    """The same refusal, for the field the hash did not used to cover.

    Fails-before: with ``split_by_arm`` outside the hashed payload this
    write was accepted, because the restated block hashed identically to
    the pinned one.
    """
    write_study_manifest(tmp_path, _pinned_manifest())
    restated = _minimal_manifest().model_copy(
        update={
            "pre_registration": _pre_registration(
                split_by_arm={"copro": None, "gepa": (40, 48)}
            )
        }
    )
    with pytest.raises(PreRegistrationViolationError, match="split_by_arm"):
        write_study_manifest(tmp_path, restated, replace=True)
    assert read_study_manifest(tmp_path).pre_registration == (
        _pre_registration()
    )


def test_a_later_write_may_not_drop_the_pre_registration(
    tmp_path: Path,
) -> None:
    write_study_manifest(tmp_path, _pinned_manifest())
    with pytest.raises(PreRegistrationViolationError, match="drops the block"):
        write_study_manifest(tmp_path, _minimal_manifest(), replace=True)


def test_an_unchanged_pre_registration_writes_through(tmp_path: Path) -> None:
    """Stages 1 and 2 rewrite the manifest constantly; that must still work."""
    write_study_manifest(tmp_path, _pinned_manifest())
    again = _pinned_manifest().model_copy(
        update={"created_at": "2026-08-23T00:00:00+00:00"}
    )
    write_study_manifest(tmp_path, again, replace=True)
    assert read_study_manifest(tmp_path).created_at.startswith("2026-08-23")


def test_an_amendment_names_the_design_hash_it_replaced(
    tmp_path: Path,
) -> None:
    write_study_manifest(tmp_path, _pinned_manifest())
    original_hash = _pre_registration().design_hash
    amended = _minimal_manifest().model_copy(
        update={
            "pre_registration": _pre_registration(
                k_repeat=9,
                provenance=PROVENANCE_AMENDED,
                amended_from=original_hash,
            )
        }
    )
    write_study_manifest(
        tmp_path, amended, replace=True, amend_pre_registration=True
    )
    written = read_study_manifest(tmp_path).pre_registration
    assert written is not None
    assert written.provenance == PROVENANCE_AMENDED
    assert written.amended_from == original_hash


def test_an_amendment_that_names_the_wrong_predecessor_is_refused(
    tmp_path: Path,
) -> None:
    write_study_manifest(tmp_path, _pinned_manifest())
    amended = _minimal_manifest().model_copy(
        update={
            "pre_registration": _pre_registration(
                k_repeat=9,
                provenance=PROVENANCE_AMENDED,
                amended_from=_hash("9"),
            )
        }
    )
    with pytest.raises(
        PreRegistrationViolationError, match="names the design"
    ):
        write_study_manifest(
            tmp_path, amended, replace=True, amend_pre_registration=True
        )


def test_an_amendment_without_its_provenance_is_refused() -> None:
    """An amendment that names no predecessor erases what it amended."""
    payload = _pre_registration().model_dump(mode="json")
    payload["provenance"] = PROVENANCE_AMENDED
    with pytest.raises(ValidationError, match="names the design hash it"):
        PreRegistrationRecord.model_validate_json(json.dumps(payload))


def test_a_design_contradicting_the_pre_registration_is_refused() -> None:
    """The overlap is checked, so pinning cannot become decorative."""
    design = DesignRecord(
        k_cal=4,
        k_repeat=9,
        k_run_by_arm={"copro": 5, "gepa": 5},
        ci_level=0.95,
        resamples=10_000,
        bootstrap_seed=0,
        correction=CORRECTION_HOLM_BONFERRONI,
        m=CORRECTION_FAMILY_SIZE,
        mde_formula="z-based",
        mde_measured=0.1,
        tau_sq=0.01,
        sigma_sq=0.02,
        completeness_rule="achieved rows / scheduled rows",
        completeness_backstop=COMPLETENESS_BACKSTOP,
    )
    payload = _pinned_manifest().model_dump(mode="json", by_alias=True)
    payload["design"] = design.model_dump(mode="json")
    with pytest.raises(ValidationError, match="contradicts the pinned"):
        StudyManifest.model_validate_json(json.dumps(payload))


# --------------------------------------------------------------------------
# The effective provider call config (Phase E item 4)
# --------------------------------------------------------------------------


def _provider_call(**overrides: str) -> ProviderCallRecord:
    fields: dict[str, str] = {
        "transport": "openrouter",
        "provider": "openrouter",
        "protocol": "chat_completions",
        "model_route": "openai/gpt-5-nano",
        "temperature": PROVIDER_CONTROL_UNSET,
        "top_p": PROVIDER_CONTROL_UNSET,
        "token_limit": PROVIDER_CONTROL_UNSET,
        "reasoning": PROVIDER_CONTROL_UNSET,
        "seed": "7",
        "extensions": "{}",
    }
    fields.update(overrides)
    return ProviderCallRecord(**fields)


def test_provider_call_wire_keys_are_pinned() -> None:
    """Persisted-format keys, pinned rather than derived from field names."""
    assert list(_provider_call().model_dump(mode="json")) == [
        "transport",
        "provider",
        "protocol",
        "model_route",
        "temperature",
        "top_p",
        "token_limit",
        "reasoning",
        "seed",
        "extensions",
    ]


def test_an_unset_control_is_named_not_omitted() -> None:
    """ "Provider default" is a state, and a consequential one.

    It is why the toy Stage 0 billed thousands of reasoning tokens per
    call, so it is stated rather than left to be inferred from a blank or
    a zero -- both of which read as "the study set this".
    """
    assert PROVIDER_CONTROL_UNSET == "provider default"
    with pytest.raises(ValidationError, match="nonblank statement"):
        _provider_call(reasoning="  ")


def test_each_transport_records_its_config_once() -> None:
    """Two configs for one transport leaves the study unable to say which.

    Fails-before: there was no such block at all -- the manifest named the
    model a study meant to run and the transport it ran on, but never what
    the transport actually bound, so neither the spend model nor the "same
    experiment" claim was auditable from the manifest.
    """
    payload = _full_manifest().models.model_dump(mode="json")

    def _with(*calls: ProviderCallRecord) -> str:
        # Via JSON, like every other round trip here: strict mode reads a
        # JSON array as a tuple but refuses a Python list.
        return json.dumps(
            payload
            | {
                "provider_calls": [
                    call.model_dump(mode="json") for call in calls
                ]
            }
        )

    # Two transports, two configs: the ordinary state of a study that
    # calibrated free and then ran paid.
    ModelsRecord.model_validate_json(
        _with(
            _provider_call(),
            _provider_call(transport="fake", provider="openai"),
        )
    )

    with pytest.raises(ValidationError, match="records its provider call"):
        ModelsRecord.model_validate_json(
            _with(
                _provider_call(),
                _provider_call(model_route="openai/gpt-4.1-nano"),
            )
        )


def test_models_default_to_no_recorded_provider_call() -> None:
    """Empty until a stage binds one, rather than a fabricated default."""
    assert _full_manifest().models.provider_calls == ()
