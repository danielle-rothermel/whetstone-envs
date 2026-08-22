"""Turn a study arm into a real run, an official score, and a held-out number.

:mod:`~whetstone_envs.optim.study.stages` takes its three provider-touching
collaborators as callables and refuses to import one. This module is where
they actually come from, and it is the seam where the study meets the shared
optimizer runner:

* :class:`StudyOptimizerRunner` drives one arm at one seed through
  :func:`~whetstone_envs.optim.run.run_optimizer`, audits the run it
  produced, copies the run's own evidence into the study's store, and
  returns the :class:`~whetstone_envs.optim.study.stages.ArmRunResult` the
  manifest records.
* :class:`RoleScorer` evaluates one candidate on one role's split through
  the *same* engine binder Stage 0 calibrated its anchors with, which is
  what makes L4's identical-procedure rule true rather than asserted.

**The transport is the study's, not this module's.** A runner is built for a
named transport and passes it straight to ``RunSpec``; nothing here reaches a
provider by default, and the fake path is fully operational end to end.

**Nulls run no optimizer.** ``null-identity`` returns the naive anchor
unchanged and ``null-random`` perturbs it through the family's own render
contract. Both still produce a real run record, a real audit verdict, and a
real held-out measurement through the identical procedure -- that is the
whole point of a control -- so they are built here beside the optimizers
rather than special-cased inside the stage harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dr_store.sync import open_sqlite
from whetstone.core.roles import EvalRole
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.schema import EvalEvidence
from whetstone.optim.contracts import OptimResult

from whetstone_envs.optim.audit.registry import audit_run
from whetstone_envs.optim.audit.schema import AUDIT_REPORT_SCHEMA
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.run import RunSpec, run_optimizer
from whetstone_envs.optim.run_cost import (
    RUN_COST_SCHEMA_NAME,
    project_run_cost,
)
from whetstone_envs.optim.study.anchors import EngineBinder
from whetstone_envs.optim.study.fanout import planned_rows_in_directory
from whetstone_envs.optim.study.gates import GEPA_MAX_METRIC_CALLS_PINNED
from whetstone_envs.optim.study.manifest import (
    EvidencePointer,
    RunRecord,
)
from whetstone_envs.optim.study.selection import (
    CandidateScore,
    HeldOutMeasurement,
    RunCandidate,
)
from whetstone_envs.optim.study.stages import ArmRunResult, StageError

if TYPE_CHECKING:
    from pydantic import JsonValue
    from whetstone.experiment.candidate import Candidate

    from whetstone_envs.optim.study.spec import ArmSpec

__all__ = [
    "NULL_IDENTITY_OPTIMIZER",
    "NULL_RANDOM_OPTIMIZER",
    "OPTIMIZER_ARM_IDS",
    "RoleScorer",
    "StudyOptimizerRunner",
    "arm_run_directory",
    "null_random_template",
]

#: The two controls, named where they are dispatched on. They are arm ids
#: and optimizer names at once, because a null is defined by what it does
#: rather than by an optimizer that implements it.
NULL_IDENTITY_OPTIMIZER = "null-identity"
NULL_RANDOM_OPTIMIZER = "null-random"

#: The arms whose runs go through the shared optimizer runner. Anything
#: outside this set and the two nulls is refused rather than silently
#: dispatched, so a new arm id cannot quietly run as a control.
OPTIMIZER_ARM_IDS: frozenset[str] = frozenset(
    {"copro", "miprov2", "gepa", "codex"}
)

#: Where a study keeps the run directories its arms produce. One directory
#: per run beneath it, so a resumed stage can find what it already paid for.
RUNS_DIRECTORY_NAME = "runs"

#: The evaluation purpose recorded on a selection score and on a held-out
#: measurement. These reach persisted evidence metadata, so they are owned
#: constants rather than inline strings.
OFFICIAL_SELECTION_PURPOSE = "study-official-selection"
HELD_OUT_REPORT_PURPOSE = "study-held-out-report"


def arm_run_directory(study_dir: Path, run_id: str) -> Path:
    """Where one run's artifacts live inside a study directory."""
    return study_dir / RUNS_DIRECTORY_NAME / run_id


def _run_id_for(arm: ArmSpec, seed: int) -> str:
    """The deterministic run id for one arm at one seed.

    Deterministic on purpose: a resumed stage recognises a run it already
    paid for by finding its directory, and a random id would make every
    resume re-run everything.
    """
    return f"{arm.arm_id}-seed{seed}"


# --------------------------------------------------------------------------
# Scoring one candidate on one role
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoleScorer:
    """Evaluate candidates on one role's split, through one bound engine.

    The engine comes from the same :class:`EngineBinder` Stage 0 calibrated
    its anchors with, so a selection score, a held-out number, and an anchor
    are produced by one procedure over one split -- which is exactly what L4
    checks after the fact and what this construction makes true beforehand.

    ``num_seeds`` is the design's ``K_REPEAT``, not the calibration's
    ``K_CAL``: what is being measured here is the design, and borrowing the
    calibration's repeat count would report a number the design never
    specified.
    """

    bind_engine: EngineBinder
    role: EvalRole
    task_ids: tuple[str, ...]
    num_seeds: int
    build_candidate: BuildCandidate

    def evidence_for(
        self, *, candidate_name: str, template: str, purpose: str
    ) -> EvalEvidence:
        """Evaluate one template on this role's split, once.

        Returns the persisted evidence rather than a scalar, because every
        caller needs the per-task vector and the row accounting: a mean
        without its completeness cannot be judged against the backstop.
        """
        engine = self.bind_engine(role=self.role, num_seeds=self.num_seeds)
        subset = engine.for_task_ids(self.task_ids)
        result = subset.evaluate(
            EvalRequest(
                request_id=f"{purpose}:{candidate_name}",
                candidate=self.build_candidate(candidate_name, template),
                metadata=_purpose_metadata(purpose),
            )
        )
        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, EvalEvidence):
            # A failed or rejected evaluation carries a failure record, not
            # evidence. Selecting or reporting from one would be reporting a
            # number that was never measured.
            raise StageError(
                f"candidate {candidate_name!r} produced no successful "
                f"evaluation on the {self.role.value} split: "
                f"{type(evidence).__name__}"
            )
        return evidence

    def eval_config_hash(self) -> str:
        """This role's Eval Config hash, without evaluating anything."""
        engine = self.bind_engine(role=self.role, num_seeds=self.num_seeds)
        subset = engine.for_task_ids(self.task_ids)
        return str(subset.eval_config_ref.config_hash)

    def score_official(self, candidate: RunCandidate) -> CandidateScore:
        """The official scorer ``report_arm`` selects on."""
        if self.role is not EvalRole.OFFICIAL:
            raise StageError(
                "selection scores on the official split; this scorer is "
                f"bound to {self.role.value}"
            )
        evidence = self.evidence_for(
            candidate_name=candidate.candidate_name,
            template=candidate.template,
            purpose=OFFICIAL_SELECTION_PURPOSE,
        )
        return CandidateScore(
            run_id=candidate.run_id,
            score=_mean_of(evidence),
            per_task=evidence.per_task_values,
            eval_config_hash=str(evidence.eval_config_ref.config_hash),
            completeness=_completeness_of(evidence),
        )

    def evaluate_held_out(
        self, *, candidate_name: str, template: str
    ) -> HeldOutMeasurement:
        """The held-out evaluator ``report_arm`` measures through."""
        if self.role is not EvalRole.HELD_OUT:
            raise StageError(
                "held-out numbers come from the held-out split; this "
                f"scorer is bound to {self.role.value}"
            )
        evidence = self.evidence_for(
            candidate_name=candidate_name,
            template=template,
            purpose=HELD_OUT_REPORT_PURPOSE,
        )
        return HeldOutMeasurement(
            candidate_name=candidate_name,
            per_task=evidence.per_task_values,
            mean=_mean_of(evidence),
            eval_config_hash=str(evidence.eval_config_ref.config_hash),
            repeats=evidence.num_seeds,
            completeness=_completeness_of(evidence),
            # The measured per-task row counts, so O7's weighting reads
            # what each task actually achieved rather than spreading one
            # study-wide completeness evenly across tasks that did not
            # fail evenly.
            per_task_counts=evidence.per_task_counts,
        )


def _purpose_metadata(purpose: str) -> dict[str, str]:
    """The purpose stamp carried onto persisted evidence."""
    return {"purpose": purpose}


def _mean_of(evidence: EvalEvidence) -> float:
    """The evidence's aggregate, or its per-task mean when absent.

    The aggregate is preferred because it is what whetstone computed and
    persisted; the fallback exists so an evidence record whose aggregate
    could not be formed still reports the number its rows support rather
    than failing the whole stage.
    """
    if evidence.aggregate_value is not None:
        return float(evidence.aggregate_value)
    values = evidence.per_task_values
    if not values:
        raise StageError("an evaluation produced no per-task values")
    return sum(values) / len(values)


def _completeness_of(evidence: EvalEvidence) -> float:
    """Achieved rows over scheduled rows, as the design's rule defines it.

    A run whose provider dropped rows has measured less of the split than
    it planned to, and the completeness backstop is what stops that from
    silently becoming a claim.
    """
    accounting = evidence.row_accounting
    if accounting.planned <= 0:
        return 0.0
    return accounting.present / accounting.planned


# --------------------------------------------------------------------------
# Running one arm
# --------------------------------------------------------------------------


class BuildCandidate:
    """Build a family candidate from a name and a template.

    A tiny callable protocol rather than a direct family import, so a
    scorer stays testable and the family lookup happens once per study.
    """

    def __init__(self, family_id: str) -> None:
        self._family = family_spec(family_id)

    def __call__(self, candidate_id: str, template: str) -> Candidate:
        contract = self._family.render_contract()
        contract.validate_template(template)
        return self._family.build_candidate(
            candidate_id=candidate_id, template=template
        )


def null_random_template(*, naive_template: str, seed: int) -> str:
    """A null-A perturbation of the naive prompt, seeded and reproducible.

    Null-A asks what best-on-official selection produces when the
    "optimizer" contributes nothing but variation, so the perturbation must
    change the prompt's surface without changing what it asks for. Reordering
    nothing and appending a seeded, meaning-free suffix does exactly that,
    and it keeps the family's placeholders untouched, so the render contract
    still validates it.
    """
    # A deterministic, meaning-free suffix: the same seed always produces
    # the same prompt, which is what makes a null run reproducible.
    return f"{naive_template}\n\n(variant {seed})"


@dataclass(frozen=True, slots=True)
class StudyOptimizerRunner:
    """Run one arm at one seed and record what it produced.

    This satisfies :class:`~whetstone_envs.optim.study.stages.OptimizerRunner`
    and is the only place the study reaches the shared optimizer runner. It
    owns four things the stage harness deliberately does not: the run's
    artifact directory, its audit, the projection of its cost, and the copy
    of its evidence into the study's own store.

    **Evidence is copied, not referenced across stores.** A run writes its
    result, cost, and audit into its own ``runtime.sqlite``; the manifest's
    pointers are resolved against the *study's* store by ``manifest check``
    and by the report. Copying the three records the manifest cites keeps
    both true without making the study's store a second copy of every
    evaluation row.
    """

    study_dir: Path
    family_id: str
    transport: str
    split_sizes: tuple[int, int, int]
    n_per_stratum: int
    pool_seed_start: int
    task_model: str
    proposer_model: str
    num_seeds: int
    naive_template: str
    store_path: Path
    codex_capacity: int | None = None
    #: The run-time half of the real-Codex spend authorization, carried
    #: from ``whetstone-study run --allow-real-codex``.
    #:
    #: This is an *authorization to spend on this invocation*, not part of
    #: the study's design: it is deliberately not an ``ArmSpec`` field and
    #: never enters the pre-registration hash, because whether a stage was
    #: allowed to bill a Codex session says nothing about what the study
    #: pre-registered. Forwarded onto the Codex arm's ``RunSpec`` only, and
    #: still only half the gate --
    #: :data:`~whetstone_envs.optim.codex.ALLOW_REAL_CODEX_ENV` must also
    #: name the opt-in in the process environment.
    allow_real_codex: bool = False

    def __call__(
        self, *, arm: ArmSpec, seed: int, study_dir: Path
    ) -> ArmRunResult:
        run_id = _run_id_for(arm, seed)
        run_dir = arm_run_directory(study_dir, run_id)
        if arm.optimizer in {NULL_IDENTITY_OPTIMIZER, NULL_RANDOM_OPTIMIZER}:
            return self._run_null(arm=arm, seed=seed, run_id=run_id)
        if arm.optimizer not in OPTIMIZER_ARM_IDS:
            known = sorted(OPTIMIZER_ARM_IDS)
            raise StageError(
                f"arm {arm.arm_id!r} names optimizer {arm.optimizer!r}, "
                f"which is neither a study optimizer {known} nor a null; "
                "refusing rather than guessing how to run it"
            )
        if not run_dir.exists():
            run_optimizer(self._spec_for(arm, seed=seed, run_dir=run_dir))
        return self._result_from(arm=arm, seed=seed, run_dir=run_dir)

    def load_recorded_run(
        self, *, arm: ArmSpec, run: RunRecord
    ) -> ArmRunResult | None:
        """Re-read a run an earlier stage recorded, or report it gone.

        This is what makes Stage 2 continue from Stage 1 rather than refuse:
        the terminal candidate of a run this process never executed is read
        back out of the run's own artifacts, so the arg-max covers the arm's
        whole ``K_RUN`` without re-paying for anything.

        A control has no run directory to re-read -- its whole evidence is
        the record already in the study's store -- so it is rebuilt from the
        record's own seed, deterministically, which is the same template it
        produced the first time.
        """
        if run.seed is None:
            # Every run this runner records carries the seed it was asked
            # for, so a seedless record cannot be matched to one of this
            # stage's seeds. Reporting it as unloadable beats guessing.
            return None
        if arm.optimizer in {NULL_IDENTITY_OPTIMIZER, NULL_RANDOM_OPTIMIZER}:
            return self._run_null(arm=arm, seed=run.seed, run_id=run.run_id)
        run_dir = Path(run.artifact_dir)
        if not (run_dir / "result.json").is_file():
            return None
        return self._result_from(arm=arm, seed=run.seed, run_dir=run_dir)

    def _spec_for(self, arm: ArmSpec, *, seed: int, run_dir: Path) -> RunSpec:
        return RunSpec(
            optimizer=arm.optimizer,
            transport=self.transport,
            family=self.family_id,
            split_sizes=self.split_sizes,
            output_dir=run_dir,
            run_id=_run_id_for(arm, seed),
            model=self.task_model,
            proposer_model=self.proposer_model,
            num_seeds=self.num_seeds,
            n_per_stratum=self.n_per_stratum,
            pool_seed_start=self.pool_seed_start,
            seed=seed,
            **(
                {"demo_mode": arm.demo_mode}
                if arm.demo_mode is not None
                else {}
            ),
            # Only forwarded when the arm sets them, so an unset arm keeps
            # the runner's own default rather than pinning it here twice.
            **{
                field: value
                for field, value in (
                    ("miprov2_num_trials", arm.miprov2_num_trials),
                    ("miprov2_num_candidates", arm.miprov2_num_candidates),
                    ("train_size", arm.train_size),
                    ("val_size", arm.val_size),
                )
                if value is not None
            },
            # The pinned metric-call budget, not the run's default. Left
            # unset, ``build_gepa_control`` resolves ``auto`` to roughly
            # ``train + val + 1`` -- about 89 on the study's 44/44 split --
            # while the Stage-1 call-count gate and the power design are
            # both built on ``GEPA_MAX_METRIC_CALLS_PINNED``. A GEPA arm
            # that ran at the default would be judged against a ceiling it
            # never had, so the pin is forwarded here rather than left to a
            # default that agrees with it only by accident.
            **(
                {"gepa_max_metric_calls": GEPA_MAX_METRIC_CALLS_PINNED}
                if arm.optimizer == "gepa"
                else {}
            ),
            **(
                {"codex_capacity": self.codex_capacity}
                if arm.optimizer == "codex" and self.codex_capacity is not None
                else {}
            ),
            # Scoped to the Codex arm because every other optimizer refuses
            # the setting outright; forwarding it unconditionally would turn
            # one authorized stage into a validation failure on every arm.
            **(
                {"allow_real_codex": True}
                if arm.optimizer == "codex" and self.allow_real_codex
                else {}
            ),
        )

    def _result_from(
        self, *, arm: ArmSpec, seed: int, run_dir: Path
    ) -> ArmRunResult:
        """Read one completed run directory into the manifest's record.

        Split out from the run itself so a resumed stage reads a directory
        it already paid for by exactly the path a fresh run takes -- there
        is one projection from artifacts to record, not two.
        """
        result = _read_optim_result(run_dir)
        report = audit_run(run_dir)
        template = _terminal_template(result, run_dir=run_dir)
        cost = project_run_cost(result, run_id=report.run_id)
        pointers = self._copy_evidence(
            run_dir=run_dir,
            result=result,
            report_payload=report.model_dump(mode="json"),
            cost_payload=(
                None if cost is None else cost.model_dump(mode="json")
            ),
        )
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=report.run_id,
                seed=seed,
                candidate_name=f"{arm.arm_id}-{report.run_id}",
                template=template,
            ),
            record=RunRecord(
                run_id=report.run_id,
                seed=seed,
                artifact_dir=str(run_dir),
                result_ref=pointers["result"],
                audit_ref=pointers["audit"],
                cost_ref=pointers["cost"],
                audit_passed=report.passed,
                spend=(() if cost is None else tuple(cost.spend)),
                # The run's own transport, not the stage's. A resumed
                # stage keeps runs an earlier invocation paid for, so the
                # stage row and its runs can disagree -- and the
                # cross-transport refusal checks the runs.
                transport=self.transport,
            ),
            observed_task_calls=_observed_task_calls(result, run_dir=run_dir),
        )

    def _run_null(
        self, *, arm: ArmSpec, seed: int, run_id: str
    ) -> ArmRunResult:
        """A control's run: a candidate, an honest record, no optimizer.

        A null runs no search, so there is no optimizer result to audit and
        nothing an optimizer-fidelity invariant could be asked about. The
        record says so rather than borrowing an optimizer's evidence: its
        pointers address a purpose-built control record in the study's own
        store, and ``audit_passed`` is true because the control did exactly
        what a control is defined to do.
        """
        template = (
            self.naive_template
            if arm.optimizer == NULL_IDENTITY_OPTIMIZER
            else null_random_template(
                naive_template=self.naive_template, seed=seed
            )
        )
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "arm_id": arm.arm_id,
            "optimizer": arm.optimizer,
            "seed": seed,
            "template": template,
            "note": (
                "a control run: no optimizer searched, so this record is "
                "the whole of the run's evidence"
            ),
        }
        pointer = self._put(NULL_RUN_SCHEMA, record)
        return ArmRunResult(
            candidate=RunCandidate(
                run_id=run_id,
                seed=seed,
                candidate_name=f"{arm.arm_id}-{run_id}",
                template=template,
            ),
            record=RunRecord(
                run_id=run_id,
                seed=seed,
                artifact_dir=str(arm_run_directory(self.study_dir, run_id)),
                result_ref=pointer,
                audit_ref=pointer,
                cost_ref=pointer,
                audit_passed=True,
                spend=(),
                transport=self.transport,
            ),
            observed_task_calls=0,
        )

    def _copy_evidence(
        self,
        *,
        run_dir: Path,
        result: OptimResult,
        report_payload: JsonValue,
        cost_payload: JsonValue | None,
    ) -> dict[str, EvidencePointer]:
        """Copy the three records the manifest cites into the study store.

        ``cost.json`` is absent when a run carried no cost report -- a fake
        transport prices nothing -- and the manifest still needs a resolvable
        pointer, so an explicit "this run reported no cost" record is stored
        rather than a pointer to nothing.
        """
        del run_dir
        result_pointer = self._put(
            OPTIM_RESULT_COPY_SCHEMA, result.model_dump(mode="json")
        )
        audit_pointer = self._put(AUDIT_REPORT_SCHEMA, report_payload)
        if cost_payload is None:
            cost_pointer = self._put(
                RUN_COST_SCHEMA_NAME,
                {
                    "schema_version": 1,
                    "run_id": str(result.run.record.run_id),
                    "spend": [],
                    "note": "this run reported no cost",
                },
            )
        else:
            cost_pointer = self._put(RUN_COST_SCHEMA_NAME, cost_payload)
        return {
            "result": result_pointer,
            "audit": audit_pointer,
            "cost": cost_pointer,
        }

    def _put(self, schema: str, record: JsonValue) -> EvidencePointer:
        with open_sqlite(str(self.store_path)) as store:
            reference, _status = store.put(schema, record)
        return EvidencePointer(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )


#: The schema a study's copy of a run's ``OptimResult`` is stored under. It
#: is deliberately not whetstone's own ``OPTIM_RESULT_SCHEMA``: this is the
#: study's copy of a record whose authority lives in the run's store, and
#: giving it its own name keeps the two distinguishable.
OPTIM_RESULT_COPY_SCHEMA = "whetstone_envs.study_run_result/v1"

#: The schema a control's run record is stored under. A null has no
#: ``OptimResult`` to copy, and pretending otherwise would put a fabricated
#: optimizer result in the study's store.
NULL_RUN_SCHEMA = "whetstone_envs.study_null_run/v1"


def _read_optim_result(run_dir: Path) -> OptimResult:
    path = run_dir / "result.json"
    if not path.is_file():
        raise StageError(f"run directory {run_dir} has no result.json")
    return OptimResult.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _terminal_template(result: OptimResult, *, run_dir: Path) -> str:
    """The prompt this run ended on, read from its own evidence.

    The last accepted candidate wins, falling back to the terminal step's
    retained candidate: an optimizer that accepted nothing still ends on
    the seed it retained, and reporting *that* is the honest description of
    what the run produced.
    """
    template: str | None = None
    for step_ref in result.step_results:
        step = step_ref.record
        for candidate in step.accepted_candidates:
            found = _template_of(candidate)
            if found is not None:
                template = found
        if step.retained_candidate_ref is not None and template is None:
            template = _template_of(step.retained_candidate_ref)
    if template is None:
        raise StageError(
            f"the run at {run_dir} accepted and retained no candidate "
            "carrying a prompt template, so it has no terminal prompt to "
            "score"
        )
    return template


def _template_of(candidate: object) -> str | None:
    record = getattr(candidate, "record", None)
    payload = getattr(record, "payload", None)
    if payload is None:
        return None
    value = payload.get("prompt_template")
    return value if type(value) is str else None


def _observed_task_calls(result: OptimResult, *, run_dir: Path) -> int:
    """Task-model **rows** this run actually scheduled.

    Counted from the run's own eval evidence rather than from its budget,
    because the Stage-1 gate exists to catch a fan-out whose budget
    accounting was itself wrong.

    The count is delegated to :func:`planned_rows_in_directory` so the
    gate's numerator is the same number the F16 fan-out measurement
    reports. The unit is rows -- ``EvalEvidence.row_accounting.planned``,
    which is what the pre-spend estimate is expressed in -- and each
    eval-evidence record is counted once no matter how many steps cite it.

    ``result`` is accepted so callers keep passing the record they already
    read, but the rows live in the run's store rather than inline on it,
    which is why the directory is what gets read.
    """
    del result
    return planned_rows_in_directory(run_dir)
