"""The study's leakage checks, L1-L6, as mechanical predicates.

Each rule is a pure function over recorded evidence, so a violation is found
by running the check rather than by reading the code that was supposed to
prevent it. :func:`study_leakage_check` runs all five and fails the study
loudly before any report is generated -- that is L6.

**Detection is not prevention.** L2 and L3 are enforced structurally in
:mod:`whetstone_envs.optim.study.selection`, where the ordering makes a
violation unreachable. Re-checking them here catches a *different* failure:
evidence that disagrees with the ledger, which means something wrote held-out
evaluations outside ``report_arm``. A study that relied on L6 alone would
discover its leak only after paying for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from typing import TYPE_CHECKING

from whetstone.optim.contracts import IntentOutcome

from whetstone_envs.optim.audit._evidence import load_run_evidence

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from whetstone_envs.optim.audit._evidence import RunEvidence, StepEvidence

__all__ = [
    "HeldOutObservation",
    "LeakageCheckError",
    "LeakageFinding",
    "LeakageReport",
    "LeakageRule",
    "OptimizerEvalObservation",
    "SplitIdentity",
    "check_l1_optimizer_internal_only",
    "check_l2_selection_once_per_arm",
    "check_l3_held_out_once_per_candidate",
    "check_l4_identical_held_out_procedure",
    "check_l5_splits_disjoint",
    "optimizer_observations_for_study",
    "optimizer_observations_from_run",
    "study_leakage_check",
]


@verify(UNIQUE)
class LeakageRule(StrEnum):
    """The five substantive rules, plus the check that runs them.

    These identifiers are persisted in the manifest's ``leakage_check``
    block and cited by the report, so they are an owned enum rather than
    ad-hoc strings.
    """

    L1_OPTIMIZER_INTERNAL_ONLY = "L1"
    L2_SELECTION_ONCE_PER_ARM = "L2"
    L3_HELD_OUT_ONCE_PER_CANDIDATE = "L3"
    L4_IDENTICAL_HELD_OUT_PROCEDURE = "L4"
    L5_SPLITS_DISJOINT = "L5"
    L6_CHECK_RAN = "L6"


class LeakageCheckError(RuntimeError):
    """The leakage check failed; the study must not report."""


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """One rule's verdict and the evidence behind it.

    ``checked`` distinguishes "this rule held" from "this rule had no
    evidence to hold against". The difference is load-bearing: every one of
    these predicates is vacuously true over an empty observation set, and a
    vacuous truth reported as a pass is exactly how a study comes to claim a
    leakage rule nobody verified. An unchecked finding never counts as
    passed.
    """

    rule: LeakageRule
    passed: bool
    detail: str
    offenders: tuple[str, ...] = ()
    checked: bool = True

    def __post_init__(self) -> None:
        if not self.checked and self.offenders:
            raise ValueError(
                f"{self.rule.value} reports offenders, so it was checked"
            )

    @property
    def holds(self) -> bool:
        """Whether this rule was checked *and* held."""
        return self.checked and self.passed


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Every rule's verdict. ``passed`` is the study's permission to report."""

    findings: tuple[LeakageFinding, ...]

    @property
    def passed(self) -> bool:
        """The study's permission to report: every rule checked and held."""
        return all(finding.holds for finding in self.findings)

    def failures(self) -> tuple[LeakageFinding, ...]:
        """Rules that were checked and did not hold."""
        return tuple(
            finding
            for finding in self.findings
            if finding.checked and not finding.passed
        )

    def unchecked(self) -> tuple[LeakageFinding, ...]:
        """Rules that had no evidence to be checked against."""
        return tuple(
            finding for finding in self.findings if not finding.checked
        )

    def finding(self, rule: LeakageRule) -> LeakageFinding:
        for finding in self.findings:
            if finding.rule is rule:
                return finding
        raise ValueError(f"no finding recorded for {rule.value}")


@dataclass(frozen=True, slots=True)
class OptimizerEvalObservation:
    """One evaluation an optimizer run caused, as L1 reads it.

    ``eval_role`` is the role recorded on the persisted ``EvalEvidence``
    and ``resolved_eval_config_hash`` is the ``config_hash`` of the Eval
    Config that evaluation resolved against -- read from the resolution
    on the intent path, and from the cited evidence on the tool path.

    L1 needs both: matching the config alone would pass an
    evaluation that reached the right config under the wrong role, and
    matching the role alone would pass one that reached a second internal
    config the run never declared.
    """

    run_id: str
    step_index: int
    resolution_index: int
    eval_role: str
    resolved_eval_config_hash: str

    @property
    def location(self) -> str:
        return (
            f"{self.run_id}:step{self.step_index}:"
            f"resolution{self.resolution_index}"
        )


@dataclass(frozen=True, slots=True)
class HeldOutObservation:
    """One held-out evaluation, as L3 and L4 read it."""

    candidate_name: str
    eval_config_hash: str
    repeats: int


@dataclass(frozen=True, slots=True)
class SplitIdentity:
    """One split's content-addressed task identity, as L5 reads it."""

    role: str
    task_hashes: tuple[str, ...]


#: The evaluation role an optimizer is allowed to see. Every other role is
#: a leak by definition.
INTERNAL_ROLE = "internal"


def check_l1_optimizer_internal_only(
    observations: Iterable[OptimizerEvalObservation],
    *,
    internal_eval_config_hash: str,
) -> LeakageFinding:
    """L1: an optimizer sees the internal split and nothing else.

    Upstream already refuses a non-internal role inside ``CoproControl`` and
    the runner passes ``EvalRole.INTERNAL`` explicitly, so this check is
    expected to be redundant. It is here because "structurally prevented"
    and "observed to have held" are different claims, and the study reports
    the second.
    """
    seen = tuple(observations)
    if not seen:
        # Vacuously true, which is not a check. L1's evidence is each
        # run's own evaluations -- intent resolutions, and for a
        # TOOL_USING run its tool evidence -- so a caller holding none
        # has not verified the rule and must not be told it passed.
        return LeakageFinding(
            rule=LeakageRule.L1_OPTIMIZER_INTERNAL_ONLY,
            passed=False,
            detail=(
                "no optimizer evaluations were supplied, so this rule had "
                "nothing to check"
            ),
            checked=False,
        )
    offenders = tuple(
        observation.location
        for observation in seen
        if observation.eval_role != INTERNAL_ROLE
        or observation.resolved_eval_config_hash != internal_eval_config_hash
    )
    return LeakageFinding(
        rule=LeakageRule.L1_OPTIMIZER_INTERNAL_ONLY,
        passed=not offenders,
        detail=(
            f"all {len(seen)} optimizer evaluations resolved the internal "
            "Eval Config under the internal role"
            if not offenders
            else f"{len(offenders)} optimizer evaluations left the internal "
            "split"
        ),
        offenders=offenders,
    )


def check_l2_selection_once_per_arm(
    *,
    selected_arm_ids: Sequence[str],
    expected_arm_ids: Iterable[str],
) -> LeakageFinding:
    """L2: exactly one selection entry per arm, and none for a stranger."""
    expected = tuple(expected_arm_ids)
    counts: dict[str, int] = {}
    for arm_id in selected_arm_ids:
        counts[arm_id] = counts.get(arm_id, 0) + 1
    duplicated = tuple(
        f"{arm_id} selected {count} times"
        for arm_id, count in sorted(counts.items())
        if count > 1
    )
    missing = tuple(
        f"{arm_id} never selected"
        for arm_id in expected
        if arm_id not in counts
    )
    unexpected = tuple(
        f"{arm_id} is not a study arm"
        for arm_id in sorted(counts)
        if arm_id not in set(expected)
    )
    offenders = duplicated + missing + unexpected
    return LeakageFinding(
        rule=LeakageRule.L2_SELECTION_ONCE_PER_ARM,
        passed=not offenders,
        detail=(
            f"{len(expected)} arms each selected exactly once on official"
            if not offenders
            else "selection did not happen exactly once per arm"
        ),
        offenders=offenders,
    )


def check_l3_held_out_once_per_candidate(
    candidate_names: Iterable[str],
) -> LeakageFinding:
    """L3: each reported candidate has exactly one held-out evaluation.

    Takes names rather than observations because what L3 limits is
    evaluations *issued*. An evaluation that was issued and never returned
    carries no procedure to describe, but it has still spent the
    candidate's one shot, so it must count here even though L4 has nothing
    to compare it against.
    """
    counts: dict[str, int] = {}
    for name in candidate_names:
        counts[name] = counts.get(name, 0) + 1
    offenders = tuple(
        f"{name} evaluated {count} times on held-out"
        for name, count in sorted(counts.items())
        if count != 1
    )
    return LeakageFinding(
        rule=LeakageRule.L3_HELD_OUT_ONCE_PER_CANDIDATE,
        passed=not offenders,
        detail=(
            f"{len(counts)} reported candidates each evaluated once on "
            "held-out"
            if not offenders
            else "a reported candidate was evaluated more than once"
        ),
        offenders=offenders,
    )


def check_l4_identical_held_out_procedure(
    observations: Iterable[HeldOutObservation],
) -> LeakageFinding:
    """L4: anchors, nulls, and arms share one held-out procedure.

    One Eval Config hash and one repeat count across every held-out
    evaluation is what makes the paired comparison genuinely paired: a
    candidate measured under a different config is being compared to the
    anchors on different tasks or with different provider controls.
    """
    seen = tuple(observations)
    if not seen:
        # Vacuously true, which is not a check -- the same distinction L1
        # draws. It still fails closed: a caller holding no held-out
        # evidence has not verified that one procedure was shared, and must
        # not be told it was.
        return LeakageFinding(
            rule=LeakageRule.L4_IDENTICAL_HELD_OUT_PROCEDURE,
            passed=False,
            detail=(
                "no held-out evaluations were recorded, so this rule had "
                "nothing to check"
            ),
            checked=False,
        )
    configs = sorted({observation.eval_config_hash for observation in seen})
    repeats = sorted({observation.repeats for observation in seen})
    offenders: list[str] = []
    if len(configs) > 1:
        offenders.extend(f"eval_config_hash {value}" for value in configs)
    if len(repeats) > 1:
        offenders.extend(f"repeats {value}" for value in repeats)
    return LeakageFinding(
        rule=LeakageRule.L4_IDENTICAL_HELD_OUT_PROCEDURE,
        passed=not offenders,
        detail=(
            f"all {len(seen)} held-out evaluations share one Eval Config "
            f"and {repeats[0]} repeats"
            if not offenders
            else "held-out evaluations did not share one procedure"
        ),
        offenders=tuple(offenders),
    )


def check_l5_splits_disjoint(
    splits: Iterable[SplitIdentity],
) -> LeakageFinding:
    """L5: no task hash appears in two splits.

    Task hashes are content-addressed over ``{task_id, prompt_inputs,
    gold}``, so hash-disjointness is the real property; id-disjointness
    implies it but not the reverse, which is why this reads hashes.
    """
    identities = tuple(splits)
    offenders: list[str] = []
    for index, left in enumerate(identities):
        for right in identities[index + 1 :]:
            shared = set(left.task_hashes) & set(right.task_hashes)
            if shared:
                offenders.append(
                    f"{left.role} and {right.role} share "
                    f"{len(shared)} task hashes"
                )
    offenders.extend(
        f"{identity.role} repeats a task hash"
        for identity in identities
        if len(set(identity.task_hashes)) != len(identity.task_hashes)
    )
    return LeakageFinding(
        rule=LeakageRule.L5_SPLITS_DISJOINT,
        passed=not offenders,
        detail=(
            f"{len(identities)} splits are pairwise disjoint by task hash"
            if not offenders
            else "splits overlap by task hash"
        ),
        offenders=tuple(offenders),
    )


def check_held_out_nesting(
    *, smaller: tuple[str, ...], larger: tuple[str, ...]
) -> LeakageFinding:
    """D5: one held-out split must be contained in the other.

    Two held-out splits of different sizes describe the same population
    only if the smaller one's tasks are all still in the larger. Otherwise
    anchors measured on the first do not describe the second, and a
    comparison across them silently changes what is being measured.

    The study now pre-registers held-out at 440, so this is no longer a
    gate-time decision about whether to grow the split. It remains a
    checkable invariant: the deterministic split makes held-220 a prefix
    of held-440 by construction, and this proves that construction held
    for whichever two splits it is handed.
    """
    missing = tuple(sorted(set(smaller) - set(larger)))
    return LeakageFinding(
        rule=LeakageRule.L5_SPLITS_DISJOINT,
        passed=not missing,
        detail=(
            f"the {len(smaller)}-task held-out split is contained in the "
            f"{len(larger)}-task one"
            if not missing
            else f"{len(missing)} tasks were dropped when held-out grew"
        ),
        offenders=missing,
    )


def study_leakage_check(  # noqa: PLR0913
    *,
    optimizer_observations: Iterable[OptimizerEvalObservation],
    internal_eval_config_hash: str,
    selected_arm_ids: Sequence[str],
    expected_arm_ids: Iterable[str],
    held_out_candidate_names: Iterable[str],
    held_out_observations: Iterable[HeldOutObservation],
    splits: Iterable[SplitIdentity],
    strict: bool = True,
) -> LeakageReport:
    """L6: run L1-L5 over the study's artifacts and fail loudly.

    ``strict`` raises on the first failing report rather than returning it.
    That is the default because the caller this exists for is report
    generation, and a report generated from leaking evidence is worse than
    no report. Pass ``strict=False`` only to inspect the findings.
    """
    findings = [
        check_l1_optimizer_internal_only(
            optimizer_observations,
            internal_eval_config_hash=internal_eval_config_hash,
        ),
        check_l2_selection_once_per_arm(
            selected_arm_ids=selected_arm_ids,
            expected_arm_ids=expected_arm_ids,
        ),
        check_l3_held_out_once_per_candidate(held_out_candidate_names),
        check_l4_identical_held_out_procedure(held_out_observations),
        check_l5_splits_disjoint(splits),
    ]
    ran = tuple(finding.rule.value for finding in findings if finding.checked)
    skipped = tuple(
        finding.rule.value for finding in findings if not finding.checked
    )
    failed = sum(
        1 for finding in findings if finding.checked and not finding.passed
    )
    detail = f"{', '.join(ran) or 'no rules'} ran; {failed} failed"
    if skipped:
        detail += f"; {', '.join(skipped)} had no evidence to check"
    findings.append(
        LeakageFinding(
            rule=LeakageRule.L6_CHECK_RAN,
            # L6 is the roll-up, so it holds only when every rule it rolls
            # up was both checked and passing.
            passed=all(finding.holds for finding in findings),
            detail=detail,
        )
    )
    report = LeakageReport(findings=tuple(findings))
    if strict and not report.passed:
        summary = "; ".join(
            f"{finding.rule.value}: {finding.detail}"
            for finding in (*report.failures(), *report.unchecked())
        )
        raise LeakageCheckError(f"study leakage check failed -- {summary}")
    return report


def optimizer_observations_from_run(
    run_dir: Path,
) -> tuple[OptimizerEvalObservation, ...]:
    """Every evaluation one optimizer run caused, as L1 reads it.

    L1's evidence is per-run and lives in the run's own store, not in the
    study manifest, which is why the manifest-only check could never do
    more than report the rule unchecked. This reads it: each completed
    intent resolution names the Eval Config it resolved against, and the
    evidence that resolution addresses names the role it ran under. L1
    needs both -- the config alone would pass an evaluation that reached
    the right config under the wrong role, and the role alone would pass
    one that reached a second internal config the run never declared.

    A rejected or failed intent is skipped, because whetstone deliberately
    writes no eval evidence for one; demanding evidence there would make an
    honest refusal look like a leak.

    **A tool-mediated evaluation is evidence too.** A ``TOOL_USING`` run --
    Codex is the only one -- resolves no intent and mints no search
    evidence *by design*: its paid evaluations are cited from
    ``tool_evidence`` instead. Reading only the intent path therefore
    yields nothing for such a run, and L1 reports itself unchecked rather
    than passed, which fails ``leakage-check`` and blocks the whole study
    from reporting. That is not a conservative default here -- it is a
    rule with evidence in front of it declining to look. So both paths
    are read, and each is resolved the same way: the same reasoning as
    ``reported_numbers_resolve`` in ``optim/audit/registry.py``.
    """
    evidence = load_run_evidence(run_dir)
    observations: list[OptimizerEvalObservation] = []
    for step in evidence.steps:
        observations.extend(_tool_observations(evidence, step))
        for index, resolution in enumerate(step.resolved_intents):
            if resolution.outcome is not IntentOutcome.COMPLETED:
                continue
            reference = resolution.eval_result_ref
            if reference is None:
                continue
            found = evidence.eval_evidence(reference)
            if found is None:
                # A completed resolution addressing something that is not
                # eval evidence is an auditable defect, and
                # ``reported_numbers_resolve`` is the invariant that reports
                # it. L1 is about which split was seen, so it reads the
                # resolutions that carry one.
                continue
            observations.append(
                OptimizerEvalObservation(
                    run_id=evidence.run_id,
                    step_index=step.index,
                    resolution_index=index,
                    eval_role=str(found.eval_role.value),
                    resolved_eval_config_hash=str(
                        resolution.resolved_eval_config.config_hash
                    ),
                )
            )
    return tuple(observations)


def _tool_observations(
    evidence: RunEvidence, step: StepEvidence
) -> tuple[OptimizerEvalObservation, ...]:
    """L1's evidence from one step's tool-mediated evaluations.

    Both fields come from the persisted ``EvalEvidence`` the Tool Result
    cites: the role, exactly as on the intent path, and the Eval Config's
    ``config_hash``.

    The config is read from the evidence rather than from the Tool
    Config, even though a call is admitted only against the config its
    Tool Config is bound to. The Tool Config carries a ``TypedRef`` -- a
    *content* hash over the stored record -- while the manifest and L1
    both speak in ``EvalConfigRef.config_hash``, the config's own
    identity. They are two different hashes of the same config, so
    comparing one against the other reports every honest evaluation as
    having left the internal split.

    A refused call is absent from ``tool_evidence`` and a rejected one
    cites no evidence ref, so neither contributes: the same treatment a
    rejected intent gets, and for the same reason.
    """
    observations: list[OptimizerEvalObservation] = []
    for index, entry in enumerate(step.tool_evidence):
        for reference in entry.result.record.evaluation_evidence_refs:
            found = evidence.eval_evidence(reference)
            if found is None:
                continue
            observations.append(
                OptimizerEvalObservation(
                    run_id=evidence.run_id,
                    step_index=step.index,
                    resolution_index=index,
                    eval_role=str(found.eval_role.value),
                    resolved_eval_config_hash=str(
                        found.eval_config_ref.config_hash
                    ),
                )
            )
    return tuple(observations)


def optimizer_observations_for_study(
    run_dirs: Iterable[Path],
) -> tuple[OptimizerEvalObservation, ...]:
    """L1's evidence across every run a study recorded.

    A run directory that no longer holds its artifacts is skipped rather
    than raising: the caller reports L1 as unchecked when the result is
    empty, which is the same verdict a missing run produces, and a crash
    would replace a legible finding with a traceback.
    """
    observations: list[OptimizerEvalObservation] = []
    for run_dir in run_dirs:
        if not (run_dir / RUN_RESULT_NAME).is_file():
            continue
        observations.extend(optimizer_observations_from_run(run_dir))
    return tuple(observations)


#: The artifact whose presence marks a readable run directory.
RUN_RESULT_NAME = "result.json"


def held_out_observations_from_counts(
    counts: Mapping[str, int], *, eval_config_hash: str, repeats: int
) -> tuple[HeldOutObservation, ...]:
    """Expand a per-candidate count into the observations L3 reads.

    A convenience for callers holding a tally rather than a record per
    evaluation; the checks themselves stay count-free so that a duplicate is
    visible as two observations rather than as an integer a caller chose.
    """
    return tuple(
        HeldOutObservation(
            candidate_name=name,
            eval_config_hash=eval_config_hash,
            repeats=repeats,
        )
        for name, count in counts.items()
        for _ in range(count)
    )
