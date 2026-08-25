"""COPRO's fidelity invariants, judged only by what the run persisted.

COPRO is the simplest of the four optimizers and its algorithm is fully
determined by two numbers: it runs ``depth`` proposal rounds, each measuring
exactly ``breadth`` candidate occurrences on the internal split, and it
finalizes by ranking every measured occurrence and keeping the best. These
invariants check that the persisted run did that, and nothing else.

Two structural facts about COPRO's evidence shape drive the whole module,
and both are easy to get wrong by reading the algorithm description alone:

1. **``proposed_candidates`` is not the round's occurrence set.** On the seed
   round the adapter plans ``breadth - 1`` drafts and re-measures the initial
   candidate as the round's last occurrence
   (``CoproDriver.plan_round``: ``proposal_count=breadth - 1``,
   ``include_initial_candidate=True``). The initial candidate is filtered out
   of ``proposed_candidates`` because it was not proposed. So the durable
   witness of round cardinality is the *measured occurrence* count -- the
   resolved intents -- not the proposal list. ``COPRO_BREADTH_PER_DEPTH``
   counts intents for exactly this reason.
2. **The run has ``depth + 1`` steps, and the last one is finalization.**
   Steps ``0..depth-1`` propose and evaluate; step ``depth`` consumes no
   budget, issues no intents, and only ranks the measured history into the
   accepted candidate. An invariant that expected evaluations on every step
   would fail on the finalizing step of every honest run.
3. **A terminal step may keep the seed and accept nothing.** When the run's
   own seed ties or wins the terminal ranking, COPRO terminalizes
   ``seed_retained`` with a ``retained_candidate_ref`` and an empty
   ``accepted_candidates`` -- the same mechanism GEPA and MIPROv2 use. Ties
   are ordinary rather than exotic, because an exact-match reward over
   ``N`` internal tasks quantizes to ``k/N``. This reaches the audit at
   both of COPRO's terminal emission points: the ordinary finalize, which
   still records ``depth + 1`` steps, and the early terminal a round taken
   without a valid proposal, which stops short of the configured depth. An
   invariant that read "accepted nothing" as "selected nothing" would fail
   every honest retention, so three of the seven below carry an explicit
   retention branch -- checked against the retained candidate, never waved
   through.

Each invariant is a pure function over :class:`RunEvidence` and cites an
evidence ref for every judgment it makes. Missing evidence is reported FAIL,
never raised: an audit that crashed on a defective run would leave that run
unjudged, which is the one outcome worse than a failing finding.

The module carries no task-family vocabulary. See
:func:`copro_terminal_provenance` for the one place the Step 10 assignment's
invariant text names a family probe, and why this audit implements the
family-agnostic half only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.contracts import IntentOutcome, IntentResolution
from whetstone.optim.copro.control import (
    COPRO_ALGORITHM_VERSION,
    CoproControl,
)

from whetstone_envs.optim.audit._evidence import evidence_ref
from whetstone_envs.optim.audit.schema import (
    AuditFinding,
    AuditStatus,
    EvidenceRef,
    InvariantId,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from whetstone.core.identity import TypedRef

    from whetstone_envs.optim.audit._evidence import RunEvidence, StepEvidence

#: How many failing specifics a ``detail`` names before it elides the rest.
#: A finding is read by a human triaging one run, so it states what was seen
#: rather than only that something failed -- but a hundred of them is noise.
_DETAIL_LIMIT = 3

#: The smallest proposal count that can contain a duplicate base. A round
#: below it has no pair, so ``COPRO_DISTINCT_BASES`` has nothing to compare.
_MIN_COMPARABLE_PROPOSALS = 2


def _finding(
    invariant: InvariantId,
    status: AuditStatus,
    detail: str,
    refs: tuple[EvidenceRef, ...] = (),
) -> AuditFinding:
    return AuditFinding(
        invariant_id=invariant,
        status=status,
        detail=detail,
        evidence_refs=refs,
    )


def _eval_refs(refs: Iterable[TypedRef | None]) -> list[EvidenceRef]:
    """Project the non-null refs in ``refs`` onto the report's wire shape."""
    return [evidence_ref(ref) for ref in refs if ref is not None]


def _elide(problems: list[str]) -> str:
    shown = "; ".join(problems[:_DETAIL_LIMIT])
    if len(problems) > _DETAIL_LIMIT:
        shown += f"; and {len(problems) - _DETAIL_LIMIT} more"
    return shown


def _control(evidence: RunEvidence) -> CoproControl | None:
    """The run's persisted COPRO control, or None when it is not readable.

    Read from the store at the ref the run binds itself to, never from a
    state delta: a step's ``copro_config`` echo is written by the same code
    path an audit is checking, so trusting it would let a defect certify
    itself. Returning None rather than raising keeps a run whose control is
    unreadable auditable -- every invariant that needs it reports FAIL.
    """
    if evidence.control_record is None:
        return None
    try:
        return CoproControl.model_validate(evidence.control_record)
    except ValueError:
        return None


def _control_refs(evidence: RunEvidence) -> tuple[EvidenceRef, ...]:
    return (evidence_ref(evidence.control_ref),)


def _no_control(invariant: InvariantId, evidence: RunEvidence) -> AuditFinding:
    return _finding(
        invariant,
        AuditStatus.FAIL,
        (
            f"the run's optimizer_config at "
            f"{evidence.control_ref.schema_name}:"
            f"{evidence.control_ref.content_hash[:12]} does not resolve to a "
            f"readable COPRO control, so the invariant cannot be checked "
            f"against the configured search"
        ),
        _control_refs(evidence),
    )


def _measuring_steps(evidence: RunEvidence) -> tuple[StepEvidence, ...]:
    """Steps that ran a proposal round, i.e. everything but finalization.

    Identified by evidence, not by index: a step that issued eval intents
    measured a round. The finalizing step issues none, so it drops out
    without the audit having to assume where it sits.
    """
    return tuple(entry for entry in evidence.steps if entry.resolved_intents)


def _terminal_step(evidence: RunEvidence) -> StepEvidence | None:
    return evidence.steps[-1] if evidence.steps else None


def _candidate_refs(entry: StepEvidence) -> tuple[TypedRef, ...]:
    return tuple(
        wrapper.record_ref for wrapper in entry.step.accepted_candidates
    )


def _measured_ref(resolution: IntentResolution) -> TypedRef:
    """The content address of the candidate one intent measured.

    An intent carries its candidate inline rather than by ref, so the ref is
    recomputed here. Recomputing is the point: the accepted-candidate
    wrappers carry a *stored* ref, and comparing the two is what makes
    "this step accepted a candidate it measured" a check rather than an
    assumption.
    """
    request = resolution.optim_eval_request.eval_request
    return candidate_reference(request.candidate).record_ref


# --- The seven invariants --------------------------------------------------


def _seed_retained(evidence: RunEvidence) -> bool:
    """Whether this run terminalized by keeping its own seed.

    **Read from the result, which is this module's one convention.** A step
    carries its own ``seed_retained`` flag and GEPA's audit branches on that
    one, so the two modules look inconsistent side by side. They are not
    reading different facts: ``OptimResult._validate`` refuses a result whose
    ``seed_retained`` differs from the final step's
    (``optim/contracts.py:1493-1497``), so in any schema-valid artifact the
    two are the same bit and neither reading can be wrong.

    The result is preferred here because it is the grain COPRO's invariants
    already judge on. This module reaches for ``evidence.result`` for the
    other terminal facts it needs -- ``terminal_failure`` in three places --
    and ``_empty_terminal_finding`` already reads retention off the result.
    Branching on the step would leave one module consulting two authorities
    for one terminal outcome, which is how a later edit ends up widening an
    exemption on one of them and not the other. GEPA's step-flag reading is
    correct for GEPA, which selects its terminal step by status rather than
    taking the run's last, and is left alone.
    """
    return evidence.result.seed_retained


def _retained_best_so_far(
    invariant: InvariantId,
    evidence: RunEvidence,
    best: float | None,
    refs: list[EvidenceRef],
) -> AuditFinding:
    """Judge a seed retention against the same best-so-far claim.

    Not a pass-through. The retention is honest only if the thing retained
    really is this run's declared seed and really did hold the maximum
    measured reward, so both are checked against recomputed evidence rather
    than taken from the run's own say-so.
    """
    terminal = _terminal_step(evidence)
    seed = evidence.result.run.record.initial_candidate_ref
    if terminal is None or seed is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                "the run reports seed retention but names no terminal step "
                "and declared seed to check the retained candidate against"
            ),
            tuple(refs),
        )
    seed_ref = seed.record_ref
    retention_refs = (*refs, evidence_ref(seed_ref))
    retained = terminal.step.retained_candidate_ref
    if retained is None or retained.record_ref != seed_ref:
        named = (
            "no retained candidate"
            if retained is None
            else f"retained candidate {retained.record_ref.content_hash[:12]}"
        )
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run reports seed retention but names {named}, not its "
                f"declared seed {seed_ref.content_hash[:12]}, so the kept "
                f"candidate is not the one the run claims to have started "
                f"from"
            ),
            retention_refs,
        )
    if best is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run retained its declared seed "
                f"{seed_ref.content_hash[:12]} but measured no internal "
                f"reward, so no best-so-far ranking decided the retention"
            ),
            retention_refs,
        )
    _reward_refs, reward_by_ref = _measured_rewards(evidence)
    seed_reward = reward_by_ref.get(seed_ref)
    if seed_reward is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run retained its declared seed "
                f"{seed_ref.content_hash[:12]} over {best}, but never "
                f"measured that seed, so the retention rests on no reward "
                f"this run recorded"
            ),
            retention_refs,
        )
    if seed_reward < best:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run retained its declared seed "
                f"{seed_ref.content_hash[:12]} scoring {seed_reward} while "
                f"{best} was measured, so it discarded a strictly better "
                f"candidate it had already measured"
            ),
            retention_refs,
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"the run retained its declared seed "
            f"{seed_ref.content_hash[:12]}, whose measured reward "
            f"{seed_reward} equals the maximum over the "
            f"{len(evidence.steps)} steps' measured candidates, so nothing "
            f"it measured beat the starting point"
        ),
        retention_refs,
    )


def copro_breadth_per_depth(evidence: RunEvidence) -> AuditFinding:
    """Every proposal round measured between 1 and ``breadth`` occurrences.

    The occurrence count, not the proposal count: COPRO's seed round plans
    ``breadth - 1`` drafts and re-measures the initial candidate as the
    round's last occurrence, so a round of ``breadth`` occurrences contains
    only ``breadth - 1`` proposals. Counting ``proposed_candidates`` would
    therefore report an honest seed round as short by one.

    Checking the *configured* breadth is deliberate. A non-default breadth is
    a budget choice the study records in its manifest, not an infidelity.

    **The breadth is a ceiling, not an equality.** A round fills its slots
    from what the proposer returned, and a draft can go missing for
    reasons that say nothing about the search: an infra failure on one
    proposer call, a draft that failed template validation, a duplicate of
    a template already in the round. Demanding the exact count made one
    such draft fatal to the whole run -- at breadth 6 and depth 3 that is
    15 proposer calls per run, so a couple of percent of bad drafts
    compounds into a majority chance of losing a Stage-2 arm outright.

    So a short round is measurement, not infidelity: upstream records the
    gap as ``proposal_shortfall`` and proceeds on what it realized. What
    stays a defect is a round that measured *nothing* -- there was no
    search that round -- or one that measured *more* than the configured
    breadth, which is a round carrying candidates nobody budgeted for.
    The round count against the configured depth stays exact and is
    :func:`copro_search_depth`'s to check; the pre-registered breadth
    remains what the design *requested*, and the realized count is what
    the run measured.
    """
    invariant = InvariantId.COPRO_BREADTH_PER_DEPTH
    control = _control(evidence)
    if control is None:
        return _no_control(invariant, evidence)

    refs: list[EvidenceRef] = list(_control_refs(evidence))
    problems: list[str] = []
    rounds = _measuring_steps(evidence)
    for entry in rounds:
        occurrences = len(entry.resolved_intents)
        refs.extend(
            _eval_refs(
                resolution.eval_result_ref
                for resolution in entry.resolved_intents
            )
        )
        if occurrences > control.breadth:
            problems.append(
                f"step {entry.index} measured {occurrences} occurrences, "
                f"more than the configured breadth {control.breadth}"
            )
        elif occurrences < 1:
            problems.append(
                f"step {entry.index} measured no occurrence at all, so no "
                f"search happened in that round"
            )

    if not rounds:
        return _no_rounds_finding(
            invariant, evidence, breadth=control.breadth, refs=tuple(refs)
        )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} of {len(rounds)} proposal rounds measured "
                f"an occurrence count outside 1..{control.breadth}: "
                f"{_elide(problems)}"
            ),
            tuple(refs),
        )
    realized = [len(entry.resolved_intents) for entry in rounds]
    short = [count for count in realized if count < control.breadth]
    if short:
        # Reported, not failed. The shortfall is a measurement the claim
        # carries, so the reader sees a search that ran narrower than it
        # asked to rather than a search that silently did.
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"all {len(rounds)} proposal rounds measured within the "
                f"configured breadth {control.breadth}, and {len(short)} "
                f"realized fewer than requested "
                f"(occurrences per round: {realized})"
            ),
            tuple(refs),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {len(rounds)} proposal rounds measured exactly "
            f"{control.breadth} occurrences, the configured breadth"
        ),
        tuple(refs),
    )


def _no_rounds_finding(
    invariant: InvariantId,
    evidence: RunEvidence,
    *,
    breadth: int,
    refs: tuple[EvidenceRef, ...],
) -> AuditFinding:
    """Judge a run that measured no proposal round at all.

    A defect unless the run declared why it stopped: a reported terminal
    failure, in which case ``COPRO_DEPTH_STEPS`` owns the verdict and
    duplicating it here would report one fault twice, or a declared seed
    retention, which is a completion rather than a fault. This exempts only
    the *absence of rounds* -- a run that measured rounds is still judged on
    their cardinality, retention or not.
    """
    if evidence.result.terminal_failure is not None:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run measured no proposal round and reported terminal "
                f"failure {evidence.result.terminal_failure.code!r}, so no "
                f"round was expected to fill breadth {breadth}"
            ),
            refs,
        )
    if _seed_retained(evidence):
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run measured no proposal round and terminalized on its "
                f"retained seed, so no round was expected to fill breadth "
                f"{breadth}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.FAIL,
        (
            f"the run measured no proposal round at all yet reported neither "
            f"a terminal failure nor retention of its seed, so breadth "
            f"{breadth} was never filled"
        ),
        refs,
    )


def copro_depth_steps(evidence: RunEvidence) -> AuditFinding:
    """Step count is exactly ``control.depth + 1``, or fewer with a failure.

    The ``+ 1`` is finalization: after ``depth`` proposal rounds the adapter
    is invoked once more, consumes no budget, issues no intents, and ranks
    the measured history into the accepted candidate. A run that stopped
    early is honest only when it says so, in one of two recorded ways: a
    ``terminal_failure``, or a declared seed retention. Both must be
    recorded rather than inferred from the short step count.

    The retention case is the early terminal specifically -- a round that
    realizes no valid proposal terminalizes on the run's best-so-far
    without spending the remaining depth. COPRO's ordinary finalize still
    records ``depth + 1`` steps and takes the exact-count branch above, so
    widening here does not loosen the common path.

    A run with *more* steps than ``depth + 1`` is a defect regardless of any
    failure: the search ran longer than it was configured to.
    """
    invariant = InvariantId.COPRO_DEPTH_STEPS
    control = _control(evidence)
    if control is None:
        return _no_control(invariant, evidence)

    refs = _control_refs(evidence)
    observed = len(evidence.steps)
    expected = control.depth + 1
    failure = evidence.result.terminal_failure

    if observed == expected:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run recorded {observed} steps, exactly the configured "
                f"depth {control.depth} plus one finalizing step"
            ),
            refs,
        )
    if observed > expected:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run recorded {observed} steps, more than the "
                f"configured depth {control.depth} plus one finalizing step "
                f"({expected})"
            ),
            refs,
        )
    if failure is not None:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run stopped at {observed} of {expected} steps and "
                f"recorded terminal failure {failure.code!r}, so the short "
                f"search is declared rather than silent"
            ),
            refs,
        )
    if _seed_retained(evidence):
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the run stopped at {observed} of {expected} steps (depth "
                f"{control.depth} plus finalization) and terminalized on its "
                f"retained seed, so the short search is declared rather than "
                f"silent"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.FAIL,
        (
            f"the run recorded {observed} of the expected {expected} "
            f"steps (depth {control.depth} plus finalization) but reported "
            f"neither a terminal failure nor retention of its seed to "
            f"explain the shortfall"
        ),
        refs,
    )


def copro_internal_only(evidence: RunEvidence) -> AuditFinding:
    """Every evaluation used the control's internal Eval Config and role.

    This is leakage rule L1 at the run level: an optimizer that measured a
    candidate on the official or held-out split has seen the data the study
    reserves for reporting, and its efficacy number means nothing.

    Both halves are checked because either alone is bypassable. The
    ``resolved_eval_config`` is what the harness bound the intent to; the
    dereferenced ``EvalEvidence.eval_role`` is what the evaluation actually
    recorded. A config ref that matched while the evidence recorded a
    non-internal role would be exactly the drift worth catching.

    An intent that did not complete is exempt -- whetstone writes no eval
    evidence for a rejected or failed one, and demanding it would read an
    honest refusal as leakage. So is an intent whose ``eval_result_ref``
    does not resolve: that is a dangling number,
    ``REPORTED_NUMBERS_RESOLVE``'s finding, and reporting it here as well
    would make one defect look like two -- including two independent
    reasons a run "leaked", which is a materially worse thing to be told
    while triaging.
    """
    invariant = InvariantId.COPRO_INTERNAL_ONLY
    control = _control(evidence)
    if control is None:
        return _no_control(invariant, evidence)

    expected_config = control.eval_config_ref.config_hash
    refs: list[EvidenceRef] = list(_control_refs(evidence))
    problems: list[str] = []
    checked = 0
    witnessed = 0
    for entry in evidence.steps:
        for position, resolution in enumerate(entry.resolved_intents):
            if resolution.outcome is not IntentOutcome.COMPLETED:
                continue
            checked += 1
            where = f"step {entry.index} intent {position}"
            bound = resolution.resolved_eval_config
            if bound is None:
                problems.append(f"{where} is bound to no Eval Config")
            elif str(bound.config_hash) != expected_config:
                problems.append(
                    f"{where} is bound to Eval Config "
                    f"{str(bound.config_hash)[:12]}, not the control's "
                    f"internal {expected_config[:12]}"
                )
            ref = resolution.eval_result_ref
            if ref is None:
                continue
            refs.append(evidence_ref(ref))
            found = evidence.eval_evidence(ref)
            if found is None:
                continue
            witnessed += 1
            if found.eval_role is not EvalRole.INTERNAL:
                problems.append(
                    f"{where} recorded eval role "
                    f"{found.eval_role.value!r}, not internal"
                )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} findings across {checked} completed "
                f"evaluations left the internal split: {_elide(problems)}"
            ),
            tuple(refs),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {checked} completed evaluations bound the control's "
            f"internal Eval Config {expected_config[:12]}, and all "
            f"{witnessed} whose evidence resolved recorded the internal role"
        ),
        tuple(refs),
    )


def _measured_rewards(
    evidence: RunEvidence,
) -> tuple[list[EvidenceRef], dict[TypedRef, float]]:
    """Every candidate this run measured, and its best internal reward.

    A candidate measured more than once ranks on its best occurrence,
    matching ``_unique_measured_attempts`` -- COPRO's own ranking keys on
    the instruction text and keeps the highest-scoring attempt, so an audit
    that took the last or the mean would judge against a different ranking
    than the one the optimizer used.
    """
    refs: list[EvidenceRef] = []
    reward_by_ref: dict[TypedRef, float] = {}
    for entry in evidence.steps:
        for resolution in entry.resolved_intents:
            if resolution.outcome is not IntentOutcome.COMPLETED:
                continue
            reward = resolution.reward_ref
            if reward is None:
                continue
            refs.append(evidence_ref(reward.record_ref))
            value = float(reward.record.value)
            measured = _measured_ref(resolution)
            prior = reward_by_ref.get(measured)
            if prior is None or value > prior:
                reward_by_ref[measured] = value
    return refs, reward_by_ref


def copro_best_so_far(evidence: RunEvidence) -> AuditFinding:
    """The selected candidate holds the maximum internal reward measured.

    COPRO's whole claim is best-so-far selection, so this is the invariant
    that decides whether a reported improvement was selected or merely
    reported. It recomputes the maximum over every reward the run's intents
    recorded and requires the finalizing step's accepted candidate to hold
    it.

    **Selection happens once, on the finalizing step.** A proposing step
    sets ``accepted_candidates`` to the drafts it just minted -- the adapter
    writes ``accepted_candidates=tuple(proposed)`` before any of them is
    measured -- so those are entrants, not selections. Judging them as
    selections would fail every honest run, and judging the finalizing step
    is what actually witnesses the claim. The finalizing step is identified
    by evidence: it accepts candidates and issues no eval intents.

    Ties resolve either way, because ``rank_attempt_history`` sorts on
    reward alone: the check is that the selected candidate's reward *equals*
    the maximum, not that it is the unique argmax. Requiring a particular
    tie-break would assert an ordering the persisted evidence does not fix.

    A selected candidate this run never measured is a FAIL -- selecting on a
    number with no backing reward is exactly the infidelity here.

    **Seed retention is a selection, not an absence of one.** When the run's
    own seed ties or wins the terminal ranking, COPRO keeps it and accepts
    nothing, so there is no finalizing step to read
    ``accepted_candidates`` from. Ties are not exotic here: an exact-match
    reward over ``N`` internal tasks quantizes to ``k/N``, so a draft
    drawing level with the seed is an ordinary outcome rather than a
    degenerate one. Reading that shape as "no selection happened" would fail
    the invariant on an honest run, and the arm would be demoted for
    reporting its result truthfully.

    The retention branch is therefore checked rather than waved through. A
    vacuous PASS would retire the invariant precisely on the runs where
    best-so-far is doing its only interesting work -- deciding that nothing
    beat the starting point. So it re-derives the same claim against the
    retained candidate: the retained ref must be the run's declared seed,
    and the seed's own measured reward must *equal* the maximum this run
    recorded. Equality, not dominance, for the reason the live-draft branch
    uses it -- ``rank_attempt_history`` sorts on reward alone, so a seed that
    tied the best draft is a legitimate retention, while a seed strictly
    below one is a run that discarded a better candidate it had already
    paid to measure.
    """
    invariant = InvariantId.COPRO_BEST_SO_FAR
    refs, reward_by_ref = _measured_rewards(evidence)
    best = max(reward_by_ref.values(), default=None)

    selecting = [
        entry
        for entry in evidence.steps
        if entry.step.accepted_candidates and not entry.resolved_intents
    ]
    if not selecting:
        if _seed_retained(evidence):
            return _retained_best_so_far(invariant, evidence, best, refs)
        if evidence.result.terminal_failure is not None:
            return _finding(
                invariant,
                AuditStatus.PASS,
                (
                    f"the run selected no candidate and reported terminal "
                    f"failure "
                    f"{evidence.result.terminal_failure.code!r}, so there is "
                    f"no selection claiming to be best-so-far"
                ),
                tuple(refs),
            )
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                "the run finished without a finalizing step that selected "
                "from measured history, and reported neither a terminal "
                "failure nor retention of its declared seed"
            ),
            tuple(refs),
        )
    if best is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"step {selecting[-1].index} selected a candidate but the "
                f"run measured no internal reward to select on"
            ),
            tuple(refs),
        )

    problems: list[str] = []
    for entry in selecting:
        for ref in _candidate_refs(entry):
            measured_value = reward_by_ref.get(ref)
            if measured_value is None:
                problems.append(
                    f"step {entry.index} selected candidate "
                    f"{ref.content_hash[:12]} which this run never measured"
                )
            elif measured_value < best:
                problems.append(
                    f"step {entry.index} selected a candidate scoring "
                    f"{measured_value} while {best} was measured"
                )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} selections did not keep the best measured "
                f"candidate: {_elide(problems)}"
            ),
            tuple(refs),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"every candidate selected across {len(selecting)} finalizing "
            f"steps holds the maximum internal reward {best} over "
            f"{len(reward_by_ref)} measured candidates"
        ),
        tuple(refs),
    )


def copro_distinct_bases(evidence: RunEvidence) -> AuditFinding:
    """Candidates proposed within one round have pairwise distinct bases.

    Two drafts in one round sharing a *candidate id* would mean the round
    explored one direction while charging for two, so this checks that
    proposals within a round are pairwise distinct candidates.

    It deliberately does not check distinct *bases*, and the reason is a
    fact about the optimizer rather than a simplification. Whetstone's
    COPRO adapter binds one base per round for every draft in it --
    ``base = initial``, taken before the round's drafts are read, with the
    ranked history reaching the proposer as prompt context rather than as
    per-draft bases. Every proposal in every round therefore carries the
    initial candidate's base by construction, in the seed round and the
    history rounds alike. An invariant demanding distinct bases would fail
    every honest run at a ``breadth`` above 2 and pass only where it had
    nothing to compare, which is the vacuous-audit failure mode this whole
    module exists to avoid.

    Rounds proposing fewer than two candidates pass vacuously; there is no
    pair to be distinct. That case is real, not hypothetical: at the
    smallest admissible ``breadth`` of 2 the seed round plans one draft and
    re-measures the initial candidate, so it proposes exactly one candidate
    and this invariant has nothing to compare. The finding says so
    explicitly rather than reporting a bare PASS, because a vacuous pass
    that reads like a checked one is how an audit stops meaning anything.
    """
    invariant = InvariantId.COPRO_DISTINCT_BASES
    problems: list[str] = []
    refs: list[EvidenceRef] = []
    rounds = 0
    for entry in evidence.steps:
        proposed = entry.step.proposed_candidates
        if len(proposed) < _MIN_COMPARABLE_PROPOSALS:
            continue
        rounds += 1
        seen: dict[str, int] = {}
        for position, wrapper in enumerate(proposed):
            refs.append(evidence_ref(wrapper.record_ref))
            identity = wrapper.identity_hash
            first = seen.get(identity)
            if first is None:
                seen[identity] = position
            else:
                problems.append(
                    f"step {entry.index} proposals {first} and {position} "
                    f"are the same candidate {identity[:12]}"
                )

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} proposal pairs across {rounds} "
                f"multi-draft rounds duplicate a candidate: "
                f"{_elide(problems)}"
            ),
            tuple(refs),
        )
    if rounds == 0:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"none of the run's {len(evidence.steps)} steps proposed "
                f"two or more candidates, so no round could duplicate one"
            ),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"every proposal in each of {rounds} multi-draft rounds is a "
            f"distinct candidate"
        ),
        tuple(refs),
    )


def copro_no_search_evals(evidence: RunEvidence) -> AuditFinding:
    """COPRO records no ``search_evidence`` on any step.

    COPRO evaluates only through resolved intents; ``search_evidence`` is
    the channel MIPROv2 and GEPA use for their own search-driven
    evaluations. An entry here means a COPRO run bought evaluations off its
    own accounted path, so its measured cost understates what it spent and
    its round cardinality is no longer witnessed by the intents alone --
    which is what ``COPRO_BREADTH_PER_DEPTH`` counts.
    """
    invariant = InvariantId.COPRO_NO_SEARCH_EVALS
    refs: list[EvidenceRef] = []
    offenders: list[str] = []
    for entry in evidence.steps:
        if not entry.search_evidence:
            continue
        offenders.append(
            f"step {entry.index} carries {len(entry.search_evidence)} "
            f"search-evidence entries"
        )
        refs.extend(
            _eval_refs(
                search.eval_result_ref for search in entry.search_evidence
            )
        )

    if offenders:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"COPRO recorded search evidence on {len(offenders)} steps, "
                f"a channel it does not use: {_elide(offenders)}"
            ),
            tuple(refs),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"none of the run's {len(evidence.steps)} steps recorded search "
            f"evidence, so every evaluation ran through a resolved intent"
        ),
    )


def copro_terminal_provenance(evidence: RunEvidence) -> AuditFinding:
    """The terminal candidate's ancestry traces to the run's declared seed.

    A COPRO run declares one starting point,
    ``OptimRun.initial_candidate_ref``, and every study number about that
    run is read as "what the optimizer did starting from that prompt". This
    walks the terminal candidate's
    ``base_ref`` chain back through the candidates this run proposed and
    requires it to end at that declared seed.

    **What the persisted format already guarantees, and what it does not.**
    ``OptimStepResult`` and ``OptimResult`` structurally enforce a great deal
    of provenance on their own: a terminal ``proposals`` list must equal the
    final step's accepted candidates; every proposed candidate must bind an
    exact *request* candidate as its base and differ from it on the mutation
    field alone; every resolved intent must cite a request or proposed
    candidate; a ``seed_retained`` step must accept nothing and must name the
    exact run seed. Those clauses are schema invariants, so an audit
    restating them could never fail and would not be an audit.

    What no schema clause ties together is the run's *declared* seed and the
    candidate the search actually started from. Each step request carries its
    own candidate list, and nothing requires it to be the run's
    ``initial_candidate_ref``. A run can therefore validate perfectly while
    reporting improvement over a seed it never optimized from -- which
    silently misattributes the whole delta. That is the gap this invariant
    covers, and it is the one provenance fact here with a failing fixture.

    **Not implemented: the probe comparison.** The Step 10 assignment's text
    also requires the terminal candidate never be ``PROBES.ceiling_template``.
    That half is deliberately absent for two independent reasons. It is a
    task-family literal, and an audit carrying family vocabulary is a
    recorded design defect by the assignment's own Section 3.5. And every
    fake-transport run proposes the ceiling template by construction --
    ``FamilySpec.proposal_bodies`` scripts it as the first draft -- so the
    check would fail on every CI artifact, which is where Section 3.5
    requires these audits to run. Detecting a ceiling-template terminal is a
    study-level concern with the family in scope, not a run-level fidelity
    invariant.
    """
    invariant = InvariantId.COPRO_TERMINAL_PROVENANCE
    terminal = _terminal_step(evidence)
    if terminal is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            "the run recorded no steps, so it has no terminal candidate",
        )

    seed = evidence.result.run.record.initial_candidate_ref
    if seed is None:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                "the run names no initial_candidate_ref, so no terminal "
                "candidate's ancestry can be traced to a declared seed"
            ),
        )
    seed_ref = seed.record_ref
    accepted = _candidate_refs(terminal)
    if not accepted:
        return _empty_terminal_finding(invariant, evidence, terminal, seed_ref)

    #: Every candidate this run proposed, by its own ref, so the walk can
    #: follow a base_ref from a proposal to the proposal it mutated.
    proposed_by_ref = {
        wrapper.record_ref: wrapper.record
        for entry in evidence.steps
        for wrapper in entry.step.proposed_candidates
    }
    refs = [evidence_ref(seed_ref), *(evidence_ref(ref) for ref in accepted)]
    problems: list[str] = []
    for ref in accepted:
        problem = _seed_ancestry_problem(
            start=ref,
            seed_ref=seed_ref,
            proposed_by_ref=proposed_by_ref,
        )
        if problem is not None:
            problems.append(problem)

    if problems:
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"{len(problems)} of {len(accepted)} terminal candidates do "
                f"not descend from the run's declared seed "
                f"{seed_ref.content_hash[:12]}: {_elide(problems)}"
            ),
            tuple(refs),
        )
    return _finding(
        invariant,
        AuditStatus.PASS,
        (
            f"all {len(accepted)} terminal candidates trace through this "
            f"run's {len(proposed_by_ref)} proposals back to its declared "
            f"seed {seed_ref.content_hash[:12]}"
        ),
        tuple(refs),
    )


def _empty_terminal_finding(
    invariant: InvariantId,
    evidence: RunEvidence,
    terminal: StepEvidence,
    seed_ref: TypedRef,
) -> AuditFinding:
    """Judge a run whose terminal step accepted no candidate.

    Two shapes are honest: a declared seed retention, and a declared
    terminal failure. Anything else is a run that returned nothing without
    saying why, which is a silent stop rather than a reported one.
    """
    refs = (evidence_ref(seed_ref),)
    if _seed_retained(evidence):
        retained = terminal.step.retained_candidate_ref
        if retained is not None and retained.record_ref == seed_ref:
            return _finding(
                invariant,
                AuditStatus.PASS,
                (
                    f"the run retained its declared seed "
                    f"{seed_ref.content_hash[:12]} and proposed nothing in "
                    f"its place"
                ),
                refs,
            )
        return _finding(
            invariant,
            AuditStatus.FAIL,
            (
                f"the run reports seed_retained but names no retained "
                f"candidate equal to its declared seed "
                f"{seed_ref.content_hash[:12]}"
            ),
            refs,
        )
    if evidence.result.terminal_failure is not None:
        return _finding(
            invariant,
            AuditStatus.PASS,
            (
                f"the terminal step accepted no candidate and the run "
                f"reported terminal failure "
                f"{evidence.result.terminal_failure.code!r}, so no candidate "
                f"claims descent from seed "
                f"{seed_ref.content_hash[:12]}"
            ),
            refs,
        )
    return _finding(
        invariant,
        AuditStatus.FAIL,
        (
            f"the terminal step accepted no candidate yet the run reported "
            f"neither a terminal failure nor retention of its declared seed "
            f"{seed_ref.content_hash[:12]}"
        ),
        refs,
    )


def _seed_ancestry_problem(
    *,
    start: TypedRef,
    seed_ref: TypedRef,
    proposed_by_ref: dict[TypedRef, Candidate],
) -> str | None:
    """Walk ``start`` back to ``seed_ref``, naming why it did not arrive.

    Returns None when the chain reaches the seed. The walk is bounded by the
    proposal count and refuses to revisit a ref, so a cyclic ``base_ref``
    chain reports as one rather than hanging the audit.
    """
    cursor = start
    seen: set[TypedRef] = set()
    while cursor != seed_ref:
        if cursor in seen:
            return (
                f"candidate {start.content_hash[:12]} sits on a cyclic "
                f"base chain at {cursor.content_hash[:12]}"
            )
        seen.add(cursor)
        record = proposed_by_ref.get(cursor)
        if record is None:
            return (
                f"candidate {start.content_hash[:12]} reaches "
                f"{cursor.content_hash[:12]}, which is neither a proposal of "
                f"this run nor its declared seed"
            )
        cursor = record.base_ref
    return None


#: Every COPRO invariant, in the order the assignment's Section 3.4 table
#: lists them. The registry splices the shared invariants in; this tuple is
#: only COPRO's own.
COPRO_INVARIANTS = (
    copro_breadth_per_depth,
    copro_depth_steps,
    copro_internal_only,
    copro_best_so_far,
    copro_distinct_bases,
    copro_no_search_evals,
    copro_terminal_provenance,
)

__all__ = [
    "COPRO_ALGORITHM_VERSION",
    "COPRO_INVARIANTS",
    "copro_best_so_far",
    "copro_breadth_per_depth",
    "copro_depth_steps",
    "copro_distinct_bases",
    "copro_internal_only",
    "copro_no_search_evals",
    "copro_terminal_provenance",
]
