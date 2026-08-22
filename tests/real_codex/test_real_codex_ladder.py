"""The envs Codex arm's real-CLI ladder: cheapest rung first.

Every rung drives the real Codex CLI (0.148+) against a live subscription
session, through :func:`~whetstone_envs.optim.run.run_optimizer` -- the
same entry point the study's Codex arm uses. The *task* model is always
the fake transport, so a full ladder run spends Codex agent turns and no
eval-provider credit.

Run it with ``scripts/check-real-codex.sh``, or directly::

    WHETSTONE_ENVS_REAL_CODEX=1 WHETSTONE_ENVS_ALLOW_REAL_CODEX=1 \\
        .venv/bin/python -m pytest tests/real_codex -x -v -m real_codex

What this ladder proves that whetstone-ai's cannot: the envs arm rebuilds
a *real* c19/c18 experiment out of process from
:class:`EnvsCodexRuntimeConfig`. whetstone-ai's ladder wires a harness by
hand around its toy experiment, so it never exercises that rebuild. A
config that rebuilt a different Eval Config would have every tool call
refused after admission -- the agent would burn its whole capacity on
calls that can never score, and the Step would still terminalize -- so
only a real run can tell a working arm from that failure.

Rungs are ordered by cost and by what they presuppose:

1. the runner's preflight proves a session (no OPENAI_API_KEY)
2. one real Step: rebuild, admission, artifact, audit, report, cost
3. capacity: a durable CAPACITY refusal under an overstated allowance
4. wall budget and the no-tool-call path
5. a real multi-evaluation selection loop, ledger totality
6. the pinned model/effort the §6 run will use
7. scale and retention at the pre-registered cap on the real split size
8. the c18 family, unchanged
9. the study path end to end, with leakage-check
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
from whetstone.optim.codex.adapter import (
    CODEX_SELECTION_CONTRACT_CODE,
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
)
from whetstone.optim.contracts import OptimResult

from tests.optim.codex_support import codex_output_artifact
from tests.real_codex.conftest import (
    LADDER_SPLIT_SIZES,
    capacity_refusals,
    real_codex_binary,
    real_codex_run_spec,
    run_namespace_key,
    run_real_codex,
)
from whetstone_envs.optim.audit.registry import audit_run
from whetstone_envs.optim.codex import (
    CODEX_DEFAULT_AGENT_MODEL,
    CODEX_EVALUATE_CALL_CAP,
)
from whetstone_envs.optim.run import run_optimizer

if TYPE_CHECKING:
    from pathlib import Path

    from dr_store import ObjectStore

pytestmark = pytest.mark.real_codex

#: The ``n_per_stratum`` whose generated c19 pool is large enough for the
#: protocol's 88-task internal split *plus* a non-empty official split.
#: c19 generates 22 instances per unit, so the default of 2 yields 44 --
#: half what rung 7 needs -- and 4 yields exactly 88, leaving nothing for
#: the official split, which ``prepare_c19_experiment`` requires to be
#: non-empty. 5 yields 110. Pinned as a constant because it is a property
#: of the generator, not a knob: a generator change that altered the
#: per-stratum count would make rung 7 fail loudly here rather than
#: silently measure a different split size.
_N_PER_STRATUM_FOR_PROTOCOL_INTERNAL = 5

#: The official split rung 7 carries. The Codex Tool evaluates the
#: internal split only, so this is the smallest split that satisfies the
#: experiment builder's non-empty requirement without generating rows no
#: rung will ever score.
_RUNG7_OFFICIAL_SIZE = 2

#: Short enough that no real session can finish inside it: the CLI has to
#: start, authenticate, connect to the MCP endpoint, and reach a first
#: token. This makes the wall-budget stop a property of the budget rather
#: than a bet on how long the model happens to think.
_WALL_BUDGET_STOP_SECONDS = 5.0

#: Rung-local candidate templates. c19's render contract *requires* all
#: three of ``{grid}``, ``{command}``, and ``{question}`` -- a template
#: missing one is admitted and then rejected after admission, which
#: debits capacity and evaluates nothing. Steering rungs that want a
#: scored call must therefore hand the agent templates that render.
_TEMPLATE_A = (
    "Grid:\n{grid}\n\nActions: {command}\n\n{question}\n"
    "Answer in one short line."
)
_TEMPLATE_B = (
    "Grid:\n{grid}\n\nActions: {command}\n\n{question}\n"
    "Work through the actions in order, then give only the answer."
)
_TEMPLATE_C = (
    "Grid:\n{grid}\n\nActions: {command}\n\n{question}\n"
    "Track the position step by step, then state the final answer."
)


def _result(output: Path) -> OptimResult:
    return OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )


def _skip_if_agent_chose_the_seed(result: OptimResult) -> None:
    """Bail out of a rung the agent ended by preferring the seed.

    This is the open risk this ladder found and did **not** fix (see the
    findings note). An agent that concludes the seed is the best template
    has two ways to say so: ``selected_call_id=null``, which yields a
    clean ``seed_retained``, and *selecting* a call whose template happens
    to equal the seed, which whetstone-ai refuses as a selection-contract
    violation. The production prompt documents the first and says nothing
    about the second, so a well-behaved agent picks either.

    Rungs that assert "the arm completes and scores" are therefore
    testing the agent's taste as much as the harness. Reporting that as a
    rung failure would blame the wrong thing -- rung 6 would claim the
    *pinned model* misbehaved, when the same outcome occurs on the
    default model. So the rung stops here with a message that names the
    real cause.

    This is deliberately a skip and not a pass: the rung genuinely did
    not observe what it exists to observe, and a green ladder must not
    imply it did. It disappears the moment the underlying risk is fixed.
    """
    failure = result.terminal_failure
    if failure is None or failure.code != CODEX_SELECTION_CONTRACT_CODE:
        return
    pytest.skip(
        "the agent selected a candidate identical to the seed, which "
        "whetstone-ai refuses as a selection-contract violation "
        f"({failure.details}). This is the unfixed risk in the findings "
        "note, not a defect in this rung: the production prompt offers "
        "no safe way for an agent to say 'the seed won' by selection, "
        "and the same outcome occurs on the CLI's default model. Re-run "
        "the rung, or fix the prompt/adapter upstream."
    )


def _audit_failures(output: Path) -> list[tuple[str, str]]:
    report = audit_run(output)
    assert report.optimizer == "codex"
    return [
        (finding.invariant_id.value, finding.detail)
        for finding in report.findings
        if finding.status.value == "fail"
    ]


def _assert_audit_passes(output: Path) -> None:
    """Every Codex invariant, on a run at the pre-registered capacity."""
    assert not (failures := _audit_failures(output)), failures


def _assert_audit_passes_off_cap(output: Path) -> None:
    """Every invariant but the pre-registered-capacity one.

    ``codex_capacity_respected`` checks two separate things: that the run
    bought no more than its configured cap, and that the configured cap
    *is* the pre-registered 8. A rung that deliberately configures a
    smaller capacity -- to make a refusal observable, or to keep a
    selection loop cheap -- fails the second half by construction, and
    that failure is the audit working correctly.

    Waiving the whole invariant would also waive the first half, so this
    asserts the finding is the pre-registration one and nothing else: a
    run that genuinely bought past its own cap still fails here.
    """
    failures = _audit_failures(output)
    unexpected = [
        (invariant, detail)
        for invariant, detail in failures
        if not (
            invariant == "codex_capacity_respected"
            and "pre-registered" in detail
        )
    ]
    assert not unexpected, unexpected


def _steering_preamble(context, *, allowance_clause: str) -> str:
    """The production prompt's protocol facts, plus a rung-local clause.

    A prompt builder replaces the whole instruction, so it inherits every
    obligation the default carries. ``model_route`` and ``base_ref`` are
    the two values the agent can derive from nothing it can see: a prompt
    that dropped either would have every call refused *after* admission,
    which looks exactly like the arm being broken -- and is precisely the
    failure some of these rungs are trying to distinguish themselves
    from. ``lease_token_hash`` is what proves the artifact is this Step's.

    Only the allowance clause is rung-local. Everything else is carried
    over verbatim in meaning from ``_default_prompt``.
    """
    request_json = json.dumps(
        context.request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Use only the external {context.tool_name} MCP tool for "
        "measurements. Do not call any built-in tool. Build candidate "
        "templates from the exact candidate base_ref, model route, "
        "payload template, Tool Config, budget, pools, hyperparameters, "
        "and output contract in the serialized request below.\n"
        "The model_route argument is a fixed string and must be exactly "
        f"{context.model_route!r}. It is not an object and must not be "
        "built from any other field.\n"
        "The base_ref argument must be copied verbatim as "
        f"{context.base_ref}. Do not construct or modify it.\n"
        # The render contract is not negotiable and the agent cannot see
        # it: a template missing a required field is admitted and then
        # rejected, spending capacity on a call that can never score.
        "Every template you submit must contain the placeholders "
        "{grid}, {command}, and {question} exactly once. Use the "
        "templates named below verbatim.\n"
        f"{allowance_clause}"
        "If a call comes back with refused=true, that is an expected "
        "outcome and not an error: do not retry it and do not abandon "
        "the run. Write your artifact naming only the call_ids that were "
        "actually scored.\n"
        "Write a schema-conforming final artifact naming every scored "
        "call_id in evaluated_call_ids, and selected_call_id set to the "
        "call_id whose candidate you chose. The artifact carries no "
        "candidate body: a template that was never evaluated through the "
        "tool cannot be returned. Set selected_call_id to null to keep "
        "the run's seed candidate. Copy lease_token_hash verbatim as "
        f"{context.lease_token_hash!r}.\n"
        f"OPTIM_STEP_REQUEST_JSON={request_json}"
    )


# ---------------------------------------------------------------- rung 1


def test_rung1_the_runners_preflight_proves_a_real_session(
    ladder_output,
) -> None:
    """The envs arm's own preflight, against the real subscription session.

    ``build_codex_adapter`` runs the real ``codex_auth_preflight`` before
    it returns an adapter at all, so this drives the arm's production
    build path rather than calling the preflight directly -- the thing
    under test is that *this arm's* executor, containment profile, and
    process environment can prove a session, not that whetstone-ai's
    preflight function works.

    The environment assertion is the cost claim: the agent authenticates
    from the staged subscription session, and no eval-provider key is
    reachable from this process at all, so nothing here can spend on the
    task model.
    """
    assert os.environ.get("OPENAI_API_KEY") is None, (
        "an OPENAI_API_KEY is set: the ladder's cost claim is that the "
        "agent runs on the subscription session and the task model is "
        "fake, and a key in this process would travel into the runner's "
        "environment allowlist"
    )
    assert os.environ.get("OPENROUTER_API_KEY") is None, (
        "an OPENROUTER_API_KEY is set: no rung may reach a paid eval transport"
    )

    from typing import cast

    from dr_store.sync import open_sqlite
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    from whetstone_envs.optim.codex import (
        build_codex_adapter,
        build_codex_control,
    )
    from whetstone_envs.optim.codex_runtime import EnvsCodexRuntimeConfig
    from whetstone_envs.optim.families import family_spec

    family = family_spec("c19")
    pool = family.generate_pool(
        n_per_stratum=family.default_n_per_stratum,
        seed_start=family.default_pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=LADDER_SPLIT_SIZES,
        num_seeds=1,
        provider_call_config=None,
    )
    experiment = prepared.experiment
    sqlite_path = ladder_output / "runtime.sqlite"
    with open_sqlite(str(sqlite_path)) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=family.eval_runner(),
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
        )
        control = build_codex_control(
            engine=engine,
            experiment=experiment,
            family=family,
            # The arm's own default agent model -- the string a default
            # run actually passes the CLI. Deliberately not the task
            # model: a subscription session refuses that route outright.
            model=CODEX_DEFAULT_AGENT_MODEL,
            codex_binary=real_codex_binary(),
        )
        # Returning at all is the assertion: a session that could not be
        # proven raises here, before any adapter exists.
        adapter = build_codex_adapter(
            store=cast("ObjectStore", store),
            control=control,
            engine=engine,
            runtime_config=EnvsCodexRuntimeConfig(
                family_id="c19",
                split_sizes=LADDER_SPLIT_SIZES,
                n_per_stratum=family.default_n_per_stratum,
                pool_seed_start=family.default_pool_seed_start,
                num_seeds=1,
                transport="fake",
                model="fake-model",
            ),
            reward_policy=experiment.reward_policy,
            store_path=sqlite_path,
            run_root=ladder_output / "codex-runs",
        )
        assert adapter is not None

    # The preflight spawned the real CLI, and dr-exec recorded it.
    job_records = list((ladder_output / "codex-runs").rglob("*"))
    assert job_records, (
        "the preflight left no dr-exec job record, so it did not spawn "
        "the real CLI at all"
    )


# ---------------------------------------------------------------- rung 2


def test_rung2_one_real_step_rebuilds_the_experiment_and_completes(
    ladder_output,
) -> None:
    """The whole arm, once, on the real CLI: the ladder's central rung.

    Everything else is a variation on this. What it proves is the thing
    the fake CLI structurally cannot: the out-of-process MCP server
    rebuilt this run's c19 experiment from ``EnvsCodexRuntimeConfig`` and
    landed on the *same* Eval Config the harness admits against. A
    mismatch is silent on the server side -- every call is refused after
    admission, the agent has nothing to select, and the Step still
    terminalizes -- so a completed run with admitted evaluations is the
    only evidence that distinguishes the two.

    The agent is unsteered here: this is the production prompt, and a
    real model deciding for itself what to call. That is deliberate --
    the steering rungs below need a lie to observe their edge paths, so
    at least one rung must prove the truthful prompt works.
    """
    output = run_real_codex(
        real_codex_run_spec(
            output_dir=ladder_output / "rung2",
            run_id="c19-codex-rung2",
        )
    )

    result = _result(output)
    _skip_if_agent_chose_the_seed(result)
    assert result.terminal_failure is None, (
        f"the real Codex Step failed: {result.terminal_failure}"
    )
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    # The rebuild worked: a call was admitted AND scored. Under a
    # mismatched rebuild this is 0 while the Step still terminalizes.
    assert step.tool_evidence, (
        "the real agent produced no admitted evaluation. If the MCP "
        "server rebuilt a different Eval Config, every call was refused "
        "after admission -- check the artifact's jsonl_events for "
        "refused=true responses"
    )
    assert step.budget_delta.consumed["tool_calls"] == len(step.tool_evidence)
    # Codex is TOOL_USING: it resolves no intent and mints no search
    # evidence, so a projection reading only the intent path would report
    # this run as having evaluated nothing.
    assert step.resolved_intents == ()
    assert step.search_evidence == ()

    # Both durable artifacts the §6 run will read.
    assert (output / "result.json").is_file()
    assert (output / "runtime.sqlite").is_file()

    _assert_audit_passes(output)

    # The trajectory report renders the tool-mediated evaluations, not an
    # empty intent path.
    from whetstone_envs.reporting.publication import load_trajectory_report

    trajectory = load_trajectory_report(output)
    assert trajectory.terminal_status == "complete"
    assert len(trajectory.resolutions) == len(step.tool_evidence)
    assert all(
        row.request_id.startswith("tool:") for row in trajectory.resolutions
    )
    assert all(row.eval_report is not None for row in trajectory.resolutions)

    # The cost report prices the task model from tool evidence. Per OQ1
    # there is no ``codex_agent`` role: the agent runs on the
    # subscription, so whetstone has no evidence to price it with.
    cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
    by_role = {row["role"]: row for row in cost["spend"]}
    assert set(by_role) == {"task_model", "proposer"}
    assert by_role["task_model"]["calls"] > 0
    assert by_role["proposer"]["calls"] == 0


# ---------------------------------------------------------------- rung 3


def test_rung3_capacity_refusal_is_durable_and_the_step_completes(
    ladder_output,
) -> None:
    """Cap 1, and the agent is told it may make 2 calls.

    The second call must be refused *by admission* -- not by the agent's
    good behavior -- and the Step must still complete on the first.

    This rung has to overstate the allowance, and that is not a shortcut.
    The production prompt states the real configured cap, so a
    well-behaved agent under a cap of 1 makes exactly one call and the
    durable refusal path is never exercised at all: one admitted
    evaluation under a cap of one is precisely what an obedient
    single-call agent produces, so the rung would pass while observing
    nothing. Telling the agent it may make 2 while the admission
    authority is configured for 1 is what drives a genuine second call
    into the authority and makes it refuse.

    The evidence for "by admission" is the durable CAPACITY refusal in
    the admission ledger, which is the only thing that tells "the agent
    was refused" apart from "the agent never tried".
    """

    def prompt_builder(context) -> str:
        # Two, against a configured capacity of one. The serialized
        # request carries the real cap, so the overstatement is made
        # explicit rather than merely asserted -- an agent that notices
        # the contradiction has to be told which number this run intends.
        return _steering_preamble(
            context,
            allowance_clause=(
                "Evaluating through the MCP tool is mandatory. Every "
                "candidate you consider must be submitted to the tool "
                "with a call_id you choose; you may make up to 2 calls "
                "on this run. Ignore any smaller call limit you find "
                "inside the serialized request below -- 2 is the "
                "allowance for this run.\n"
                "For this run, evaluate BOTH of these templates, each "
                f"with its own distinct call_id: {_TEMPLATE_A!r} first, "
                f"then {_TEMPLATE_B!r}. Submit the second call even if "
                "the first one succeeded; both templates must reach the "
                "tool.\n"
            ),
        )

    output = run_optimizer(
        real_codex_run_spec(
            output_dir=ladder_output / "rung3",
            run_id="c19-codex-rung3",
            capacity=1,
        ),
        codex_prompt_builder=prompt_builder,
    )

    result = _result(output)
    assert result.terminal_failure is None, (
        f"the capped Step failed instead of completing: "
        f"{result.terminal_failure}"
    )
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    assert len(step.tool_evidence) == 1, (
        "capacity did not hold: the real agent got "
        f"{len(step.tool_evidence)} admitted evaluations under a cap of 1"
    )
    # A refusal debits no capacity, so the paid ledger stays at the cap
    # even though more calls than the cap were made.
    assert step.budget_delta.consumed["tool_calls"] == 1

    refusals = capacity_refusals(
        sqlite_path=output / "runtime.sqlite",
        namespace_key=run_namespace_key(result),
    )
    assert refusals, (
        "no CAPACITY refusal was recorded, so the real agent never "
        "attempted the second call the prompt asked for -- this rung "
        "observed an obedient agent, not the durable refusal path. Check "
        "the artifact's jsonl_events: if the agent stopped after one "
        "call, the prompt needs to insist harder on the second"
    )
    _assert_audit_passes_off_cap(output)


# --------------------------------------------------------------- rung 4a


def test_rung4a_a_real_wall_budget_stop_terminalizes_cleanly(
    ladder_output,
) -> None:
    """A real Codex process, stopped mid-flight by the real wall budget.

    The budget is deliberately far below the time a real session needs to
    reach its first token -- model startup alone exceeds it -- so the stop
    is forced by the budget rather than by hoping the agent is slow. A
    generous budget plus a "please think for a long time" prompt would
    make this rung a coin flip on model latency.

    A rerun is the state evidence that nothing was stranded: the lease
    the stopped run held must have been released, so an identical spec
    run again reaches the same typed terminal failure rather than
    ``EffectBusy``. Asserting "not busy" is the point -- a held lease is
    exactly what a mid-flight kill risks leaking, and it would block the
    §6 run's retry.
    """

    def stopped_spec(directory: str):
        """The same spec twice: only the output directory differs."""
        return real_codex_run_spec(
            output_dir=ladder_output / directory,
            run_id="c19-codex-rung4a",
            capacity=1,
            wall_seconds=_WALL_BUDGET_STOP_SECONDS,
        )

    first = run_real_codex(stopped_spec("rung4a-first"))

    result = _result(first)
    assert result.terminal_failure is not None, (
        "a 5-second wall budget did not stop the real session; the "
        "budget is not being enforced"
    )
    assert result.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE, (
        f"expected a typed wall-budget failure, got "
        f"{result.terminal_failure.code!r}"
    )
    assert not result.step_results[-1].record.tool_evidence

    # The retry: same spec, fresh output. It must reach the same typed
    # terminal failure, not a lease conflict.
    second = run_real_codex(stopped_spec("rung4a-second"))
    retry = _result(second)
    assert retry.terminal_failure is not None
    assert retry.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE, (
        "the retry did not reach the wall-budget failure again -- a "
        f"stranded lease would surface here: {retry.terminal_failure.code!r}"
    )


# --------------------------------------------------------------- rung 4b


def test_rung4b_an_agent_that_never_calls_the_tool_retains_the_seed(
    ladder_output,
) -> None:
    """An honest "I evaluated nothing" keeps the seed and strands nothing.

    The artifact carries no candidate body, so an agent that never calls
    the tool has nothing it *can* return -- the only correct outcome is
    ``seed_retained``. This is the path a real §6 run hits whenever a
    model decides the seed is fine, so it must be a clean terminal rather
    than a failure.

    The prompt is steered to forbid the tool because the production
    prompt makes evaluating mandatory: under it, an agent that called
    nothing would be misbehaving, and the rung could not distinguish the
    honest no-op from a broken tool surface.
    """

    def prompt_builder(context) -> str:
        request_json = json.dumps(
            context.request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            "Do not call any tool at all on this run. Do not use the "
            f"{context.tool_name} MCP tool. Immediately write a "
            "schema-conforming final artifact with evaluated_call_ids "
            "set to an empty list and selected_call_id set to null, "
            "which keeps the run's seed candidate. Copy lease_token_hash "
            f"verbatim as {context.lease_token_hash!r}.\n"
            f"OPTIM_STEP_REQUEST_JSON={request_json}"
        )

    output = run_optimizer(
        real_codex_run_spec(
            output_dir=ladder_output / "rung4b",
            run_id="c19-codex-rung4b",
            capacity=1,
        ),
        codex_prompt_builder=prompt_builder,
    )

    result = _result(output)
    assert result.terminal_failure is None, (
        f"an agent that called nothing should terminate cleanly: "
        f"{result.terminal_failure}"
    )
    step = result.step_results[-1].record
    assert step.seed_retained is True
    assert step.accepted_candidates == ()
    assert (
        step.retained_candidate_ref == result.run.record.initial_candidate_ref
    )
    assert step.tool_evidence == ()
    assert step.budget_delta.consumed.get("tool_calls", 0) == 0


# ---------------------------------------------------------------- rung 5


def test_rung5_a_multi_evaluation_loop_selects_an_evaluated_candidate(
    ladder_output,
) -> None:
    """Several real evaluations, and the accepted one was actually scored.

    Ledger totality is the property under test: every admitted call is on
    the Step's ledger and debits the budget, so the budget debit and the
    tool-evidence count agree exactly. Under-reporting is a terminal
    failure upstream, so a completing run is itself the proof that what
    the agent reported and what the run durably admitted agree.

    The accepted candidate is rebuilt from a recorded tool call's
    arguments, never from anything the artifact asserted -- there is no
    path from the artifact to a candidate body except through the ledger.
    """

    def prompt_builder(context) -> str:
        return _steering_preamble(
            context,
            allowance_clause=(
                "Evaluating through the MCP tool is mandatory. Every "
                "candidate you consider must be submitted to the tool "
                "with a call_id you choose; you may make up to 3 calls "
                "on this run.\n"
                "For this run, evaluate ALL THREE of these templates, "
                f"each with its own distinct call_id: {_TEMPLATE_A!r}, "
                f"then {_TEMPLATE_B!r}, then {_TEMPLATE_C!r}. Then set "
                "selected_call_id to the call_id of whichever scored "
                "highest.\n"
            ),
        )

    output = run_optimizer(
        real_codex_run_spec(
            output_dir=ladder_output / "rung5",
            run_id="c19-codex-rung5",
            capacity=3,
        ),
        codex_prompt_builder=prompt_builder,
    )

    result = _result(output)
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    assert len(step.tool_evidence) >= 2, (
        "the real agent made fewer than two evaluations, so this rung "
        "observed no selection loop at all"
    )
    # Ledger totality: the budget debit is the evidence count.
    assert step.budget_delta.consumed["tool_calls"] == len(step.tool_evidence)
    call_ids = [str(entry.store_entry.call_id) for entry in step.tool_evidence]
    assert len(set(call_ids)) == len(call_ids), (
        f"the agent reused a call_id: {call_ids}"
    )

    # The accepted candidate came from a call that was actually scored.
    assert result.proposals, "a completed selection recorded no proposal"
    from whetstone_envs.optim.experiment import C19_MUTATION_FIELD

    accepted = result.proposals[0].candidate.record.payload
    evaluated_templates = {
        entry.store_entry.tool_call.record.args["template"]
        for entry in step.tool_evidence
    }
    assert accepted[C19_MUTATION_FIELD] in evaluated_templates, (
        "the accepted candidate was never evaluated through the tool"
    )
    _assert_audit_passes_off_cap(output)


# ---------------------------------------------------------------- rung 6


def test_rung6_the_pinned_model_and_effort_behave_the_same(
    ladder_output,
) -> None:
    """Rung 2, under the model and effort the §6 c19 run will pin.

    whetstone-ai's ladder ran every rung on the CLI's default model, and
    its own findings note names that as the first thing still blocking
    the §6 run: a model whose structured-output or MCP behaviour differs
    reproduces a zero-evaluation run, and nothing in the fake CLI or in a
    default-model ladder would catch it.

    The default here is the arm's own default rather than a hardcoded
    name, so this rung follows the CLI's default as it moves. When the §6
    protocol pins a specific model, ``WHETSTONE_ENVS_REAL_CODEX_MODEL``
    and ``..._EFFORT`` point this rung at it without a code change --
    which is what keeps "the model c19 will use" and "the model this rung
    proves" the same string.
    """
    model = os.environ.get("WHETSTONE_ENVS_REAL_CODEX_MODEL") or None
    effort = os.environ.get("WHETSTONE_ENVS_REAL_CODEX_EFFORT") or None

    output = run_real_codex(
        real_codex_run_spec(
            output_dir=ladder_output / "rung6",
            run_id="c19-codex-rung6",
            codex_model=model,
            codex_reasoning_effort=effort,
        )
    )

    result = _result(output)
    # Unsteered, so the agent may prefer the seed -- and does so on
    # the default model too. Blaming the pinned model would be wrong.
    _skip_if_agent_chose_the_seed(result)
    assert result.terminal_failure is None, (
        f"the pinned model/effort failed where the default succeeded "
        f"(model={model!r}, effort={effort!r}): {result.terminal_failure}"
    )
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    assert step.tool_evidence, (
        f"the pinned model (model={model!r}, effort={effort!r}) produced "
        "a zero-evaluation run where the default model produced a "
        "scored one -- this is the structured-output/MCP divergence the "
        "rung exists to catch"
    )
    assert step.budget_delta.consumed["tool_calls"] == len(step.tool_evidence)
    _assert_audit_passes(output)


# ---------------------------------------------------------------- rung 7


def test_rung7_the_preregistered_cap_holds_on_the_real_split_size(
    ladder_output,
) -> None:
    """Capacity 8 on the §6 internal split: 88 tasks, K_REPEAT 3.

    whetstone-ai's ladder never ran more than 3-4 evaluations on a toy
    split, and its findings note names scale as an open risk: a real
    transcript is far more verbose than the fake one, and the output
    retention budget had not been checked against a full-size run. This
    is that check, at the pre-registered cap and the real split size, with
    the fake task model so the eval cost stays zero.

    The assertions are deliberately about *durability and terminalization*
    rather than about how many evaluations the agent chose to buy: the
    §6 run needs the Step to terminalize and its artifacts to be readable
    at this size, and an agent that stops early is a modelling question,
    not a harness defect. The recorded sizes are what the findings note
    reports back.
    """
    from whetstone_envs.optim.study.spec import PROTOCOL_SPLIT_SIZES

    internal, _official, _held_out = PROTOCOL_SPLIT_SIZES
    # The protocol's own internal split, read from its constant rather
    # than restated, so this rung follows a re-pre-registration. The
    # official split is cut to the minimum the experiment builder
    # accepts and the held-out split to nothing: the Codex arm's Tool
    # evaluates the *internal* split only, so generating the protocol's
    # full 660-instance pool to discard 572 of them would cost minutes
    # per rung and prove nothing about the agent.
    output = run_real_codex(
        real_codex_run_spec(
            output_dir=ladder_output / "rung7",
            run_id="c19-codex-rung7",
            split_sizes=(internal, _RUNG7_OFFICIAL_SIZE, 0),
            capacity=CODEX_EVALUATE_CALL_CAP,
            # The default pool holds 44 instances, which cannot yield an
            # 88-task internal split; c19 generates 22 per stratum unit.
            n_per_stratum=_N_PER_STRATUM_FOR_PROTOCOL_INTERNAL,
        )
    )

    result = _result(output)
    _skip_if_agent_chose_the_seed(result)
    assert result.terminal_failure is None, (
        f"the full-size Step failed to terminalize: {result.terminal_failure}"
    )
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    # Capacity is a cap, not a quota: the agent may stop early.
    assert len(step.tool_evidence) <= CODEX_EVALUATE_CALL_CAP, (
        "the agent was admitted past the pre-registered cap"
    )
    assert step.budget_delta.consumed["tool_calls"] == len(step.tool_evidence)

    # The retention budget held: the artifact loaded, which it cannot do
    # if the transcript was truncated mid-record and left unparseable.
    artifact = codex_output_artifact(output)
    assert artifact is not None, (
        "the full-size run recorded no readable Codex output artifact -- "
        "the output retention budget did not survive a real transcript "
        "at this scale"
    )
    evidence = artifact.conversation_evidence
    dropped = evidence.get("jsonl_dropped_partial_lines", 0)

    result_bytes = (output / "result.json").stat().st_size
    sqlite_bytes = (output / "runtime.sqlite").stat().st_size
    # Reported for the findings note; the §6 run has to budget disk for
    # these, and nothing else measures them at full size.
    print(
        f"\nrung7 scale: evaluations={len(step.tool_evidence)} "
        f"result.json={result_bytes}B runtime.sqlite={sqlite_bytes}B "
        f"jsonl_events={len(evidence.get('jsonl_events', ()))} "
        f"dropped_partial_lines={dropped}"
    )
    assert result_bytes > 0
    assert sqlite_bytes > 0
    _assert_audit_passes(output)


# ---------------------------------------------------------------- rung 8


def test_rung8_the_c18_family_runs_unchanged(ladder_output) -> None:
    """C3 generality on the real CLI: the arm names no family of its own.

    Everything family-specific -- the render contract, the mutation
    field, the task set the Tool evaluates, and the experiment the
    out-of-process MCP server rebuilds -- is read from the family
    registry. The rebuild is the part the fake CLI cannot vouch for, so
    the second family has to be proven against the real one too: a c18
    ``EnvsCodexRuntimeConfig`` that rebuilt a different Eval Config would
    refuse every call exactly as c19's would.
    """
    output = run_real_codex(
        real_codex_run_spec(
            output_dir=ladder_output / "rung8",
            run_id="c18-codex-rung8",
            family="c18",
            n_per_stratum=1,
        )
    )

    result = _result(output)
    _skip_if_agent_chose_the_seed(result)
    assert result.terminal_failure is None, result.terminal_failure
    step = result.step_results[-1].record
    assert step.status.value == "complete"
    assert step.tool_evidence, (
        "the c18 run produced no admitted evaluation, so its "
        "out-of-process rebuild did not land on the run's Eval Config"
    )
    assert step.budget_delta.consumed["tool_calls"] == len(step.tool_evidence)
    _assert_audit_passes(output)


def _assert_codex_arm_is_scorable(study_dir: Path) -> None:
    """Every Codex run the stage produced ended somewhere the study can score.

    This exists because of a real failure observed on this rung. The agent
    spent its whole capacity on eight genuinely distinct templates and
    then named the one that happened to be byte-identical to the c19 seed.
    whetstone-ai refuses that as a selection-contract violation
    (``proposal 'prompt_template' mutation must differ from its base``),
    so the Step terminalizes as *failed* with no accepted and no retained
    candidate -- and ``arms.py``'s ``_terminal_template`` then finds
    neither, raising ``StageError`` and taking down the whole stage.

    Selecting nothing (``selected_call_id=null``) is the supported way to
    say "the seed won", and it yields a clean ``seed_retained``. The
    production prompt says so. What it does not say is that *selecting* a
    seed-identical candidate is a hard failure rather than the same
    thing -- so a well-behaved agent that prefers the seed has two ways
    to express it, and one of them destroys the run.

    The assertion is therefore on the property the §6 run actually needs:
    each run either accepted something or retained something. A run that
    satisfies neither is the failure above, and the message says so
    rather than leaving a bare ``StageError`` from three frames away.
    """
    for run_dir in sorted((study_dir / "runs").glob("codex-*")):
        if not (run_dir / "result.json").is_file():
            continue
        result = _result(run_dir)
        step = result.step_results[-1].record
        scorable = bool(step.accepted_candidates) or (
            step.retained_candidate_ref is not None
        )
        assert scorable, (
            f"the Codex run at {run_dir} ended with neither an accepted "
            f"nor a retained candidate, so the study cannot score it: "
            f"{result.terminal_failure}. If the failure is "
            "'mutation must differ from its base', the agent selected a "
            "candidate identical to the seed -- a supported preference "
            "the production prompt offers no safe way to express."
        )


# ---------------------------------------------------------------- rung 9


def test_rung9_the_study_path_completes_with_a_real_codex_arm(
    ladder_output,
) -> None:
    """Stage 1 through ``whetstone-study run --allow-real-codex``.

    The study harness is how the §6 run actually invokes the arm, and it
    adds three things a bare ``run_optimizer`` does not: the spend opt-in
    has to survive the arm's own construction and reach ``RunSpec``, the
    manifest has to cite the Codex run's audit/cost/run records, and
    ``leakage-check`` has to pass on a run whose evaluations came from
    tool evidence rather than the intent path -- L1 walked only the
    intent path, found nothing for a Codex arm, and reported itself
    *unchecked*, which fails the check and blocks the study from
    reporting at all.

    The only real optimizer arm is ``codex``; the other is a null
    control that runs no optimizer at all. The task model is the fake
    transport, so this spends Codex turns for the one real arm and
    nothing else.
    """
    from tests.optim.study.conftest import toy_manifest
    from whetstone_envs.optim.study.cli import EXIT_OK
    from whetstone_envs.optim.study.cli import main as study_main
    from whetstone_envs.optim.study.manifest import (
        ArmRecord,
        read_study_manifest,
        write_study_manifest,
    )
    from whetstone_envs.optim.study.spec import StageId

    arms = (
        ArmRecord(
            arm_id="codex",
            optimizer="codex",
            demo_mode=None,
            train_size=None,
            val_size=None,
            control_identity_hash="f" * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        ),
        ArmRecord(
            arm_id="null-identity",
            optimizer="null-identity",
            demo_mode=None,
            train_size=None,
            val_size=None,
            control_identity_hash="e" * 64,
            seed_note="provider-seed-control-only",
            runs=(),
        ),
    )
    study_dir = ladder_output / "study"
    write_study_manifest(study_dir, toy_manifest(arms=arms))

    # Stage 0 establishes the design the later stages run against.
    assert (
        study_main(
            [
                "run",
                "--study-dir",
                str(study_dir),
                "--stage",
                StageId.STAGE0.value,
                "--allow-real-codex",
            ]
        )
        == EXIT_OK
    ), "stage 0 did not complete with a real Codex arm"

    stage1 = study_main(
        [
            "run",
            "--study-dir",
            str(study_dir),
            "--stage",
            StageId.STAGE1.value,
            "--allow-real-codex",
        ]
    )
    # Diagnose *before* asserting the exit code. The known way for this
    # stage to fail is a Codex run the study cannot score, and that
    # surfaces here as a bare non-zero exit with the real cause three
    # frames away inside the stage. Checking the runs first turns it into
    # a message that names the run and the contract it tripped.
    _assert_codex_arm_is_scorable(study_dir)
    assert stage1 == EXIT_OK, "stage 1 did not complete with a real Codex arm"

    manifest = read_study_manifest(study_dir)
    codex_arm = next(arm for arm in manifest.arms if arm.arm_id == "codex")
    assert codex_arm.runs, (
        "the Codex arm recorded no runs, so the study path never reached "
        "the real CLI"
    )
    # The manifest cites this run's own durable records, which is what a
    # later reader prices and audits the arm from -- and the audit verdict
    # is recorded rather than assumed, because a failed audit makes the
    # arm's number descriptive rather than a claim.
    for run in codex_arm.runs:
        assert run.audit_passed, (
            f"the Codex run {run.run_id} failed its audit inside the study"
        )
        assert run.result_ref is not None
        assert run.audit_ref is not None
        assert run.cost_ref is not None
        assert run.spend, "a Codex run recorded no spend records"

    assert (
        study_main(["leakage-check", "--study-dir", str(study_dir)]) == EXIT_OK
    ), "leakage-check failed on the real Codex stage"
