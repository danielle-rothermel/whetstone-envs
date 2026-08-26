"""Bind a study directory to a running population, engine, and store.

The stage harness takes every provider-touching collaborator as a callable
so it can be exercised without one. This module is where those callables
actually come from: it reads the manifest's ``population`` and ``models``
blocks, regenerates the pool the study pre-registered, and hands back a
:class:`~whetstone_envs.optim.study.stages.StageEnvironment` bound to a
per-role evaluation engine over one open store.

**The population is regenerated, never re-chosen, and the result is
checked.** Pool generation is deterministic in
``(n_per_stratum, pool_seed_start)`` and both are recorded in the manifest,
so binding an environment reproduces the exact tasks the study
pre-registered rather than drawing a fresh sample of the same size. Binding
then verifies that: the regenerated splits' content-addressed task hashes
must equal the ones the manifest recorded, and a mismatch refuses the bind
rather than proceeding to evaluate a different population under the study's
name. That check is cheap and it is the only thing standing between a
changed generator and a study whose Stage-2 numbers describe different
tasks than its Stage-0 anchors did.

The environment is a context manager because the store and every engine it
opens are resources: leaving the block closes them on every exit path,
including the failures a stage re-raises.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from dr_providers import ProviderKind, ReasoningEffort
from dr_store.sync import open_sqlite
from whetstone.core.roles import EvalRole
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import (
    DEFAULT_CONCURRENCY as WHETSTONE_DEFAULT_CONCURRENCY,
)
from whetstone.eval.runtime_engine import RuntimeEvalEngine

from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    DRIVER_MAX_ATTEMPTS,
    RETRY_BASE_SECONDS,
    RETRY_JITTER_FRACTION,
    RETRY_MAX_SECONDS,
    RETRY_MULTIPLIER,
    TASK_CALL_MAX_ATTEMPTS,
    TASK_CALL_TIMEOUT_SECONDS,
    bind_openrouter_transport,
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    validate_provider_concurrency,
    widened_execution_policy,
)
from whetstone_envs.optim.rows import task_rows_from_instances
from whetstone_envs.optim.study.manifest import (
    PROVIDER_CONTROL_UNSET,
    PROVIDER_SEED_DERIVED_PER_CALL,
    STUDY_STORE_NAME,
    ModelsRecord,
    ProviderCallRecord,
    SplitsRecord,
    TransportName,
    read_study_manifest,
    write_study_manifest,
)
from whetstone_envs.optim.study.spec import (
    CODEX_EVALUATE_CALL_CAP,
    CODEX_WALL_SECONDS,
)
from whetstone_envs.optim.study.stages import StageEnvironment, StageError
from whetstone_envs.reporting.schema import SPLIT_ROLE_BY_REPORT_ROLE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dr_providers.modeling.call import ProviderCallConfig
    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.candidate import Candidate
    from whetstone.provider.policy import ProviderExecutionPolicy

    from whetstone_envs.pools import PoolSplit

__all__ = [
    "FAKE_TRANSPORT",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_TRANSPORT",
    "SPLIT_ROLE_BY_EVAL_ROLE",
    "STUDY_STORE_NAME",
    "TOY_API_KEY_ENV",
    "anchor_candidates",
    "assert_default_concurrency_matches",
    "bound_stage_environment",
    "require_transport_credentials",
]

#: The default transport: offline, answering from the experiment's own
#: gold. Named rather than inline so the default and every comparison
#: against it cannot drift apart.
FAKE_TRANSPORT = TransportName.FAKE.value

#: The billed transport, the one a paid stage names explicitly.
OPENROUTER_TRANSPORT = TransportName.OPENROUTER.value

#: Where each transport reads its key from. The fake transport still needs
#: a named variable because the reference runtime always asks for one; it
#: is never read, which is why the toy name is not a credential.
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
TOY_API_KEY_ENV = "WHETSTONE_TOY_API_KEY"

#: Re-exported from the manifest, which owns a study directory's layout.
#: Anchor evaluations are the study's own records, not a run's, so they live
#: beside ``study.json`` rather than inside any one run directory.

#: The split each evaluation role binds to, spelled as whetstone spells it.
SPLIT_ROLE_BY_EVAL_ROLE: dict[EvalRole, str] = {
    EvalRole.INTERNAL: SPLIT_ROLE_BY_REPORT_ROLE["internal"],
    EvalRole.OFFICIAL: SPLIT_ROLE_BY_REPORT_ROLE["official"],
    EvalRole.HELD_OUT: SPLIT_ROLE_BY_REPORT_ROLE["held_out"],
}

#: The names the anchors are reported under. Persisted into the manifest's
#: held-out rows, so they are owned constants rather than inline strings.
NAIVE_ANCHOR_NAME = "naive"
CEILING_ANCHOR_NAME = "ceiling"


def assert_default_concurrency_matches() -> None:
    """Refuse to bind if whetstone's default has moved off ours.

    :data:`~whetstone_envs.optim.provider.DEFAULT_PROVIDER_CONCURRENCY`
    is persisted identity, so it is a literal rather than an import of
    whetstone's
    ``DEFAULT_CONCURRENCY``. That leaves exactly one way for the two to
    disagree silently -- a dependency bump -- and this is the check that
    turns that into a loud failure at bind time instead of a stage that
    records one width and runs at another.
    """
    if DEFAULT_PROVIDER_CONCURRENCY != WHETSTONE_DEFAULT_CONCURRENCY:
        raise RuntimeError(
            "whetstone's DEFAULT_CONCURRENCY is "
            f"{WHETSTONE_DEFAULT_CONCURRENCY}, but this package records "
            f"{DEFAULT_PROVIDER_CONCURRENCY} as the width an invocation "
            "that names none ran at. Update "
            "DEFAULT_PROVIDER_CONCURRENCY and confirm what existing "
            "stage records mean."
        )


def anchor_candidates(family_id: str) -> tuple[Candidate, Candidate]:
    """The family's naive and ceiling anchor candidates, in that order.

    Both are built through the family's own render contract, so an anchor
    that would not render is refused here rather than at the first paid
    evaluation.
    """
    family = family_spec(family_id)
    contract = family.render_contract()
    naive = family.probes.naive_template
    ceiling = family.probes.ceiling_template
    contract.validate_template(naive)
    contract.validate_template(ceiling)
    return (
        family.build_candidate(candidate_id=NAIVE_ANCHOR_NAME, template=naive),
        family.build_candidate(
            candidate_id=CEILING_ANCHOR_NAME, template=ceiling
        ),
    )


def _require_recorded_population(
    split: PoolSplit, recorded: SplitsRecord
) -> None:
    """Refuse a bind whose regenerated tasks are not the recorded ones.

    Compared by content-addressed task hash, not by size or id: the hash is
    over ``{task_id, prompt_inputs, gold}``, so it is the only comparison
    that catches a generator whose output changed while its shape did not.
    """
    for role, instances, record in (
        ("internal", split.internal_eval, recorded.internal),
        ("official", split.official, recorded.official),
        ("held_out", split.held_out, recorded.held_out),
    ):
        if not record.task_hashes:
            # A manifest written before its splits were measured records
            # sizes only; there is nothing to disagree with yet.
            continue
        regenerated = tuple(
            row.task_hash for row in task_rows_from_instances(instances)
        )
        if regenerated != record.task_hashes:
            raise ValueError(
                f"the regenerated {role} split does not match the one this "
                "study recorded; the population or its generator changed"
            )


def require_transport_credentials(transport: str) -> None:
    """Refuse an unknown transport, or a paid one with no key, up front.

    Both refusals are ``ValueError`` and both happen before any store is
    opened, any pool is generated, and any provider is reached. That
    ordering is the point: a stage that cannot legitimately spend must fail
    without having written anything into the study directory, because a
    partially-initialized study is harder to reason about than one that
    never started.

    The key is checked for *presence*, never read into a message. A wrong
    key is the provider's refusal to make, and an error that echoed a
    credential would be worse than the missing one it described.
    """
    if transport not in {FAKE_TRANSPORT, OPENROUTER_TRANSPORT}:
        raise ValueError(
            f"unsupported transport {transport!r}; the study harness runs "
            f"on {FAKE_TRANSPORT!r} or {OPENROUTER_TRANSPORT!r}"
        )
    if transport != OPENROUTER_TRANSPORT:
        return
    if not os.environ.get(OPENROUTER_API_KEY_ENV, "").strip():
        raise ValueError(
            f"transport {OPENROUTER_TRANSPORT!r} needs "
            f"{OPENROUTER_API_KEY_ENV} in the environment; nothing was "
            "written and no provider was called"
        )


def _control_text(value: object) -> str:
    """One request control as the manifest states it.

    An unset control is :data:`PROVIDER_CONTROL_UNSET` rather than an
    omission or a zero: "left to the provider's default" is a real state
    with real consequences -- it is why the toy Stage 0 billed thousands
    of reasoning tokens per call -- so it is said out loud.
    """
    if value is None:
        return PROVIDER_CONTROL_UNSET
    return str(value)


def _provider_call_record(
    *,
    transport: str,
    config: ProviderCallConfig,
    policy: ProviderExecutionPolicy,
) -> ProviderCallRecord:
    """The bound config, flattened into the manifest's own record.

    Every field is a string because this block is a *statement of what was
    bound*, not a typed re-declaration of the provider's own model: the
    controls are heterogeneous, an unset one is meaningful, and a reader
    comparing two stages compares text.
    """
    route = config.route
    controls = config.controls
    return ProviderCallRecord(
        transport=transport,
        provider=str(route.provider.value),
        protocol=str(route.protocol.value),
        model_route=str(route.model),
        temperature=_control_text(controls.temperature),
        top_p=_control_text(controls.top_p),
        token_limit=_control_text(controls.token_limit),
        # Recorded verbatim, never set from here: whether the design pins a
        # task-model reasoning effort is an open decision, and a field that
        # looked like a knob would answer it by accident.
        reasoning=_control_text(controls.reasoning),
        # Not the statically bound control, which is unset here and would
        # read as "the provider chose the seed". Every eval call goes out
        # seeded: whetstone's ``provider_call_config_with_parameters``
        # sets ``seed`` unconditionally from ``derive_rng_seed``, and
        # refuses a definition that cannot transport it. What is bound at
        # this point is therefore not what reaches the wire.
        seed=PROVIDER_SEED_DERIVED_PER_CALL,
        # Serialized rather than repr'd: an extension carries provider
        # request body the study did not otherwise name, and a reader
        # comparing two stages needs to read it, not a container's repr.
        extensions=json.dumps(
            config.extensions.model_dump(mode="json"), sort_keys=True
        ),
        # Read off the policy that was actually bound, not off the
        # constants it was built from: the paid route is hardened and the
        # fake route is not, and the record must say which one ran.
        timeout_seconds=repr(policy.transport_policy.timeout_seconds),
        max_attempts=str(_effective_max_attempts(policy)),
        retry_backoff=_retry_backoff_text(policy),
    )


def _effective_max_attempts(policy: ProviderExecutionPolicy) -> int:
    """How many times one logical call is really attempted.

    Not ``policy.max_attempts`` alone. On the paid route the retry budget
    deliberately does not live on the policy: the driver is pinned to a
    single attempt and :class:`~whetstone_envs.optim.provider.
    RetryingTransport` -- the layer that actually waits -- spends
    :data:`~whetstone_envs.optim.provider.TASK_CALL_MAX_ATTEMPTS` inside
    one driver attempt. Recording the driver's number would report a
    hardened route as making one attempt when it makes five, which is
    exactly the count an operator reconciles spend against.

    The two are multiplied rather than picked between because that is
    what the two loops do to each other; the pinning to one is what
    keeps the product equal to the wrapper's budget rather than 25x.
    """
    if policy.max_attempts == DRIVER_MAX_ATTEMPTS and _is_hardened(policy):
        return TASK_CALL_MAX_ATTEMPTS
    return policy.max_attempts


def _is_hardened(policy: ProviderExecutionPolicy) -> bool:
    """Whether this policy is the paid route's hardened one.

    Keyed off the reasoning-sized timeout, which is the other half of
    :func:`~whetstone_envs.optim.provider.hardened_execution_policy` and
    the only thing distinguishing a hardened policy from a fake-route one
    that happens to allow a single attempt.
    """
    return policy.transport_policy.timeout_seconds == TASK_CALL_TIMEOUT_SECONDS


def _retry_backoff_text(policy: ProviderExecutionPolicy) -> str:
    """The waits between attempts, as the operator would read them.

    Spelled as the actual delay sequence rather than as the schedule's
    three parameters, because what a reader wants to know is how long a
    rate-limited row could have been held -- and because the sequence is
    what makes the retry budget and the timeout comparable at a glance.

    Driven by :func:`_effective_max_attempts` rather than by
    ``policy.max_attempts``, so the hardened route -- whose attempts are
    spent inside the transport wrapper, not the driver -- reports the
    schedule it will really wait rather than "none".
    """
    attempts = _effective_max_attempts(policy)
    if attempts <= 1:
        return "none (single attempt)"
    schedule = ", ".join(
        repr(
            min(
                RETRY_BASE_SECONDS * RETRY_MULTIPLIER ** (attempt - 1),
                RETRY_MAX_SECONDS,
            )
        )
        for attempt in range(1, attempts)
    )
    return (
        f"exponential {schedule} s "
        f"(+/-{int(RETRY_JITTER_FRACTION * 100)}% jitter, "
        "Retry-After delta honoured; HTTP-date ignored)"
    )


def _require_pinned_reasoning_effort(
    *,
    transport: str,
    record: ProviderCallRecord,
    models: ModelsRecord,
) -> None:
    """Refuse a paid bind whose effort disagrees with the design.

    The manifest already *records* what was bound, which makes a
    disagreement visible to a reader. That is not enough: a reader has to
    look, and the looking happens after the stage has spent. This turns
    the same comparison into a refusal, so a paid stage that would have
    evaluated at an effort the pre-registration does not name fails before
    it bills rather than after.

    Applied to paid transports only. The fake transport binds the
    reference default and never reaches a provider, so its recorded
    effort is not a claim about the study's treatment.
    """
    if transport != OPENROUTER_TRANSPORT:
        return
    expected = models.task_reasoning_effort
    if record.reasoning == expected:
        return
    raise StageError(
        "the bound task route disagrees with the pre-registered reasoning "
        f"effort: the design names {expected!r} and this transport bound "
        f"{record.reasoning!r}. A stage that ran this way would measure a "
        "task model the pre-registration does not describe."
    )


def _record_provider_call_config(
    study_dir: Path,
    *,
    transport: str,
    config: ProviderCallConfig,
    policy: ProviderExecutionPolicy,
) -> None:
    """Record what this transport actually bound, replacing its entry.

    Replaced rather than appended: one effective config per transport, so
    a re-run on a transport already recorded restates it instead of
    leaving the study unable to say which of two configs its numbers came
    from. A rebind that produced the identical record writes nothing, so
    an ordinary resume does not touch the manifest.
    """
    manifest = read_study_manifest(study_dir)
    record = _provider_call_record(
        transport=transport, config=config, policy=policy
    )
    _require_pinned_reasoning_effort(
        transport=transport, record=record, models=manifest.models
    )
    existing = manifest.models.provider_calls
    if record in existing:
        return
    updated = (
        *(entry for entry in existing if entry.transport != transport),
        record,
    )
    write_study_manifest(
        study_dir,
        manifest.model_copy(
            update={
                "models": manifest.models.model_copy(
                    update={"provider_calls": updated}
                )
            }
        ),
        replace=True,
    )


@contextmanager
def bound_stage_environment(  # noqa: PLR0913
    study_dir: Path,
    *,
    transport: str = FAKE_TRANSPORT,
    allow_real_codex: bool = False,
    discard_stale_runs: bool = False,
    provider_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY,
    allow_width_change: bool = False,
) -> Iterator[StageEnvironment]:
    """Open a study's store and bind one engine per evaluation role.

    ``transport="fake"`` is the default because every stage in this package
    is exercised without provider calls; a paid stage names ``openrouter``
    explicitly, so no code path reaches a provider by omission.

    **The credential check happens first, before anything is opened.**
    ``openrouter`` without a key in the environment is refused here, ahead
    of the store, the pool, and every engine -- so an unauthorized paid run
    cannot leave a half-written study directory behind, and the operator
    learns what is missing rather than watching the first evaluation fail.

    ``allow_real_codex`` is the run-time authorization to spend on a real,
    billed Codex session, carried from ``whetstone-study run
    --allow-real-codex``. It defaults off for the same reason the transport
    does, and it is a property of *this invocation* rather than of the
    study: it reaches the runner and the harness's early refusal, and never
    the manifest or the pre-registration hash.

    ``discard_stale_runs`` is the third of the same kind: the operator's
    authorization to discard a run directory whose own artifacts say it is
    not this invocation's run, rather than refusing. It defaults off
    because such a directory may be paid evidence.

    ``provider_concurrency`` is how many task evaluations run against the
    provider at once. It is an invocation property like the transport --
    it changes how long the stage takes, not what it measures, and it
    never enters the pre-registration hash -- but unlike the Codex
    authorization it is *recorded*, because a stage's wall time and its
    rate-limit failures are only interpretable against the width it ran
    at. It reaches both bounds that matter: the engine's worker pool and
    the HTTP client's connection pool, which is raised to match so the
    workers are not queued behind sockets.

    ``allow_width_change`` is the operator's authorization to resume an
    arm stage at a different width than its surviving runs were produced
    at. It defaults off because a resume reuses those run directories and
    a run does not persist the width it ran at, so an unauthorized change
    would record runs under a width they never ran at. Like the other
    three authorizations it belongs to the invocation, and unlike them it
    leaves a recorded note -- the width is what a stage's wall time is
    read against, so a stage whose runs span two widths says so.
    """
    require_transport_credentials(transport)
    validate_provider_concurrency(provider_concurrency)
    assert_default_concurrency_matches()
    # Imported inside the binder, not at module scope. ``arms`` reaches the
    # shared optimizer runner, which reaches ``optim.run_cost``, which reads
    # this package's ``RunSpendRecord`` -- so a module-level import here
    # would close a cycle through ``study/__init__``. Deferring it at the one
    # place the runner is actually constructed keeps every other importer of
    # ``study.manifest`` free of the optimizer stack.
    from whetstone_envs.optim.study.arms import (  # noqa: PLC0415
        BuildCandidate,
        RoleScorer,
        StudyOptimizerRunner,
    )
    from whetstone_envs.optim.study.spend import (  # noqa: PLC0415
        ReportSpendLedger,
    )

    manifest = read_study_manifest(study_dir)
    population = manifest.population
    family = family_spec(population.family)
    pool = family.generate_pool(
        n_per_stratum=population.n_per_stratum,
        seed_start=population.pool_seed_start,
    )
    split_sizes = (
        manifest.splits.internal.size,
        manifest.splits.official.size,
        manifest.splits.held_out.size,
    )
    naive, ceiling = anchor_candidates(population.family)
    paid = transport == OPENROUTER_TRANSPORT
    # The route every task evaluation takes. ``None`` on the fake
    # transport, so the fake path's prepared experiment -- and therefore
    # every Eval Config hash it derives -- is byte-for-byte what it was
    # before a paid path existed.
    provider_call_config = (
        openrouter_seeded_call_config(
            model=manifest.models.task_model,
            # The pre-registered effort, read off the manifest rather than
            # off the protocol module: the manifest is what this study was
            # initialised with, and a stage that reached past it could bind
            # an effort the pre-registration does not name.
            reasoning_effort=ReasoningEffort(
                manifest.models.task_reasoning_effort
            ),
        )
        if paid
        else None
    )
    # The split is a deterministic function of the pool and the sizes, so
    # one reference preparation names every role's tasks whatever repeat
    # count a later engine binds at.
    reference = family.build_experiment(
        pool,
        split_sizes=split_sizes,
        num_seeds=1,
        provider_call_config=provider_call_config,
    )
    split = reference.split
    _require_recorded_population(split, manifest.splits)
    # The policy the engines below will be bound with, derived here so the
    # manifest records the settings that actually ran rather than the
    # defaults they were built from. It is rebuilt, not shared, because
    # the store is not open yet; the two agree by construction because
    # both apply the same two transforms in the same order to the same
    # reference config.
    runtime_config_for_role = {
        role: ReferenceEvalRuntimeConfig(
            split_role=SPLIT_ROLE_BY_EVAL_ROLE[role],
            transport_api_key_env=(
                OPENROUTER_API_KEY_ENV if paid else TOY_API_KEY_ENV
            ),
            **({"provider_kind": ProviderKind.OPENROUTER} if paid else {}),
        )
        for role in SPLIT_ROLE_BY_EVAL_ROLE
    }
    # The policy every engine and the live transport share. Derived from
    # the internal role's config because all three roles build the same
    # policy from the same key and provider kind; binding one and using it
    # everywhere is what keeps the transport's pool, the engines' worker
    # counts, and the manifest's record describing a single decision.
    execution_policy = widened_execution_policy(
        runtime_config_for_role[EvalRole.INTERNAL].execution_policy,
        concurrency=provider_concurrency,
    )
    if paid:
        # Only the billed route: the fake transport answers from the
        # experiment's own gold, so a reasoning-sized timeout and a retry
        # budget would describe a provider it never reaches -- and would
        # change the fake path's recorded policy identity for no reason.
        execution_policy = hardened_execution_policy(execution_policy)
    # Read off the prepared experiment rather than off the argument above:
    # on the fake transport that argument is ``None`` and the effective
    # config is the reference default the experiment builds for itself, so
    # recording the argument would record "nothing" for a path that does
    # in fact bind one.
    _record_provider_call_config(
        study_dir,
        transport=transport,
        config=reference.experiment.rollout_graph.provider_call_config,
        policy=execution_policy,
    )
    with open_sqlite(str(study_dir / STUDY_STORE_NAME)) as store:
        # One live transport for the whole stage, not one per engine
        # binding. A stage binds an engine per (role, repeat count) and
        # rebinds on every scored candidate, so a factory that built a
        # fresh HTTP client each time would open one connection pool per
        # evaluation against a provider the study is rate-limited by.
        # The transport itself is kept, not just its factory: it is the
        # sole owner of the retry budget and therefore the only record of
        # the attempts it spent, which no persisted row can be read back
        # for. The stage reports them off this object at the end.
        retrying_transport = (
            bind_openrouter_transport(execution_policy)[0] if paid else None
        )
        openrouter_factory = (
            None
            if retrying_transport is None
            else (lambda _policy: retrying_transport)
        )

        def bind_engine(*, role: EvalRole, num_seeds: int) -> EvalEngine:
            # One prepared experiment per (role, repeat count): the only
            # thing that differs between the three engines is which split
            # they are bound to, which is what "one procedure, three roles"
            # means for L4.
            prepared = family.build_experiment(
                pool,
                split_sizes=split_sizes,
                num_seeds=num_seeds,
                provider_call_config=provider_call_config,
            )
            config = runtime_config_for_role[role]
            transport_factory = (
                openrouter_factory
                if openrouter_factory is not None
                else fake_transport_factory(
                    gold_by_prompt=fake_gold_by_prompt(
                        prepared.experiment,
                        render_contract=family.render_contract(),
                        ceiling_template=family.probes.ceiling_template,
                    )
                )
            )
            # Built here rather than through ``config.build_engine``,
            # which takes neither a concurrency nor a policy and so always
            # yields whetstone's default width over the unwidened policy.
            # This is the same in-process driver that helper assembles for
            # ``driver_mode="in_process"``, which is the mode every study
            # config above binds; the study needs the two arguments the
            # helper does not forward, so it constructs the pair itself
            # rather than building an engine and discarding it.
            driver = GraphRolloutEvalDriver(
                eval_runner=family.eval_runner(),
                mutation_field=family.mutation_field,
                render_contract=family.render_contract(),
                transport_factory=transport_factory,
            )
            return RuntimeEvalEngine(
                store=cast("ObjectStore", store),
                experiment=prepared.experiment,
                sampling=prepared.experiment.eval_configs.split_for(
                    config.split_role
                ),
                execution_policy=execution_policy,
                driver=driver,
                concurrency=provider_concurrency,
            )

        task_ids_by_role = {
            EvalRole.INTERNAL: tuple(
                instance.id for instance in split.internal_eval
            ),
            EvalRole.OFFICIAL: tuple(
                instance.id for instance in split.official
            ),
            EvalRole.HELD_OUT: tuple(
                instance.id for instance in split.held_out
            ),
        }
        # The design's repeat count, not the calibration's: an arm stage
        # measures the design, and Stage 0 records what that design is. A
        # manifest with no design yet has no arm stage to run either, so
        # falling back to one repeat only affects Stage 0's own bind.
        k_repeat = 1 if manifest.design is None else manifest.design.k_repeat
        build_candidate = BuildCandidate(population.family)
        # One ledger for both scorers, because the reporting pass is one
        # bill: official-selection scoring and held-out evaluation reach
        # the same provider on the same invocation, and splitting them
        # would make the stage fold two partial totals that must then be
        # kept in step.
        report_spend = ReportSpendLedger(cast("ObjectStore", store))
        official = RoleScorer(
            bind_engine=bind_engine,
            role=EvalRole.OFFICIAL,
            task_ids=task_ids_by_role[EvalRole.OFFICIAL],
            num_seeds=k_repeat,
            build_candidate=build_candidate,
            spend_ledger=report_spend,
        )
        held_out = RoleScorer(
            bind_engine=bind_engine,
            role=EvalRole.HELD_OUT,
            task_ids=task_ids_by_role[EvalRole.HELD_OUT],
            num_seeds=k_repeat,
            build_candidate=build_candidate,
            spend_ledger=report_spend,
        )
        runner = StudyOptimizerRunner(
            study_dir=study_dir,
            family_id=population.family,
            transport=transport,
            split_sizes=split_sizes,
            n_per_stratum=population.n_per_stratum,
            pool_seed_start=population.pool_seed_start,
            task_model=manifest.models.task_model,
            # The pinned effort, reaching the *in-search* evaluations.
            # The engines bound above cover the reporting pass only; the
            # runner builds its own ``RunSpec`` per arm, so without this
            # the optimizers' own evaluations -- the K_REPEAT-multiplied
            # majority of the study's paid calls -- would run at the
            # provider's default under a design that pre-registered
            # otherwise.
            task_reasoning_effort=ReasoningEffort(
                manifest.models.task_reasoning_effort
            ),
            proposer_model=manifest.models.proposer_model,
            num_seeds=k_repeat,
            naive_template=family.probes.naive_template,
            store_path=study_dir / STUDY_STORE_NAME,
            # The protocol's pinned admission cap, passed rather than
            # left to the runner's own default. The two constants agree
            # today, which is exactly the problem: the cap is design --
            # it is what equalizes the Codex arm's eval budget against
            # the others (D12) -- so it has to *reach* the RunSpec from
            # the design rather than be reconstructed by a default that
            # happens to match.
            codex_capacity=CODEX_EVALUATE_CALL_CAP,
            # The wall the cap above needs to be reachable. Eight admitted
            # calls at ~120 s each is ~960 s of evaluation, and the
            # dependency's default wall is 600 s -- so leaving this to the
            # default did not merely risk a tight fit, it made the
            # pre-registered cap impossible to spend and terminalized the
            # arm partway through. Forwarded from the design for exactly
            # the reason the cap is.
            codex_wall_seconds=CODEX_WALL_SECONDS,
            allow_real_codex=allow_real_codex,
            discard_stale_runs=discard_stale_runs,
            # The same width the engines above were bound with. The runner
            # builds its own ``RunSpec`` per arm, so an unforwarded width
            # left every in-search evaluation at the ``RunSpec`` default
            # while the reporting pass ran at the operator's -- one stage
            # running at two widths, recorded as one.
            provider_concurrency=provider_concurrency,
            # The in-search route reports itself into the same witness the
            # reporting pass writes to, which is also where the
            # pre-registration refusal lives -- so an arm that would
            # evaluate at an unpinned effort is refused before it bills.
            record_provider_call=lambda config, policy: (
                _record_provider_call_config(
                    study_dir,
                    transport=transport,
                    config=config,
                    policy=policy,
                )
            ),
        )
        yield StageEnvironment(
            bind_engine=bind_engine,
            naive_candidate=naive,
            ceiling_candidate=ceiling,
            task_ids_by_role=task_ids_by_role,
            pool_ceiling=sum(split_sizes),
            run_optimizer=runner,
            score_official=official.score_official,
            evaluate_held_out=held_out.evaluate_held_out,
            load_recorded_run=runner.load_recorded_run,
            real_codex_authorized=allow_real_codex,
            transport=transport,
            provider_concurrency=provider_concurrency,
            # Read only by the arm stage's pre-dispatch width refusal.
            # It never reaches the runner: the runner records this
            # invocation's width on the runs it executes, and whether the
            # *reused* runs may carry a different one is a question about
            # the manifest, settled before anything is dispatched.
            allow_width_change=allow_width_change,
            # Read by the same pre-dispatch width refusal, for the case
            # the manifest cannot see: run directories surviving with no
            # stage row are of unrecoverable width, and this is the
            # operator having already said such directories are not
            # evidence to preserve.
            discard_stale_runs=discard_stale_runs,
            # The stage's own store, so a stage that evaluates through the
            # engine can price what it evaluated. It is the same connection
            # every engine writes into, which is what makes reading the
            # rows back a read of this stage's own evidence.
            store=cast("ObjectStore", store),
            report_spend=report_spend,
            # ``None`` on the fake transport, which reaches no provider
            # and so has no attempts to report.
            provider_attempts=retrying_transport,
        )
