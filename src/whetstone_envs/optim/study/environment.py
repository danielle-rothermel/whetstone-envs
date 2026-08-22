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

from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from dr_store.sync import open_sqlite
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.provider import (
    fake_gold_by_prompt,
    fake_transport_factory,
)
from whetstone_envs.optim.rows import task_rows_from_instances
from whetstone_envs.optim.study.arms import (
    BuildCandidate,
    RoleScorer,
    StudyOptimizerRunner,
)
from whetstone_envs.optim.study.manifest import (
    STUDY_STORE_NAME,
    SplitsRecord,
    read_study_manifest,
)
from whetstone_envs.optim.study.stages import StageEnvironment
from whetstone_envs.reporting.schema import SPLIT_ROLE_BY_REPORT_ROLE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dr_store import ObjectStore
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.candidate import Candidate

    from whetstone_envs.pools import PoolSplit

__all__ = [
    "FAKE_TRANSPORT",
    "SPLIT_ROLE_BY_EVAL_ROLE",
    "STUDY_STORE_NAME",
    "anchor_candidates",
    "bound_stage_environment",
]

#: The only transport the study harness binds today. Named rather than
#: inline so the refusal below and the default above cannot drift apart.
FAKE_TRANSPORT = "fake"

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


@contextmanager
def bound_stage_environment(
    study_dir: Path, *, transport: str = FAKE_TRANSPORT
) -> Iterator[StageEnvironment]:
    """Open a study's store and bind one engine per evaluation role.

    ``transport="fake"`` is the default because every stage in this package
    is exercised without provider calls; a paid stage names ``openrouter``
    explicitly, so no code path reaches a provider by omission.
    """
    if transport != FAKE_TRANSPORT:
        # The provider-backed binder belongs with the stage that spends, and
        # spending is authorized at a gate rather than by a default. Naming
        # the gap beats a partially-wired live path that looks ready.
        raise ValueError(
            f"transport {transport!r} is not wired into the study harness; "
            "only fake-transport stages run today"
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
    # The split is a deterministic function of the pool and the sizes, so
    # one reference preparation names every role's tasks whatever repeat
    # count a later engine binds at.
    split = family.build_experiment(
        pool,
        split_sizes=split_sizes,
        num_seeds=1,
        provider_call_config=None,
    ).split
    _require_recorded_population(split, manifest.splits)
    with open_sqlite(str(study_dir / STUDY_STORE_NAME)) as store:

        def bind_engine(*, role: EvalRole, num_seeds: int) -> EvalEngine:
            # One prepared experiment per (role, repeat count): the only
            # thing that differs between the three engines is which split
            # they are bound to, which is what "one procedure, three roles"
            # means for L4.
            prepared = family.build_experiment(
                pool,
                split_sizes=split_sizes,
                num_seeds=num_seeds,
                provider_call_config=None,
            )
            config = ReferenceEvalRuntimeConfig(
                split_role=SPLIT_ROLE_BY_EVAL_ROLE[role],
                transport_api_key_env="WHETSTONE_TOY_API_KEY",
            )
            return config.build_engine(
                cast("ObjectStore", store),
                experiment=prepared.experiment,
                eval_runner=family.eval_runner(),
                mutation_field=family.mutation_field,
                render_contract=family.render_contract(),
                transport_factory=fake_transport_factory(
                    gold_by_prompt=fake_gold_by_prompt(
                        prepared.experiment,
                        render_contract=family.render_contract(),
                        ceiling_template=family.probes.ceiling_template,
                    )
                ),
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
        official = RoleScorer(
            bind_engine=bind_engine,
            role=EvalRole.OFFICIAL,
            task_ids=task_ids_by_role[EvalRole.OFFICIAL],
            num_seeds=k_repeat,
            build_candidate=build_candidate,
        )
        held_out = RoleScorer(
            bind_engine=bind_engine,
            role=EvalRole.HELD_OUT,
            task_ids=task_ids_by_role[EvalRole.HELD_OUT],
            num_seeds=k_repeat,
            build_candidate=build_candidate,
        )
        runner = StudyOptimizerRunner(
            study_dir=study_dir,
            family_id=population.family,
            transport=transport,
            split_sizes=split_sizes,
            n_per_stratum=population.n_per_stratum,
            pool_seed_start=population.pool_seed_start,
            task_model=manifest.models.task_model,
            proposer_model=manifest.models.proposer_model,
            num_seeds=k_repeat,
            naive_template=family.probes.naive_template,
            store_path=study_dir / STUDY_STORE_NAME,
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
        )
