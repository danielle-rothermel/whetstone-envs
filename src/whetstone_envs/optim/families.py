"""The task-family registry the shared optimizer runner reads.

``run_optimizer`` drives whichever family a run names, so every piece of
family-specific knowledge it needs -- how to generate the pool, how to
prepare the experiment, which probe pair anchors it, which candidate field
optimizers mutate, which placeholders a proposed template must keep, and how
a generation is scored -- lives behind one ``FamilySpec`` rather than inline
in the runner.

This is the C3 generality boundary. A second family is admitted by
registering a ``FamilySpec``; if adding one required a change anywhere else
under ``whetstone_envs/optim/``, that is the domain leak the study is looking
for, not a change to absorb quietly.

Both ``c19`` and ``c18`` are registered here, and their registrations are the
only place either family's name appears in the runner's path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING, Protocol

from whetstone_envs.c18 import PROBES as C18_PROBES
from whetstone_envs.c19 import PROBES as C19_PROBES
from whetstone_envs.c19 import generate_pool as c19_generate_pool
from whetstone_envs.optim.c18_experiment import (
    C18_CONTRACT,
    C18_DEFAULT_N_PER_STRATUM,
    C18_DEFAULT_POOL_SEED_START,
    C18_TASK_CONTEXT,
    C18VerdictEvalProcedureRunner,
    c18_generate_pool,
    prepare_c18_experiment,
)
from whetstone_envs.optim.experiment import (
    C19_CONTRACT,
    ExperimentContract,
    prepare_c19_experiment,
)
from whetstone_envs.optim.scoring_runner import ExactMatchEvalProcedureRunner
from whetstone_envs.probes import ProbePair

if TYPE_CHECKING:
    from dr_providers import ProviderCallConfig
    from whetstone.eval.eval_procedure import EvalProcedureRunner
    from whetstone.experiment.candidate import TemplateRenderContract

    from whetstone_envs.optim.experiment import PreparedExperiment
    from whetstone_envs.pools import TaskPool

__all__ = [
    "KNOWN_FAMILY_IDS",
    "EvalRunnerFactory",
    "ExperimentBuilder",
    "FamilyId",
    "FamilySpec",
    "PoolGenerator",
    "family_spec",
    "register_family",
    "registered_family_ids",
]


@verify(UNIQUE)
class FamilyId(StrEnum):
    """Every task family the shared optimizer runner knows by name.

    Membership here is knowledge of the identifier, not of a registration:
    a family is runnable only once :func:`register_family` has bound a
    :class:`FamilySpec` to its id.
    """

    C19 = auto()
    C18 = auto()


#: Every family identifier the runner recognises, registered or not.
KNOWN_FAMILY_IDS: tuple[str, ...] = tuple(member.value for member in FamilyId)


class PoolGenerator(Protocol):
    """Generate a family's deterministic task pool."""

    def __call__(self, *, n_per_stratum: int, seed_start: int) -> TaskPool: ...


class ExperimentBuilder(Protocol):
    """Prepare a family's Experiment from one of its pools."""

    def __call__(
        self,
        pool: TaskPool,
        *,
        split_sizes: tuple[int, int, int],
        num_seeds: int,
        provider_call_config: ProviderCallConfig | None,
    ) -> PreparedExperiment: ...


class EvalRunnerFactory(Protocol):
    """Build the eval-node runner that scores this family's generations."""

    def __call__(self) -> EvalProcedureRunner: ...


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Everything the shared runner needs to drive one task family.

    The runner reads only this record, so it carries no family literal of
    its own. ``run_id_prefix`` names the family in a generated run id, which
    keeps artifact directories self-describing when two families share an
    output root.

    ``contract`` is the single owner of the family's persisted identity --
    namespace, mutation field, placeholders, and probe templates -- so the
    experiment builder, the render contract, and the optimizer adapters all
    read one record rather than three copies of the same facts.
    """

    family_id: str
    contract: ExperimentContract
    #: One sentence naming the task, shown to a proposer as context.
    task_context: str
    #: How a rendered task is described to an instruction proposer.
    rendering_rules: str
    #: How a generation becomes a score, described to that same proposer.
    example_execution: str
    #: Naive and ceiling anchors, and the scripted fake-transport bodies.
    probes: ProbePair
    generate_pool: PoolGenerator
    build_experiment: ExperimentBuilder
    #: Scores one generation against a task's frozen gold.
    eval_runner: EvalRunnerFactory
    #: Default pool size per stratum when a run does not choose one.
    default_n_per_stratum: int
    #: Default first generator seed when a run does not choose one.
    default_pool_seed_start: int
    run_id_prefix: str

    def __post_init__(self) -> None:
        if not self.prompt_fields:
            msg = f"family {self.family_id!r} declares no prompt fields"
            raise ValueError(msg)
        if self.default_n_per_stratum < 1:
            msg = (
                f"family {self.family_id!r} default_n_per_stratum must be "
                f"at least 1, got {self.default_n_per_stratum}"
            )
            raise ValueError(msg)

    @property
    def namespace(self) -> str:
        """The family's persisted namespace, owned by its contract."""
        return self.contract.namespace

    @property
    def mutation_field(self) -> str:
        """The candidate payload field every optimizer mutates."""
        return self.contract.mutation_field

    @property
    def prompt_fields(self) -> tuple[str, ...]:
        """The placeholders a proposed template must keep, in order."""
        return self.contract.prompt_fields

    @property
    def response_field(self) -> str:
        """The key this family's generated component answers under.

        A labeled demonstration files the task's gold under this key, so
        the demo teaches the answer rather than an empty string.
        """
        return self.contract.response_field

    def render_contract(self) -> TemplateRenderContract:
        """The contract every template for this family must satisfy."""
        return self.contract.render_contract()

    def proposal_bodies(self) -> tuple[str, ...]:
        """Scripted proposer bodies for a fake-transport run.

        A seed optimizer asks for one draft and keeps the naive initial
        candidate, so the first body must differ from that seed or the
        optimizer rejects a no-op mutation. Every body satisfies the
        family's render contract, which the runner's proposal path
        re-validates.
        """
        return (self.probes.ceiling_template, self.probes.naive_template)


_C19_SPEC = FamilySpec(
    family_id=FamilyId.C19.value,
    contract=C19_CONTRACT,
    task_context="Predict the MiniGrid fact asked by the question.",
    rendering_rules=(
        "Render the template with the task's grid, command, and question "
        "substituted for its placeholders."
    ),
    example_execution=(
        "The rendered prompt is sent to the task model and its reply is "
        "scored by exact match against the MiniGrid oracle answer."
    ),
    probes=C19_PROBES,
    generate_pool=c19_generate_pool,
    build_experiment=prepare_c19_experiment,
    eval_runner=ExactMatchEvalProcedureRunner,
    # The historical runner default, retained so an unparameterised run
    # keeps generating the pool it always generated.
    default_n_per_stratum=2,
    default_pool_seed_start=765_432,
    run_id_prefix=FamilyId.C19.value,
)

_C18_SPEC = FamilySpec(
    family_id=FamilyId.C18.value,
    contract=C18_CONTRACT,
    task_context=C18_TASK_CONTEXT,
    rendering_rules=(
        "Render the template with the task's facts-and-rules text and its "
        "query statement substituted for its placeholders."
    ),
    example_execution=(
        "The rendered prompt is sent to the task model, its final verdict "
        "line is extracted, and that verdict is scored against the "
        "entailment oracle's True/False label."
    ),
    probes=C18_PROBES,
    generate_pool=c18_generate_pool,
    build_experiment=prepare_c18_experiment,
    eval_runner=C18VerdictEvalProcedureRunner,
    default_n_per_stratum=C18_DEFAULT_N_PER_STRATUM,
    default_pool_seed_start=C18_DEFAULT_POOL_SEED_START,
    run_id_prefix=FamilyId.C18.value,
)

_REGISTRY: dict[str, FamilySpec] = {}


def register_family(spec: FamilySpec) -> None:
    """Admit one family to the runner.

    Registration is rejected for an unknown identifier and for a duplicate,
    so a second family cannot silently shadow the first.
    """
    if spec.family_id not in set(KNOWN_FAMILY_IDS):
        msg = (
            f"unknown family {spec.family_id!r}; "
            f"known families are {KNOWN_FAMILY_IDS}"
        )
        raise ValueError(msg)
    if spec.family_id in _REGISTRY:
        msg = f"family {spec.family_id!r} is already registered"
        raise ValueError(msg)
    _REGISTRY[spec.family_id] = spec


def family_spec(family_id: str) -> FamilySpec:
    """Return the registered spec for ``family_id``.

    A known-but-unregistered family reports that distinctly from an
    unrecognised name: the first is a wiring gap, the second is a typo.
    """
    spec = _REGISTRY.get(family_id)
    if spec is not None:
        return spec
    if family_id in set(KNOWN_FAMILY_IDS):
        msg = (
            f"family {family_id!r} is known but not registered; "
            "import the module that registers it"
        )
        raise ValueError(msg)
    msg = (
        f"unsupported family {family_id!r}; "
        f"registered families are {registered_family_ids()}"
    )
    raise ValueError(msg)


def registered_family_ids() -> tuple[str, ...]:
    """Every family currently runnable, in registration order."""
    return tuple(_REGISTRY)


register_family(_C19_SPEC)
register_family(_C18_SPEC)
