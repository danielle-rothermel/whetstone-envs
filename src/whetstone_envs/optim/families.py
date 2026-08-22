"""The task-family registry the shared optimizer runner reads.

``run_optimizer`` drives whichever family a run names, so every piece of
family-specific knowledge it needs -- how to generate the pool, how to
prepare the experiment, which probe pair anchors it, which candidate field
optimizers mutate, and which placeholders a proposed template must keep --
lives behind one ``FamilySpec`` rather than inline in the runner.

This is the C3 generality boundary. A second family is admitted by
registering a ``FamilySpec``; if adding one required a change anywhere else
under ``whetstone_envs/optim/``, that is the domain leak the study is looking
for, not a change to absorb quietly.

Only ``c19`` is registered today. ``c18`` is registered by its own module
when that family's experiment builder lands. ``KNOWN_FAMILY_IDS`` already
records the identifier and :func:`register_family` already admits it, so an
unregistered-but-planned family reports a wiring gap rather than a typo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING, Protocol

from whetstone_envs.c19 import PROBES as C19_PROBES
from whetstone_envs.c19 import generate_pool as c19_generate_pool
from whetstone_envs.optim.experiment import (
    C19_MUTATION_FIELD,
    C19_NAMESPACE,
    C19_PROMPT_FIELDS,
    c19_render_contract,
    prepare_c19_experiment,
)
from whetstone_envs.probes import ProbePair

if TYPE_CHECKING:
    from dr_providers import ProviderCallConfig
    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.experiment.env import Experiment

    from whetstone_envs.pools import PoolSplit, TaskPool


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


class PreparedExperiment(Protocol):
    """What a family's experiment builder must hand the runner.

    ``prepare_c19_experiment`` already returns exactly this shape, and the
    reporting projection reads the same two attributes, so a second family
    satisfies the protocol by returning its own prepared pair.
    """

    @property
    def experiment(self) -> Experiment: ...

    @property
    def split(self) -> PoolSplit: ...


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


class RenderContractFactory(Protocol):
    """Build the family's template render contract."""

    def __call__(self) -> TemplateRenderContract: ...


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Everything the shared runner needs to drive one task family.

    The runner reads only this record, so it carries no family literal of
    its own. ``run_id_prefix`` names the family in a generated run id, which
    keeps artifact directories self-describing when two families share an
    output root.
    """

    family_id: str
    namespace: str
    #: The candidate payload field every optimizer mutates.
    mutation_field: str
    #: The placeholders a proposed template must keep, in contract order.
    prompt_fields: tuple[str, ...]
    #: One sentence naming the task, shown to a proposer as context.
    task_context: str
    #: Naive and ceiling anchors, and the scripted fake-transport bodies.
    probes: ProbePair
    generate_pool: PoolGenerator
    build_experiment: ExperimentBuilder
    render_contract: RenderContractFactory
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
    namespace=C19_NAMESPACE,
    mutation_field=C19_MUTATION_FIELD,
    prompt_fields=C19_PROMPT_FIELDS,
    task_context="Predict the MiniGrid fact asked by the question.",
    probes=C19_PROBES,
    generate_pool=c19_generate_pool,
    build_experiment=prepare_c19_experiment,
    render_contract=c19_render_contract,
    # The historical runner default, retained so an unparameterised run
    # keeps generating the pool it always generated.
    default_n_per_stratum=2,
    default_pool_seed_start=765_432,
    run_id_prefix=FamilyId.C19.value,
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


__all__ = [
    "KNOWN_FAMILY_IDS",
    "ExperimentBuilder",
    "FamilyId",
    "FamilySpec",
    "PoolGenerator",
    "PreparedExperiment",
    "RenderContractFactory",
    "family_spec",
    "register_family",
    "registered_family_ids",
]
