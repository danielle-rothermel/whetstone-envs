"""The shared optimizer runner every task family reaches through.

``run_optimizer`` drives one optimizer over one family's prepared
experiment and writes the run's artifacts off-repo. It reads family-specific
knowledge only from the :mod:`whetstone_envs.optim.families` registry, so it
carries no family literal of its own -- that is the C3 generality property
the second family exercises.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dr_providers import ProviderKind
from dr_store.sync import open_sqlite
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    copro_run_request,
    prepare_codex_run,
    prepare_copro_run,
    prepare_gepa_run,
    prepare_miprov2_run,
)
from whetstone.core.identity import (
    compute_identity_hash,
)
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.core.roles import EvalRole
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.optim.adapters import MappingAdapterRegistry, OptimizerAdapter
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
from whetstone.optim.copro.control import (
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optim.copro.proposal_contract import CoproProposalContractRecord
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    Miprov2Adapter,
)
from whetstone.optim.miprov2.control import Miprov2Control
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    ProviderProposerTransport,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import ToolAdmissionAuthority
from whetstone.provider.language_model import PlainPromptAdapter

from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
    CODEX_ADAPTER_KEY,
    CODEX_DEFAULT_BINARY,
    CODEX_EVALUATE_CALL_CAP,
    CODEX_REASONING_EFFORTS,
    CodexReasoningEffort,
    CodexTestSeam,
    RealCodexRefusedError,
    build_codex_adapter,
    build_codex_control,
    codex_run_root,
    refuse_unauthorized_real_codex,
    resolve_codex_agent_model,
)
from whetstone_envs.optim.codex_runtime import EnvsCodexRuntimeConfig
from whetstone_envs.optim.experiment import provider_call_config_ref
from whetstone_envs.optim.families import (
    KNOWN_FAMILY_IDS,
    FamilyId,
    FamilySpec,
    family_spec,
    registered_family_ids,
)
from whetstone_envs.optim.gepa import build_gepa_adapter
from whetstone_envs.optim.miprov2 import (
    DEFAULT_MIPROV2_FULL_EVAL_STEPS,
    DEFAULT_MIPROV2_MINIBATCH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    DEMO_MODES,
    Miprov2DemoMode,
    build_miprov2_adapter,
    build_miprov2_control,
    build_miprov2_state,
    miprov2_run_ref,
)
from whetstone_envs.optim.nulls import (
    NULL_RANDOM_OPTIMIZER,
    NullRandomTransport,
)
from whetstone_envs.optim.provider import (
    DEFAULT_PROVIDER_CONCURRENCY,
    bind_openrouter_transport,
    fake_gold_by_prompt,
    fake_transport_factory,
    hardened_execution_policy,
    openrouter_seeded_call_config,
    widened_execution_policy,
)
from whetstone_envs.optim.run_cost import project_run_cost, write_run_cost
from whetstone_envs.optim.split import (
    COPRO_SHAPED_OPTIMIZERS,
    GEPA_OPTIMIZER,
    MIN_COPRO_BREADTH,
    TRAIN_VAL_OPTIMIZERS,
    partition_internal_split,
)
from whetstone_envs.reporting.projection import project_trajectory_report
from whetstone_envs.reporting.publication import (
    durable_run_boundary,
    prepare_output_root,
    publish_trajectory_report,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_store import ObjectStore
    from whetstone.coordination.harness_run_controller import OptimRunLaunch
    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.env import Experiment
    from whetstone.optim.codex.adapter import CodexAdapter
    from whetstone.optim.codex.runner import CodexPromptContext
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.proposal.proposer import ProposerTransport

    from whetstone_envs.optim.codex_runtime import CodexRuntimeTransport

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm" / "data" / "runs" / ("whetstone-envs")
)

#: Every optimizer the shared runner can drive today.
#:
#: ``null-random`` (null-A) is here because it is a *control for selection*:
#: it must spend the same proposal budget and fill the same slots as the
#: optimizer it stands in for, and it must produce the same evidence -- a
#: result, a store, an audit, priced cost rows -- or it cannot be compared
#: against the arms it controls. It is COPRO's own search shape with
#: :class:`~whetstone_envs.optim.nulls.NullRandomTransport` substituted for
#: the proposer, so it reaches that evidence through this runner rather than
#: through a parallel path of its own.
#:
#: ``null-identity`` (null-B) is absent: it proposes nothing, so it has no
#: search to drive and no optimizer-fidelity invariant to audit. The study
#: harness synthesizes its record directly. Admitting a name the runner
#: cannot drive would fail late, inside a durable run boundary, instead of
#: at spec validation.
OPTIMIZERS = ("codex", "copro", "gepa", "miprov2", NULL_RANDOM_OPTIMIZER)

TRANSPORTS = ("fake", "openrouter")

#: Retained COPRO search shape: two drafts per step, one step of depth.
DEFAULT_COPRO_BREADTH = 2
DEFAULT_COPRO_DEPTH = 1
#: Each optimizer's own seed default, used when a spec names none. These
#: mirror the values ``configure_gepa`` and ``configure_miprov2`` already
#: default to, so an unseeded run keeps the control identity it always had.
GEPA_DEFAULT_SEED = 0
MIPROV2_DEFAULT_SEED = 9
#: null-A's own seed default. It matches the study spec's ``null-random``
#: arm seed, so a single run reproduces the control the study would have
#: drawn at that arm's first seed.
NULL_RANDOM_DEFAULT_SEED = 5000

#: The MIPROv2 candidate count below which minibatching exhausted the
#: search space and raised inside the durable run boundary (D3's defect
#: (d), whetstone-ai #137). Two candidates and a minibatch let consecutive
#: full-eval steps promote every observed combination, and
#: ``select_promotion`` fell out of its ranked loop into a bare
#: ``ValueError: No valid program found in param_score_dict``.
MIPROV2_MINIBATCH_MIN_CANDIDATES = 3

#: The first whetstone-ai release whose ``select_promotion`` degrades to
#: DSPy's own behaviour rather than raising. At or above it the shape is
#: runnable and the refusal below lifts; below it the refusal is the only
#: thing standing between a study arm and a run that dies mid-flight.
MIPROV2_SPENT_COMBINATION_FIX_VERSION = (0, 1, 9)

#: Where the refusal points a reader who wants the upstream story.
MIPROV2_SPENT_COMBINATION_ISSUE = (
    "https://github.com/danielle-rothermel/whetstone-ai/pull/137"
)

#: The default split, kept small so an unparameterised run stays a smoke run.
DEFAULT_SPLIT_SIZES = (2, 2, 0)


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One optimizer run over one task family.

    Every field the study varies is explicit here, so a run is fully
    described by its spec and nothing is read from module state.
    """

    optimizer: str
    transport: str
    #: The registered task family this run drives.
    family: str = FamilyId.C19.value
    split_sizes: tuple[int, int, int] = DEFAULT_SPLIT_SIZES
    output_dir: Path | None = None
    run_id: str | None = None
    model: str = "openai/gpt-4.1-nano"
    #: The proposer's model. ``None`` reuses ``model`` for both roles.
    proposer_model: str | None = None
    #: MIPROv2's demonstration regime; ignored by COPRO and GEPA.
    demo_mode: str = Miprov2DemoMode.FEWSHOT.value
    #: Repeats per task (K_REPEAT). One seed keeps a run deterministic.
    num_seeds: int = 1
    #: Instances generated per stratum. ``None`` takes the family default.
    n_per_stratum: int | None = None
    #: First generator seed for the pool. ``None`` takes the family default.
    pool_seed_start: int | None = None
    #: This run's algorithmic seed. ``None`` keeps each optimizer's own
    #: default, which is what an unparameterised smoke run wants.
    #:
    #: GEPA and MIPROv2 carry it into their controls as an explicit field.
    #: ``CoproControl`` has no seed: COPRO's stochasticity is the proposer
    #: LM, so its effective seed is the provider ``SEED`` control plus
    #: proposal ordering. The field is still recorded for a COPRO run so the
    #: study manifest can state what was requested and how it was honoured;
    #: :func:`seed_disposition` names that difference. The study assigns
    #: disjoint per-optimizer ranges; choosing them is the study's concern,
    #: not the runner's, so the runner accepts any integer.
    seed: int | None = None
    #: COPRO candidates proposed per step.
    copro_breadth: int = DEFAULT_COPRO_BREADTH
    #: COPRO search depth; step count is ``depth + 1``.
    copro_depth: int = DEFAULT_COPRO_DEPTH
    #: GEPA's paid metric-call ceiling. ``None`` keeps the family default of
    #: one full pass over the trainset plus one reflection minibatch.
    gepa_max_metric_calls: int | None = None
    #: Traces GEPA's reflection proposer consumes per reflection round.
    #: ``None`` keeps the family's own single-trace default. A study pins
    #: this, because how many traces the reflection step sees is part of
    #: the proposer's input rather than a runtime detail.
    gepa_reflection_minibatch_size: int | None = None
    #: Whether MIPROv2 evaluates each trial on a sampled minibatch rather
    #: than the whole validation split. Off by default, which is the
    #: schedule this runner has always produced; the protocol's auto-light
    #: configuration turns it on.
    miprov2_minibatch: bool = DEFAULT_MIPROV2_MINIBATCH
    #: Tasks per minibatched trial. ``None`` takes the whole validation
    #: split, which is what a non-minibatched trial evaluates.
    miprov2_minibatch_size: int | None = None
    #: Trials between full-validation re-evaluations of the incumbent.
    miprov2_minibatch_full_eval_steps: int = DEFAULT_MIPROV2_FULL_EVAL_STEPS
    #: How many task evaluations run against the provider at once.
    #:
    #: An execution property, not an algorithmic one: it changes how long
    #: the run takes, never what it computes, so it is not part of any
    #: control record and no optimizer reads it. It sets the evaluation
    #: engine's worker pool and widens the HTTP connection pool to match.
    provider_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY
    #: MIPROv2 optimization trials. The default is this runner's own shape,
    #: which is below the protocol's auto-light 10; Wave 3's measured call
    #: counts are the cost of the default, and raising this raises them.
    miprov2_num_trials: int = DEFAULT_MIPROV2_NUM_TRIALS
    #: MIPROv2 instruction/fewshot candidates per component. The default is
    #: below the protocol's auto-light 6, for the same reason.
    miprov2_num_candidates: int = DEFAULT_MIPROV2_NUM_CANDIDATES
    #: The explicit train/val partition of the internal split, required by
    #: every optimizer with a train/val concept (``miprov2`` and ``gepa``)
    #: and refused on the others. The trainset is the first ``train_size``
    #: tasks of the internal split and the valset the next ``val_size``, so
    #: the two sets are disjoint and reproducible from the spec alone. They
    #: have no default: an in-search improvement measured on tasks the
    #: optimizer trained on cannot be told apart from memorization, so a
    #: run must state the partition it is claiming.
    train_size: int | None = None
    val_size: int | None = None
    #: Extra scripted proposer bodies for a fake-transport run, appended to
    #: the family's own. The family scripts a ceiling draft and the naive
    #: seed; the seed is rejected as a no-op mutation, so a fake round can
    #: only ever land one accepted draft. Supplying further distinct bodies
    #: is what lets a ``breadth`` above 2 produce a genuinely multi-draft
    #: round. Each must satisfy the family's render contract, which the
    #: proposal path re-validates. Refused on a real transport, where the
    #: proposer -- not the runner -- writes the bodies.
    extra_proposal_bodies: tuple[str, ...] = ()
    #: The Codex arm's admitted evaluate-call cap: the per-run
    #: ``ToolCapacity.max_accepted_calls`` and the Step's ``tool_calls``
    #: budget at once. ``None`` takes the arm's own cap,
    #: :data:`~whetstone_envs.optim.codex.CODEX_EVALUATE_CALL_CAP`.
    #: Rejected on other optimizers so it cannot look honoured when
    #: nothing reads it.
    codex_capacity: int | None = None
    #: The Codex CLI this run spawns. The default is the real binary,
    #: resolved on the run PATH; a test overrides it to the scripted fake.
    #: Rejected on other optimizers for the same reason as the cap.
    codex_binary: str = CODEX_DEFAULT_BINARY
    #: Codex's own model, and how hard it reasons. ``None`` means the
    #: arm's own default agent model -- *not* the run's ``model``, which
    #: is the task model. The two are different products on different
    #: routes: the task model is an OpenRouter route the Codex CLI cannot
    #: run at all, and a subscription session refuses it before the agent
    #: emits a token. See
    #: :data:`~whetstone_envs.optim.codex.CODEX_DEFAULT_AGENT_MODEL`.
    codex_model: str | None = None
    codex_reasoning_effort: str = CodexReasoningEffort.MEDIUM.value
    #: The Codex agent's wall budget in seconds. ``None`` keeps
    #: whetstone-ai's own default.
    codex_wall_seconds: float | None = None
    #: Half of the deliberate opt-in to a real, billed Codex session. A
    #: Codex run without a test seam is refused unless this is set *and*
    #: :data:`~whetstone_envs.optim.codex.ALLOW_REAL_CODEX_ENV` names the
    #: opt-in in the process environment -- see
    #: :func:`~whetstone_envs.optim.codex.refuse_unauthorized_real_codex`.
    #: This field alone cannot authorize spend, which is why it is safe
    #: for a serialized spec to carry it. Rejected on other optimizers,
    #: like every other Codex-scoped setting.
    allow_real_codex: bool = False


#: How a run's ``seed`` reaches the optimizer, recorded per optimizer.
#:
#: These are manifest values, not free-form prose: the study records which
#: arm carried an explicit control seed and which did not.
SEED_DISPOSITION_CONTROL_FIELD = "control-seed-field"
SEED_DISPOSITION_PROVIDER_ONLY = "provider-seed-control-only"


def installed_whetstone_ai_version() -> str | None:
    """The installed whetstone-ai version string, or ``None`` if absent.

    Absent is a real state on this repo's base install: the ``optim``
    extra installs whetstone-ai only on Python 3.13+, and this module is
    reachable from a spec-validation test that never builds a run.
    """
    try:
        return package_version("whetstone-ai")
    except PackageNotFoundError:
        return None


def _version_text(parts: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in parts)


def _release_tuple(text: str) -> tuple[int, ...]:
    """The leading numeric release of a version string.

    Only the release segment is compared, so a pre-release or local
    version of the fixed release still reads as the fixed release. A
    segment that is not an integer stops the parse rather than raising:
    the caller's fallback -- refuse -- is the safe answer for a version
    this function cannot rank.
    """
    parts: list[int] = []
    for segment in text.split("."):
        digits = ""
        for character in segment:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(segment):
            break
    return tuple(parts)


def _miprov2_spent_combination_fixed() -> bool:
    """Whether the installed whetstone-ai carries the #137 fallback.

    An unreadable or absent version reads as *not fixed*: the refusal is
    what stops a study arm from dying inside a durable run boundary, so
    the uncertain case keeps it rather than lifting it on a guess.
    """
    installed = installed_whetstone_ai_version()
    if installed is None:
        return False
    release = _release_tuple(installed)
    if not release:
        return False
    return release >= MIPROV2_SPENT_COMBINATION_FIX_VERSION


def _validate_miprov2_settings(spec: RunSpec) -> None:
    """Refuse MIPROv2 minibatch settings the run cannot honour.

    Like the other optimizer-scoped settings, these are refused at spec
    validation rather than inside the durable run boundary, and refused on
    another optimizer rather than silently ignored -- a setting that looks
    honoured but is not is how a study comes to misdescribe its own arm.
    """
    non_default = (
        spec.miprov2_minibatch != DEFAULT_MIPROV2_MINIBATCH
        or spec.miprov2_minibatch_size is not None
        or spec.miprov2_minibatch_full_eval_steps
        != DEFAULT_MIPROV2_FULL_EVAL_STEPS
        or spec.miprov2_num_trials != DEFAULT_MIPROV2_NUM_TRIALS
        or spec.miprov2_num_candidates != DEFAULT_MIPROV2_NUM_CANDIDATES
    )
    if non_default and spec.optimizer != "miprov2":
        raise ValueError("miprov2 settings apply only to --optimizer miprov2")
    if (
        spec.miprov2_minibatch_size is not None
        and spec.miprov2_minibatch_size < 1
    ):
        raise ValueError("miprov2_minibatch_size must be at least 1")
    if spec.miprov2_minibatch and spec.miprov2_minibatch_size is None:
        # D3's defect (e). Left unset, ``configure_miprov2`` resolves the
        # batch to the whole valset, so "minibatch on" silently means "no
        # minibatch" -- and the F16 fan-out invariant then correctly FAILs
        # the audit of a run that already spent. Refusing the combination
        # here turns a paid audit failure into a free validation error.
        raise ValueError(
            "--miprov2-minibatch requires --miprov2-minibatch-size: left "
            "unset the batch is the whole validation split, so "
            "minibatching is on in name only and the run's "
            "mipro_minibatch_sizing invariant fails after it has spent"
        )
    if (
        spec.miprov2_minibatch
        and spec.miprov2_num_candidates < MIPROV2_MINIBATCH_MIN_CANDIDATES
        and not _miprov2_spent_combination_fixed()
    ):
        raise ValueError(
            f"miprov2_num_candidates {spec.miprov2_num_candidates} with "
            f"minibatching requires whetstone-ai "
            f"{_version_text(MIPROV2_SPENT_COMBINATION_FIX_VERSION)} or "
            f"newer; the installed "
            f"{installed_whetstone_ai_version() or 'whetstone-ai'} raises "
            f"'No valid program found in param_score_dict' inside the "
            f"durable run boundary once consecutive full-eval steps have "
            f"promoted every observed combination. Use at least "
            f"{MIPROV2_MINIBATCH_MIN_CANDIDATES} candidates, turn "
            f"minibatching off, or upgrade -- see "
            f"{MIPROV2_SPENT_COMBINATION_ISSUE}"
        )
    if spec.miprov2_minibatch_full_eval_steps < 1:
        raise ValueError(
            "miprov2_minibatch_full_eval_steps must be at least 1"
        )
    if spec.miprov2_num_trials < 1:
        raise ValueError("miprov2_num_trials must be at least 1")
    if spec.miprov2_num_candidates < 1:
        raise ValueError("miprov2_num_candidates must be at least 1")


def _validate_train_val_split(spec: RunSpec) -> None:
    """Refuse a train/val split the run cannot honestly claim.

    Required for every optimizer in :data:`TRAIN_VAL_OPTIMIZERS` and
    refused on the others, for the same reason the other optimizer-scoped
    settings are: a size that looks honoured but is not is how a study
    comes to misdescribe its own arm.

    Both sizes are checked against the *internal* split, which is the only
    split these optimizers may see -- the official and held-out splits are
    not theirs to train on. Refused here, at pure spec validation, so an
    unrunnable partition never reaches the durable run boundary.
    """
    supplied = spec.train_size is not None or spec.val_size is not None
    if spec.optimizer not in TRAIN_VAL_OPTIMIZERS:
        if supplied:
            raise ValueError(
                "train_size and val_size apply only to "
                f"--optimizer {{{', '.join(TRAIN_VAL_OPTIMIZERS)}}}"
            )
        return
    if spec.train_size is None or spec.val_size is None:
        raise ValueError(
            f"--optimizer {spec.optimizer} requires an explicit "
            "--train-size and --val-size partition of the internal split"
        )
    if spec.train_size < 1:
        raise ValueError("train_size must be at least 1")
    if spec.val_size < 1:
        raise ValueError("val_size must be at least 1")
    internal = spec.split_sizes[0]
    if spec.train_size + spec.val_size > internal:
        raise ValueError(
            f"train_size {spec.train_size} + val_size {spec.val_size} "
            f"exceeds the internal split of {internal}"
        )
    if (
        spec.optimizer == GEPA_OPTIMIZER
        and spec.train_size + spec.val_size != internal
    ):
        # GEPA is stricter than the others by construction: whetstone's
        # GEPA factory builds its data registry from the whole internal
        # split and then requires the control's trainset and valset to
        # *cover* it exactly, so a partition that merely fits inside the
        # split is rejected. That rejection happens inside the durable run
        # boundary, after the run directory exists, so the same rule is
        # restated here at pure spec validation -- a partial partition then
        # refuses before anything is written.
        raise ValueError(
            f"--optimizer {spec.optimizer} requires train_size + val_size "
            f"to cover the internal split exactly: {spec.train_size} + "
            f"{spec.val_size} = {spec.train_size + spec.val_size}, not "
            f"{internal}. GEPA's data registry is built from the whole "
            "internal split and its trainset and valset must partition it"
        )


def _validate_codex_settings(spec: RunSpec) -> None:
    """Refuse Codex settings the run cannot honour.

    Refused on another optimizer rather than silently ignored, like every
    other optimizer-scoped setting: a setting that looks honoured but is
    not is how a study comes to misdescribe its own arm.
    """
    non_default = (
        spec.codex_capacity is not None
        or spec.codex_binary != CODEX_DEFAULT_BINARY
        or spec.codex_model is not None
        or spec.codex_reasoning_effort != CodexReasoningEffort.MEDIUM.value
        or spec.codex_wall_seconds is not None
        or spec.allow_real_codex
    )
    if non_default and spec.optimizer != "codex":
        raise ValueError("codex settings apply only to --optimizer codex")
    if spec.codex_capacity is not None and spec.codex_capacity < 1:
        raise ValueError("codex_capacity must be at least 1")
    if spec.codex_wall_seconds is not None and spec.codex_wall_seconds <= 0:
        raise ValueError("codex_wall_seconds must be positive")
    if spec.codex_reasoning_effort not in CODEX_REASONING_EFFORTS:
        raise ValueError(
            "codex_reasoning_effort must be one of "
            f"{list(CODEX_REASONING_EFFORTS)}"
        )


def seed_disposition(optimizer: str) -> str:
    """Name how ``optimizer`` honours a run's requested seed.

    COPRO is the honest exception: ``CoproControl`` carries no seed field,
    so a COPRO run's reproducibility rests on the provider ``SEED`` control
    and proposal ordering rather than on an algorithmic seed. Recording that
    beats faking a seed the control never reads.
    """
    if optimizer == "copro":
        return SEED_DISPOSITION_PROVIDER_ONLY
    return SEED_DISPOSITION_CONTROL_FIELD


def default_output_dir(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / run_id


def _provider_config_resolver(experiment: Experiment):
    provider_config = experiment.rollout_graph.provider_call_config

    def resolve(_ref: object):
        return provider_config

    return resolve


def _proposal_contract(family: FamilySpec) -> CoproProposalContractRecord:
    placeholders = ", ".join(f"{{{field}}}" for field in family.prompt_fields)
    return CoproProposalContractRecord(
        target_name=f"{family.family_id}_prompt_template",
        task_context=family.task_context,
        output_rule=(
            f"Return one non-empty prompt template that uses {placeholders}."
        ),
    )


#: The family-namespaced schema name for COPRO's inline executor policy.
COPRO_EXECUTOR_SCHEMA_SUFFIX = "copro_proposal_executor"


def _copro_adapter(  # noqa: PLR0913
    *,
    engine: EvalEngine,
    control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    family: FamilySpec,
    extra_proposal_bodies: tuple[str, ...] = (),
) -> CoproAdapter:
    """The COPRO adapter this run drives, scripted when transport is fake.

    ``extra_proposal_bodies`` extends the family's scripted bodies so a
    fake run at a wider ``breadth`` proposes genuinely distinct drafts
    rather than re-offering the seed.
    """
    transport = proposer_transport or FakeProposerTransport(
        {},
        default=family.proposal_bodies(extra_proposal_bodies),
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
    )
    return CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema=(f"{family.namespace}.{COPRO_EXECUTOR_SCHEMA_SUFFIX}"),
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ValidatedSpec:
    """A spec proven runnable, with its family and demo mode resolved."""

    family: FamilySpec
    demo_mode: Miprov2DemoMode
    n_per_stratum: int
    pool_seed_start: int


def _validate_gepa_settings(spec: RunSpec) -> None:
    """Reject GEPA controls a non-GEPA run could not honour.

    Refused rather than ignored, for ``_validate_miprov2_settings``'s
    reason: a control that looks honoured and is not is how a run comes to
    misdescribe itself. Both settings are refused here, at spec validation,
    rather than inside the durable run boundary where the failure would
    leave a run directory behind.
    """
    settings = (
        ("gepa_max_metric_calls", spec.gepa_max_metric_calls),
        (
            "gepa_reflection_minibatch_size",
            spec.gepa_reflection_minibatch_size,
        ),
    )
    for name, value in settings:
        if value is None:
            continue
        if spec.optimizer != GEPA_OPTIMIZER:
            raise ValueError(f"{name} applies only to --optimizer gepa")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")


def _validate_spec(spec: RunSpec) -> _ValidatedSpec:
    """Reject an unrunnable spec before any durable effect happens."""
    if spec.optimizer not in set(OPTIMIZERS):
        raise ValueError(f"unsupported optimizer {spec.optimizer!r}")
    if spec.transport not in set(TRANSPORTS):
        raise ValueError(f"unsupported transport {spec.transport!r}")
    if spec.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if spec.copro_breadth < MIN_COPRO_BREADTH:
        # Refusing here keeps the failure at spec validation instead of
        # inside the durable run boundary, where it would leave a run
        # directory behind.
        raise ValueError(f"copro_breadth must be at least {MIN_COPRO_BREADTH}")
    if spec.copro_depth < 0:
        raise ValueError("copro_depth must be non-negative")
    _validate_gepa_settings(spec)
    _validate_miprov2_settings(spec)
    _validate_train_val_split(spec)
    if spec.extra_proposal_bodies and spec.transport != "fake":
        raise ValueError(
            "extra_proposal_bodies applies only to --transport fake"
        )
    _validate_codex_settings(spec)
    family = family_spec(spec.family)
    n_per_stratum = (
        family.default_n_per_stratum
        if spec.n_per_stratum is None
        else spec.n_per_stratum
    )
    if n_per_stratum < 1:
        raise ValueError("n_per_stratum must be at least 1")
    pool_seed_start = (
        family.default_pool_seed_start
        if spec.pool_seed_start is None
        else spec.pool_seed_start
    )
    try:
        demo_mode = Miprov2DemoMode(spec.demo_mode)
    except ValueError as error:
        raise ValueError(
            f"unsupported demo mode {spec.demo_mode!r}"
        ) from error
    return _ValidatedSpec(
        family=family,
        demo_mode=demo_mode,
        n_per_stratum=n_per_stratum,
        pool_seed_start=pool_seed_start,
    )


@dataclass(frozen=True, slots=True)
class _BoundOptimizer:
    """The one adapter a run registers, plus what its launch needs."""

    adapter_key: str
    adapter: OptimizerAdapter
    gepa_control: GepaControl | None = None
    miprov2_control: Miprov2Control | None = None
    miprov2_adapter: Miprov2Adapter | None = None
    codex_control: CodexControl | None = None


def build_codex_runtime_config(
    *, spec: RunSpec, validated: _ValidatedSpec
) -> EnvsCodexRuntimeConfig:
    """The runtime config the Codex MCP evaluation server rebuilds from.

    Derived from the same spec the in-process engine was built from, so
    the two cannot drift: every generation parameter is read from one
    place. ``build_codex_adapter`` then proves the rebuild lands on the
    same Eval Config before anything is spawned.
    """
    return EnvsCodexRuntimeConfig(
        family_id=validated.family.family_id,
        split_sizes=spec.split_sizes,
        n_per_stratum=validated.n_per_stratum,
        pool_seed_start=validated.pool_seed_start,
        num_seeds=spec.num_seeds,
        transport=cast("CodexRuntimeTransport", spec.transport),
        model=spec.model,
        # Forwarded rather than defaulted: the server rebuilds from this
        # config alone, so an unforwarded width would leave the Codex arm
        # evaluating at whetstone's default while every other arm ran at
        # the operator's.
        provider_concurrency=spec.provider_concurrency,
    )


def _bind_optimizer(  # noqa: PLR0913
    *,
    spec: RunSpec,
    validated: _ValidatedSpec,
    store: ObjectStore,
    engine: EvalEngine,
    experiment: Experiment,
    run_id: str,
    copro_control: CoproControl,
    prompt_adapter: PlainPromptAdapter,
    proposer_transport: ProposerTransport | None,
    output_dir: Path,
    sqlite_path: Path,
    codex_test_seam: CodexTestSeam | None = None,
    codex_prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> _BoundOptimizer:
    """Build exactly the adapter this run drives.

    Registry membership is part of controller identity, so a run registers
    its own optimizer and nothing else.
    """
    if spec.optimizer == "codex":
        codex_control = build_codex_control(
            engine=engine,
            experiment=experiment,
            family=validated.family,
            model=resolve_codex_agent_model(spec.codex_model),
            max_tool_calls=spec.codex_capacity,
            codex_binary=spec.codex_binary,
            reasoning_effort=CodexReasoningEffort(spec.codex_reasoning_effort),
            wall_seconds=spec.codex_wall_seconds,
        )
        return _BoundOptimizer(
            adapter_key=CODEX_ADAPTER_KEY,
            # The preflight runs inside this call, so a broken Codex
            # session fails here -- before the runtime exists and before
            # any capacity or eval budget is committed.
            adapter=build_codex_adapter(
                store=store,
                control=codex_control,
                engine=engine,
                runtime_config=build_codex_runtime_config(
                    spec=spec, validated=validated
                ),
                reward_policy=experiment.reward_policy,
                store_path=sqlite_path,
                run_root=codex_run_root(output_dir),
                test_seam=codex_test_seam,
                prompt_builder=codex_prompt_builder,
            ),
            codex_control=codex_control,
        )
    if spec.optimizer in COPRO_SHAPED_OPTIMIZERS:
        # null-A is COPRO's search shape with an uninformative proposer.
        # Substituting only the transport is what makes it a control for
        # *selection*: the budget, the slots, the selection rule, and the
        # recorded evidence are COPRO's own, and the single thing that
        # differs is that the drafts carry no information.
        bound_transport = proposer_transport
        if spec.optimizer == NULL_RANDOM_OPTIMIZER:
            bound_transport = NullRandomTransport(
                seed=(
                    NULL_RANDOM_DEFAULT_SEED
                    if spec.seed is None
                    else spec.seed
                ),
                render_contract=validated.family.render_contract(),
                execution_policy_hash=(
                    engine.execution_policy_identity_hash()
                ),
                prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                    prompt_adapter
                ),
            )
        return _BoundOptimizer(
            adapter_key=COPRO_ADAPTER_KEY,
            adapter=_copro_adapter(
                engine=engine,
                control=copro_control,
                prompt_adapter=prompt_adapter,
                proposer_transport=bound_transport,
                family=validated.family,
                extra_proposal_bodies=spec.extra_proposal_bodies,
            ),
        )
    # Both remaining optimizers take a train/val split; ``_validate_spec``
    # has already proven the two sizes are present and fit the internal
    # split, so this is a pure re-derivation of the same partition the
    # spec names.
    trainset_task_hashes, valset_task_hashes = partition_internal_split(
        tuple(engine.sampling.task_hashes),
        train_size=cast("int", spec.train_size),
        val_size=cast("int", spec.val_size),
    )
    if spec.optimizer == "gepa":
        gepa_adapter = build_gepa_adapter(
            store=store,
            engine=engine,
            experiment=experiment,
            family=validated.family,
            run_id=run_id,
            proposer_transport=proposer_transport,
            max_metric_calls=spec.gepa_max_metric_calls,
            reflection_minibatch_size=spec.gepa_reflection_minibatch_size,
            seed=GEPA_DEFAULT_SEED if spec.seed is None else spec.seed,
            trainset_task_hashes=trainset_task_hashes,
            valset_task_hashes=valset_task_hashes,
        )
        return _BoundOptimizer(
            adapter_key=GEPA_ADAPTER_KEY,
            adapter=gepa_adapter,
            gepa_control=gepa_adapter.control,
        )
    miprov2_control = build_miprov2_control(
        engine=engine,
        experiment=experiment,
        family=validated.family,
        demo_mode=validated.demo_mode,
        seed=MIPROV2_DEFAULT_SEED if spec.seed is None else spec.seed,
        minibatch=spec.miprov2_minibatch,
        minibatch_size=spec.miprov2_minibatch_size,
        minibatch_full_eval_steps=spec.miprov2_minibatch_full_eval_steps,
        num_trials=spec.miprov2_num_trials,
        num_candidates=spec.miprov2_num_candidates,
        trainset_task_hashes=trainset_task_hashes,
        valset_task_hashes=valset_task_hashes,
    )
    miprov2_adapter = build_miprov2_adapter(
        store=store,
        engine=engine,
        control=miprov2_control,
        family=validated.family,
        proposer_transport=proposer_transport,
    )
    return _BoundOptimizer(
        adapter_key=MIPROV2_ADAPTER_KEY,
        adapter=miprov2_adapter,
        miprov2_control=miprov2_control,
        miprov2_adapter=miprov2_adapter,
    )


def _codex_preflight_already_proven() -> None:
    """The preflight ``prepare_codex_run`` requires, already satisfied.

    ``build_codex_adapter`` runs the real ``codex_auth_preflight`` before
    it returns an adapter at all, so by the time the launch is prepared
    the session has been proven against this run's own executor and
    process environment. ``prepare_codex_run`` still requires a callable --
    it has no default, deliberately, so no caller can forget one -- and
    this names the proof that already happened. Running a second probe
    here would spend wall time re-proving the same session.

    This is not a way to skip the check: the only path that reaches it
    went through the real preflight moments earlier, and a failure there
    raises before the runtime is built.
    """


def _codex_tool_executor(
    *,
    engine: EvalEngine,
    experiment: Experiment,
    effect_authority: EffectLeaseAuthority,
    owner_id: str,
) -> EvaluatingToolExecutor:
    """The in-harness executor that re-issues Codex's out-of-process calls.

    Codex evaluates through its own MCP server, which admits and persists
    each call against the same durable store. The harness still needs an
    executor so the Step's Issued Tool Call ledger can record every call;
    for an already-terminal call it reads the durable result rather than
    evaluating again.
    """
    return EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        experiment.reward_policy,
        effect_authority,
        owner_id=owner_id,
        replay_policy=ReplayPolicy.IDEMPOTENT,
    )


def _build_run_runtime(  # noqa: PLR0913
    *,
    store,
    engine: EvalEngine,
    experiment: Experiment,
    bound: _BoundOptimizer,
    run_id: str,
    sqlite_path: Path,
) -> RegisteredRuntime:
    """Assemble the runtime this run drives its one optimizer through.

    Effect leases are durable, not per-process: they live in the run's own
    ``runtime.sqlite`` beside the object store, mirroring whetstone-ai's
    platform CLI, which hands ``EffectLeaseAuthority.sqlite`` the same path
    it opened the store from. The lease authority owns
    ``dr_store_lease_authority*`` while the object store owns
    ``objects``/``bindings``, so one file carries both without a name
    collision. A memory authority would discard every terminal at process
    exit, so a re-run against a completed run directory would re-execute
    effects that already happened.

    A Codex run needs two more things, and they are the reason this is a
    function rather than one call: its Tool Calls are admitted from an
    out-of-process MCP evaluation server, so its per-run capacity must be
    durable too -- an in-memory authority would let that process admit
    past the cap, and the cap is what bounds paid evaluations -- and being
    the one ``TOOL_USING`` run, it needs a tool executor so the Step's
    Issued Tool Call ledger records every call the agent made outside.
    """
    effect_authority = EffectLeaseAuthority.sqlite(sqlite_path)
    is_codex = bound.codex_control is not None
    owner_id = f"{run_id}-owner"
    runtime = build_runtime(
        store=store,
        engine=engine,
        adapter_registry=MappingAdapterRegistry(
            {bound.adapter_key: bound.adapter}
        ),
        effect_authority=effect_authority,
        owner_id=owner_id,
        admission=(
            ToolAdmissionAuthority.sqlite(sqlite_path) if is_codex else None
        ),
        tool_executor=(
            _codex_tool_executor(
                engine=engine,
                experiment=experiment,
                effect_authority=effect_authority,
                owner_id=owner_id,
            )
            if is_codex
            else None
        ),
    )
    if is_codex:
        # The adapter reads durable Tool Call entries from the exact store
        # the harness admits through, so it is bound to the runtime's own
        # rather than to a second one built here. ``bind_tool_store`` is
        # the Codex adapter's, not the ``OptimizerAdapter`` protocol's --
        # only a TOOL_USING adapter has one -- and ``_bind_optimizer``
        # guarantees this branch holds a ``CodexAdapter``.
        codex_adapter = cast("CodexAdapter", bound.adapter)
        codex_adapter.bind_tool_store(runtime.tool_store)
    return runtime


def _prepare_launch(  # noqa: PLR0913
    *,
    runtime: RegisteredRuntime,
    bound: _BoundOptimizer,
    run_id: str,
    experiment: Experiment,
    copro_control: CoproControl,
    engine: EvalEngine,
    family: FamilySpec,
) -> OptimRunLaunch:
    """Bind the run for whichever optimizer this run registered."""
    render_contract = family.render_contract()
    if bound.codex_control is not None:
        return prepare_codex_run(
            runtime,
            run_id=run_id,
            control=bound.codex_control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
            # The session was already proven when the adapter was built,
            # before the runtime existed. ``prepare_codex_run`` requires a
            # callable and has no default, so this names the proof that
            # already happened rather than re-spawning a second probe on
            # the run's wall budget.
            preflight=_codex_preflight_already_proven,
        )
    if bound.gepa_control is not None:
        return prepare_gepa_run(
            runtime,
            run_id=run_id,
            control=bound.gepa_control,
            experiment=experiment,
            render_contract=render_contract,
            mutation_field=family.mutation_field,
        )
    if bound.miprov2_control is not None:
        control = bound.miprov2_control
        miprov2_adapter = bound.miprov2_adapter
        assert miprov2_adapter is not None
        return prepare_miprov2_run(
            runtime,
            run_id=run_id,
            control=control,
            experiment=experiment,
            initial_state=build_miprov2_state(
                run=miprov2_run_ref(
                    run_id=run_id,
                    control=control,
                    experiment=experiment,
                ),
                control=control,
                engine=engine,
                experiment=experiment,
                adapter=miprov2_adapter,
                family=family,
            ),
            render_contract=render_contract,
            mutation_field=family.mutation_field,
        )
    return prepare_copro_run(
        runtime,
        run_id=run_id,
        control=copro_control,
        experiment=experiment,
        render_contract=render_contract,
        mutation_field=family.mutation_field,
    )


def run_optimizer(  # noqa: PLR0915
    spec: RunSpec,
    *,
    codex_test_seam: CodexTestSeam | None = None,
    codex_prompt_builder: Callable[[CodexPromptContext], str] | None = None,
) -> Path:
    """Run one optimizer over one family's split, writing artifacts off-repo.

    Every optimizer reaches the same runtime entry point. MIPROv2
    additionally binds an opening durable state, and its ``demo_mode``
    selects the demonstration regime; Codex is the one ``TOOL_USING`` run,
    so it alone gets a durable admission authority and a tool executor.
    Every family-specific decision is read from the family registry, so
    this function names no family.

    ``codex_test_seam`` exists only so a test can drive the scripted fake
    Codex CLI. It is keyword-only, absent from :class:`RunSpec`, and has
    no CLI flag, so no production path or serialized spec can select one --
    see :class:`~whetstone_envs.optim.codex.CodexTestSeam`.

    ``codex_prompt_builder`` is the same shape of seam for the agent's
    instruction, and exists for the real-CLI ladder: its capacity and
    no-tool-call rungs cannot observe what they assert under the truthful
    production prompt, because an agent correctly told its real allowance
    obeys it and the durable refusal path is never reached. Like the
    seam it is keyword-only, absent from :class:`RunSpec`, and has no CLI
    flag, so a study arm or a serialized spec cannot change what the
    agent is told.

    A Codex run that supplies neither a seam nor the deliberate real-Codex
    opt-in is refused before any effect happens: the preflight spawns the
    billed CLI to prove a session, so it cannot be the thing that stops an
    accidental paid run. Every caller -- the CLI, a study arm, a
    parametrized test -- reaches Codex through this function, so one gate
    covers all of them.

    The spend guard runs *after* spec validation and before anything else.
    Both are pure and neither spends, so the order is only about which
    message a caller gets: an unrunnable spec should be told what is wrong
    with it rather than that it could not afford to run, and a spec that
    is merely unaffordable has nothing else to report.
    """
    if codex_test_seam is not None and spec.optimizer != "codex":
        raise ValueError("codex_test_seam applies only to --optimizer codex")
    if codex_prompt_builder is not None and spec.optimizer != "codex":
        raise ValueError(
            "codex_prompt_builder applies only to --optimizer codex"
        )
    validated = _validate_spec(spec)
    if spec.optimizer == "codex":
        refuse_unauthorized_real_codex(
            test_seam=codex_test_seam,
            allow_real_codex=spec.allow_real_codex,
        )
    family = validated.family
    resolved_run_id = spec.run_id or (
        f"{family.run_id_prefix}-{spec.optimizer}-{uuid4().hex[:8]}"
    )
    provider = None
    api_key_env = "WHETSTONE_TOY_API_KEY"
    if spec.transport == "openrouter":
        provider = openrouter_seeded_call_config(model=spec.model)
        api_key_env = "OPENROUTER_API_KEY"
    pool = family.generate_pool(
        n_per_stratum=validated.n_per_stratum,
        seed_start=validated.pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=spec.split_sizes,
        num_seeds=spec.num_seeds,
        provider_call_config=provider,
    )
    experiment = prepared.experiment
    if spec.transport == "openrouter":
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
            provider_kind=ProviderKind.OPENROUTER,
        )
    else:
        runtime_config = ReferenceEvalRuntimeConfig(
            transport_api_key_env=api_key_env,
        )
    # Widened once, here, so the engine's worker pool, the live transport's
    # connection pool, and the proposer all describe one decision.
    execution_policy = widened_execution_policy(
        runtime_config.execution_policy,
        concurrency=spec.provider_concurrency,
    )
    if spec.transport == "openrouter":
        execution_policy = hardened_execution_policy(execution_policy)
    live_transport = None
    if spec.transport == "openrouter":
        live_transport, transport_factory = bind_openrouter_transport(
            execution_policy
        )
    else:
        transport_factory = fake_transport_factory(
            gold_by_prompt=fake_gold_by_prompt(
                experiment,
                render_contract=family.render_contract(),
                ceiling_template=family.probes.ceiling_template,
            )
        )
    resolved_output = prepare_output_root(
        spec.output_dir or default_output_dir(resolved_run_id)
    )
    sqlite_path = resolved_output / "runtime.sqlite"
    with (
        durable_run_boundary(resolved_output),
        open_sqlite(str(sqlite_path)) as store,
    ):
        # Built directly rather than through ``build_engine``, which
        # forwards neither a concurrency nor a policy and so would run at
        # whetstone's default width over the unwidened pool. This is the
        # same in-process driver that helper assembles for the default
        # ``driver_mode``.
        engine = RuntimeEvalEngine(
            store=cast("ObjectStore", store),
            experiment=experiment,
            sampling=experiment.eval_configs.split_for(
                runtime_config.split_role
            ),
            execution_policy=execution_policy,
            driver=GraphRolloutEvalDriver(
                eval_runner=family.eval_runner(),
                mutation_field=family.mutation_field,
                render_contract=family.render_contract(),
                transport_factory=transport_factory,
            ),
            concurrency=spec.provider_concurrency,
        )
        prompt_adapter = PlainPromptAdapter()
        proposer_transport = None
        if live_transport is not None:
            proposer_transport = ProviderProposerTransport(
                resolve_provider_call_config=_proposer_config_resolver(
                    experiment=experiment,
                    proposer_model=spec.proposer_model,
                ),
                transport=live_transport,
                execution_policy=execution_policy,
                prompt_adapter=prompt_adapter,
            )
        defaults = CoproInjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=provider_call_config_ref(experiment),
                temperature=None,
            ),
            proposal_contract=_proposal_contract(family),
            eval_config_ref=engine.eval_config_ref,
            eval_role=EvalRole.INTERNAL,
            provider_execution_policy_ref=(
                engine.provider_execution_policy_ref
            ),
            expected_reward_policy_hash=(
                experiment.reward_policy.identity_hash()
            ),
            provider_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            prompt_adapter=prompt_adapter,
        )
        copro_control = configure_copro(
            breadth=spec.copro_breadth,
            depth=spec.copro_depth,
            track_stats=False,
            defaults=defaults,
        )
        bound = _bind_optimizer(
            spec=spec,
            validated=validated,
            store=cast("ObjectStore", store),
            engine=engine,
            experiment=experiment,
            run_id=resolved_run_id,
            copro_control=copro_control,
            prompt_adapter=prompt_adapter,
            proposer_transport=proposer_transport,
            output_dir=resolved_output,
            sqlite_path=sqlite_path,
            codex_test_seam=codex_test_seam,
            codex_prompt_builder=codex_prompt_builder,
        )
        runtime = _build_run_runtime(
            store=store,
            engine=engine,
            experiment=experiment,
            bound=bound,
            run_id=resolved_run_id,
            sqlite_path=sqlite_path,
        )
        # ``RegisteredRuntime.close`` releases the eval engine and the
        # authority's sqlite connection on every exit path, including the
        # failures ``durable_run_boundary`` re-raises.
        with runtime:
            launch = _prepare_launch(
                runtime=runtime,
                bound=bound,
                run_id=resolved_run_id,
                experiment=experiment,
                copro_control=copro_control,
                engine=engine,
                family=family,
            )
            request = copro_run_request(
                launch,
                controller_identity_hash=runtime.controller.runtime_hash,
            )
            result_ref = runtime.controller.drive(request)
            if result_ref.schema_name != OPTIM_RESULT_SCHEMA:
                raise ValueError(
                    "optimizer run did not produce an OptimResult"
                )
            result = OptimResult.model_validate(
                runtime.store.get(result_ref.reference)
            )
            (resolved_output / "result.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            # ``cost.json`` beside ``result.json``, so a reader can price a
            # run without parsing a whole optimization result. It is written
            # only when the result carries a cost report: an all-zero
            # document would claim the run was free rather than unmeasured.
            cost = project_run_cost(result, run_id=resolved_run_id)
            if cost is not None:
                write_run_cost(resolved_output, cost, store=store)
            trajectory = project_trajectory_report(
                store=cast("ObjectStore", store),
                prepared=prepared,
                result_ref=result_ref,
                result=result,
                transport=spec.transport,
                model=spec.model,
                split_sizes=spec.split_sizes,
            )
            publish_trajectory_report(resolved_output, trajectory)
    return resolved_output


def _proposer_config_resolver(
    *,
    experiment: Experiment,
    proposer_model: str | None,
):
    """Resolve the proposal route, which may differ from the task route.

    A study runs a cheap task model against a stronger proposer, so the two
    routes are separable. ``None`` reuses the experiment's own route, which
    keeps a single-model run byte-identical to one that never named a
    proposer.
    """
    if proposer_model is None:
        return _provider_config_resolver(experiment)
    proposer_config = openrouter_seeded_call_config(model=proposer_model)

    def resolve(_ref: object):
        return proposer_config

    return resolve


__all__ = [
    "ALLOW_REAL_CODEX_ENV",
    "ALLOW_REAL_CODEX_ENV_VALUE",
    "CODEX_DEFAULT_BINARY",
    "CODEX_EVALUATE_CALL_CAP",
    "CODEX_REASONING_EFFORTS",
    "COPRO_EXECUTOR_SCHEMA_SUFFIX",
    "DEFAULT_COPRO_BREADTH",
    "DEFAULT_COPRO_DEPTH",
    "DEFAULT_MIPROV2_FULL_EVAL_STEPS",
    "DEFAULT_MIPROV2_MINIBATCH",
    "DEFAULT_MIPROV2_NUM_CANDIDATES",
    "DEFAULT_MIPROV2_NUM_TRIALS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SPLIT_SIZES",
    "DEMO_MODES",
    "GEPA_DEFAULT_SEED",
    "KNOWN_FAMILY_IDS",
    "MIPROV2_DEFAULT_SEED",
    "MIPROV2_MINIBATCH_MIN_CANDIDATES",
    "MIPROV2_SPENT_COMBINATION_FIX_VERSION",
    "MIPROV2_SPENT_COMBINATION_ISSUE",
    "OPTIMIZERS",
    "SEED_DISPOSITION_CONTROL_FIELD",
    "SEED_DISPOSITION_PROVIDER_ONLY",
    "TRAIN_VAL_OPTIMIZERS",
    "TRANSPORTS",
    "CodexReasoningEffort",
    "CodexTestSeam",
    "RealCodexRefusedError",
    "RunSpec",
    "default_output_dir",
    "installed_whetstone_ai_version",
    "registered_family_ids",
    "run_optimizer",
    "seed_disposition",
]
