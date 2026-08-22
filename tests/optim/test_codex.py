"""The Codex arm's control, runtime config, and preflight seam."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_store.sync import open_sqlite
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.optim.codex import (
    CODEX_ADAPTER_KEY,
    CODEX_EVALUATE_CALL_CAP,
    CODEX_REASONING_EFFORTS,
    CODEX_RUN_ROOT_NAME,
    CodexReasoningEffort,
    CodexTestSeam,
    build_codex_adapter,
    build_codex_control,
    codex_run_root,
)
from whetstone_envs.optim.codex_runtime import (
    ENVS_CODEX_RUNTIME_CONFIG_CLASS,
    EnvsCodexRuntimeConfig,
)
from whetstone_envs.optim.families import family_spec

if TYPE_CHECKING:
    from dr_store import ObjectStore

SPLIT_SIZES = (2, 2, 0)


@pytest.fixture
def c19_world(tmp_path):
    """One prepared c19 experiment and the engine a run would build."""
    family = family_spec("c19")
    pool = family.generate_pool(
        n_per_stratum=family.default_n_per_stratum,
        seed_start=family.default_pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=SPLIT_SIZES,
        num_seeds=1,
        provider_call_config=None,
    )
    with open_sqlite(str(tmp_path / "world.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            cast("ObjectStore", store),
            experiment=prepared.experiment,
            eval_runner=family.eval_runner(),
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
        )
        yield family, prepared.experiment, engine, store


# --------------------------------------------------------------------------
# The control
# --------------------------------------------------------------------------


def test_the_capacity_cap_defaults_to_the_studys_own(c19_world) -> None:
    """D2 fixes the arm's admitted evaluate-call cap at 8."""
    family, experiment, engine, _store = c19_world
    control = build_codex_control(
        engine=engine, experiment=experiment, family=family, model="agent"
    )
    assert CODEX_EVALUATE_CALL_CAP == 8
    assert control.max_tool_calls == CODEX_EVALUATE_CALL_CAP


def test_the_control_binds_the_familys_mutation_field(c19_world) -> None:
    """The family owns the field, exactly as the other builders do."""
    family, experiment, engine, _store = c19_world
    control = build_codex_control(
        engine=engine, experiment=experiment, family=family, model="agent"
    )
    assert control.mutation_field == family.mutation_field
    assert control.mutation_field != "user_prompt_template"


def test_the_control_pins_every_binding_it_is_measured_against(
    c19_world,
) -> None:
    """A control that disagreed with the engine would misreport the run."""
    family, experiment, engine, _store = c19_world
    control = build_codex_control(
        engine=engine, experiment=experiment, family=family, model="agent"
    )
    assert control.eval_config_ref == engine.eval_config_ref
    assert control.reward_policy_hash == (
        experiment.reward_policy.identity_hash()
    )
    assert control.evaluation_execution_policy_hash == (
        engine.execution_policy_identity_hash()
    )
    assert control.task_model_identity_hash == (
        engine.task_model_identity_hash()
    )
    assert control.internal_task_hashes == (
        experiment.eval_configs.internal.task_set.task_hashes
    )


def test_an_explicit_capacity_overrides_the_default(c19_world) -> None:
    family, experiment, engine, _store = c19_world
    control = build_codex_control(
        engine=engine,
        experiment=experiment,
        family=family,
        model="agent",
        max_tool_calls=3,
    )
    assert control.max_tool_calls == 3


def test_a_control_off_the_internal_split_is_refused(c19_world) -> None:
    """Codex evaluates the internal split; anything else is a mismatch."""
    family, experiment, _engine, store = c19_world
    # An engine built over a different pool evaluates a different task
    # set, which is exactly the disagreement this refuses.
    other_pool = family.generate_pool(
        n_per_stratum=family.default_n_per_stratum,
        seed_start=family.default_pool_seed_start + 7,
    )
    other = family.build_experiment(
        other_pool,
        split_sizes=SPLIT_SIZES,
        num_seeds=1,
        provider_call_config=None,
    )
    other_engine = ReferenceEvalRuntimeConfig().build_engine(
        cast("ObjectStore", store),
        experiment=other.experiment,
        eval_runner=family.eval_runner(),
        mutation_field=family.mutation_field,
        render_contract=family.render_contract(),
    )
    with pytest.raises(ValueError, match="internal eval split"):
        build_codex_control(
            engine=other_engine,
            experiment=experiment,
            family=family,
            model="agent",
        )


def test_the_reasoning_efforts_are_the_enum_projection() -> None:
    """The CLI's choices are the owning enum's, not a second list."""
    assert (
        tuple(member.value for member in CodexReasoningEffort)
        == CODEX_REASONING_EFFORTS
    )
    assert "medium" in CODEX_REASONING_EFFORTS


# --------------------------------------------------------------------------
# The out-of-process runtime config
# --------------------------------------------------------------------------


def test_the_runtime_config_rebuilds_the_same_eval_config(
    c19_world, tmp_path
) -> None:
    """The whole point: the MCP server's engine must be the run's engine.

    whetstone-ai's ``ReferenceEvalRuntimeConfig`` always rebuilds the toy
    experiment, so a Codex run wired to it has every tool call refused as
    "not bound to the engine's exact Eval Config". This config carries the
    generation parameters instead and rebuilds the identical experiment.
    """
    _family, _experiment, engine, _store = c19_world
    config = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=family_spec("c19").default_n_per_stratum,
        pool_seed_start=family_spec("c19").default_pool_seed_start,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    with open_sqlite(str(tmp_path / "rebuilt.sqlite")) as store:
        rebuilt = config.build_engine(cast("ObjectStore", store))
        assert rebuilt.eval_config_ref == engine.eval_config_ref
        assert rebuilt.sampling.task_hashes == engine.sampling.task_hashes


def test_the_reference_config_alone_would_not_match(
    c19_world, tmp_path
) -> None:
    """The negative that motivates the module: the toy engine differs.

    Without this, the failure is silent on the server side and shows up
    only as every tool call being refused.
    """
    _family, _experiment, engine, _store = c19_world
    with open_sqlite(str(tmp_path / "toy.sqlite")) as store:
        toy = ReferenceEvalRuntimeConfig().build_engine(
            cast("ObjectStore", store)
        )
        assert toy.eval_config_ref != engine.eval_config_ref


def test_the_runtime_config_round_trips_through_json() -> None:
    """The MCP server validates this out of an environment variable."""
    config = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=4,
        pool_seed_start=765_432,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    assert (
        EnvsCodexRuntimeConfig.model_validate_json(config.model_dump_json())
        == config
    )


def test_the_runtime_config_class_path_resolves() -> None:
    """A persisted ``module:Class`` string the server imports by name."""
    from whetstone.eval.protocol import load_runtime_config

    assert ENVS_CODEX_RUNTIME_CONFIG_CLASS == (
        "whetstone_envs.optim.codex_runtime:EnvsCodexRuntimeConfig"
    )
    config = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=4,
        pool_seed_start=765_432,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    loaded = load_runtime_config(
        class_path=ENVS_CODEX_RUNTIME_CONFIG_CLASS,
        raw=config.model_dump_json().encode(),
    )
    assert loaded == config


def test_the_runtime_config_carries_the_launch_rendering_settings() -> None:
    """The MCP server refuses a config missing either of these."""
    config = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=4,
        pool_seed_start=765_432,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    assert config.mutation_field == family_spec("c19").mutation_field
    assert config.render_contract is not None


# --------------------------------------------------------------------------
# The preflight seam
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)
def test_the_adapter_refuses_a_runtime_config_that_would_not_match(
    c19_world, tmp_path
) -> None:
    """A mismatched rebuild is caught here, not by every call being refused.

    On the server side the mismatch is silent: it comes up on the toy
    experiment and refuses each call for a reason that names the Eval
    Config rather than the config that produced it. Proving the rebuild
    before anything spawns is what makes that failure legible.
    """
    family, _experiment, engine, store = c19_world
    control = build_codex_control(
        engine=engine, experiment=_experiment, family=family, model="agent"
    )
    wrong = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=family.default_n_per_stratum,
        # A different generator seed rebuilds a different task set.
        pool_seed_start=family.default_pool_seed_start + 7,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    calls: list[str] = []

    with pytest.raises(ValueError, match="different Eval Config"):
        build_codex_adapter(
            store=store,
            control=control,
            engine=engine,
            runtime_config=wrong,
            reward_policy=_experiment.reward_policy,
            store_path=tmp_path / "store.sqlite",
            run_root=tmp_path / "runs",
            test_seam=CodexTestSeam(
                preflight=lambda **_kwargs: calls.append("preflight"),
                environment={},
            ),
        )

    # Refused before the preflight, so a mismatched config never spends
    # wall time proving a session it cannot use.
    assert calls == []


def test_the_adapter_key_is_whetstones_own() -> None:
    assert CODEX_ADAPTER_KEY == "codex"


def test_the_run_root_lives_beneath_the_output_directory(tmp_path) -> None:
    """A run's spawn evidence stays beside the artifacts it produced."""
    assert codex_run_root(tmp_path) == tmp_path / CODEX_RUN_ROOT_NAME
    assert CODEX_RUN_ROOT_NAME == "codex-runs"


def test_no_production_module_imports_the_testing_package() -> None:
    """The scripted preflight must be unreachable from the shipped path.

    A preflight that production could substitute is not a preflight. The
    stand-in lives in ``whetstone.testing``; if any module under ``src/``
    imported that package, a production run could reach it, and the
    guarantee that no budgeted Codex run starts without a proven session
    would rest on nobody having wired it up yet.

    An ``import`` statement is the violation, not a mention: the module
    docstring that explains why the seam exists names the package on
    purpose.
    """
    import re

    import whetstone_envs

    root = Path(whetstone_envs.__file__).resolve().parent
    importing = re.compile(
        r"^\s*(?:from|import)\s+whetstone\.testing\b", re.MULTILINE
    )
    offenders = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if importing.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []


def test_the_run_spec_carries_no_preflight_seam() -> None:
    """A serialized spec must not be able to name a stand-in."""
    from dataclasses import fields

    from whetstone_envs.optim.run import RunSpec

    names = {field.name for field in fields(RunSpec)}
    assert not any("seam" in name or "preflight" in name for name in names)


def test_no_test_drives_codex_without_the_scripted_seam() -> None:
    """A test that runs Codex must not reach the real, paid CLI.

    ``run_optimizer`` spawns whatever ``codex_binary`` names, and the
    default is the real binary on the run PATH -- correct for production
    and a hazard in a suite. The hazard is not hypothetical: a test that
    parametrized over ``OPTIMIZERS`` and called ``run_optimizer`` did
    invoke the real CLI, and only its authentication preflight stopped
    the run.

    So every module that *calls* ``run_optimizer`` while naming the Codex
    arm must also import the scripted seam. Checking the pairing rather
    than each call site keeps the rule readable: a module holding one
    without the other is the shape that goes wrong. A module that only
    builds a spec, or only mentions the runner in prose, calls nothing
    and is not the hazard.
    """
    import re

    tests_root = Path(__file__).resolve().parent.parent
    calls_runner = re.compile(r"run_optimizer\s*\(")
    # A spec whose optimizer really is Codex. A module that only names the
    # string -- to exclude it from a parametrization, say -- drives nothing.
    drives_codex = re.compile(r"""optimizer\s*=\s*["']codex["']""")
    seam = re.compile(r"codex_test_seam|CodexTestSeam")
    unguarded = []
    for path in sorted(tests_root.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if not calls_runner.search(source):
            continue
        if not drives_codex.search(source):
            continue
        if seam.search(source):
            continue
        unguarded.append(str(path.relative_to(tests_root)))
    assert unguarded == []
