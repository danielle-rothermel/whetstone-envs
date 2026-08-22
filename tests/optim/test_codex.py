"""The Codex arm's control, runtime config, and preflight seam."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("whetstone.experiment.env")

from dr_providers import ProviderKind
from dr_store.sync import open_sqlite
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

from whetstone_envs.optim.codex import (
    ALLOW_REAL_CODEX_ENV,
    ALLOW_REAL_CODEX_ENV_VALUE,
    CODEX_ADAPTER_KEY,
    CODEX_EVALUATE_CALL_CAP,
    CODEX_REASONING_EFFORTS,
    CODEX_RUN_ROOT_NAME,
    FORBID_REAL_CODEX_ENV,
    CodexReasoningEffort,
    CodexTestSeam,
    RealCodexRefusedError,
    build_codex_adapter,
    build_codex_control,
    codex_run_root,
    refuse_unauthorized_real_codex,
)
from whetstone_envs.optim.codex_runtime import (
    ENVS_CODEX_RUNTIME_CONFIG_CLASS,
    EnvsCodexRuntimeConfig,
)
from whetstone_envs.optim.families import family_spec
from whetstone_envs.optim.provider import openrouter_seeded_call_config
from whetstone_envs.optim.run import RunSpec, run_optimizer

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


def test_the_runtime_config_field_names_are_pinned() -> None:
    """The golden the class docstring promises.

    This model is validated out of a subprocess environment variable by
    the MCP evaluation server, so its field spellings are a persisted
    cross-process wire format, not internal naming. Renaming one is a
    silent break: the server would refuse the config it was handed, and
    the only symptom is a Codex run that never gets an engine. Deriving
    the expectation from the class would assert nothing, so the literals
    are written out.
    """
    assert tuple(EnvsCodexRuntimeConfig.model_fields) == (
        "family_id",
        "split_sizes",
        "n_per_stratum",
        "pool_seed_start",
        "num_seeds",
        "transport",
        "model",
        "partial_log_path",
        "prompt_cache_path",
        "row_job_entrypoint",
        "unit_deadline_seconds",
    )
    # Frozen and closed: the server must not silently accept a config
    # carrying a field this side never wrote.
    assert EnvsCodexRuntimeConfig.model_config["frozen"] is True
    assert EnvsCodexRuntimeConfig.model_config["extra"] == "forbid"


def test_the_runtime_config_serialized_keys_are_the_field_names() -> None:
    """What actually crosses the process boundary, pinned as JSON keys."""
    config = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=4,
        pool_seed_start=765_432,
        num_seeds=1,
        transport="fake",
        model="openai/gpt-4.1-nano",
    )
    assert set(json.loads(config.model_dump_json())) == {
        "family_id",
        "split_sizes",
        "n_per_stratum",
        "pool_seed_start",
        "num_seeds",
        "transport",
        "model",
        "partial_log_path",
        "prompt_cache_path",
        "row_job_entrypoint",
        "unit_deadline_seconds",
    }


def test_the_runtime_config_class_path_literal_is_pinned() -> None:
    """The ``module:Class`` string the runner records for the server.

    It is resolved by import on the far side, so a module move or a class
    rename breaks it with no compile-time signal. The docstring says a
    golden test pins it; this is that test, and
    ``test_the_runtime_config_class_path_resolves`` proves the string
    still loads.
    """
    assert ENVS_CODEX_RUNTIME_CONFIG_CLASS == (
        "whetstone_envs.optim.codex_runtime:EnvsCodexRuntimeConfig"
    )
    module, _, class_name = ENVS_CODEX_RUNTIME_CONFIG_CLASS.partition(":")
    assert module == EnvsCodexRuntimeConfig.__module__
    assert class_name == EnvsCodexRuntimeConfig.__name__


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


def test_the_eval_config_alone_does_not_pin_the_model_route(
    tmp_path,
) -> None:
    """Why the adapter checks the task model too, and not only the config.

    On the openrouter transport the task model is carried by the provider
    call config, not by the Eval Config -- so two runtime configs naming
    different models rebuild to the *same* ``eval_config_ref``. A check
    that stopped at the config would admit a run measured on a route the
    study never asked for, and the run would complete and report a
    perfectly coherent trajectory about the wrong model.
    """

    def rebuilt(model: str, name: str):
        config = EnvsCodexRuntimeConfig(
            family_id="c19",
            split_sizes=SPLIT_SIZES,
            n_per_stratum=family_spec("c19").default_n_per_stratum,
            pool_seed_start=family_spec("c19").default_pool_seed_start,
            num_seeds=1,
            transport="openrouter",
            model=model,
        )
        with open_sqlite(str(tmp_path / f"{name}.sqlite")) as store:
            engine = config.build_engine(cast("ObjectStore", store))
            return engine.eval_config_ref, engine.task_model_identity_hash()

    asked_for = rebuilt("openai/gpt-4.1-nano", "asked")
    drifted = rebuilt("openai/gpt-5-nano", "drifted")

    assert asked_for[0] == drifted[0]
    assert asked_for[1] != drifted[1]


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)
def test_the_adapter_refuses_a_runtime_config_on_another_model_route(
    tmp_path,
) -> None:
    """A drifted model route is refused before anything is spawned.

    The Eval Config agrees here -- that is the point -- so this is the
    mismatch the ``eval_config_ref`` check cannot see, and the reason the
    adapter asserts the task-model identity as well.
    """
    family = family_spec("c19")
    pool = family.generate_pool(
        n_per_stratum=family.default_n_per_stratum,
        seed_start=family.default_pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=SPLIT_SIZES,
        num_seeds=1,
        provider_call_config=openrouter_seeded_call_config(
            model="openai/gpt-4.1-nano"
        ),
    )
    experiment = prepared.experiment
    drifted = EnvsCodexRuntimeConfig(
        family_id="c19",
        split_sizes=SPLIT_SIZES,
        n_per_stratum=family.default_n_per_stratum,
        pool_seed_start=family.default_pool_seed_start,
        num_seeds=1,
        transport="openrouter",
        # The one difference: another model route, on an otherwise
        # identical experiment.
        model="openai/gpt-5-nano",
    )
    calls: list[str] = []
    with open_sqlite(str(tmp_path / "world.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig(
            transport_api_key_env="OPENROUTER_API_KEY",
            provider_kind=ProviderKind.OPENROUTER,
        ).build_engine(
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
            model="agent",
        )
        with pytest.raises(ValueError, match="different task model"):
            build_codex_adapter(
                store=cast("ObjectStore", store),
                control=control,
                engine=engine,
                runtime_config=drifted,
                reward_policy=experiment.reward_policy,
                store_path=tmp_path / "store.sqlite",
                run_root=tmp_path / "runs",
                test_seam=CodexTestSeam(
                    preflight=lambda **_kwargs: calls.append("preflight"),
                    environment={},
                ),
            )

    # Refused ahead of the preflight, like the Eval Config mismatch.
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


# --------------------------------------------------------------------------
# The real-Codex spend guard
# --------------------------------------------------------------------------


def _tripwire_binary(directory: Path) -> str:
    """A "Codex CLI" that must never be reached at all.

    The refusal has to happen before a subprocess exists, so the run is
    pointed at a binary that only exists to be evidence: if the guard
    regresses, the preflight resolves and spawns this instead of the real
    ``codex``, which is what keeps the test itself off the paid CLI.

    The spawn is proved absent by :func:`_assert_nothing_spawned` rather
    than by a marker this script writes -- the Codex containment profile
    is a sandbox, and a blocked write would look identical to never having
    run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "codex"
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _assert_nothing_spawned(output_dir: Path) -> None:
    """No subprocess, and nothing durable, came of a refused run.

    ``build_codex_executor`` creates the run's ``codex-runs`` root before
    it can spawn anything and writes one job record per spawn beneath it,
    so an absent directory is positive evidence that no Codex process was
    started. The output directory being absent says the refusal also
    landed ahead of the durable run boundary, so a refused run leaves no
    artifacts to clean up.
    """
    assert not output_dir.exists()
    assert not (output_dir / CODEX_RUN_ROOT_NAME).exists()


def _codex_spec(binary: str, output_dir: Path, **overrides) -> RunSpec:
    return RunSpec(
        optimizer="codex",
        transport="fake",
        family="c19",
        split_sizes=SPLIT_SIZES,
        output_dir=output_dir,
        run_id="codex-guard",
        codex_binary=binary,
        **overrides,
    )


def test_the_env_var_name_matches_the_suites_own() -> None:
    """The conftest fixture spells it rather than importing it.

    ``tests/conftest.py`` loads for every suite, including an install
    without the ``optim`` extra, so it cannot import the owning module.
    Pinning the two spellings equal here is what keeps the fixture from
    clearing a variable nobody reads.
    """
    from tests.conftest import (
        ALLOW_REAL_CODEX_ENV as CONFTEST_ENV,
    )

    assert CONFTEST_ENV == ALLOW_REAL_CODEX_ENV
    assert ALLOW_REAL_CODEX_ENV == "WHETSTONE_ENVS_ALLOW_REAL_CODEX"
    assert ALLOW_REAL_CODEX_ENV_VALUE == "1"


def test_a_codex_run_without_a_seam_or_the_opt_in_is_refused(
    tmp_path, monkeypatch
) -> None:
    """The default Codex run must never reach the real, billed CLI.

    The authentication preflight is not a spend guard: it proves a session
    by *spawning* the CLI, and on a machine with a Codex login that spawn
    succeeds and is billed. So a run that names the Codex arm and supplies
    nothing else is refused outright -- before any preflight, adapter,
    admission authority, or subprocess exists.

    The session tripwire is lifted here so the refusal under test is the
    *opt-in* one rather than the tripwire's, which would otherwise answer
    first and leave this assertion unable to fail. The run is still
    pointed at the tripwire binary, so a regression spawns evidence
    instead of the real CLI.
    """
    monkeypatch.delenv(FORBID_REAL_CODEX_ENV, raising=False)
    binary = _tripwire_binary(tmp_path / "bin")
    output_dir = tmp_path / "run"

    with pytest.raises(RealCodexRefusedError, match="costs money"):
        run_optimizer(_codex_spec(binary, output_dir))

    _assert_nothing_spawned(output_dir)


def test_the_flag_alone_does_not_authorize_a_paid_run(tmp_path) -> None:
    """A serialized spec must not be able to buy a session by itself.

    ``allow_real_codex`` travels in a spec, a study arm, or a copied
    command line, so it is deliberately only half of the opt-in. The
    session fixture guarantees the environment half is unset here.
    """
    binary = _tripwire_binary(tmp_path / "bin")
    output_dir = tmp_path / "run"

    with pytest.raises(RealCodexRefusedError):
        run_optimizer(_codex_spec(binary, output_dir, allow_real_codex=True))

    _assert_nothing_spawned(output_dir)


def test_the_env_var_alone_does_not_authorize_a_paid_run(
    tmp_path, monkeypatch
) -> None:
    """An exported variable must not turn every Codex run into a paid one."""
    binary = _tripwire_binary(tmp_path / "bin")
    output_dir = tmp_path / "run"
    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)

    with pytest.raises(RealCodexRefusedError):
        run_optimizer(_codex_spec(binary, output_dir))

    _assert_nothing_spawned(output_dir)


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE"])
def test_only_the_exact_env_value_opts_in(
    tmp_path, monkeypatch, value: str
) -> None:
    """A half-remembered spelling refuses rather than spends."""
    binary = _tripwire_binary(tmp_path / "bin")
    output_dir = tmp_path / "run"
    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, value)

    with pytest.raises(RealCodexRefusedError):
        run_optimizer(_codex_spec(binary, output_dir, allow_real_codex=True))

    _assert_nothing_spawned(output_dir)


def test_a_seam_admits_the_run_without_any_opt_in() -> None:
    """The scripted path is the other admissible ground, and needs nothing.

    Checked on the guard directly rather than by driving a whole run: the
    e2e suite already proves a seamed run completes, and this states the
    one fact the guard owns -- a seam is sufficient on its own.
    """
    seam = CodexTestSeam(preflight=lambda **_kwargs: None, environment={})
    refuse_unauthorized_real_codex(test_seam=seam, allow_real_codex=False)


def test_both_halves_of_the_opt_in_admit_the_run(monkeypatch) -> None:
    """The deliberate paid path exists, and takes both halves to reach.

    The session tripwire is lifted for this one assertion, because the
    fact under test is precisely what the tripwire otherwise masks: that
    the two halves together *do* admit a run. Nothing is spawned -- the
    guard is called directly and returns without building anything.
    """
    monkeypatch.delenv(FORBID_REAL_CODEX_ENV, raising=False)
    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)
    refuse_unauthorized_real_codex(test_seam=None, allow_real_codex=True)


# --------------------------------------------------------------------------
# The test-process tripwire
# --------------------------------------------------------------------------


def test_the_forbid_var_name_matches_the_suites_own() -> None:
    """The conftest arms the variable this module's gate reads."""
    from tests.conftest import (
        FORBID_REAL_CODEX_ENV as CONFTEST_ENV,
    )

    assert CONFTEST_ENV == FORBID_REAL_CODEX_ENV
    assert FORBID_REAL_CODEX_ENV == "WHETSTONE_ENVS_FORBID_REAL_CODEX"


def test_the_suite_arms_the_tripwire_for_its_whole_session() -> None:
    """The fixture's effect, checked rather than assumed.

    Every test below relies on the tripwire being armed by the time it
    runs; if the fixture stopped setting it, those tests would keep
    passing for the wrong reason.
    """
    import os

    assert os.environ.get(FORBID_REAL_CODEX_ENV)


def test_the_tripwire_refuses_even_both_opt_in_halves(
    tmp_path, monkeypatch
) -> None:
    """The opt-in is process state, so it cannot be the last defence.

    Fails-before: a test that monkeypatched the allow variable to prove a
    gate lifts also lifted the real gate, and the study harness's early
    session probe -- which runs *after* the opt-in is satisfied -- spawned
    the real CLI. With the tripwire armed the same setup refuses.
    """
    binary = _tripwire_binary(tmp_path / "bin")
    output_dir = tmp_path / "run"
    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)

    with pytest.raises(RealCodexRefusedError, match="forbids it"):
        run_optimizer(_codex_spec(binary, output_dir, allow_real_codex=True))

    _assert_nothing_spawned(output_dir)


def test_the_tripwire_also_covers_the_standalone_preflight(
    tmp_path, monkeypatch
) -> None:
    """The study harness's early probe routes through the same gate.

    The probe is a real session probe, so reaching it around the gate
    would be exactly the bypass the tripwire exists to prevent. Pointed at
    the tripwire binary, so a regression spawns evidence rather than the
    real CLI.
    """
    from whetstone_envs.optim.codex import preflight_codex_session

    scratch = tmp_path / "preflight"
    monkeypatch.setenv(ALLOW_REAL_CODEX_ENV, ALLOW_REAL_CODEX_ENV_VALUE)

    with pytest.raises(RealCodexRefusedError, match="forbids it"):
        preflight_codex_session(
            scratch_root=scratch,
            codex_binary=_tripwire_binary(tmp_path / "bin"),
            allow_real_codex=True,
        )

    _assert_nothing_spawned(scratch)


def test_the_tripwire_does_not_block_the_scripted_path(tmp_path) -> None:
    """A seamed run reaches no real CLI, so there is nothing to forbid.

    This is what keeps the tripwire from disarming the suite: every
    scripted Codex test still runs under it.
    """
    from whetstone_envs.optim.codex import preflight_codex_session

    seen: list[str] = []

    def _record(**_kwargs: object) -> None:
        seen.append("preflight")

    preflight_codex_session(
        scratch_root=tmp_path / "preflight",
        codex_binary=_tripwire_binary(tmp_path / "bin"),
        test_seam=CodexTestSeam(preflight=_record, environment={}),
    )
    assert seen == ["preflight"]


def test_the_tool_input_schema_ordering_the_projection_unpacks() -> None:
    """The projection destructures this tuple positionally, so pin it.

    ``reporting/projection.py`` rebuilds a Codex run's evaluated candidate
    from the Tool Call's args, reading the keys out of whetstone-ai's
    ``CODEX_EVAL_INPUT_FIELDS`` rather than re-spelling them -- which is
    what keeps the report and ``optim/audit/codex.py`` agreeing about the
    surface the agent was handed. It unpacks the tuple positionally, so a
    reordering upstream would silently swap ``base_ref`` and ``template``
    and attribute every evaluation to the wrong candidate. Neither type
    checking nor the arity check catches that, so the order is a golden.
    """
    from whetstone.optim.codex.mcp_bridge import (
        CODEX_EVAL_INPUT_FIELDS,
    )

    assert CODEX_EVAL_INPUT_FIELDS == ("base_ref", "model_route", "template")
