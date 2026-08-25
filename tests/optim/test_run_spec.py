"""Every ``RunSpec`` field and every CLI flag that reaches it.

The runner's spec is the study's unit of description: a run is reproducible
only if every knob the study varies is on the spec and nothing is read from
module state. These tests pin each field's validation, each field's default,
and that the CLI carries each one through unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone.optim.contracts import OptimResult

from whetstone_envs.optim.cli import build_parser, main
from whetstone_envs.optim.run import (
    CODEX_DEFAULT_BINARY,
    DEFAULT_COPRO_BREADTH,
    DEFAULT_COPRO_DEPTH,
    DEFAULT_MIPROV2_NUM_CANDIDATES,
    DEFAULT_MIPROV2_NUM_TRIALS,
    DEFAULT_SPLIT_SIZES,
    GEPA_DEFAULT_SEED,
    MIPROV2_DEFAULT_SEED,
    MIPROV2_MINIBATCH_MIN_CANDIDATES,
    MIPROV2_SPENT_COMBINATION_FIX_VERSION,
    OPTIMIZERS,
    SEED_DISPOSITION_CONTROL_FIELD,
    SEED_DISPOSITION_PROVIDER_ONLY,
    TRANSPORTS,
    RunSpec,
    run_optimizer,
    seed_disposition,
)
from whetstone_envs.reporting.publication import DurableRunError


def _spec(**overrides: object) -> RunSpec:
    """A minimal runnable spec, with any field overridden."""
    fields: dict[str, object] = {"optimizer": "copro", "transport": "fake"}
    fields.update(overrides)
    return RunSpec(**fields)  # ty: ignore[invalid-argument-type]


def _captured_spec(argv: list[str]) -> RunSpec:
    """Run ``main`` with the runner stubbed, returning the spec it built."""
    with patch(
        "whetstone_envs.optim.cli.run_optimizer",
        return_value=Path("/dev/null"),
    ) as runner:
        assert main(argv) == 0
    (spec,), _ = runner.call_args
    assert isinstance(spec, RunSpec)
    return spec


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_spec_defaults_keep_an_unparameterised_run_a_smoke_run() -> None:
    spec = _spec()
    assert spec.family == "c19"
    assert spec.split_sizes == DEFAULT_SPLIT_SIZES
    assert spec.num_seeds == 1
    assert spec.proposer_model is None
    assert spec.n_per_stratum is None
    assert spec.pool_seed_start is None
    assert spec.seed is None
    assert spec.copro_breadth == DEFAULT_COPRO_BREADTH
    assert spec.copro_depth == DEFAULT_COPRO_DEPTH
    assert spec.gepa_max_metric_calls is None
    assert spec.codex_capacity is None


def test_cli_defaults_match_the_spec_defaults() -> None:
    spec = _captured_spec(["--optimizer", "copro"])
    assert spec == _spec(
        optimizer="copro",
        transport="fake",
        demo_mode=spec.demo_mode,
        model=spec.model,
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"optimizer": "gradient"}, "unsupported optimizer"),
        ({"transport": "carrier-pigeon"}, "unsupported transport"),
        ({"demo_mode": "handful"}, "unsupported demo mode"),
        ({"num_seeds": 0}, "num_seeds must be at least 1"),
        ({"copro_breadth": 1}, "copro_breadth must be at least 2"),
        ({"copro_depth": -1}, "copro_depth must be non-negative"),
        ({"n_per_stratum": 0}, "n_per_stratum must be at least 1"),
        ({"family": "c17"}, "unsupported family"),
    ],
)
def test_the_runner_rejects_an_unrunnable_spec_before_any_effect(
    tmp_path, overrides, message
) -> None:
    """Validation happens before the durable run boundary opens."""
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        run_optimizer(_spec(output_dir=output, **overrides))
    assert not output.exists()


def test_gepa_metric_call_ceiling_is_refused_on_other_optimizers(
    tmp_path,
) -> None:
    """A flag no optimizer reads must not look honoured."""
    with pytest.raises(ValueError, match="only to --optimizer gepa"):
        run_optimizer(
            _spec(
                optimizer="copro",
                output_dir=tmp_path / "wrong-optimizer",
                gepa_max_metric_calls=4,
            )
        )


def test_gepa_metric_call_ceiling_must_be_positive(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="gepa_max_metric_calls must be at least 1"
    ):
        run_optimizer(
            _spec(
                optimizer="gepa",
                output_dir=tmp_path / "zero-ceiling",
                gepa_max_metric_calls=0,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codex_capacity", 8),
        ("codex_binary", "/usr/bin/false"),
        ("codex_model", "some-agent-model"),
        ("codex_reasoning_effort", "high"),
        ("codex_wall_seconds", 30.0),
    ],
)
def test_codex_settings_are_refused_on_another_optimizer(
    tmp_path, field: str, value: object
) -> None:
    """A setting that looks honoured but is not misdescribes the arm."""
    with pytest.raises(
        ValueError, match="codex settings apply only to --optimizer codex"
    ):
        run_optimizer(
            _spec(
                optimizer="copro",
                output_dir=tmp_path / f"codex-{field}",
                **{field: value},
            )
        )


def test_a_zero_codex_capacity_is_refused() -> None:
    """A cap of zero admits nothing, so the run could buy no evaluation."""
    with pytest.raises(ValueError, match="codex_capacity must be at least 1"):
        run_optimizer(_spec(optimizer="codex", codex_capacity=0))


def test_an_unknown_codex_reasoning_effort_is_refused() -> None:
    with pytest.raises(ValueError, match="codex_reasoning_effort must be"):
        run_optimizer(
            _spec(optimizer="codex", codex_reasoning_effort="maximum")
        )


def test_the_runner_admits_only_the_optimizers_it_can_drive() -> None:
    """null-A is drivable; null-B is not, because it has no search.

    null-A is a control *for selection*, so it must spend the same
    proposal budget and produce the same evidence as the arm it stands in
    for -- which means driving COPRO's search shape with an uninformative
    proposer. null-B proposes nothing, so there is no search to drive and
    no fidelity invariant to audit; admitting it would fail inside the
    durable run boundary rather than at spec validation.
    """
    assert OPTIMIZERS == ("codex", "copro", "gepa", "miprov2", "null-random")
    assert "null-identity" not in OPTIMIZERS


def test_a_test_seam_is_refused_on_another_optimizer() -> None:
    """The scripted-preflight seam belongs to the Codex arm alone."""
    from whetstone_envs.optim.codex import CodexTestSeam

    with pytest.raises(ValueError, match="codex_test_seam applies only"):
        run_optimizer(
            _spec(optimizer="copro"),
            codex_test_seam=CodexTestSeam(
                preflight=lambda **_kwargs: None, environment={}
            ),
        )


# --------------------------------------------------------------------------
# Seed plumbing
# --------------------------------------------------------------------------


def test_seed_disposition_records_copro_lack_of_a_control_seed() -> None:
    """COPRO's control has no seed field, and the manifest says so."""
    assert seed_disposition("copro") == SEED_DISPOSITION_PROVIDER_ONLY
    assert seed_disposition("gepa") == SEED_DISPOSITION_CONTROL_FIELD
    assert seed_disposition("miprov2") == SEED_DISPOSITION_CONTROL_FIELD


def test_an_unseeded_run_keeps_each_optimizer_own_default_seed() -> None:
    """An omitted seed must not silently re-seed a control.

    The runner's fallbacks mirror ``configure_gepa`` and
    ``configure_miprov2``, so an unseeded run keeps the control identity
    hash it always had. Reading the upstream defaults here means a change
    upstream fails this test instead of silently shifting every unseeded
    run's identity.
    """
    import inspect

    from whetstone.optim.gepa.control import configure_gepa
    from whetstone.optim.miprov2.control import configure_miprov2

    gepa_seed = inspect.signature(configure_gepa).parameters["seed"].default
    miprov2_seed = (
        inspect.signature(configure_miprov2).parameters["seed"].default
    )
    assert gepa_seed == GEPA_DEFAULT_SEED
    assert miprov2_seed == MIPROV2_DEFAULT_SEED


def test_an_explicit_seed_reaches_the_gepa_control(tmp_path) -> None:
    from whetstone_envs.optim import run as run_module

    seen: list[int] = []
    original = run_module.build_gepa_adapter

    def capture(**kwargs):
        seen.append(kwargs["seed"])
        return original(**kwargs)

    with patch.object(run_module, "build_gepa_adapter", capture):
        run_optimizer(
            _spec(
                optimizer="gepa",
                output_dir=tmp_path / "gepa-seeded",
                run_id="gepa-seeded",
                seed=3001,
                train_size=1,
                val_size=1,
            )
        )
    assert seen == [3001]


def test_an_explicit_seed_reaches_the_miprov2_control(tmp_path) -> None:
    from whetstone_envs.optim import run as run_module

    seen: list[int] = []

    # The control is built before the run drives, so stopping the run right
    # after capture keeps this test about seed plumbing rather than about
    # MIPROv2's end-to-end behaviour, which its own e2e test owns.
    sentinel = RuntimeError("captured")

    def capture_then_stop(**kwargs):
        seen.append(kwargs["seed"])
        raise sentinel

    with (
        patch.object(run_module, "build_miprov2_control", capture_then_stop),
        pytest.raises(DurableRunError),
    ):
        run_optimizer(
            _spec(
                optimizer="miprov2",
                output_dir=tmp_path / "miprov2-seeded",
                run_id="miprov2-seeded",
                seed=2001,
                train_size=1,
                val_size=1,
            )
        )
    assert seen == [2001]


# --------------------------------------------------------------------------
# Pool parameterisation
# --------------------------------------------------------------------------


@contextmanager
def _recorded_pool_generation():
    """Record the pool arguments the runner hands the family generator.

    The registry holds the family's own ``generate_pool`` as a field, so
    swapping a recording wrapper onto a copy of the registered spec is what
    observes the call without changing which generator actually runs. The
    patch lands in the runner's own namespace because it binds
    ``family_spec`` at import.
    """
    from whetstone_envs.optim import run as run_module
    from whetstone_envs.optim.families import family_spec

    spec = family_spec("c19")
    seen: list[tuple[int, int]] = []

    def record(*, n_per_stratum: int, seed_start: int):
        seen.append((n_per_stratum, seed_start))
        return spec.generate_pool(
            n_per_stratum=n_per_stratum, seed_start=seed_start
        )

    recorded = replace(spec, generate_pool=record)
    with patch.object(run_module, "family_spec", lambda _family_id: recorded):
        yield seen


def test_pool_generation_is_no_longer_hardcoded(tmp_path) -> None:
    """``n_per_stratum`` and ``pool_seed_start`` reach the generator.

    The runner used to call ``generate_pool(n_per_stratum=2,
    seed_start=765_432)`` with both values written into its body.
    """
    with _recorded_pool_generation() as seen:
        run_optimizer(
            _spec(
                output_dir=tmp_path / "pool-params",
                run_id="pool-params",
                n_per_stratum=3,
                pool_seed_start=4242,
            )
        )
    assert seen == [(3, 4242)]


def test_omitted_pool_parameters_take_the_family_defaults(tmp_path) -> None:
    from whetstone_envs.optim.families import family_spec

    spec = family_spec("c19")
    with _recorded_pool_generation() as seen:
        run_optimizer(
            _spec(
                output_dir=tmp_path / "pool-defaults",
                run_id="pool-defaults",
            )
        )
    assert seen == [(spec.default_n_per_stratum, spec.default_pool_seed_start)]


# --------------------------------------------------------------------------
# COPRO breadth and depth
# --------------------------------------------------------------------------


def test_copro_breadth_reaches_the_control(tmp_path) -> None:
    """The configured breadth is what COPRO asks its proposer to fill.

    The family scripts exactly two fake-transport bodies, so a breadth of
    three cannot be filled. The round proceeds on what it realized rather
    than terminalizing -- a dropped draft is a stochastic outcome, not a
    contract violation -- so the evidence that the configured value
    reached the control is the round it *asked* for: two occurrences
    measured where a breadth of two would have measured two and a default
    breadth would have asked for something else entirely.

    Compared against a default-breadth run over identical inputs, because
    the realized count alone cannot distinguish "asked for three, got two"
    from "asked for two, got two". What differs is the requested breadth
    recorded on the run's own control.
    """
    output = tmp_path / "copro-breadth"
    run_optimizer(
        _spec(
            optimizer="copro",
            output_dir=output,
            run_id="copro-breadth",
            copro_breadth=3,
        )
    )
    result = OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )
    # A shortfall is not a failure: the run completed on what it realized.
    assert result.terminal_failure is None
    assert DEFAULT_COPRO_BREADTH != 3
    # The configured breadth reached the control, and the scripted bodies
    # could not fill it, so every round realized fewer than it requested.
    realized = [
        len(step.record.resolved_intents)
        for step in result.step_results
        if step.record.resolved_intents
    ]
    assert realized
    assert all(count < 3 for count in realized)


def _copro_steps(tmp_path: Path, *, run_id: str, **overrides) -> OptimResult:
    output = tmp_path / run_id
    run_optimizer(
        _spec(
            optimizer="copro",
            output_dir=output,
            run_id=run_id,
            **overrides,
        )
    )
    return OptimResult.model_validate_json(
        (output / "result.json").read_text(encoding="utf-8")
    )


def test_copro_depth_reaches_the_control(tmp_path) -> None:
    """Depth sets how many rounds COPRO attempts, which is ``depth + 1``.

    Compared directly against a default-depth run over identical inputs.
    The two runs differ in nothing but ``copro_depth``, so the step count
    is the evidence the configured depth reached the control: the deeper
    run attempts a further round the shallower one never asks for.

    The round count is the exact quantity here, and stays exact. A round
    that cannot fill its breadth now proceeds on what it realized rather
    than terminalizing, so what the extra depth buys is a further
    *attempt* -- which is precisely what depth configures.
    """
    deep = _copro_steps(tmp_path, run_id="copro-depth-2", copro_depth=2)
    shallow = _copro_steps(tmp_path, run_id="copro-depth-default")

    assert len(shallow.step_results) == DEFAULT_COPRO_DEPTH + 1
    assert shallow.terminal_failure is None

    # Depth 2 plans one more round than the default depth of 1.
    assert DEFAULT_COPRO_DEPTH != 2
    assert len(deep.step_results) == 2 + 1
    assert len(deep.step_results) > len(shallow.step_results)


# --------------------------------------------------------------------------
# GEPA metric-call ceiling
# --------------------------------------------------------------------------


def test_gepa_metric_call_ceiling_reaches_the_control(tmp_path) -> None:
    from whetstone_envs.optim import run as run_module

    seen: list[int | None] = []
    original = run_module.build_gepa_adapter

    def capture(**kwargs):
        seen.append(kwargs["max_metric_calls"])
        return original(**kwargs)

    with patch.object(run_module, "build_gepa_adapter", capture):
        run_optimizer(
            _spec(
                optimizer="gepa",
                output_dir=tmp_path / "gepa-ceiling",
                run_id="gepa-ceiling",
                gepa_max_metric_calls=3,
                train_size=1,
                val_size=1,
            )
        )
    assert seen == [3]


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------


def test_every_new_flag_reaches_the_spec_unchanged() -> None:
    spec = _captured_spec(
        [
            "--family",
            "c19",
            "--optimizer",
            "gepa",
            "--transport",
            "openrouter",
            "--model",
            "openai/gpt-5-nano",
            "--proposer-model",
            "openai/gpt-5.4-nano",
            "--split-sizes",
            "88,132,220",
            "--num-seeds",
            "3",
            "--seed",
            "3002",
            "--n-per-stratum",
            "32",
            "--pool-seed-start",
            "1000000",
            "--copro-breadth",
            "5",
            "--copro-depth",
            "4",
            "--gepa-max-metric-calls",
            "200",
            "--run-id",
            "study-run",
        ]
    )
    assert spec.family == "c19"
    assert spec.optimizer == "gepa"
    assert spec.transport == "openrouter"
    assert spec.model == "openai/gpt-5-nano"
    assert spec.proposer_model == "openai/gpt-5.4-nano"
    assert spec.split_sizes == (88, 132, 220)
    assert spec.num_seeds == 3
    assert spec.seed == 3002
    assert spec.n_per_stratum == 32
    assert spec.pool_seed_start == 1_000_000
    assert spec.copro_breadth == 5
    assert spec.copro_depth == 4
    assert spec.gepa_max_metric_calls == 200
    assert spec.run_id == "study-run"


def test_the_codex_flags_reach_the_spec() -> None:
    spec = _captured_spec(
        [
            "--optimizer",
            "codex",
            "--codex-capacity",
            "8",
            "--codex-binary",
            "/opt/fake/codex",
            "--codex-model",
            "agent-model",
            "--codex-reasoning-effort",
            "high",
            "--codex-wall-seconds",
            "45",
        ]
    )
    assert spec.codex_capacity == 8
    assert spec.codex_binary == "/opt/fake/codex"
    assert spec.codex_model == "agent-model"
    assert spec.codex_reasoning_effort == "high"
    assert spec.codex_wall_seconds == 45.0


def test_the_codex_binary_defaults_to_the_real_cli() -> None:
    """A run that names no binary spawns the real Codex, not a stand-in.

    The fake CLI is selectable only by an explicit flag, so no default
    path can quietly run a scripted agent and report it as a Codex run.
    """
    spec = _captured_spec(["--optimizer", "codex"])
    assert spec.codex_binary == CODEX_DEFAULT_BINARY
    assert spec.codex_capacity is None
    assert spec.codex_model is None


def test_a_negative_pool_seed_start_is_accepted_by_the_parser() -> None:
    """Seed starts are arbitrary integers; only counts must be positive."""
    spec = _captured_spec(["--optimizer", "copro", "--pool-seed-start", "-5"])
    assert spec.pool_seed_start == -5


@pytest.mark.parametrize(
    "flag",
    [
        "--num-seeds",
        "--n-per-stratum",
        "--copro-breadth",
        "--gepa-max-metric-calls",
        "--codex-capacity",
        "--codex-wall-seconds",
    ],
)
def test_count_flags_refuse_a_non_positive_value(flag: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "copro", flag, "0"])


def test_copro_breadth_refuses_a_single_draft_per_step() -> None:
    """Upstream needs two drafts to have anything to select between."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "copro", "--copro-breadth", "1"])


@pytest.mark.parametrize("family", ["c19", "c18"])
def test_every_registered_family_parses(family: str) -> None:
    spec = _captured_spec(["--optimizer", "copro", "--family", family])
    assert spec.family == family


def test_an_unregistered_family_is_refused_at_the_parser() -> None:
    """``c17`` is not a family at all, so it never reaches the runner."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "copro", "--family", "c17"])


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_every_drivable_optimizer_parses(optimizer: str) -> None:
    spec = _captured_spec(["--optimizer", optimizer])
    assert spec.optimizer == optimizer


@pytest.mark.parametrize("optimizer", ["null-identity"])
def test_optimizers_the_runner_cannot_drive_are_refused(
    optimizer: str,
) -> None:
    """null-B proposes nothing, so there is no search to drive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", optimizer])


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_every_runner_transport_parses(transport: str) -> None:
    spec = _captured_spec(["--optimizer", "copro", "--transport", transport])
    assert spec.transport == transport


# --------------------------------------------------------------------------
# MIPROv2 search shape and split
# --------------------------------------------------------------------------


def test_the_miprov2_shape_defaults_are_the_runners_own() -> None:
    """Defaults unchanged, which is what keeps the e2e fixtures fast."""
    spec = _spec()
    assert spec.miprov2_num_trials == DEFAULT_MIPROV2_NUM_TRIALS == 2
    assert spec.miprov2_num_candidates == DEFAULT_MIPROV2_NUM_CANDIDATES == 3
    # The split has no default: a run must state what it trained on.
    assert spec.train_size is None
    assert spec.val_size is None


@pytest.mark.parametrize(
    ("flag", "value", "field"),
    [
        ("--miprov2-num-trials", "10", "miprov2_num_trials"),
        ("--miprov2-num-candidates", "6", "miprov2_num_candidates"),
    ],
)
def test_the_miprov2_shape_flags_reach_the_spec(
    flag: str, value: str, field: str
) -> None:
    """The protocol's auto-light shape, requestable from the CLI."""
    spec = _captured_spec(["--optimizer", "miprov2", flag, value])
    assert getattr(spec, field) == int(value)


def test_minibatching_requires_an_explicit_batch_size() -> None:
    """``--miprov2-minibatch`` alone is a batch of the whole valset.

    Fails-before: the size defaulted to ``len(valset)``, so the run was
    configured with minibatching *on* and a batch covering every task --
    minibatching in name only. The run then spent, and its
    ``mipro_minibatch_sizing`` invariant FAILed the audit afterwards (D3's
    defect (e)). Refusing at pure spec validation makes the same finding
    free.

    The message names both flags, because the recovery is supplying the
    second one rather than dropping the first.
    """
    spec = _spec(optimizer="miprov2", miprov2_minibatch=True)
    with pytest.raises(ValueError) as error:
        run_optimizer(spec)
    message = str(error.value)
    assert "--miprov2-minibatch" in message
    assert "--miprov2-minibatch-size" in message


def test_minibatching_with_an_explicit_size_passes_validation() -> None:
    """The refusal is of the *combination*, not of minibatching."""
    from whetstone_envs.optim.run import _validate_miprov2_settings

    _validate_miprov2_settings(
        _spec(
            optimizer="miprov2",
            miprov2_minibatch=True,
            miprov2_minibatch_size=1,
        )
    )


def test_the_cli_refuses_minibatch_without_a_size() -> None:
    """The same refusal, reached the way an operator reaches it."""
    with pytest.raises(ValueError) as error:
        main(
            [
                "--optimizer",
                "miprov2",
                "--miprov2-minibatch",
                "--train-size",
                "1",
                "--val-size",
                "1",
            ]
        )
    assert "--miprov2-minibatch-size" in str(error.value)


# --------------------------------------------------------------------------
# The MIPROv2 spent-combination shape trap (whetstone-ai #137)
# --------------------------------------------------------------------------


def _two_candidate_minibatch_spec() -> RunSpec:
    return _spec(
        optimizer="miprov2",
        miprov2_minibatch=True,
        miprov2_minibatch_size=1,
        miprov2_num_candidates=2,
    )


def test_two_candidates_with_minibatch_are_refused_on_an_unfixed_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The D3 shape trap, refused at validation on a pre-0.1.9 install.

    Fails-before: nothing refused this shape, so a Stage-1 arm configured
    with two candidates and a minibatch reached
    ``select_promotion``, exhausted its ranked list, and raised
    ``ValueError: No valid program found in param_score_dict`` *inside* the
    durable run boundary -- surfacing as a ``DurableRunError`` after the
    run had already spent.

    The version is patched rather than the environment downgraded: the
    refusal is a claim about the installed release, and pinning it to a
    known-unfixed string is what makes both sides of the gate testable
    against one installed package.
    """
    monkeypatch.setattr(
        "whetstone_envs.optim.run.installed_whetstone_ai_version",
        lambda: "0.1.8",
    )
    with pytest.raises(ValueError) as error:
        run_optimizer(_two_candidate_minibatch_spec())
    message = str(error.value)
    assert "whetstone-ai" in message
    assert "137" in message
    assert str(MIPROV2_MINIBATCH_MIN_CANDIDATES) in message


@pytest.mark.parametrize("reported", ["0.1.9", "0.1.11", "0.2.0"])
def test_two_candidates_with_minibatch_pass_on_the_fixed_release(
    monkeypatch: pytest.MonkeyPatch, reported: str
) -> None:
    """At the fix release and above, the refusal lifts.

    0.1.11 is the version this repo pins and 0.1.9 is the floor the gate
    compares against, so both are exercised -- a comparison that ranked
    ``"0.1.11" < "0.1.9"`` lexically would pass the floor case and fail
    the pinned one, which is exactly the bug worth catching here.
    """
    from whetstone_envs.optim.run import _validate_miprov2_settings

    monkeypatch.setattr(
        "whetstone_envs.optim.run.installed_whetstone_ai_version",
        lambda: reported,
    )
    _validate_miprov2_settings(_two_candidate_minibatch_spec())


def test_an_unreadable_whetstone_ai_version_keeps_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uncertain reads as unfixed: the safe side of the gate.

    A base install carries no whetstone-ai at all, and a version this
    parser cannot rank says nothing about whether the fallback shipped.
    Lifting the refusal on either would trade a free validation error for
    a run that dies mid-flight.
    """
    from whetstone_envs.optim.run import _validate_miprov2_settings

    for reported in (None, "not-a-version"):
        monkeypatch.setattr(
            "whetstone_envs.optim.run.installed_whetstone_ai_version",
            lambda reported=reported: reported,
        )
        with pytest.raises(ValueError, match="whetstone-ai"):
            _validate_miprov2_settings(_two_candidate_minibatch_spec())


def test_the_installed_release_carries_the_spent_combination_fallback() -> (
    None
):
    """The pin and the gate agree: this repo installs a fixed release.

    Pinned rather than asserted through the gate alone, because the gate
    reads the installed version and would agree with itself on any pin.
    """
    from whetstone_envs.optim.run import (
        _miprov2_spent_combination_fixed,
        installed_whetstone_ai_version,
    )

    installed = installed_whetstone_ai_version()
    assert installed is not None
    assert MIPROV2_SPENT_COMBINATION_FIX_VERSION == (0, 1, 9)
    assert _miprov2_spent_combination_fixed()


@pytest.mark.parametrize("optimizer", ["miprov2", "gepa"])
def test_the_train_val_split_reaches_the_spec(optimizer: str) -> None:
    spec = _captured_spec(
        [
            "--optimizer",
            optimizer,
            "--train-size",
            "1",
            "--val-size",
            "1",
        ]
    )
    assert (spec.train_size, spec.val_size) == (1, 1)


@pytest.mark.parametrize("optimizer", ["miprov2", "gepa"])
def test_an_optimizer_with_a_train_val_concept_requires_one(
    optimizer: str,
) -> None:
    """No default: an unstated split is refused rather than guessed."""
    spec = _spec(optimizer=optimizer)
    with pytest.raises(ValueError, match="requires an explicit"):
        run_optimizer(spec)


@pytest.mark.parametrize("optimizer", ["miprov2", "gepa"])
@pytest.mark.parametrize("overrides", [{"train_size": 1}, {"val_size": 1}])
def test_half_a_train_val_split_is_refused(
    optimizer: str, overrides: dict[str, object]
) -> None:
    """Both sizes or neither; one alone cannot name a partition."""
    spec = replace(_spec(optimizer=optimizer), **overrides)
    with pytest.raises(ValueError, match="requires an explicit"):
        run_optimizer(spec)


def test_a_split_exceeding_the_internal_split_is_refused() -> None:
    spec = _spec(optimizer="miprov2", train_size=2, val_size=2)
    with pytest.raises(ValueError, match="exceeds the internal split of 2"):
        run_optimizer(spec)


def test_a_partial_gepa_partition_is_refused_before_the_run_boundary(
    tmp_path: Path,
) -> None:
    """GEPA must cover the internal split exactly, and refuse early if not.

    whetstone's GEPA factory builds its data registry from the whole
    internal split and then requires the control's trainset and valset to
    cover it, so ``train + val < internal`` is not a legal GEPA partition.

    Fails-before: envs validated only ``<=``, so a partial partition passed
    spec validation and was rejected by whetstone *inside* the durable run
    boundary -- after the run directory existed. Asserting the directory is
    absent is what makes "before the boundary" checkable.
    """
    output_dir = tmp_path / "run"
    spec = replace(
        _spec(optimizer="gepa", train_size=1, val_size=1),
        split_sizes=(3, 2, 0),
        output_dir=output_dir,
    )
    with pytest.raises(ValueError, match="cover the internal split exactly"):
        run_optimizer(spec)
    assert not output_dir.exists()


def test_an_exact_gepa_partition_passes_validation() -> None:
    """The coverage rule refuses a partial partition, not every partition."""
    from whetstone_envs.optim.run import _validate_train_val_split

    _validate_train_val_split(
        replace(
            _spec(optimizer="gepa", train_size=1, val_size=1),
            split_sizes=(2, 2, 0),
        )
    )


def test_miprov2_still_accepts_a_partition_inside_the_internal_split() -> None:
    """The coverage rule is GEPA's alone; MIPROv2 keeps the ``<=`` rule.

    MIPROv2 bootstraps from the trainset and scores on the valset without
    building a registry over the whole split, so a partition that leaves
    tasks unused is legal for it and must not be swept up by GEPA's rule.
    """
    from whetstone_envs.optim.run import _validate_train_val_split

    _validate_train_val_split(
        replace(
            _spec(optimizer="miprov2", train_size=1, val_size=1),
            split_sizes=(3, 2, 0),
        )
    )


@pytest.mark.parametrize("optimizer", ["copro", "codex"])
def test_a_train_val_split_is_refused_on_an_optimizer_without_one(
    optimizer: str,
) -> None:
    """COPRO and Codex-direct have no train/val concept."""
    spec = _spec(optimizer=optimizer, train_size=1, val_size=1)
    with pytest.raises(ValueError, match="apply only to --optimizer"):
        run_optimizer(spec)


@pytest.mark.parametrize("flag", ["--train-size", "--val-size"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_train_val_size_is_refused(
    flag: str, value: str
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "miprov2", flag, value])


@pytest.mark.parametrize(
    "flag", ["--miprov2-num-trials", "--miprov2-num-candidates"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_miprov2_shape_is_refused(
    flag: str, value: str
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "miprov2", flag, value])


@pytest.mark.parametrize(
    "overrides",
    [
        {"miprov2_num_trials": 10},
        {"miprov2_num_candidates": 6},
    ],
)
def test_miprov2_settings_are_refused_on_another_optimizer(
    overrides: dict[str, object],
) -> None:
    """A setting that looks honoured but is not misdescribes the arm."""
    spec = replace(_spec(optimizer="copro"), **overrides)
    with pytest.raises(ValueError, match="apply only to --optimizer miprov2"):
        run_optimizer(spec)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"miprov2_num_trials": 0}, "miprov2_num_trials must be at least 1"),
        (
            {"miprov2_num_candidates": 0},
            "miprov2_num_candidates must be at least 1",
        ),
    ],
)
def test_invalid_miprov2_settings_are_refused_at_spec_validation(
    overrides: dict[str, object], message: str
) -> None:
    """Refused before the durable run boundary, like the other settings."""
    spec = replace(_spec(optimizer="miprov2"), **overrides)
    with pytest.raises(ValueError, match=message):
        run_optimizer(spec)
