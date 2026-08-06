from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.c19 import generation
from whetstone_envs.c19._minigrid import clone_state, pprint_grid, trace_script
from whetstone_envs.c19.generation import (
    DEFAULT_N_PER_STRATUM,
    DEFAULT_SPLIT_SIZES,
    MAX_N_PER_STRATUM,
    generate_pool,
    strata_labels,
)
from whetstone_envs.c19.model import (
    ACTION_ALPHABET,
    Action,
    C19Fact,
    DoorState,
    ObjectKind,
    WorldSnapshot,
    parse_command,
)
from whetstone_envs.c19.oracle import derive_fact
from whetstone_envs.c19.scenarios import (
    BuiltScenario,
    C19Scenario,
    C19Size,
    build_scenario,
)
from whetstone_envs.instances import public_prompt_identity
from whetstone_envs.manifests import content_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

    from whetstone_envs.instances import Instance
    from whetstone_envs.pools import PoolSplit, TaskPool


@pytest.fixture(scope="module")
def default_pool() -> TaskPool:
    return generate_pool()


@pytest.fixture(scope="module")
def default_split(default_pool: TaskPool) -> PoolSplit:
    return default_pool.split(*DEFAULT_SPLIT_SIZES)


def _counts(instances: Iterable[Instance]) -> Counter[str]:
    return Counter(instance.strata[0] for instance in instances)


def test_default_pool_is_exactly_22_by_16(default_pool: TaskPool) -> None:
    assert DEFAULT_N_PER_STRATUM == 16
    assert len(strata_labels()) == 22
    assert len(default_pool) == 352
    assert default_pool.stratum_counts() == dict.fromkeys(
        strata_labels(),
        16,
    )


def test_actual_default_split_is_exactly_4_6_6_per_stratum(
    default_split: PoolSplit,
) -> None:
    assert DEFAULT_SPLIT_SIZES == (88, 132, 132)
    assert set(_counts(default_split.internal_eval).values()) == {4}
    assert set(_counts(default_split.official).values()) == {6}
    assert set(_counts(default_split.held_out).values()) == {6}


def test_every_stratum_has_multiple_gold_values_in_every_default_split(
    default_split: PoolSplit,
) -> None:
    for subset in (
        default_split.internal_eval,
        default_split.official,
        default_split.held_out,
    ):
        for label in strata_labels():
            gold_values = {
                instance.gold
                for instance in subset
                if instance.strata == (label,)
            }
            assert len(gold_values) >= 2, (label, gold_values)


def test_carrying_quotas_hold_overall_and_in_every_actual_split(
    default_pool: TaskPool,
    default_split: PoolSplit,
) -> None:
    carrying_labels = tuple(
        label for label in strata_labels() if label.endswith("|carrying")
    )
    for label in carrying_labels:
        assert Counter(
            instance.gold for instance in default_pool.in_stratum(label)
        ) == {"yes": 8, "no": 8}

    for subset, expected in (
        (default_split.internal_eval, {"yes": 2, "no": 2}),
        (default_split.official, {"yes": 3, "no": 3}),
        (default_split.held_out, {"yes": 3, "no": 3}),
    ):
        for label in carrying_labels:
            assert (
                Counter(
                    instance.gold
                    for instance in subset
                    if instance.strata == (label,)
                )
                == expected
            )


def test_public_inputs_and_identities_are_exact_and_unique(
    default_pool: TaskPool,
) -> None:
    identities = [
        public_prompt_identity(instance) for instance in default_pool.instances
    ]

    assert all(
        set(instance.prompt_inputs) == {"grid", "command", "question"}
        for instance in default_pool.instances
    )
    assert len(identities) == len(set(identities)) == 352


def test_one_scene_seed_is_shared_across_applicable_fact_projections(
    default_pool: TaskPool,
) -> None:
    projections: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    scenes: dict[tuple[str, str, int], set[tuple[str, str]]] = defaultdict(set)
    for instance in default_pool.instances:
        scenario, size, fact = instance.strata[0].split("|")
        scene = (scenario, size, instance.seed)
        projections[scene].add(fact)
        scenes[scene].add(
            (
                instance.prompt_inputs["grid"],
                instance.prompt_inputs["command"],
            )
        )

    for scene, facts in projections.items():
        scenario, _size, _seed = scene
        expected = (
            {"coordinate", "heading", "front"}
            if scenario == "navigation"
            else {"coordinate", "heading", "front", "carrying"}
        )
        assert facts == expected
        assert len(scenes[scene]) == 1


def test_generation_is_deterministic_and_does_not_touch_global_rng() -> None:
    random.seed(912_345)
    before = random.getstate()
    first = generate_pool(n_per_stratum=2, seed_start=765_432)
    middle = random.getstate()
    second = generate_pool(n_per_stratum=2, seed_start=765_432)

    assert before == middle == random.getstate()
    assert content_hash(first) == content_hash(second)
    assert first.instances == second.instances


def test_generation_is_hash_seed_stable() -> None:
    code = """
from whetstone_envs.c19.generation import generate_pool
from whetstone_envs.manifests import content_hash
print(content_hash(generate_pool(n_per_stratum=2, seed_start=765432)))
"""
    hashes: set[str] = set()
    for hash_seed in ("0", "1", "927451"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        hashes.add(completed.stdout.strip())

    assert len(hashes) == 1


def _rebuild(instance: Instance) -> BuiltScenario:
    scenario_text, size_text, _fact = instance.strata[0].split("|")
    scenario = C19Scenario(scenario_text)
    size = C19Size.SMALL if size_text == "small" else C19Size.MEDIUM
    carrying = (
        None
        if scenario is C19Scenario.NAVIGATION
        else not instance.prompt_inputs["command"].endswith("D")
    )
    built = build_scenario(
        scenario,
        size,
        instance.seed,
        carrying=carrying,
    )
    assert pprint_grid(built.state) == instance.prompt_inputs["grid"]
    assert built.command == instance.prompt_inputs["command"]
    return built


def _assert_split_causal_witnesses(  # noqa: PLR0912
    instances: tuple[Instance, ...],
) -> None:
    changed_actions: set[Action] = set()
    successful_forward = False
    blocked_forward = False
    successful_drop = False
    door_witness = False
    seen_scenes: set[tuple[str, int]] = set()

    for instance in instances:
        scene = (instance.strata[0].rsplit("|", maxsplit=1)[0], instance.seed)
        if scene in seen_scenes:
            continue
        seen_scenes.add(scene)
        built = _rebuild(instance)
        actions = parse_command(built.command)
        trace = trace_script(clone_state(built.state), built.command)
        for action, before, after in zip(
            actions,
            trace[:-1],
            trace[1:],
            strict=True,
        ):
            if before != after:
                changed_actions.add(action)
            if action is Action.FORWARD:
                moved = (before.agent_row, before.agent_column) != (
                    after.agent_row,
                    after.agent_column,
                )
                successful_forward |= moved
                blocked_forward |= not moved
            if action is Action.DROP and before != after:
                successful_drop = True

        if instance.strata[0].startswith("door|"):
            initial = trace[0]
            initial_doors = {
                (cell.row, cell.column): cell.object.door_state
                for cell in initial.cells
                if cell.object is not None
                and cell.object.kind is ObjectKind.DOOR
            }
            toggled_positions: set[tuple[int, int]] = set()
            traversed_positions: set[tuple[int, int]] = set()
            for action, before, after in zip(
                actions,
                trace[:-1],
                trace[1:],
                strict=True,
            ):
                if action is Action.TOGGLE and before != after:
                    for position in initial_doors:
                        index = position[0] * before.width + position[1]
                        if before.cells[index] != after.cells[index]:
                            toggled_positions.add(position)
                if action is Action.FORWARD:
                    position = (after.agent_row, after.agent_column)
                    if position != (before.agent_row, before.agent_column):
                        index = position[0] * after.width + position[1]
                        obj = after.cells[index].object
                        if (
                            obj is not None
                            and obj.kind is ObjectKind.DOOR
                            and obj.door_state is DoorState.OPEN
                        ):
                            traversed_positions.add(position)
            door_witness |= (
                set(initial_doors.values())
                == {DoorState.LOCKED, DoorState.CLOSED}
                and toggled_positions == set(initial_doors)
                and traversed_positions == set(initial_doors)
            )

    assert {Action.LEFT, Action.RIGHT, Action.FORWARD}.issubset(
        changed_actions,
    )
    assert Action.PICKUP in changed_actions
    assert Action.DROP in changed_actions
    assert Action.TOGGLE in changed_actions
    assert successful_forward
    assert blocked_forward
    assert successful_drop
    assert door_witness


def test_every_actual_split_has_causal_action_and_motif_witnesses(
    default_split: PoolSplit,
) -> None:
    _assert_split_causal_witnesses(default_split.internal_eval)
    _assert_split_causal_witnesses(default_split.official)
    _assert_split_causal_witnesses(default_split.held_out)


def _counterfactually_relevant_actions(
    instances: tuple[Instance, ...],
) -> set[Action]:
    relevant: set[Action] = set()
    for instance in instances:
        grid_text = instance.prompt_inputs["grid"]
        command = instance.prompt_inputs["command"]
        fact = C19Fact(instance.strata[0].rsplit("|", maxsplit=1)[1])
        for index, action in enumerate(parse_command(command)):
            without_action = command[:index] + command[index + 1 :]
            if derive_fact(grid_text, without_action, fact) != instance.gold:
                relevant.add(action)
    return relevant


def test_every_action_is_counterfactually_relevant_in_each_split(
    default_split: PoolSplit,
) -> None:
    expected = set(ACTION_ALPHABET)

    assert (
        _counterfactually_relevant_actions(
            default_split.internal_eval,
        )
        == expected
    )
    assert (
        _counterfactually_relevant_actions(default_split.official) == expected
    )
    assert (
        _counterfactually_relevant_actions(default_split.held_out) == expected
    )


def test_every_default_gold_is_independently_derived(
    default_pool: TaskPool,
) -> None:
    for instance in default_pool.instances:
        fact = instance.strata[0].rsplit("|", maxsplit=1)[1]
        assert instance.gold == derive_fact(
            instance.prompt_inputs["grid"],
            instance.prompt_inputs["command"],
            generation.C19Fact(fact),
        )


def test_deliberate_prefix_mismatch_fails_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = build_scenario(
        C19Scenario.NAVIGATION,
        C19Size.SMALL,
        707,
        carrying=None,
    )
    original_simulate = generation.simulate

    def mismatching_simulate(
        grid_text: str,
        command: str,
    ) -> WorldSnapshot:
        return replace(
            original_simulate(grid_text, command),
            agent_row=99,
        )

    monkeypatch.setattr(generation, "simulate", mismatching_simulate)

    with pytest.raises(AssertionError):
        generation._assert_prefix_agreement(
            state=built.state,
            grid_text=pprint_grid(built.state),
            command=built.command,
            context=generation._SceneContext(
                scenario=C19Scenario.NAVIGATION,
                size=C19Size.SMALL,
                seed=707,
            ),
        )


@pytest.mark.parametrize("value", [0, -1])
def test_generation_rejects_nonpositive_counts(value: int) -> None:
    with pytest.raises(ValueError):
        generate_pool(n_per_stratum=value)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_generation_rejects_noninteger_counts(value: object) -> None:
    with pytest.raises(TypeError):
        generate_pool(
            n_per_stratum=value,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("value", [False, 1.0, "1"])
def test_generation_rejects_noninteger_seed_start(value: object) -> None:
    with pytest.raises(TypeError):
        generate_pool(
            seed_start=value,  # ty: ignore[invalid-argument-type]
        )


def test_generation_accepts_maximum_count() -> None:
    pool = generate_pool(n_per_stratum=MAX_N_PER_STRATUM)

    assert len(pool) == len(strata_labels()) * MAX_N_PER_STRATUM
    assert set(pool.stratum_counts().values()) == {MAX_N_PER_STRATUM}


def test_generation_rejects_count_above_maximum_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_build_scenario(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        pytest.fail("validation must precede scenario generation")

    monkeypatch.setattr(
        generation, "build_scenario", unexpected_build_scenario
    )

    with pytest.raises(ValueError):
        generate_pool(n_per_stratum=MAX_N_PER_STRATUM + 1)
