from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from whetstone_envs.c19._minigrid import (
    clone_state,
    pprint_grid,
    trace_script,
)
from whetstone_envs.c19.model import C19Fact, WorldSnapshot
from whetstone_envs.c19.oracle import derive_fact, simulate
from whetstone_envs.c19.scenarios import (
    SCENARIO_ORDER,
    SIZE_ORDER,
    C19Scenario,
    C19Size,
    build_scenario,
)
from whetstone_envs.instances import Instance, make_instance
from whetstone_envs.manifests import Manifest
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from whetstone_envs.c19._minigrid import MiniGridState


GENERATOR_VERSION = "c19-custom-v2"
DEFAULT_N_PER_STRATUM = 16
MAX_N_PER_STRATUM = 128
DEFAULT_SEED_START = 1_000_000
DEFAULT_SPLIT_SIZES: tuple[int, int, int] = (88, 132, 132)

_COMMON_FACTS: tuple[C19Fact, ...] = (
    C19Fact.COORDINATE,
    C19Fact.HEADING,
    C19Fact.FRONT,
)
_CARRYING_FACTS: tuple[C19Fact, ...] = (
    C19Fact.COORDINATE,
    C19Fact.HEADING,
    C19Fact.FRONT,
    C19Fact.CARRYING,
)
_QUESTIONS: dict[C19Fact, str] = {
    C19Fact.COORDINATE: "What is the agent's final coordinate?",
    C19Fact.HEADING: "What is the agent's final heading?",
    C19Fact.FRONT: (
        "What is in the cell directly in front of the agent at the end?"
    ),
    C19Fact.CARRYING: (
        "Is the agent carrying an object at the end? Answer yes or no."
    ),
}


@dataclass(frozen=True, slots=True)
class _SceneContext:
    scenario: C19Scenario
    size: C19Size
    seed: int


def _facts(scenario: C19Scenario) -> tuple[C19Fact, ...]:
    if scenario is C19Scenario.NAVIGATION:
        return _COMMON_FACTS
    return _CARRYING_FACTS


def stratum_label(
    scenario: C19Scenario,
    size: C19Size,
    fact: C19Fact,
) -> str:
    """Return the stable family|size|fact task-stratum label."""
    return f"{scenario.value}|{size.name.lower()}|{fact.value}"


def strata_labels() -> tuple[str, ...]:
    """Return all 22 default strata in persisted generation order."""
    return tuple(
        stratum_label(scenario, size, fact)
        for scenario in SCENARIO_ORDER
        for size in SIZE_ORDER
        for fact in _facts(scenario)
    )


def _assert_prefix_agreement(
    *,
    state: MiniGridState,
    grid_text: str,
    command: str,
    context: _SceneContext,
) -> tuple[WorldSnapshot, ...]:
    live_trace = trace_script(clone_state(state), command)
    for prefix_length, live_snapshot in enumerate(live_trace[1:], start=1):
        command_prefix = command[:prefix_length]
        oracle_snapshot = simulate(grid_text, command_prefix)
        if live_snapshot != oracle_snapshot:
            msg = (
                "C19 live/oracle mismatch for "
                f"family={context.scenario.value} "
                f"size={context.size.name.lower()} "
                f"seed={context.seed} prefix={prefix_length} "
                f"command_prefix={command_prefix!r}: "
                f"live={live_snapshot!r} oracle={oracle_snapshot!r}"
            )
            raise AssertionError(msg)
    if not any(before != after for before, after in pairwise(live_trace)):
        msg = (
            "C19 constructive command changed no state for "
            f"family={context.scenario.value} "
            f"size={context.size.name.lower()} seed={context.seed}"
        )
        raise AssertionError(msg)
    return live_trace


def _build_scene(
    scenario: C19Scenario,
    size: C19Size,
    seed: int,
    *,
    carrying: bool | None,
) -> tuple[str, str]:
    context = _SceneContext(scenario=scenario, size=size, seed=seed)
    built = build_scenario(
        scenario,
        size,
        seed,
        carrying=carrying,
    )
    grid_text = pprint_grid(built.state)
    trace = _assert_prefix_agreement(
        state=built.state,
        grid_text=grid_text,
        command=built.command,
        context=context,
    )
    if carrying is not None:
        actual_carrying = trace[-1].carrying is not None
        if actual_carrying is not carrying:
            msg = (
                "C19 carrying schedule disagreement for "
                f"family={scenario.value} size={size.name.lower()} "
                f"seed={seed}: expected={carrying} actual={actual_carrying}"
            )
            raise AssertionError(msg)
    return grid_text, built.command


def _instance(
    *,
    context: _SceneContext,
    fact: C19Fact,
    grid_text: str,
    command: str,
) -> Instance:
    return make_instance(
        id=(
            f"c19-{context.scenario.value}-"
            f"{context.size.name.lower()}-{fact.value}-{context.seed}"
        ),
        seed=context.seed,
        strata=stratum_label(context.scenario, context.size, fact),
        prompt_inputs={
            "grid": grid_text,
            "command": command,
            "question": _QUESTIONS[fact],
        },
        gold=derive_fact(grid_text, command, fact),
    )


def _validate_generation_inputs(
    n_per_stratum: int,
    seed_start: int,
) -> None:
    if type(n_per_stratum) is not int:
        msg = "n_per_stratum must be an int"
        raise TypeError(msg)
    if n_per_stratum <= 0:
        msg = "n_per_stratum must be positive"
        raise ValueError(msg)
    if n_per_stratum > MAX_N_PER_STRATUM:
        msg = (
            "n_per_stratum must be at most "
            f"{MAX_N_PER_STRATUM}, got {n_per_stratum}"
        )
        raise ValueError(msg)
    if type(seed_start) is not int:
        msg = "seed_start must be an int"
        raise TypeError(msg)


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate the deterministic 22-stratum C19 task pool.

    ``n_per_stratum`` must be between 1 and ``MAX_N_PER_STRATUM`` (128),
    inclusive.

    Each family-size scene seed is reused across all applicable fact
    projections. Pool order is scene-index-major so the actual TaskPool split
    schedule receives alternating carrying=yes/no outcomes within every
    carrying stratum.
    """
    _validate_generation_inputs(n_per_stratum, seed_start)
    instances: list[Instance] = []
    family_sizes = tuple(
        (scenario, size) for scenario in SCENARIO_ORDER for size in SIZE_ORDER
    )

    for scene_index in range(n_per_stratum):
        for group_index, (scenario, size) in enumerate(family_sizes):
            seed = seed_start + group_index * n_per_stratum + scene_index
            carrying = (
                None
                if scenario is C19Scenario.NAVIGATION
                else scene_index % 2 == 0
            )
            grid_text, command = _build_scene(
                scenario,
                size,
                seed,
                carrying=carrying,
            )
            instances.extend(
                _instance(
                    context=_SceneContext(
                        scenario=scenario,
                        size=size,
                        seed=seed,
                    ),
                    fact=fact,
                    grid_text=grid_text,
                    command=command,
                )
                for fact in _facts(scenario)
            )
    return TaskPool(instances)


def build_manifest(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    seed_start: int = DEFAULT_SEED_START,
) -> Manifest:
    """Generate one C19 pool and build its manifest."""
    pool = generate_pool(
        n_per_stratum=n_per_stratum,
        seed_start=seed_start,
    )
    seeds = tuple(instance.seed for instance in pool.instances)
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(min(seeds), max(seeds) + 1),
    )
