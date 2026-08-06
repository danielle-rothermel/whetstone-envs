from __future__ import annotations

import random
from functools import cache
from typing import TYPE_CHECKING

from whetstone_envs.c23._domain import (
    GeneratedTask,
    GenerationConfiguration,
    Hypothesis,
    RuleConfiguration,
    RuleFamily,
    StratumConfiguration,
)
from whetstone_envs.c23._inductionbench import (
    characteristic_inputs,
    sample_hypothesis,
)
from whetstone_envs.c23._prompts import render_demonstrations
from whetstone_envs.c23._selection import select_task
from whetstone_envs.c23._transducers import apply_reference
from whetstone_envs.instances import (
    Instance,
    make_instance,
    public_prompt_identity,
)
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

GENERATOR_VERSION = "c23-generate-4"
DEFAULT_N_PER_STRATUM = 50
_DEFAULT_SEED_START = 555_000_000
_DEFAULT_SPLIT_PER_STRATUM = (10, 20, 20)

DEFAULT_CONFIG = GenerationConfiguration(
    vocab=("a", "b", "c", "d"),
    strata=(
        StratumConfiguration(
            "S1",
            RuleConfiguration(RuleFamily.ISL, 2),
            _DEFAULT_SEED_START,
        ),
        StratumConfiguration(
            "S2",
            RuleConfiguration(RuleFamily.L_OSL, 2),
            _DEFAULT_SEED_START + 1,
        ),
        StratumConfiguration(
            "S3",
            RuleConfiguration(RuleFamily.R_OSL, 2),
            _DEFAULT_SEED_START + 2,
        ),
        StratumConfiguration(
            "S4",
            RuleConfiguration(RuleFamily.ISL, 3),
            _DEFAULT_SEED_START + 3,
        ),
    ),
    demonstrations_per_instance=6,
    maximum_query_length=8,
    attempts_per_instance=100,
)


class GenerationExhaustedError(RuntimeError):
    """Raised when bounded generation cannot produce a unique public task."""


def generate_pool(*, n_per_stratum: int = DEFAULT_N_PER_STRATUM) -> TaskPool:
    """Generate the deterministic C23 pool with one public size override."""
    if type(n_per_stratum) is not int:
        raise TypeError("n_per_stratum must be an integer")
    if n_per_stratum < 1:
        raise ValueError("n_per_stratum must be positive")
    return _generate_pool(DEFAULT_CONFIG, n_per_stratum=n_per_stratum)


def _generate_pool(
    config: GenerationConfiguration,
    *,
    n_per_stratum: int,
) -> TaskPool:
    @cache
    def apply_hypothesis(hypothesis: Hypothesis, value: str) -> str:
        return apply_reference(hypothesis, value)

    inputs = characteristic_inputs(
        config.vocab,
        config.maximum_query_length,
    )
    seen_identities: set[tuple[tuple[str, str], ...]] = set()
    per_stratum: list[list[Instance]] = []
    for stratum in config.strata:
        rng = random.Random(stratum.seed)  # noqa: S311 - deterministic data
        retained: list[Instance] = []
        attempts = 0
        attempt_limit = config.attempts_per_instance * n_per_stratum
        while len(retained) < n_per_stratum and attempts < attempt_limit:
            task = _make_task(
                stratum,
                config,
                rng,
                inputs,
                apply_hypothesis,
            )
            attempts += 1
            if task is None:
                continue
            instance = _make_instance(
                task,
                stratum=stratum,
                retained_index=len(retained),
            )
            identity = public_prompt_identity(instance)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            retained.append(instance)
        if len(retained) != n_per_stratum:
            raise GenerationExhaustedError(
                f"stratum {stratum.label} retained {len(retained)} unique "
                f"tasks after {attempts} attempts; requested {n_per_stratum}",
            )
        per_stratum.append(retained)
    instances = (
        per_stratum[stratum_index][row]
        for row in range(n_per_stratum)
        for stratum_index in range(len(per_stratum))
    )
    return TaskPool(instances)


def _make_task(
    stratum: StratumConfiguration,
    config: GenerationConfiguration,
    rng: random.Random,
    inputs: tuple[str, ...],
    apply_hypothesis: Callable[[Hypothesis, str], str],
) -> GeneratedTask | None:
    hypothesis = sample_hypothesis(stratum.rule, config.vocab, rng)
    return select_task(hypothesis, config, rng, inputs, apply_hypothesis)


def _make_instance(
    task: GeneratedTask,
    *,
    stratum: StratumConfiguration,
    retained_index: int,
) -> Instance:
    return make_instance(
        id=f"c23-{stratum.label}-{stratum.seed}-{retained_index:04d}",
        seed=stratum.seed,
        strata=stratum.label,
        prompt_inputs={
            "demos_block": render_demonstrations(task.demonstrations),
            "query": task.query,
        },
        gold=task.gold,
    )


def default_split_sizes(pool: TaskPool) -> tuple[int, int, int]:
    """Validate a balanced C23 pool and return fixed 10/20/20 allocations."""
    expected_labels = tuple(stratum.label for stratum in DEFAULT_CONFIG.strata)
    if pool.strata != expected_labels:
        raise ValueError(
            "C23 pool strata must be "
            f"{expected_labels!r}, got {pool.strata!r}",
        )
    if any(len(instance.strata) != 1 for instance in pool.instances):
        raise ValueError("C23 instances must belong to exactly one stratum")
    counts = pool.stratum_counts()
    if len(set(counts.values())) != 1:
        raise ValueError(f"C23 pool strata must be balanced, got {counts!r}")
    required = sum(_DEFAULT_SPLIT_PER_STRATUM)
    available = next(iter(counts.values()))
    if available < required:
        raise ValueError(
            f"C23 split requires {required} instances per stratum, "
            f"but the pool has {available}",
        )
    stratum_count = len(expected_labels)
    internal, official, held_out = _DEFAULT_SPLIT_PER_STRATUM
    return (
        internal * stratum_count,
        official * stratum_count,
        held_out * stratum_count,
    )
