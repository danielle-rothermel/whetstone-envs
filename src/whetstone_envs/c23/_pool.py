from __future__ import annotations

import random
from functools import cache
from typing import TYPE_CHECKING

from whetstone_envs.c23._domain import (
    CategoryConfiguration,
    GeneratedTask,
    GenerationConfiguration,
    Hypothesis,
    RuleConfiguration,
    RuleFamily,
)
from whetstone_envs.c23._inductionbench import (
    characteristic_inputs,
    sample_hypothesis,
)
from whetstone_envs.c23._prompts import render_demonstrations
from whetstone_envs.c23._selection import select_task
from whetstone_envs.c23._transducers import apply_reference
from whetstone_envs.instances import Instance, InstanceGroup, InstancePool

if TYPE_CHECKING:
    from collections.abc import Callable

GENERATOR_VERSION = "c23-generate-4"
DEFAULT_N_PER_CATEGORY = 50
_DEFAULT_SEED_START = 555_000_000
_DEFAULT_SPLIT_PER_CATEGORY = (10, 20, 20)

DEFAULT_CONFIG = GenerationConfiguration(
    vocab=("a", "b", "c", "d"),
    categories=(
        CategoryConfiguration(
            "S1",
            RuleConfiguration(RuleFamily.ISL, 2),
            _DEFAULT_SEED_START,
        ),
        CategoryConfiguration(
            "S2",
            RuleConfiguration(RuleFamily.L_OSL, 2),
            _DEFAULT_SEED_START + 1,
        ),
        CategoryConfiguration(
            "S3",
            RuleConfiguration(RuleFamily.R_OSL, 2),
            _DEFAULT_SEED_START + 2,
        ),
        CategoryConfiguration(
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
    pass


def generate_pool(
    *, n_per_category: int = DEFAULT_N_PER_CATEGORY
) -> InstancePool:
    """Generate the deterministic C23 pool with one public size override."""
    if type(n_per_category) is not int:
        raise TypeError("n_per_category must be an integer")
    if n_per_category < 1:
        raise ValueError("n_per_category must be positive")
    return _generate_pool(DEFAULT_CONFIG, n_per_category=n_per_category)


def _generate_pool(
    config: GenerationConfiguration,
    *,
    n_per_category: int,
) -> InstancePool:
    @cache
    def apply_hypothesis(hypothesis: Hypothesis, value: str) -> str:
        return apply_reference(hypothesis, value)

    inputs = characteristic_inputs(
        config.vocab,
        config.maximum_query_length,
    )
    seen_identities: set[tuple[tuple[str, str], ...]] = set()
    per_category: list[list[Instance]] = []
    for category in config.categories:
        rng = random.Random(category.seed)  # noqa: S311 - deterministic data
        retained: list[Instance] = []
        attempts = 0
        attempt_limit = config.attempts_per_instance * n_per_category
        while len(retained) < n_per_category and attempts < attempt_limit:
            task = _make_task(
                category,
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
                category=category,
                retained_index=len(retained),
            )
            identity = instance.public_prompt_identity
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            retained.append(instance)
        if len(retained) != n_per_category:
            raise GenerationExhaustedError(
                f"category {category.label} retained {len(retained)} unique "
                f"tasks after {attempts} attempts; requested {n_per_category}",
            )
        per_category.append(retained)
    instances = (
        per_category[category_index][row]
        for row in range(n_per_category)
        for category_index in range(len(per_category))
    )
    return InstancePool(
        instance_group=InstanceGroup(name="c23", instances=tuple(instances)),
    )


def _make_task(
    category: CategoryConfiguration,
    config: GenerationConfiguration,
    rng: random.Random,
    inputs: tuple[str, ...],
    apply_hypothesis: Callable[[Hypothesis, str], str],
) -> GeneratedTask | None:
    hypothesis = sample_hypothesis(category.rule, config.vocab, rng)
    return select_task(hypothesis, config, rng, inputs, apply_hypothesis)


def _make_instance(
    task: GeneratedTask,
    *,
    category: CategoryConfiguration,
    retained_index: int,
) -> Instance:
    return Instance(
        id=f"c23-{category.label}-{category.seed}-{retained_index:04d}",
        seed=category.seed,
        categories=(category.label,),
        prompt_inputs={
            "demos_block": render_demonstrations(task.demonstrations),
            "query": task.query,
        },
        gold=task.gold,
    )


def default_split_sizes(pool: InstancePool) -> tuple[int, int, int]:
    """Validate a balanced C23 pool and return fixed 10/20/20 allocations."""
    expected_labels = tuple(
        category.label for category in DEFAULT_CONFIG.categories
    )
    if pool.categories != expected_labels:
        raise ValueError(
            "C23 pool categories must be "
            f"{expected_labels!r}, got {pool.categories!r}",
        )
    if any(
        len(instance.categories) != 1
        for instance in pool.instance_group.instances
    ):
        raise ValueError("C23 instances must belong to exactly one category")
    counts = pool.category_counts()
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"C23 pool categories must be balanced, got {counts!r}"
        )
    required = sum(_DEFAULT_SPLIT_PER_CATEGORY)
    available = next(iter(counts.values()))
    if available < required:
        raise ValueError(
            f"C23 split requires {required} instances per category, "
            f"but the pool has {available}",
        )
    category_count = len(expected_labels)
    internal, official, held_out = _DEFAULT_SPLIT_PER_CATEGORY
    return (
        internal * category_count,
        official * category_count,
        held_out * category_count,
    )
