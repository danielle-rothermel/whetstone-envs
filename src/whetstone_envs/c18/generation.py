from __future__ import annotations

from whetstone_envs.c18 import oracle, upstream
from whetstone_envs.c18.config import (
    DEFAULT_CONFIG,
    DistractorMode,
    GenerationConfig,
)
from whetstone_envs.instances import Instance, InstanceGroup, InstancePool
from whetstone_envs.manifests import Manifest


def _validated_count(
    config: GenerationConfig,
    n_per_category: int | None,
) -> int:
    count = config.n_per_category if n_per_category is None else n_per_category
    if type(count) is not int:
        msg = "n_per_category must be an int"
        raise TypeError(msg)
    if count <= 0:
        msg = f"n_per_category must be positive, got {count}"
        raise ValueError(msg)
    return count


def _build_category(
    *,
    hops: int,
    distractors: DistractorMode,
    seed: int,
    count: int,
) -> tuple[Instance, ...]:
    raw_instances = upstream.generate_raw(
        hops=hops,
        seed=seed,
        num_trials=count,
        distractors=distractors,
    )
    if len(raw_instances) != count:
        msg = (
            f"C18 D{hops} seed {seed} generated {len(raw_instances)} "
            f"instances, expected {count}"
        )
        raise AssertionError(msg)

    instances: list[Instance] = []
    for index, raw in enumerate(raw_instances):
        derived = oracle.entailment_label(raw.question, raw.query)
        if derived != raw.answer:
            msg = (
                f"C18 oracle disagreement at D{hops} seed {seed} "
                f"instance {index}: upstream={raw.answer!r}, "
                f"oracle={derived!r}"
            )
            raise AssertionError(msg)
        instances.append(
            Instance(
                id=f"c18-D{hops}-{seed}-{index:04d}",
                seed=seed,
                categories=(f"D{hops}",),
                prompt_inputs={
                    "question": raw.question,
                    "query": raw.query,
                },
                gold=raw.answer,
            )
        )
    return tuple(instances)


def generate_pool(
    config: GenerationConfig = DEFAULT_CONFIG,
    *,
    n_per_category: int | None = None,
) -> InstancePool:
    """Generate a deterministic C18 pool for ``config``.

    One fresh seed is consumed per configured depth. Instances use
    depth-interleaved order because pool order participates in the persisted
    dataset identity.
    """
    count = _validated_count(config, n_per_category)
    blocks = tuple(
        _build_category(
            hops=category.hops,
            distractors=category.distractors,
            seed=config.seed_start + index,
            count=count,
        )
        for index, category in enumerate(config.categories)
    )
    return InstancePool(
        instance_group=InstanceGroup(
            name="c18",
            instances=tuple(
                block[row] for row in range(count) for block in blocks
            ),
        ),
    )


def build_manifest(
    pool: InstancePool,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> Manifest:
    """Build the canonical retained-pool manifest for ``config``."""
    return Manifest.from_pool(
        pool,
        generator_version=config.generator_version,
        seed_range=config.seed_range,
    )


def default_split_sizes(
    pool: InstancePool,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> tuple[int, int, int]:
    """Return split sizes scaled to a generated pool's uniform categories."""
    expected_labels = tuple(category.label for category in config.categories)
    counts = pool.category_counts()
    if tuple(counts) != expected_labels:
        msg = (
            f"C18 pool categories {tuple(counts)!r} do not match "
            f"configuration {expected_labels!r}"
        )
        raise ValueError(msg)
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        msg = f"C18 split sizing requires uniform categories, got {counts!r}"
        raise ValueError(msg)
    per_category = unique_counts.pop()
    internal, official, held_out = config.split.scale(per_category)
    category_count = len(config.categories)
    return (
        internal * category_count,
        official * category_count,
        held_out * category_count,
    )


__all__ = [
    "build_manifest",
    "default_split_sizes",
    "generate_pool",
]
