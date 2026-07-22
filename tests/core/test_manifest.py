"""Tests for the diffable pool manifest."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    content_hash,
)
from whetstone_envs.core.pool import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from whetstone_envs.core.instance import Instance


def _build_pool(factory: Callable[..., Instance]) -> TaskPool:
    instances = [factory(i, "easy") for i in range(2)] + [
        factory(i, "hard") for i in range(2, 4)
    ]
    return TaskPool(instances)


def test_content_hash_is_deterministic(
    synthetic_instance: Callable[..., Instance],
) -> None:
    # Regenerating the pool twice must yield a byte-identical hash.
    assert content_hash(_build_pool(synthetic_instance)) == content_hash(
        _build_pool(synthetic_instance)
    )


def test_content_hash_independent_of_prompt_input_order() -> None:
    a = make_instance(
        id="t", seed=1, strata="s", prompt_inputs={"a": "1", "b": "2"}
    )
    b = make_instance(
        id="t", seed=1, strata="s", prompt_inputs={"b": "2", "a": "1"}
    )
    assert content_hash(TaskPool([a])) == content_hash(TaskPool([b]))


def test_content_hash_changes_with_gold() -> None:
    a = TaskPool([make_instance(id="t", seed=1, strata="s", gold="A")])
    b = TaskPool([make_instance(id="t", seed=1, strata="s", gold="B")])
    assert content_hash(a) != content_hash(b)


def test_from_pool_records_counts_and_hash(
    synthetic_instance: Callable[..., Instance],
) -> None:
    pool = _build_pool(synthetic_instance)
    manifest = Manifest.from_pool(
        pool, generator_version="gen@1.0", seed_range=(1000, 1004)
    )
    assert manifest.stratum_counts == {"easy": 2, "hard": 2}
    assert manifest.content_hash == content_hash(pool)
    assert manifest.seed_range == (1000, 1004)
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION


def test_manifest_round_trips_through_json(
    synthetic_instance: Callable[..., Instance],
) -> None:
    manifest = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(0, 10),
    )
    restored = Manifest.from_dict(json.loads(manifest.to_json()))
    assert restored == manifest


def test_manifest_write_read_round_trip(
    synthetic_instance: Callable[..., Instance],
    tmp_path: Path,
) -> None:
    manifest = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(0, 10),
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert Manifest.read(path) == manifest


def test_matches_regenerated_pool(
    synthetic_instance: Callable[..., Instance],
) -> None:
    frozen = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(0, 10),
    )
    # A deterministic regeneration still matches the frozen manifest.
    assert frozen.matches_pool(_build_pool(synthetic_instance)) is True


def test_detects_drifted_pool(
    synthetic_instance: Callable[..., Instance],
) -> None:
    frozen = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(0, 10),
    )
    drifted = TaskPool(
        [synthetic_instance(i, "easy") for i in range(2)]
        + [synthetic_instance(i, "hard") for i in range(2, 4)]
        + [synthetic_instance(9, "hard")]
    )
    # An extra hard instance changes both the count and the hash.
    assert frozen.matches_pool(drifted) is False


def test_from_dict_rejects_bad_seed_range() -> None:
    with pytest.raises(TypeError, match="two-element"):
        Manifest.from_dict(
            {
                "generator_version": "g",
                "seed_range": [1, 2, 3],
                "stratum_counts": {"easy": 1},
                "content_hash": "abc",
            }
        )


def test_from_dict_rejects_non_numeric_count() -> None:
    with pytest.raises(TypeError, match="integer manifest field"):
        Manifest.from_dict(
            {
                "generator_version": "g",
                "seed_range": [1, 2],
                "stratum_counts": {"easy": "lots"},
                "content_hash": "abc",
            }
        )
