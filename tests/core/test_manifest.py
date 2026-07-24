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


@pytest.fixture
def valid_manifest_payload() -> dict[str, object]:
    """Return one known-valid payload for single-field mutation tests."""
    return {
        "schema_version": 1,
        "generator_version": "generator@1.0",
        "seed_range": [10, 20],
        "stratum_counts": {"easy": 2, "hard": 0},
        "content_hash": "a" * 64,
    }


def test_content_hash_matches_pinned_vector() -> None:
    inst = make_instance(
        id="vector-1",
        seed=7,
        strata=("easy", "short"),
        prompt_inputs={"b": "2", "a": "1"},
        gold="yes",
    )
    assert content_hash(TaskPool([inst])) == (
        "765870a223a64fdd2d4cd0f00351b8d683a864e10b78b5906da05e3058988353"
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
        + [synthetic_instance(2, "hard")]
        + [synthetic_instance(3, "hard", gold="changed")]
    )
    assert drifted.stratum_counts() == frozen.stratum_counts
    assert frozen.matches_pool(drifted) is False


def test_from_dict_accepts_known_valid_payload(
    valid_manifest_payload: dict[str, object],
) -> None:
    assert Manifest.from_dict(valid_manifest_payload).to_dict() == (
        valid_manifest_payload
    )


def test_from_dict_rejects_missing_schema_version(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload.pop("schema_version")
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*schema_version)(?=.*required)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_unsupported_schema_version(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["schema_version"] = 2
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*schema_version)(?=.*unsupported)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_non_string_generator_version(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["generator_version"] = 123
    with pytest.raises(
        TypeError,
        match=r"(?i)(?=.*generator_version)(?=.*string)",
    ):
        Manifest.from_dict(valid_manifest_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1.5),
        ("seed_range", [10.5, 20]),
        ("stratum_counts", {"easy": 2.5, "hard": 0}),
    ],
    ids=["schema-version", "seed", "count"],
)
def test_from_dict_rejects_fractional_integer_fields(
    valid_manifest_payload: dict[str, object],
    field: str,
    value: object,
) -> None:
    valid_manifest_payload[field] = value
    with pytest.raises(
        TypeError,
        match=rf"(?i)(?=.*{field})(?=.*integer)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_bad_seed_range_shape(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["seed_range"] = [10, 20, 30]
    with pytest.raises(
        TypeError,
        match=r"(?i)(?=.*seed_range)(?=.*two-element)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_reversed_seed_range(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["seed_range"] = [20, 10]
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*seed_range)(?=.*ordered)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_non_numeric_count(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["stratum_counts"] = {
        "easy": "lots",
        "hard": 0,
    }
    with pytest.raises(
        TypeError,
        match=r"(?i)(?=.*stratum_counts)(?=.*integer)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_negative_count(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["stratum_counts"] = {"easy": -1, "hard": 0}
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*stratum_counts)(?=.*non-negative)",
    ):
        Manifest.from_dict(valid_manifest_payload)


@pytest.mark.parametrize(
    "bad_hash",
    [
        "a" * 63,
        "g" * 64,
        "A" * 64,
    ],
    ids=["wrong-length", "non-hex", "uppercase"],
)
def test_from_dict_rejects_malformed_content_hash(
    valid_manifest_payload: dict[str, object],
    bad_hash: str,
) -> None:
    valid_manifest_payload["content_hash"] = bad_hash
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*content_hash)(?=.*sha-256)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_read_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(
        TypeError,
        match=r"(?i)(?=.*manifest)(?=.*object)",
    ):
        Manifest.read(path)


def test_from_dict_rejects_unknown_fields(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["unexpected"] = "value"
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*unknown)(?=.*unexpected)",
    ):
        Manifest.from_dict(valid_manifest_payload)
