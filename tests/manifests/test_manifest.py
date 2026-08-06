"""Tests for the diffable pool manifest."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.manifests import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    content_hash,
)
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from whetstone_envs.instances import Instance


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
        "stratum_counts": {"easy": 2, "hard": 1},
        "content_hash": "a" * 64,
    }


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


def test_manifest_detaches_and_freezes_stratum_counts() -> None:
    pool = TaskPool(
        [make_instance(id="easy-0", seed=0, strata="easy")],
    )
    source_counts = pool.stratum_counts()
    manifest = Manifest(
        generator_version="g",
        seed_range=(0, 1),
        stratum_counts=source_counts,
        content_hash=content_hash(pool),
    )
    serialized = manifest.to_json()

    source_counts["easy"] = 2

    assert manifest.to_json() == serialized
    assert manifest.matches_pool(pool) is True
    with pytest.raises(TypeError):
        manifest.stratum_counts["easy"] = 2  # ty: ignore[invalid-assignment]


def test_manifest_round_trips_through_json(
    synthetic_instance: Callable[..., Instance],
) -> None:
    manifest = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(1000, 1004),
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
        seed_range=(1000, 1004),
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
        seed_range=(1000, 1004),
    )
    # A deterministic regeneration still matches the frozen manifest.
    assert frozen.matches_pool(_build_pool(synthetic_instance)) is True


def test_detects_drifted_pool(
    synthetic_instance: Callable[..., Instance],
) -> None:
    frozen = Manifest.from_pool(
        _build_pool(synthetic_instance),
        generator_version="g",
        seed_range=(1000, 1004),
    )
    drifted = TaskPool(
        [synthetic_instance(i, "easy") for i in range(2)]
        + [synthetic_instance(2, "hard")]
        + [synthetic_instance(3, "hard", gold="changed")]
    )
    assert drifted.stratum_counts() == frozen.stratum_counts
    assert frozen.matches_pool(drifted) is False


@pytest.mark.parametrize(
    "seed_range",
    [(1001, 1004), (1000, 1003)],
    ids=["below-start", "at-exclusive-end"],
)
def test_from_pool_rejects_retained_seed_outside_range(
    synthetic_instance: Callable[..., Instance],
    seed_range: tuple[int, int],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*retained)(?=.*seed)(?=.*range)",
    ):
        Manifest.from_pool(
            _build_pool(synthetic_instance),
            generator_version="g",
            seed_range=seed_range,
        )


def test_matches_pool_rejects_out_of_range_retained_seed(
    synthetic_instance: Callable[..., Instance],
) -> None:
    pool = _build_pool(synthetic_instance)
    manifest = Manifest(
        generator_version="g",
        seed_range=(1000, 1003),
        stratum_counts=pool.stratum_counts(),
        content_hash=content_hash(pool),
    )

    assert manifest.matches_pool(pool) is False


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
        ("schema_version", True),
        ("seed_range", [10.5, 20]),
        ("seed_range", [10, False]),
        ("stratum_counts", {"easy": 2.5, "hard": 1}),
        ("stratum_counts", {"easy": 2, "hard": True}),
    ],
    ids=[
        "fractional-schema-version",
        "bool-schema-version",
        "fractional-seed",
        "bool-seed",
        "fractional-count",
        "bool-count",
    ],
)
def test_from_dict_rejects_non_integer_fields(
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


def test_from_dict_rejects_equal_seed_range_endpoints(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["seed_range"] = [10, 10]
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*seed_range)(?=.*start)(?=.*end)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_direct_manifest_rejects_equal_seed_range_endpoints() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*seed_range)(?=.*start)(?=.*end)",
    ):
        Manifest(
            generator_version="g",
            seed_range=(10, 10),
            stratum_counts={"easy": 1},
            content_hash="a" * 64,
        )


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
    valid_manifest_payload["stratum_counts"] = {"easy": -1, "hard": 1}
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*stratum_counts)(?=.*non-negative)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_from_dict_rejects_zero_valued_stratum(
    valid_manifest_payload: dict[str, object],
) -> None:
    valid_manifest_payload["stratum_counts"] = {"easy": 2, "hard": 0}
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*stratum_counts)(?=.*positive)",
    ):
        Manifest.from_dict(valid_manifest_payload)


def test_direct_manifest_rejects_zero_valued_stratum() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*stratum_counts)(?=.*positive)",
    ):
        Manifest(
            generator_version="g",
            seed_range=(10, 20),
            stratum_counts={"easy": 2, "hard": 0},
            content_hash="a" * 64,
        )


@pytest.mark.parametrize(
    "stratum_counts",
    [{}, {"easy": 0, "hard": 0}],
    ids=["empty", "zero-total"],
)
def test_from_dict_rejects_zero_task_manifest(
    valid_manifest_payload: dict[str, object],
    stratum_counts: dict[str, int],
) -> None:
    valid_manifest_payload["stratum_counts"] = stratum_counts
    with pytest.raises(
        ValueError,
        match=r"(?i)(?=.*stratum_counts)(?=.*positive)",
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
