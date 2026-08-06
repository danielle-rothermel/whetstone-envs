from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from whetstone_envs.c11 import (
    DEFAULT_SPLIT_SIZES,
    C11Stratum,
    build_manifest,
    canonicalize,
    generate_pool,
    generation,
)
from whetstone_envs.instances import public_prompt_identity
from whetstone_envs.manifests import Manifest
from whetstone_envs.scoring import exact_match

_EXPONENT = re.compile(r"\d(?:\.\d+)?[eE][+-]?\d+")
_MANIFEST_PATH = Path(generation.__file__).with_name("manifest.json")


def _incomplete_json_canonicalize(source: str) -> str:
    """Approximate JCS while omitting UTF-16 and ECMAScript edge rules."""

    def coerce_integral_floats(value: object) -> object:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, list):
            return [coerce_integral_floats(item) for item in value]
        if isinstance(value, dict):
            return {
                key: coerce_integral_floats(item)
                for key, item in value.items()
            }
        return value

    value = coerce_integral_floats(json.loads(source))
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_generation_is_deterministic() -> None:
    assert generate_pool(n_per_stratum=8) == generate_pool(n_per_stratum=8)


def test_generation_is_stable_across_hash_seeds() -> None:
    script = """
from whetstone_envs.c11 import build_manifest, generate_pool
print(build_manifest(generate_pool(n_per_stratum=8)).content_hash)
"""
    outputs = []
    for hash_seed in ("1", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(  # noqa: S603 - controlled interpreter
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_default_pool_matches_committed_manifest() -> None:
    pool = generate_pool()
    manifest = Manifest.read(_MANIFEST_PATH)

    assert manifest == build_manifest(pool)
    assert manifest.matches_pool(pool)


def test_manifest_regeneration_is_byte_identical(tmp_path: Path) -> None:
    regenerated = tmp_path / "manifest.json"
    build_manifest(generate_pool()).write(regenerated)

    assert regenerated.read_bytes() == _MANIFEST_PATH.read_bytes()


def test_default_pool_and_split_are_balanced() -> None:
    pool = generate_pool()
    expected_counts = {
        stratum.value: generation.DEFAULT_N_PER_STRATUM
        for stratum in C11Stratum
    }
    assert pool.stratum_counts() == expected_counts

    split = pool.split(*DEFAULT_SPLIT_SIZES)
    expected_per_stratum = (2, 40, 40)
    for group, count in zip(
        (split.internal_eval, split.official, split.held_out),
        expected_per_stratum,
        strict=True,
    ):
        assert len(group) == count * len(C11Stratum)
        assert Counter(instance.strata[0] for instance in group) == {
            stratum.value: count for stratum in C11Stratum
        }


def test_every_instance_is_unique_adversarial_and_oracle_consistent() -> None:
    pool = generate_pool()
    identities = {
        public_prompt_identity(instance) for instance in pool.instances
    }

    assert len(identities) == len(pool)
    for instance in pool.instances:
        assert set(instance.prompt_inputs) == {generation.INPUT_JSON_FIELD}
        input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
        assert input_json != instance.gold
        assert canonicalize(input_json) == instance.gold
        assert exact_match(instance.gold, instance.gold) == 1
        assert exact_match(input_json, instance.gold) == 0


def test_each_stratum_guarantees_its_named_tension() -> None:
    pool = generate_pool(n_per_stratum=12)
    by_stratum = {
        stratum: pool.in_stratum(stratum.value) for stratum in C11Stratum
    }

    for instance in by_stratum[C11Stratum.WHITESPACE]:
        input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
        compact = json.dumps(
            json.loads(input_json),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert compact == instance.gold

    for instance in by_stratum[C11Stratum.KEY_ORDER]:
        input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
        keys = list(json.loads(input_json))
        assert keys != sorted(keys, key=lambda key: key.encode("utf-16be"))

    for instance in by_stratum[C11Stratum.NUMBER]:
        input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
        assert _EXPONENT.search(input_json)

    for stratum in (C11Stratum.UNICODE, C11Stratum.MIXED):
        for instance in by_stratum[stratum]:
            input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
            assert "\\u" in input_json
            assert any(ord(character) > 127 for character in instance.gold)

    for instance in by_stratum[C11Stratum.MIXED]:
        input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
        assert _EXPONENT.search(input_json)
        keys = list(json.loads(input_json))
        assert keys != sorted(keys, key=lambda key: key.encode("utf-16be"))


def test_rfc_specific_strata_reject_an_incomplete_canonicalizer() -> None:
    pool = generate_pool()

    for stratum in (
        C11Stratum.KEY_ORDER,
        C11Stratum.NUMBER,
        C11Stratum.MIXED,
    ):
        for instance in pool.in_stratum(stratum.value):
            input_json = instance.prompt_inputs[generation.INPUT_JSON_FIELD]
            assert _incomplete_json_canonicalize(input_json) != instance.gold


@pytest.mark.parametrize("value", [True, 1.5, "4", None])
def test_generation_size_must_be_an_integer(value: object) -> None:
    with pytest.raises(TypeError, match="n_per_stratum must be an integer"):
        generate_pool(n_per_stratum=value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [0, -1])
def test_generation_size_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="n_per_stratum must be positive"):
        generate_pool(n_per_stratum=value)


@pytest.mark.parametrize("value", [True, 1.5, "4", None])
def test_seed_start_must_be_an_integer(value: object) -> None:
    with pytest.raises(TypeError, match="seed_start must be an integer"):
        generate_pool(seed_start=value)  # ty: ignore[invalid-argument-type]


def test_manifest_rejects_an_empty_pool() -> None:
    from whetstone_envs.pools import TaskPool

    with pytest.raises(ValueError, match="require a nonempty pool"):
        build_manifest(TaskPool(()))
