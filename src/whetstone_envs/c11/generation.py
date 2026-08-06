"""Deterministic generation for RFC 8785 canonicalization tasks."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from enum import UNIQUE, StrEnum, verify
from types import MappingProxyType

from whetstone_envs.c11.oracle import canonicalize
from whetstone_envs.instances import (
    Instance,
    make_instance,
    public_prompt_identity,
)
from whetstone_envs.manifests import Manifest
from whetstone_envs.pools import TaskPool

GENERATOR_VERSION = "c11-generation-1"
INPUT_JSON_FIELD = "input_json"
DEFAULT_SEED_START = 1_000_000
DEFAULT_N_PER_STRATUM = 82
DEFAULT_SPLIT_SIZES: tuple[int, int, int] = (10, 200, 200)

_MAX_ATTEMPTS_PER_INSTANCE = 100
_MESSY_SEPARATORS = (", ", ": ")
_ASCII_KEYS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "omega",
)
_ASCII_WORDS = (
    "red",
    "green",
    "blue",
    "amber",
    "cyan",
    "coral",
    "ivory",
    "slate",
)
_UNICODE_STRINGS = (
    'quote " inside-é',
    "back\\slash-π",
    "tab\tsep-ü",
    "new\nline-雪",
    "carriage\rreturn-é",
    "ctrl\x01char-λ",
    "unicode-é-accent",
    "emoji-\U0001f600-astral",
    "mix\tü\\end",
)
_NONCANONICAL_NUMBER_LEXEMES = (
    "1e2",
    "1E+2",
    "1.00e+2",
    "100e-2",
    "15e+1",
)


@verify(UNIQUE)
class C11Stratum(StrEnum):
    """The independently reported C11 task strata."""

    WHITESPACE = "c11/whitespace"
    KEY_ORDER = "c11/key-order"
    NUMBER = "c11/number"
    UNICODE = "c11/unicode"
    MIXED = "c11/mixed"


# Keep pool order explicit. Adding an enum member must not silently change
# generated identities by changing persisted payload order.
_STRATA = (
    C11Stratum.WHITESPACE,
    C11Stratum.KEY_ORDER,
    C11Stratum.NUMBER,
    C11Stratum.UNICODE,
    C11Stratum.MIXED,
)

_Builder = Callable[[random.Random], str]


def _messy_dumps(value: object, *, escape_non_ascii: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=escape_non_ascii,
        separators=_MESSY_SEPARATORS,
    )


def _force_unsorted(keys: list[str], rng: random.Random) -> list[str]:
    rng.shuffle(keys)
    if keys == sorted(keys):
        keys[0], keys[1] = keys[1], keys[0]
    return keys


def _build_whitespace(rng: random.Random) -> str:
    keys = sorted(rng.sample(_ASCII_KEYS, rng.randint(2, 4)))
    return _messy_dumps({key: rng.randint(0, 9) for key in keys})


def _build_key_order(rng: random.Random) -> str:
    keys = _force_unsorted(
        rng.sample(_ASCII_KEYS, rng.randint(4, 7)),
        rng,
    )
    return _messy_dumps({key: rng.choice(_ASCII_WORDS) for key in keys})


def _build_number(rng: random.Random) -> str:
    keys = sorted(rng.sample(_ASCII_KEYS, rng.randint(2, 4)))
    exponent_key = rng.choice(keys)
    other_lexemes = (
        "1.0",
        "100.0",
        "-0.0",
        *_NONCANONICAL_NUMBER_LEXEMES,
    )
    members = (
        f"{json.dumps(key)}: "
        + (
            rng.choice(_NONCANONICAL_NUMBER_LEXEMES)
            if key == exponent_key
            else rng.choice(other_lexemes)
        )
        for key in keys
    )
    return "{" + ", ".join(members) + "}"


def _build_unicode(rng: random.Random) -> str:
    keys = sorted(rng.sample(_ASCII_KEYS, rng.randint(2, 4)))
    value = {key: rng.choice(_UNICODE_STRINGS) for key in keys}
    return _messy_dumps(value, escape_non_ascii=True)


def _build_mixed(rng: random.Random) -> str:
    outer_keys = _force_unsorted(rng.sample(_ASCII_KEYS, 3), rng)
    nested_keys = sorted(rng.sample(_ASCII_KEYS, 2))
    nested = {key: rng.choice(_UNICODE_STRINGS) for key in nested_keys}
    values = {
        outer_keys[0]: json.dumps(
            rng.choice(_UNICODE_STRINGS),
            ensure_ascii=True,
        ),
        outer_keys[1]: (
            "["
            + ", ".join(
                (
                    rng.choice(_NONCANONICAL_NUMBER_LEXEMES),
                    "-0.0",
                    str(rng.randint(1, 1000)),
                )
            )
            + "]"
        ),
        outer_keys[2]: _messy_dumps(nested, escape_non_ascii=True),
    }
    members = (f"{json.dumps(key)}: {values[key]}" for key in outer_keys)
    return "{" + ", ".join(members) + "}"


_BUILDERS: Mapping[C11Stratum, _Builder] = MappingProxyType(
    {
        C11Stratum.WHITESPACE: _build_whitespace,
        C11Stratum.KEY_ORDER: _build_key_order,
        C11Stratum.NUMBER: _build_number,
        C11Stratum.UNICODE: _build_unicode,
        C11Stratum.MIXED: _build_mixed,
    }
)


def _build_instance(stratum: C11Stratum, seed: int) -> Instance:
    rng = random.Random(seed)  # noqa: S311 - deterministic task generation
    input_json = _BUILDERS[stratum](rng)
    gold = canonicalize(input_json)
    if input_json == gold:
        msg = f"{stratum.value} builder produced a canonical input"
        raise RuntimeError(msg)
    return make_instance(
        id=f"c11-{stratum.name.lower()}-{seed}",
        seed=seed,
        strata=stratum.value,
        prompt_inputs={INPUT_JSON_FIELD: input_json},
        gold=gold,
    )


def _validate_generation_inputs(
    n_per_stratum: object,
    seed_start: object,
) -> tuple[int, int]:
    if type(n_per_stratum) is not int:
        msg = "n_per_stratum must be an integer"
        raise TypeError(msg)
    if n_per_stratum <= 0:
        msg = "n_per_stratum must be positive"
        raise ValueError(msg)
    if type(seed_start) is not int:
        msg = "seed_start must be an integer"
        raise TypeError(msg)
    return n_per_stratum, seed_start


def generate_pool(
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    seed_start: int = DEFAULT_SEED_START,
) -> TaskPool:
    """Generate a deterministic, balanced C11 task pool."""
    n_per_stratum, seed_start = _validate_generation_inputs(
        n_per_stratum,
        seed_start,
    )
    next_seed = seed_start
    seen_identities: set[tuple[tuple[str, str], ...]] = set()
    by_stratum: list[list[Instance]] = []

    for stratum in _STRATA:
        instances: list[Instance] = []
        attempts = 0
        while len(instances) < n_per_stratum:
            if attempts >= n_per_stratum * _MAX_ATTEMPTS_PER_INSTANCE:
                msg = (
                    f"exhausted seed budget for {stratum.value}: retained "
                    f"{len(instances)} of {n_per_stratum} instances"
                )
                raise RuntimeError(msg)
            instance = _build_instance(stratum, next_seed)
            next_seed += 1
            attempts += 1
            identity = public_prompt_identity(instance)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            instances.append(instance)
        by_stratum.append(instances)

    ordered = (
        block[index] for index in range(n_per_stratum) for block in by_stratum
    )
    return TaskPool(ordered)


def build_manifest(pool: TaskPool) -> Manifest:
    """Describe the retained C11 pool with the shared manifest contract."""
    if not pool.instances:
        msg = "C11 manifests require a nonempty pool"
        raise ValueError(msg)
    seeds = tuple(instance.seed for instance in pool.instances)
    return Manifest.from_pool(
        pool,
        generator_version=GENERATOR_VERSION,
        seed_range=(min(seeds), max(seeds) + 1),
    )
