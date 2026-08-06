from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from importlib.resources import as_file, files
from random import Random

from whetstone_envs.c22._ifeval_adapter import render_constraint_block
from whetstone_envs.c22.constraints import (
    Constraint,
    ConstraintStack,
    EndPhrase,
    ExactWordCount,
    ForbiddenLetter,
    ForbiddenWord,
    HighlightedSections,
    NoComma,
    Placeholders,
    Postscript,
    Quotation,
    RequiredKeyword,
    Title,
)
from whetstone_envs.instances import Instance, make_instance
from whetstone_envs.manifests import Manifest
from whetstone_envs.pools import TaskPool

GENERATOR_VERSION = "c22-1"
PUBLISHED_KEY_MAX = 3757
DEFAULT_SEED_START = 1_000_000
HARD_SEED_START = 2_000_000
_INSTANCES_PER_STRATUM = 20
_MAX_GENERATION_ATTEMPTS = 100

_NONCE_KEYWORDS = (
    "zylthorn",
    "quarnex",
    "vopflim",
    "jaxbryn",
    "wexcorb",
    "plurnyx",
    "gwentar",
    "brimquol",
)
_END_PHRASES = (
    "THUS CONCLUDED",
    "END OF ANSWER",
    "FULLY DONE",
    "SIGNED OFF",
)
_RARE_LETTERS = ("z", "q", "x", "j", "k")
_POSTSCRIPT_MARKERS = ("P.S.", "P.P.S")


@verify(UNIQUE)
class Preset(StrEnum):
    """The two fixed C22 pool designs.

    Do not iterate this enum to build persisted payloads; each preset's
    configuration is declared explicitly below.
    """

    DEFAULT = "default"
    HARD = "hard"


class _Mix(StrEnum):
    EASY = "easy"
    MIXED = "mixed"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class _GenerationConfig:
    preset: Preset
    constraint_counts: tuple[int, ...]
    mixes: tuple[_Mix, ...]
    seed_start: int

    @property
    def generator_version(self) -> str:
        if self.preset is Preset.DEFAULT:
            return GENERATOR_VERSION
        return f"{GENERATOR_VERSION}+hard"


_DEFAULT_CONFIG = _GenerationConfig(
    preset=Preset.DEFAULT,
    constraint_counts=(3, 4, 5),
    mixes=(_Mix.EASY, _Mix.MIXED),
    seed_start=DEFAULT_SEED_START,
)
_HARD_CONFIG = _GenerationConfig(
    preset=Preset.HARD,
    constraint_counts=(3, 6, 8),
    mixes=(_Mix.HARD,),
    seed_start=HARD_SEED_START,
)

type _ConstraintFactory = Callable[[Random], Constraint]


def _required_keyword(rng: Random) -> Constraint:
    return RequiredKeyword(keyword=rng.choice(_NONCE_KEYWORDS))


def _forbidden_word(rng: Random) -> Constraint:
    return ForbiddenWord(word=rng.choice(_NONCE_KEYWORDS))


def _end_phrase(rng: Random) -> Constraint:
    return EndPhrase(phrase=rng.choice(_END_PHRASES))


def _title(_rng: Random) -> Constraint:
    return Title()


def _quotation(_rng: Random) -> Constraint:
    return Quotation()


def _no_comma(_rng: Random) -> Constraint:
    return NoComma()


def _placeholders(rng: Random) -> Constraint:
    return Placeholders(count=rng.randint(1, 3))


def _postscript(rng: Random) -> Constraint:
    return Postscript(marker=rng.choice(_POSTSCRIPT_MARKERS))


def _highlighted_sections(rng: Random) -> Constraint:
    return HighlightedSections(count=rng.randint(1, 3))


def _exact_word_count(rng: Random) -> Constraint:
    return ExactWordCount(count=rng.randint(3, 12))


def _forbidden_letter(rng: Random) -> Constraint:
    return ForbiddenLetter(letter=rng.choice(_RARE_LETTERS))


_EASY_FACTORIES: tuple[_ConstraintFactory, ...] = (
    _required_keyword,
    _end_phrase,
    _title,
    _quotation,
    _no_comma,
    _placeholders,
    _postscript,
    _highlighted_sections,
)
_HARD_FACTORIES: tuple[_ConstraintFactory, ...] = (
    _exact_word_count,
    _forbidden_letter,
    _forbidden_word,
)


def _config_for(preset: Preset) -> _GenerationConfig:
    if not isinstance(preset, Preset):
        msg = "preset must be a C22 Preset"
        raise TypeError(msg)
    if preset is Preset.DEFAULT:
        return _DEFAULT_CONFIG
    return _HARD_CONFIG


def _preset_excludes_pair(
    chosen: list[_ConstraintFactory],
    candidate: _ConstraintFactory,
) -> bool:
    """Keep each preset task to one whole-response framing constraint."""
    return {candidate, *chosen} >= {_title, _quotation}


def _sample_factories(
    rng: Random,
    count: int,
    mix: _Mix,
) -> tuple[_ConstraintFactory, ...]:
    chosen: list[_ConstraintFactory] = []
    if mix is _Mix.HARD:
        hard = list(_HARD_FACTORIES)
        rng.shuffle(hard)
        chosen.extend(hard)
        candidates = list(_EASY_FACTORIES)
    elif mix is _Mix.MIXED:
        chosen.append(rng.choice(_HARD_FACTORIES))
        candidates = [*_EASY_FACTORIES, *_HARD_FACTORIES]
    else:
        candidates = list(_EASY_FACTORIES)

    rng.shuffle(candidates)
    for candidate in candidates:
        if len(chosen) >= count:
            break
        if candidate in chosen or _preset_excludes_pair(chosen, candidate):
            continue
        chosen.append(candidate)
    if len(chosen) != count:
        msg = f"C22 {mix.value} mix cannot produce {count} constraints"
        raise ValueError(msg)
    return tuple(chosen)


def _make_instance(
    *,
    seed: int,
    count: int,
    mix: _Mix,
    used_blocks: set[str],
) -> Instance:
    rng = Random(seed)
    last_error: ValueError | None = None
    for _attempt in range(_MAX_GENERATION_ATTEMPTS):
        factories = _sample_factories(rng, count, mix)
        constraints = tuple(factory(rng) for factory in factories)
        try:
            stack = ConstraintStack(constraints=constraints)
        except ValueError as error:
            last_error = error
            continue
        block = render_constraint_block(stack.constraints)
        if block in used_blocks:
            continue
        used_blocks.add(block)
        return make_instance(
            id=f"c22-{seed}",
            seed=seed,
            strata=f"n{count}_{mix.value}",
            prompt_inputs={"constraints": block},
            gold=stack.to_gold(),
        )
    detail = (
        "duplicate public prompt" if last_error is None else str(last_error)
    )
    msg = (
        f"could not generate C22 seed={seed}, count={count}, "
        f"mix={mix.value!r} after {_MAX_GENERATION_ATTEMPTS} attempts: "
        f"{detail}"
    )
    raise RuntimeError(msg)


def _generate_pool(
    preset: Preset,
    *,
    instances_per_stratum: int,
) -> TaskPool:
    if type(instances_per_stratum) is not int or instances_per_stratum <= 0:
        msg = "instances_per_stratum must be a positive integer"
        raise ValueError(msg)
    config = _config_for(preset)
    if config.seed_start <= PUBLISHED_KEY_MAX:
        msg = "C22 seed range overlaps published IFEval keys"
        raise ValueError(msg)
    instances: list[Instance] = []
    used_blocks: set[str] = set()
    seed = config.seed_start
    for count in config.constraint_counts:
        for mix in config.mixes:
            for _ in range(instances_per_stratum):
                instances.append(
                    _make_instance(
                        seed=seed,
                        count=count,
                        mix=mix,
                        used_blocks=used_blocks,
                    )
                )
                seed += 1
    return TaskPool(instances)


def generate_pool(preset: Preset = Preset.DEFAULT) -> TaskPool:
    """Generate one of the two fixed, reproducible C22 pools."""
    return _generate_pool(
        preset,
        instances_per_stratum=_INSTANCES_PER_STRATUM,
    )


def load_manifest(preset: Preset = Preset.DEFAULT) -> Manifest:
    _config_for(preset)
    resource = files("whetstone_envs.c22.preset_manifests").joinpath(
        f"{preset.value}.json"
    )
    with as_file(resource) as path:
        return Manifest.read(path)


__all__ = [
    "Preset",
    "generate_pool",
    "load_manifest",
]
