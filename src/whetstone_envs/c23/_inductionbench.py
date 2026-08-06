# Modified from InductionBench e0b8392; see attribution/PROVENANCE.md.

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from whetstone_envs.c23._domain import (
    Demonstration,
    Hypothesis,
    RuleConfiguration,
)
from whetstone_envs.c23._transducers import apply_reference

if TYPE_CHECKING:
    import random


def characteristic_inputs(
    vocab: tuple[str, ...],
    maximum_query_length: int,
) -> tuple[str, ...]:
    """Return a bounded canonical characteristic and query sample.

    The complete layers through length four cover every supported context and
    short interaction. Longer layers contribute deterministic context-bearing
    representatives rather than materializing the exponential Cartesian
    product through the maximum query length.
    """
    complete = tuple(
        "".join(symbols)
        for length in range(1, min(4, maximum_query_length) + 1)
        for symbols in itertools.product(vocab, repeat=length)
    )
    contexts = tuple(
        "".join(symbols)
        for length in (2, 3)
        for symbols in itertools.product(vocab, repeat=length)
    )
    extended = tuple(
        vocab[0] * (length - len(context)) + context
        for length in range(5, maximum_query_length + 1)
        for context in contexts
    )
    return tuple(dict.fromkeys((*complete, *extended)))


def sample_hypothesis(
    configuration: RuleConfiguration,
    vocab: tuple[str, ...],
    rng: random.Random,
) -> Hypothesis:
    contexts = tuple(
        "".join(symbols)
        for symbols in itertools.product(
            vocab,
            repeat=configuration.context_length,
        )
    )
    context = rng.choice(contexts)
    replacements = (
        *(symbol for symbol in vocab if symbol != context[-1]),
        "",
    )
    return Hypothesis(
        configuration=configuration,
        context=context,
        replacement=rng.choice(replacements),
    )


def examples_for(
    hypothesis: Hypothesis,
    inputs: tuple[str, ...],
) -> tuple[Demonstration, ...]:
    return tuple(
        Demonstration(value, apply_reference(hypothesis, value))
        for value in inputs
    )
