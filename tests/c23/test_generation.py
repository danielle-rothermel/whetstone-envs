from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from whetstone_envs.c23 import _pool as pool_module
from whetstone_envs.c23 import generate_pool
from whetstone_envs.c23._domain import (
    Demonstration,
    GeneratedTask,
    GenerationConfiguration,
    Hypothesis,
    RuleConfiguration,
    RuleFamily,
)

VOCAB = ("a", "b", "c", "d")


def _apply(rule: Hypothesis, value: str) -> str:
    family = rule.configuration.family
    if family is RuleFamily.ISL:
        output = ""
        for index, symbol in enumerate(value):
            context = value[: index + 1][-rule.configuration.context_length :]
            output += rule.replacement if context == rule.context else symbol
        return output
    if family is RuleFamily.R_OSL:
        return _apply(
            Hypothesis(
                RuleConfiguration(RuleFamily.L_OSL, 2),
                rule.context,
                rule.replacement,
            ),
            value[::-1],
        )[::-1]
    output = ""
    for symbol in value:
        output += symbol
        if output[-rule.configuration.context_length :] == rule.context:
            output = output[:-1] + rule.replacement
    return output


def _independent_hypotheses() -> tuple[Hypothesis, ...]:
    configurations = (
        RuleConfiguration(RuleFamily.ISL, 2),
        RuleConfiguration(RuleFamily.L_OSL, 2),
        RuleConfiguration(RuleFamily.R_OSL, 2),
        RuleConfiguration(RuleFamily.ISL, 3),
    )
    return tuple(
        Hypothesis(configuration, context, replacement)
        for configuration in configurations
        for symbols in itertools.product(
            VOCAB,
            repeat=configuration.context_length,
        )
        for context in ("".join(symbols),)
        for replacement in (
            *(symbol for symbol in VOCAB if symbol != context[-1]),
            "",
        )
    )


def _parse_demos(block: str) -> tuple[Demonstration, ...]:
    return tuple(
        Demonstration(*line.split(" -> ", maxsplit=1))
        for line in block.splitlines()
    )


def _pool_projection(n_per_stratum: int) -> str:
    pool = generate_pool(n_per_stratum=n_per_stratum)
    payload = [
        {
            "id": instance.id,
            "seed": instance.seed,
            "strata": instance.strata,
            "prompt_inputs": dict(instance.prompt_inputs),
            "gold": instance.gold,
        }
        for instance in pool.instances
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode(),
    ).hexdigest()


def test_small_pool_is_determinate_nontrivial_and_immutable() -> None:
    pool = generate_pool(n_per_stratum=1)
    hypotheses = _independent_hypotheses()

    for instance in pool.instances:
        demos = _parse_demos(instance.prompt_inputs["demos_block"])
        query = instance.prompt_inputs["query"]
        assert len(demos) == 6
        assert query not in {example.input for example in demos}
        assert 2 <= len(query) <= 8
        assert instance.gold != query
        consistent = tuple(
            rule
            for rule in hypotheses
            if all(
                _apply(rule, example.input) == example.output
                for example in demos
            )
        )
        assert {_apply(rule, query) for rule in consistent} == {instance.gold}

    with pytest.raises(FrozenInstanceError):
        pool_module.DEFAULT_CONFIG.__setattr__("vocab", ("x",))


def test_generation_is_repeatable_without_global_rng_mutation() -> None:
    random.seed(982_451)
    before = random.getstate()
    first = _pool_projection(2)
    after = random.getstate()

    assert first == _pool_projection(2)
    assert after == before


def test_generation_is_stable_across_process_hash_seeds() -> None:
    script = """
import hashlib, json
from whetstone_envs.c23 import generate_pool
pool = generate_pool(n_per_stratum=2)
payload = [
    (i.id, i.seed, i.strata, dict(i.prompt_inputs), i.gold)
    for i in pool.instances
]
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
"""
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.run(  # noqa: S603 - fixed interpreter and test script
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip(),
        )
    assert outputs[0] == outputs[1]


def test_duplicate_public_identity_retries_then_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = Hypothesis(
        RuleConfiguration(RuleFamily.ISL, 2),
        "ab",
        "c",
    )
    task = GeneratedTask(
        rule,
        (
            Demonstration("a", "a"),
            Demonstration("b", "b"),
            Demonstration("c", "c"),
            Demonstration("d", "d"),
            Demonstration("ab", "ac"),
            Demonstration("ba", "ba"),
        ),
        "abab",
        "acac",
    )
    calls = 0

    def duplicate_task(*_args: object) -> GeneratedTask:
        nonlocal calls
        calls += 1
        return task

    monkeypatch.setattr(pool_module, "_make_task", duplicate_task)
    config = GenerationConfiguration(
        vocab=VOCAB,
        strata=(pool_module.DEFAULT_CONFIG.strata[0],),
        demonstrations_per_instance=6,
        maximum_query_length=8,
        attempts_per_instance=1,
    )

    with pytest.raises(
        pool_module.GenerationExhaustedError, match="retained 1"
    ):
        pool_module._generate_pool(config, n_per_stratum=2)
    assert calls == 2
