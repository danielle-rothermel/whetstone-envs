"""Import and random-state isolation regressions for C22."""

from __future__ import annotations

import random
import subprocess
import sys
from random import Random

from whetstone_envs.c22 import generate, oracle
from whetstone_envs.c22._vendor.instruction_following_eval import (
    instructions_registry,
)
from whetstone_envs.c22.atoms import EASY_POOL, HARD_POOL


def test_generation_preserves_module_global_random_state() -> None:
    before = random.getstate()
    generate.generate_pool(n_per_stratum=2)
    assert random.getstate() == before


def test_supported_description_builds_preserve_global_random_state() -> None:
    local_rng = Random(42)  # noqa: S311 - deterministic regression fixture
    before = random.getstate()
    for atom in (*EASY_POOL, *HARD_POOL):
        kwargs = atom.derive_kwargs(local_rng)
        instruction = instructions_registry.INSTRUCTION_DICT[
            atom.instruction_id
        ](atom.instruction_id)
        instruction.build_description(**kwargs)
        assert random.getstate() == before


def test_oracle_scoring_preserves_module_global_random_state() -> None:
    pools = (generate.generate_pool(), generate.HARD_PRESET.generate())
    before = random.getstate()
    for pool in pools:
        for instance in pool.instances:
            oracle.score_gold(instance.gold, "candidate response")
    assert random.getstate() == before


def test_import_is_namespaced_and_ignores_fake_top_level_package() -> None:
    program = """
import sys
import types

fake = types.ModuleType("instruction_following_eval")
fake.sentinel = object()
sys.modules["instruction_following_eval"] = fake
before = tuple(sys.path)

from whetstone_envs.c22 import generate, oracle

assert tuple(sys.path) == before
assert sys.modules["instruction_following_eval"] is fake
assert not any(
    name.startswith("instruction_following_eval.")
    for name in sys.modules
)
pool = generate.generate_pool(n_per_stratum=1)
oracle.score_gold(pool.instances[0].gold, "candidate response")
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and program
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
