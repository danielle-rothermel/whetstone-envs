"""Runtime-binding checks for c19's Minigrid layer.

Confirms the properties the candidate page marks run-verified: Minigrid is
seed-threaded (same seed -> byte-identical grid; different seed -> varies),
and the live object-model walk agrees with the independent ASCII-only
oracle across every stratum (the construction-time cross-check, verified
here on the rollout layer directly).
"""

from __future__ import annotations

from whetstone_envs.c19 import envs, oracle

_SEED = 1_000_000


def test_same_seed_reproduces_byte_identical_grid() -> None:
    # Minigrid threads the seed through reset -> self.np_random.
    a = envs.rollout("Empty-Random", "small", _SEED, command_length=6)
    b = envs.rollout("Empty-Random", "small", _SEED, command_length=6)
    assert a.grid_ascii == b.grid_ascii
    assert a.command == b.command
    assert a.facts == b.facts


def test_different_seed_varies_the_grid() -> None:
    a = envs.rollout("Empty-Random", "small", _SEED, command_length=6)
    c = envs.rollout("Empty-Random", "small", _SEED + 1, command_length=6)
    # At least one of layout or command differs across seeds.
    assert (a.grid_ascii, a.command) != (c.grid_ascii, c.command)


def test_live_walk_agrees_with_independent_oracle_all_strata() -> None:
    # The gold-agreement assertion the generator makes, verified directly on
    # the rollout for a small seed span across every env/size/fact.
    for env_id in envs.ENV_IDS:
        for size in envs.SIZE_LEVELS:
            for seed in range(_SEED, _SEED + 5):
                roll = envs.rollout(env_id, size, seed, command_length=8)
                for fact in envs.applicable_fact_types(env_id):
                    live = roll.facts[fact]
                    derived = oracle.derive_fact(
                        roll.grid_ascii,
                        roll.command,
                        fact,
                    )
                    assert derived == live, (env_id, size, seed, fact)


def test_carrying_applicable_only_to_fetch() -> None:
    assert "carrying" in envs.applicable_fact_types("Fetch")
    for env_id in ("SimpleCrossing", "FourRooms", "Empty-Random"):
        assert "carrying" not in envs.applicable_fact_types(env_id)


def test_strata_labels_cover_the_default_crossing() -> None:
    labels = envs.strata_labels()
    # 4 envs x 2 sizes x 3 shared facts = 24, plus Fetch-only carrying x2 = 26.
    assert len(labels) == 26
    assert len(set(labels)) == 26


def test_command_uses_only_vanilla_actions() -> None:
    roll = envs.rollout("Fetch", "small", _SEED, command_length=12)
    assert set(roll.command) <= oracle.VALID_COMMANDS


def test_grid_ascii_is_the_initial_state_not_post_walk() -> None:
    # The captured grid is the pre-command layout: the oracle re-walks it
    # from scratch and reproduces the live post-walk facts, which would be
    # impossible if the grid were already the post-walk state.
    roll = envs.rollout("Fetch", "small", _SEED, command_length=10)
    assert oracle.derive_fact(
        roll.grid_ascii,
        roll.command,
        "coordinate",
    ) == roll.facts["coordinate"]


def test_pprint_grid_uses_two_characters_for_every_cell() -> None:
    for env_id in envs.ENV_IDS:
        for size in envs.SIZE_LEVELS:
            grid = envs.rollout(
                env_id,
                size,
                _SEED,
                command_length=0,
            ).grid_ascii
            rows = grid.splitlines()
            assert all(len(row) == 2 * len(rows) for row in rows)
            assert any("  " in row for row in rows)
