"""Probe-prompt checks for c19: stable rendering, semantics, no gold leak.

The naive prompt is pinned byte-for-byte against a fixed hand-built
fixture. The ceiling prompt is pinned by its prefix, runtime semantics,
embedded fields, and question-line suffix. A static check asserts neither
prompt ever contains the instance's gold answer.
"""

from __future__ import annotations

from whetstone_envs.c19 import generate, prompts
from whetstone_envs.core.instance import make_instance

# A fixed, hand-built fixture instance (not generator-produced) so the
# expected rendered bytes are stable and human-auditable.
_FIXTURE_GRID = "\n".join(
    [
        "WGWGWGWGWG",
        "WG>>KY  WG",
        "WG      WG",
        "WG    GGWG",
        "WGWGWGWGWG",
    ],
)
_FIXTURE = make_instance(
    id="c19-fixture-1",
    seed=1_000_000,
    strata="Fetch|small|coordinate",
    prompt_inputs={
        "grid": _FIXTURE_GRID,
        "command": "FFRF",
        "fact_type": "coordinate",
    },
    gold="3,3",
)

_EXPECTED_NAIVE = (
    "Here is a grid and a sequence of moves for the robot.\n"
    "\n"
    "WGWGWGWGWG\n"
    "WG>>KY  WG\n"
    "WG      WG\n"
    "WG    GGWG\n"
    "WGWGWGWGWG\n"
    "\n"
    "Moves: FFRF\n"
    "\n"
    "Where does the robot end up? Answer with just the final coordinate."
)


def test_naive_render_is_byte_for_byte_fixed() -> None:
    assert prompts.render_naive(_FIXTURE) == _EXPECTED_NAIVE


def test_ceiling_render_has_stable_prefix_and_suffix() -> None:
    rendered = prompts.render_ceiling(_FIXTURE)
    assert rendered.startswith(
        "You are simulating a robot on a 2D grid. Follow these rules EXACTLY.",
    )
    # The move-semantics block is present.
    assert (
        "  - L = turn left 90 degrees in place (does not move): "
        "E->N->W->S->E.\n"
    ) in rendered
    assert (
        "  - F = step ONE cell forward in the current facing direction"
        in rendered
    )
    # The public fields are substituted verbatim.
    assert _FIXTURE_GRID in rendered
    assert "Moves: FFRF" in rendered
    # The coordinate question line ends the prompt, byte-for-byte.
    assert rendered.endswith(
        "QUESTION: What is the robot's final coordinate? Answer on the last "
        "line as: row,col",
    )


def test_ceiling_describes_pprint_grid_glyphs_exactly() -> None:
    rendered = prompts.render_ceiling(_FIXTURE)
    assert 'two space characters "  " are an empty floor cell' in rendered
    assert '"AR" = red ball' in rendered
    assert '"BG" = green box' in rendered
    assert '"BR"' not in rendered
    assert 'doubled direction arrow: ">>" faces right' in rendered


def test_ceiling_describes_runtime_action_semantics() -> None:
    rendered = prompts.render_ceiling(_FIXTURE)
    assert "solid object such as a key, ball," in rendered
    assert "box, or closed door blocks movement" in rendered
    assert "pick up a pickup-able key, ball, or box" in rendered
    assert "object types in these\n    grids do not" in rendered


def test_fact_line_variants_are_selected_by_fact_type() -> None:
    for fact, naive_tail, ceiling_tail in (
        (
            "heading",
            "Which direction is the robot facing at the end? Answer with "
            "one letter.",
            "Answer on the last line as one of: E W N S",
        ),
        (
            "carrying",
            "Is the robot carrying an object at the end? Answer yes or no.",
            "Answer on the last line as: yes or no",
        ),
        (
            "front",
            "What is directly in front of the robot at the end? Answer with "
            "one word.",
            "one word (e.g. wall, empty, key, ball, goal).",
        ),
    ):
        inst = make_instance(
            id=f"c19-fixture-{fact}",
            seed=1_000_000,
            strata=f"Fetch|small|{fact}",
            prompt_inputs={
                "grid": _FIXTURE_GRID,
                "command": "FFRF",
                "fact_type": fact,
            },
            gold="x",
        )
        assert prompts.render_naive(inst).endswith(naive_tail)
        assert ceiling_tail in prompts.render_ceiling(inst)


def test_render_is_stable_across_calls() -> None:
    # Rendering is a pure function of the instance: same bytes every call.
    assert prompts.render_naive(_FIXTURE) == prompts.render_naive(_FIXTURE)
    assert prompts.render_ceiling(_FIXTURE) == prompts.render_ceiling(_FIXTURE)


def test_probepair_dispatches_naive_and_ceiling() -> None:
    assert prompts.PROBES.render_naive(_FIXTURE) == prompts.render_naive(
        _FIXTURE,
    )
    assert prompts.PROBES.render_ceiling(_FIXTURE) == prompts.render_ceiling(
        _FIXTURE,
    )


def test_no_prompt_leaks_the_gold_answer() -> None:
    # Static no-gold-leak check (PLAN): across a full default-shape pool,
    # neither rendered prompt may contain the instance's gold answer.
    pool = generate.generate_pool(n_per_stratum=1)
    for inst in pool.instances:
        gold = inst.gold
        naive = prompts.render_naive(inst)
        ceiling = prompts.render_ceiling(inst)
        # The gold value must not appear as the prompt's answer. Guard
        # against trivially-substringy golds (single letters like "E"/"S"
        # appear inside prose) by checking the answer is not the tail.
        assert not naive.rstrip().endswith(f" {gold}"), inst.id
        assert not ceiling.rstrip().endswith(f": {gold}"), inst.id
        # The gold is never present as a standalone answer line.
        assert f"\n{gold}\n" not in naive, inst.id
        assert f"\n{gold}\n" not in ceiling, inst.id
