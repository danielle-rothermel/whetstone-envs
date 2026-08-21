from __future__ import annotations

from dataclasses import dataclass

from whetstone_envs.c19.model import Action, C19Fact
from whetstone_envs.c19.probes import PROBES
from whetstone_envs.c19.scenarios import C19Scenario, C19Size


@dataclass(frozen=True, slots=True)
class NamedDescription:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class C19Info:
    objective: str
    public_inputs: tuple[NamedDescription, ...]
    sizes: tuple[NamedDescription, ...]
    scenarios: tuple[NamedDescription, ...]
    actions: tuple[NamedDescription, ...]
    tokens: tuple[NamedDescription, ...]
    facts: tuple[NamedDescription, ...]
    scoring: str
    pool: str
    terminology: tuple[NamedDescription, ...]
    naive_template: str
    ceiling_template: str


C19_INFO = C19Info(
    objective=(
        "Predict the requested final MiniGrid state fact after applying the "
        "complete action script. C19 tests state prediction, not action "
        "planning."
    ),
    public_inputs=(
        NamedDescription("grid", "two-character initial grid"),
        NamedDescription("command", "complete LRFPDT action script"),
        NamedDescription("question", "requested final-state fact"),
        NamedDescription("private gold", "exact answer used only for scoring"),
    ),
    sizes=tuple(
        NamedDescription(size.name.lower(), f"{int(size)} by {int(size)} grid")
        for size in C19Size
    ),
    scenarios=(
        NamedDescription(C19Scenario.NAVIGATION.value, "movement and heading"),
        NamedDescription(C19Scenario.MANIPULATION.value, "pickup and drop"),
        NamedDescription(
            C19Scenario.DOOR.value, "door toggling and unlocking"
        ),
    ),
    actions=(
        NamedDescription(
            Action.LEFT.value, "turn left in place; always succeeds"
        ),
        NamedDescription(
            Action.RIGHT.value, "turn right in place; always succeeds"
        ),
        NamedDescription(
            Action.FORWARD.value,
            "move only into empty or open-door space; otherwise no-op",
        ),
        NamedDescription(
            Action.PICKUP.value,
            "pick up a key or ball ahead only with empty hands; otherwise "
            "no-op",
        ),
        NamedDescription(
            Action.DROP.value,
            "drop the carried object only into an empty cell; otherwise no-op",
        ),
        NamedDescription(
            Action.TOGGLE.value,
            "toggle a door ahead; locked doors require a same-color carried "
            "key",
        ),
    ),
    tokens=(
        NamedDescription("  ", "empty"),
        NamedDescription(">> VV << ^^", "agent facing E, S, W, N"),
        NamedDescription("WG", "grey wall"),
        NamedDescription("K* A*", "colored key or ball"),
        NamedDescription("D* L* __", "closed, locked, or open door"),
        NamedDescription(
            "coordinates", "zero-based row,column; origin at top left"
        ),
    ),
    facts=(
        NamedDescription(C19Fact.COORDINATE.value, "row,col"),
        NamedDescription(C19Fact.HEADING.value, "one of E, S, W, N"),
        NamedDescription(
            C19Fact.FRONT.value, "one of empty, wall, key, ball, door"
        ),
        NamedDescription(C19Fact.CARRYING.value, "yes or no"),
    ),
    scoring=(
        "Prediction and gold are independently stripped of surrounding "
        "whitespace and complete outer triple-backtick fences, then compared "
        "exactly for a binary score."
    ),
    pool=(
        "The default pool has 22 strata and 352 instances. Internal "
        "evaluation has 88 tasks and supports optimization; official "
        "evaluation has 132 tasks and is rewardless evidence; 132 held-out "
        "tasks are hashes only."
    ),
    terminology=(
        NamedDescription(
            "candidate",
            "one complete prompt template evaluated against a fixed task set",
        ),
        NamedDescription(
            "naive",
            "the intentionally sparse floor probe, not a claim that every "
            "model must fail",
        ),
        NamedDescription(
            "ceiling",
            "the known-good instruction-rich probe, not a guaranteed score "
            "upper bound",
        ),
    ),
    naive_template=PROBES.naive_template,
    ceiling_template=PROBES.ceiling_template,
)


__all__ = ["C19_INFO", "C19Info", "NamedDescription"]
