from __future__ import annotations

import argparse
import re
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whetstone_envs.c22 import PROBES, Preset, generate_pool
from whetstone_envs.c22.constraints import (
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
from whetstone_envs.c22.oracle import evaluate

if TYPE_CHECKING:
    from whetstone_envs.instances import Instance


def compose_response(stack: ConstraintStack, *, compact: bool) -> str:
    parts: list[str] = []
    keyword = ""
    ending = ""
    postscript = ""
    word_count: int | None = None
    quoted = False
    label = "!" if compact else "note"
    for constraint in stack.constraints:
        match constraint:
            case RequiredKeyword(keyword=value):
                keyword = value
            case EndPhrase(phrase=value):
                ending = value
            case Postscript(marker=value):
                postscript = value
            case Title():
                parts.append(f"<<{label}>>")
            case Placeholders(count=count):
                parts.extend(["[]" if compact else "[note]"] * count)
            case HighlightedSections(count=count):
                parts.extend([f"*{label}*"] * count)
            case ExactWordCount(count=count):
                word_count = count
            case Quotation():
                quoted = True
    if compact:
        # IFEval accepts keyword substrings, allowing tight word budgets.
        suffix = keyword + postscript + ending
    else:
        suffix = " ".join(
            part for part in (keyword, postscript, ending) if part
        )
    parts.append(suffix)
    used_words = len(re.findall(r"\w+", " ".join(parts)))
    if word_count is not None:
        parts[0:0] = ["note"] * max(0, word_count - used_words)
    response = " ".join(part for part in parts if part) or "note"
    return f'"{response}"' if quoted else response


def passing_response(stack: ConstraintStack) -> str:
    for compact in (False, True):
        response = compose_response(stack, compact=compact)
        if evaluate(stack, response).follow_all:
            return response
    msg = "Could not construct a passing example for this constraint stack"
    raise ValueError(msg)


def failing_response(stack: ConstraintStack, passing: str) -> str:
    match stack.constraints[0]:
        case RequiredKeyword(keyword=keyword):
            response = re.sub(
                re.escape(keyword), "", passing, flags=re.IGNORECASE
            )
        case ForbiddenWord(word=word):
            response = f"{passing} {word}"
        case EndPhrase():
            response = f"{passing}!"
        case Title():
            response = passing.replace("<<", "").replace(">>", "")
        case Quotation():
            response = passing.strip('"')
        case NoComma():
            response = f"{passing},"
        case Placeholders():
            response = re.sub(r"\[.*?\]", "", passing)
        case Postscript(marker=marker):
            response = passing.replace(marker, "")
        case HighlightedSections():
            response = passing.replace("*", "")
        case ExactWordCount():
            response = f"{passing} note"
        case ForbiddenLetter(letter=letter):
            response = f"{passing}{letter}"
    if evaluate(stack, response).follow_all:
        msg = "The failing example unexpectedly passed its constraint stack"
        raise ValueError(msg)
    return response


def render_scoring_examples(console: Console, stack: ConstraintStack) -> None:
    passing = passing_response(stack)
    failing = failing_response(stack, passing)
    passing_result = evaluate(stack, passing)
    failing_result = evaluate(stack, failing)
    outcomes = Table(title="Per-constraint outcomes")
    outcomes.add_column("#")
    outcomes.add_column("Constraint")
    for name, response, result, color in (
        ("Passing example", passing, passing_result, "green"),
        ("Failing example", failing, failing_result, "red"),
    ):
        console.print(
            Panel(
                Text(response),
                title=f"{name} | Score: {result.score}",
                border_style=color,
            )
        )
        outcomes.add_column(name)
    for index, ((kind, passing_passed), (_, failing_passed)) in enumerate(
        zip(
            passing_result.per_constraint,
            failing_result.per_constraint,
            strict=True,
        ),
        start=1,
    ):
        outcomes.add_row(
            str(index),
            Text(kind.replace("_", " ")),
            Text(
                "PASS" if passing_passed else "FAIL",
                style="green" if passing_passed else "red",
            ),
            Text(
                "PASS" if failing_passed else "FAIL",
                style="green" if failing_passed else "red",
            ),
        )
    console.print(outcomes)


def render_instance(console: Console, instance: Instance) -> None:
    console.rule(Text(instance.id, style="bold"))
    console.print(
        Text(
            f"Seed: {instance.seed} | "
            f"Categories: {', '.join(instance.categories)}"
        )
    )
    console.print(
        Panel(
            Text(instance.prompt_inputs["constraints"]),
            title="Public numbered constraints",
            border_style="cyan",
        )
    )
    stack = ConstraintStack.from_gold(instance.gold)
    table = Table(title="Private gold: scoring specification")
    table.add_column("#")
    table.add_column("Constraint")
    table.add_column("Parameters")
    for index, constraint in enumerate(stack.constraints, start=1):
        parameters = constraint.model_dump(exclude={"kind"})
        table.add_row(
            str(index),
            Text(constraint.kind.replace("_", " ")),
            Text(
                ", ".join(
                    f"{name}={value!r}" for name, value in parameters.items()
                )
                or "—"
            ),
        )
    console.print(table)
    render_scoring_examples(console, stack)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect C22 instruction-constraint instances."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--preset", type=Preset, choices=tuple(Preset), default=Preset.DEFAULT
    )
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Show both rendered probe prompts.",
    )
    arguments = parser.parse_args(argv)
    if arguments.count < 1:
        parser.error("count must be positive")
    instances = generate_pool(arguments.preset).instance_group.instances
    if arguments.count > len(instances):
        parser.error(
            f"count exceeds the fixed pool of {len(instances)} instances"
        )
    console = Console()
    console.print(
        Text(
            f"C22 | Instruction constraints | "
            f"Preset: {arguments.preset.value}",
            style="bold",
        )
    )
    for instance in instances[: arguments.count]:
        render_instance(console, instance)
        if arguments.probes:
            console.print(
                Panel(Text(PROBES.render_naive(instance)), title="Naive probe")
            )
            console.print(
                Panel(
                    Text(PROBES.render_ceiling(instance)),
                    title="Ceiling probe",
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
