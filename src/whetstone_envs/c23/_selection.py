from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING

from whetstone_envs.c23._domain import (
    Demonstration,
    GeneratedTask,
    GenerationConfiguration,
    Hypothesis,
    RuleConfiguration,
    RuleFamily,
)
from whetstone_envs.c23._inductionbench import examples_for

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterable

_MINIMUM_QUERY_LENGTH = 2
_MAXIMUM_DEMONSTRATION_LENGTH = 4


@cache
def supported_hypotheses(
    vocab: tuple[str, ...],
) -> tuple[Hypothesis, ...]:
    """Enumerate the complete finite hypothesis class in canonical order."""
    configurations = (
        RuleConfiguration(RuleFamily.ISL, 2),
        RuleConfiguration(RuleFamily.L_OSL, 2),
        RuleConfiguration(RuleFamily.R_OSL, 2),
        RuleConfiguration(RuleFamily.ISL, 3),
    )
    return tuple(
        Hypothesis(configuration, context, replacement)
        for configuration in configurations
        for context in (
            "".join(symbols)
            for symbols in itertools.product(
                vocab,
                repeat=configuration.context_length,
            )
        )
        for replacement in (
            *(symbol for symbol in vocab if symbol != context[-1]),
            "",
        )
    )


def select_task(
    hypothesis: Hypothesis,
    config: GenerationConfiguration,
    rng: random.Random,
    inputs: tuple[str, ...],
    apply_hypothesis: Callable[[Hypothesis, str], str],
) -> GeneratedTask | None:
    """Select six demos and a nontrivial determinate held-out query."""
    examples = examples_for(hypothesis, inputs)
    firing_queries = tuple(
        example
        for example in examples
        if len(example.input) >= _MINIMUM_QUERY_LENGTH
        and example.output != example.input
    )
    if not firing_queries:
        return None
    query_offset = rng.randrange(len(firing_queries))
    ordered_queries = (
        firing_queries[query_offset:] + firing_queries[:query_offset]
    )
    hypotheses = supported_hypotheses(config.vocab)
    for query in ordered_queries:
        wrong = tuple(
            candidate
            for candidate in hypotheses
            if apply_hypothesis(candidate, query.input) != query.output
        )
        candidates = tuple(
            example
            for example in examples
            if example.input != query.input
            and len(example.input) <= _MAXIMUM_DEMONSTRATION_LENGTH
        )
        selected = _cover_wrong_hypotheses(
            wrong,
            candidates,
            budget=config.demonstrations_per_instance,
            apply_hypothesis=apply_hypothesis,
        )
        if selected is None:
            continue
        selected_inputs = {example.input for example in selected}
        padding_offset = rng.randrange(len(candidates))
        padding = candidates[padding_offset:] + candidates[:padding_offset]
        demonstrations = list(selected)
        for example in padding:
            if len(demonstrations) == config.demonstrations_per_instance:
                break
            if example.input not in selected_inputs:
                demonstrations.append(example)
                selected_inputs.add(example.input)
        if len(demonstrations) != config.demonstrations_per_instance:
            continue
        demonstrations.sort(key=lambda item: (len(item.input), item.input))
        frozen = tuple(demonstrations)
        if _version_space_outputs(
            frozen,
            query.input,
            hypotheses,
            apply_hypothesis,
        ) != {
            query.output,
        }:
            raise AssertionError("demo cover did not determine query output")
        return GeneratedTask(
            hypothesis=hypothesis,
            demonstrations=frozen,
            query=query.input,
            gold=query.output,
        )
    return None


def _version_space_outputs(
    demonstrations: tuple[Demonstration, ...],
    query: str,
    hypotheses: tuple[Hypothesis, ...],
    apply_hypothesis: Callable[[Hypothesis, str], str],
) -> frozenset[str]:
    return frozenset(
        apply_hypothesis(hypothesis, query)
        for hypothesis in hypotheses
        if all(
            apply_hypothesis(hypothesis, example.input) == example.output
            for example in demonstrations
        )
    )


def _cover_wrong_hypotheses(
    wrong: tuple[Hypothesis, ...],
    candidates: tuple[Demonstration, ...],
    *,
    budget: int,
    apply_hypothesis: Callable[[Hypothesis, str], str],
) -> tuple[Demonstration, ...] | None:
    if not wrong:
        return ()
    coverage = tuple(
        (
            example,
            sum(
                1 << index
                for index, hypothesis in enumerate(wrong)
                if apply_hypothesis(hypothesis, example.input)
                != example.output
            ),
        )
        for example in candidates
    )
    coverage = tuple(item for item in coverage if item[1])
    target = (1 << len(wrong)) - 1
    if _union_masks(mask for _example, mask in coverage) != target:
        return None
    greedy = _greedy_cover(coverage, target, budget)
    if greedy is not None:
        return greedy

    representatives: dict[int, Demonstration] = {}
    for example, mask in coverage:
        representatives.setdefault(mask, example)
    masks = tuple(representatives)
    exact = _bounded_cover(masks, target, budget)
    if exact is None:
        return None
    return tuple(representatives[masks[index]] for index in exact)


def _union_masks(masks: Iterable[int]) -> int:
    union = 0
    for mask in masks:
        union |= mask
    return union


def _greedy_cover(
    coverage: tuple[tuple[Demonstration, int], ...],
    target: int,
    budget: int,
) -> tuple[Demonstration, ...] | None:
    uncovered = target
    remaining = list(coverage)
    selected: list[Demonstration] = []
    while uncovered and len(selected) < budget:
        index = max(
            range(len(remaining)),
            key=lambda candidate_index: (
                (remaining[candidate_index][1] & uncovered).bit_count(),
                -candidate_index,
            ),
        )
        example, mask = remaining.pop(index)
        if not mask & uncovered:
            break
        selected.append(example)
        uncovered &= ~mask
    return tuple(selected) if not uncovered else None


def _bounded_cover(
    masks: tuple[int, ...],
    target: int,
    budget: int,
) -> tuple[int, ...] | None:
    """Find a deterministic cover within the fixed demo budget."""

    @cache
    def search(
        uncovered: int,
        remaining_budget: int,
    ) -> tuple[int, ...] | None:
        if not uncovered:
            return ()
        if not remaining_budget:
            return None
        useful = tuple(
            index for index in range(len(masks)) if masks[index] & uncovered
        )
        if not useful:
            return None
        max_cover = max(
            (masks[index] & uncovered).bit_count() for index in useful
        )
        if (
            uncovered.bit_count() + max_cover - 1
        ) // max_cover > remaining_budget:
            return None
        pivot = uncovered & -uncovered
        covering = tuple(index for index in useful if masks[index] & pivot)
        for index in sorted(
            covering,
            key=lambda candidate: (
                -(masks[candidate] & uncovered).bit_count(),
                candidate,
            ),
        ):
            suffix = search(
                uncovered & ~masks[index],
                remaining_budget - 1,
            )
            if suffix is not None:
                return (index, *suffix)
        return None

    return search(target, budget)
