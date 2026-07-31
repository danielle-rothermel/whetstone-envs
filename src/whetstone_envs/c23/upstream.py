r"""The import boundary around the vendored InductionBench generator + oracle.

The c23 baseline spec regenerates fresh single-rule ISL/OSL instances with
the *vendored + patched* InductionBench generator and reuses its rule
transducers as the independent oracle. This module is the thin boundary
that drives the vendored code (see
``_vendor/inductionbench/PROVENANCE.md``): it marshals plain arguments into
the ``argparse.Namespace`` the vendored functions expect, sets the vendored
module-global ``config.vocab`` for the duration of one call **under a
lock** (restoring it after, so the global is never a hidden cross-call
coupling), and projects the vendored output into plain records.

Public demonstrations and the held-out query are selected jointly against
the explicitly enumerated finite benchmark hypothesis class. Every supported
hypothesis consistent with the demonstrations must produce the frozen gold
on the query; the selector does not require those hypotheses to identify the
same latent rule.

Two upstream facts from the repos review drive the design:

* the vendored generator reads the alphabet from a module-global
  ``config.vocab`` (upstream global mutable state). We set it explicitly
  per call, guarded by a lock and try/finally-restored, rather than
  mutating a shared global that outlives the call.
* the vendored generator is deterministic given ``(seed, args)`` after
  its private-RNG and canonical-order fixes; the
  boundary threads a real ``seed`` into both ``generate_rules`` and
  ``generate_data`` (patch 3/4).

The oracle re-application (:func:`apply_rule`) calls the vendored
``apply_ISL_rule`` / ``apply_L_OSL_rule`` / ``apply_R_OSL_rule``
**unmodified** -- the c23 oracle never reimplements rule application.
"""

from __future__ import annotations

import argparse
import itertools
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from whetstone_envs.c23 import prompts
from whetstone_envs.c23._vendor.inductionbench import config
from whetstone_envs.c23._vendor.inductionbench import (
    synthetic_data_generation as _sdg,
)
from whetstone_envs.core.instance import make_instance
from whetstone_envs.core.pool import public_prompt_identity

if TYPE_CHECKING:
    from collections.abc import Sequence, Set

# The three transducer families the ISL/OSL path exposes (spec Section 1
# secondary axis). ``L_OSL`` / ``R_OSL`` use the vendored ``_OSL`` spelling
# the generator's ``--type`` argument expects.
ISL = "ISL"
L_OSL = "L_OSL"
R_OSL = "R_OSL"
RULE_TYPES: tuple[str, ...] = (ISL, L_OSL, R_OSL)
MIN_QUERY_LEN = 2

# The benchmark's supported finite family/k surface. This is deliberately
# the exact set represented by the four public strata, not the Cartesian
# product of every family with every k. A hypothesis is one of these
# configurations plus one length-k context and one non-identity replacement
# (including deletion).
SUPPORTED_RULE_CONFIGURATIONS: tuple[tuple[str, int], ...] = (
    (ISL, 2),
    (L_OSL, 2),
    (R_OSL, 2),
    (ISL, 3),
)
PublicPromptIdentity = tuple[tuple[str, str], ...]

# The vendored ``config.vocab`` is a process-global read by every generator
# function. We set it per call under this lock so concurrent generations
# never race on it, and restore the prior value afterwards.
_VOCAB_LOCK = threading.Lock()

# The vendored alphabet is drawn from the start of the lowercase Latin
# letters (upstream ``list('abc...'[:vocab_size])``); pinned here so the
# boundary builds the same alphabet the generator's own ``__main__`` does.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class UpstreamError(RuntimeError):
    """Raised when the vendored generator cannot produce the asked-for pool."""


@dataclass(frozen=True, slots=True)
class RawInstance:
    """One generated instance projected onto its public + oracle fields.

    ``demos`` is the input->output demonstration mapping the model sees;
    ``query`` is a held-out input string **not** present in ``demos``;
    ``gold`` is the vendored transducer applied to ``query`` (the oracle
    output). ``rule`` is the latent rule dict, retained only so the
    generator can cross-check the frozen gold against a fresh oracle
    re-application -- it is never placed in a prompt.
    """

    rule_type: str
    k: int
    rule: Mapping[str, str]
    demos: Mapping[str, str]
    query: str
    gold: str


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One member of the supported finite single-rule hypothesis class."""

    rule_type: str
    k: int
    context: str
    replacement: str

    def apply(self, value: str) -> str:
        """Apply this hypothesis through the canonical vendored transducer."""
        return apply_rule(
            self.rule_type,
            self.k,
            {self.context: self.replacement},
            value,
        )


def _require_int(name: str, value: int) -> int:
    """Return a strict integer, rejecting booleans and numeric lookalikes."""
    if type(value) is not int:
        msg = f"{name} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def vocab_for(vocab_size: int) -> list[str]:
    """Return the vendored alphabet of ``vocab_size`` symbols (a, b, ...)."""
    vocab_size = _require_int("vocab_size", vocab_size)
    if not 1 <= vocab_size <= len(_ALPHABET):
        msg = f"vocab_size must be in 1..{len(_ALPHABET)}, got {vocab_size}"
        raise UpstreamError(msg)
    return list(_ALPHABET[:vocab_size])


def enumerate_supported_hypotheses(
    vocab_size: int,
) -> tuple[Hypothesis, ...]:
    """Enumerate the exact supported finite one-rule hypothesis class.

    For each supported family/k configuration, the rule context is every
    length-k string over the fixed alphabet. The replacement is deletion or
    any alphabet symbol other than the context's final symbol, exactly
    matching the vendored one-rule generator's output domain.
    """
    vocab = vocab_for(vocab_size)
    hypotheses: list[Hypothesis] = []
    for rule_type, k in SUPPORTED_RULE_CONFIGURATIONS:
        contexts = _strings_of_length(vocab, k)
        for context in contexts:
            replacements = [
                symbol for symbol in vocab if symbol != context[-1]
            ]
            replacements.append("")
            hypotheses.extend(
                Hypothesis(rule_type, k, context, replacement)
                for replacement in replacements
            )
    return tuple(hypotheses)


def consistent_hypotheses(
    demos: Mapping[str, str],
    *,
    vocab_size: int,
) -> tuple[Hypothesis, ...]:
    """Return every supported hypothesis consistent with public ``demos``."""
    hypotheses = enumerate_supported_hypotheses(vocab_size)
    return tuple(
        hypothesis
        for hypothesis in hypotheses
        if all(
            hypothesis.apply(demo_input) == demo_output
            for demo_input, demo_output in demos.items()
        )
    )


def version_space_outputs(
    demos: Mapping[str, str],
    query: str,
    *,
    vocab_size: int,
) -> frozenset[str]:
    """Return query outputs allowed by the demo-consistent version space."""
    return frozenset(
        hypothesis.apply(query)
        for hypothesis in consistent_hypotheses(
            demos,
            vocab_size=vocab_size,
        )
    )


def _make_args(
    *,
    rule_type: str,
    k: int,
    vocab_size: int,
    number_of_rules: int,
    num_of_datapoints: int,
    sample_size_times: int,
) -> argparse.Namespace:
    """Build the ``argparse.Namespace`` the vendored functions consume.

    Only the fields the ISL/OSL generation + oracle path reads are set;
    ``repeat`` is pinned ``False`` (the repeat branch reads prior result
    JSONs, off our path).
    """
    return argparse.Namespace(
        type=rule_type,
        k=k,
        vocab_size=vocab_size,
        number_of_rules=number_of_rules,
        num_of_datapoints=num_of_datapoints,
        sample_size_times=sample_size_times,
        shot_number=1,
        repeat=False,
    )


def apply_rule(
    rule_type: str,
    k: int,
    rule: Mapping[str, str],
    query: str,
) -> str:
    """Apply the latent ``rule`` to ``query`` via the vendored transducer.

    Dispatches to the vendored ``apply_ISL_rule`` / ``apply_L_OSL_rule`` /
    ``apply_R_OSL_rule`` **unmodified** (the c23 oracle's ground truth).
    ``rule`` must be a plain ``{context: output}`` mapping; a ``dict`` copy
    is passed so the vendored code cannot mutate the caller's mapping.
    """
    k = _require_int("k", k)
    args = argparse.Namespace(type=rule_type, k=k)
    rule_dict = dict(rule)
    if rule_type == ISL:
        return _sdg.apply_ISL_rule(args, rule_dict, query)
    if rule_type == L_OSL:
        return _sdg.apply_L_OSL_rule(args, rule_dict, query)
    if rule_type == R_OSL:
        return _sdg.apply_R_OSL_rule(args, rule_dict, query)
    msg = f"unknown rule_type {rule_type!r} (expected one of {RULE_TYPES})"
    raise UpstreamError(msg)


def _strings_of_length(
    vocab: Sequence[str],
    length: int,
) -> tuple[str, ...]:
    """Return one finite string layer in canonical product order."""
    return tuple(
        "".join(chars) for chars in itertools.product(vocab, repeat=length)
    )


def _canonical_examples(
    examples: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Freeze examples in stable shortest-input-first order."""
    return tuple(
        sorted(
            examples.items(),
            key=lambda item: (len(item[0]), item[0], item[1]),
        ),
    )


def _find_demo_cover(
    hypotheses: Sequence[Hypothesis],
    candidates: Sequence[tuple[str, str]],
    *,
    n_demos: int,
) -> tuple[tuple[str, str], ...] | None:
    """Find at most ``n_demos`` examples eliminating every hypothesis.

    ``hypotheses`` contains exactly the query-wrong portion of the version
    space. Each candidate demonstration covers the hypotheses it
    contradicts. A query-aware greedy pass handles the common case; a
    deterministic bounded exact-cover search prevents the heuristic from
    freezing ambiguity or falsely declaring a feasible budget impossible.
    """
    if not hypotheses:
        return ()

    coverage = _demo_coverage(hypotheses, candidates)
    target = (1 << len(hypotheses)) - 1
    if not _coverage_is_complete(coverage, target):
        return None

    greedy = _greedy_demo_cover(coverage, target, n_demos)
    if greedy is not None:
        return greedy

    # Examples with identical coverage are interchangeable for feasibility,
    # so retain the first canonical representative.
    representative: dict[int, tuple[str, str]] = {}
    for example, mask in coverage:
        representative.setdefault(mask, example)
    masks = tuple(representative)
    solution = _exact_cover_masks(masks, target, n_demos)
    if solution is None:
        return None
    return tuple(representative[masks[index]] for index in solution)


def _demo_coverage(
    hypotheses: Sequence[Hypothesis],
    candidates: Sequence[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], int], ...]:
    """Encode which query-wrong hypotheses each demo contradicts."""
    coverage: list[tuple[tuple[str, str], int]] = []
    for example in candidates:
        demo_input, demo_output = example
        mask = 0
        for index, hypothesis in enumerate(hypotheses):
            if hypothesis.apply(demo_input) != demo_output:
                mask |= 1 << index
        if mask:
            coverage.append((example, mask))
    return tuple(coverage)


def _coverage_is_complete(
    coverage: Sequence[tuple[tuple[str, str], int]],
    target: int,
) -> bool:
    """Return whether the candidates jointly eliminate the target set."""
    union = 0
    for _example, mask in coverage:
        union |= mask
    return union == target


def _greedy_demo_cover(
    coverage: Sequence[tuple[tuple[str, str], int]],
    target: int,
    n_demos: int,
) -> tuple[tuple[str, str], ...] | None:
    """Return a targeted greedy cover when it fits ``n_demos``."""
    uncovered = target
    greedy: list[tuple[str, str]] = []
    unused = list(coverage)
    while uncovered and len(greedy) < n_demos:
        best_index = -1
        best_count = 0
        for index, (_example, mask) in enumerate(unused):
            count = (mask & uncovered).bit_count()
            if count > best_count:
                best_index = index
                best_count = count
        if best_index < 0:
            break
        example, mask = unused.pop(best_index)
        greedy.append(example)
        uncovered &= ~mask
    if not uncovered:
        return tuple(greedy)
    return None


def _exact_cover_masks(
    masks: tuple[int, ...],
    target: int,
    n_demos: int,
) -> tuple[int, ...] | None:
    """Find a bounded exact set cover over canonical coverage masks."""

    @cache
    def search(uncovered_mask: int, budget: int) -> tuple[int, ...] | None:
        if not uncovered_mask:
            return ()
        if budget == 0 or not masks:
            return None

        useful = tuple(mask & uncovered_mask for mask in masks)
        max_cover = max(mask.bit_count() for mask in useful)
        if max_cover == 0:
            return None
        minimum_needed = (
            uncovered_mask.bit_count() + max_cover - 1
        ) // max_cover
        if minimum_needed > budget:
            return None

        pivot_candidates: tuple[int, ...] | None = None
        pending = uncovered_mask
        while pending:
            pivot = pending & -pending
            covering = tuple(
                index for index, mask in enumerate(masks) if mask & pivot
            )
            if pivot_candidates is None or len(covering) < len(
                pivot_candidates,
            ):
                pivot_candidates = covering
            pending &= ~pivot
        assert pivot_candidates is not None

        ordered = sorted(
            pivot_candidates,
            key=lambda index: (
                -(masks[index] & uncovered_mask).bit_count(),
                index,
            ),
        )
        for index in ordered:
            remainder = uncovered_mask & ~masks[index]
            suffix = search(remainder, budget - 1)
            if suffix is not None:
                return (index, *suffix)
        return None

    return search(target, n_demos)


def _public_prompt_identity(
    demos: Mapping[str, str],
    query: str,
) -> PublicPromptIdentity:
    """Return the shared pool identity for one candidate public prompt."""
    candidate = make_instance(
        id="c23-public-identity-candidate",
        seed=0,
        strata="c23",
        prompt_inputs={
            "demos_block": prompts.render_demos_block(dict(demos)),
            "query": query,
        },
    )
    return public_prompt_identity(candidate)


def _select_demos_and_query(
    full_demos: Mapping[str, str],
    *,
    rule_type: str,
    k: int,
    rule: Mapping[str, str],
    vocab_size: int,
    n_demos: int,
    max_query_len: int,
    query_offset: int,
    excluded_public_identities: Set[PublicPromptIdentity],
    seed: int,
    instance_index: int,
) -> tuple[dict[str, str], str, str]:
    """Jointly select a determinate nontrivial query and exact demos.

    Candidate queries and demonstrations come from the vendored
    characteristic-sample dataset. Queries are tried deterministically in
    shortest-input-first order and must be transformed by the latent rule.
    For each query, demo selection eliminates every supported hypothesis
    that would produce a different query output. Surviving hypotheses may
    disagree about the rule as long as they all agree on the scored answer.
    """
    hypotheses = enumerate_supported_hypotheses(vocab_size)
    examples = _canonical_examples(full_demos)
    canonical_firing_queries = tuple(
        (query, gold)
        for query, gold in examples
        if MIN_QUERY_LEN <= len(query) <= max_query_len and gold != query
    )
    offset = (
        query_offset % len(canonical_firing_queries)
        if canonical_firing_queries
        else 0
    )
    firing_queries = (
        canonical_firing_queries[offset:] + canonical_firing_queries[:offset]
    )

    duplicate_selections = 0
    for query, gold in firing_queries:
        derived_gold = apply_rule(rule_type, k, rule, query)
        if derived_gold != gold:
            msg = (
                "vendored characteristic sample disagrees with its "
                f"{rule_type} k={k} rule for query {query!r}: "
                f"sample={gold!r}, derived={derived_gold!r}"
            )
            raise UpstreamError(msg)

        candidates = tuple(
            example for example in examples if example[0] != query
        )
        if len(candidates) < n_demos:
            continue
        wrong = tuple(
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.apply(query) != gold
        )
        informative = tuple(
            example for example in candidates if example[0] != example[1]
        )
        selected = _find_demo_cover(
            wrong,
            informative,
            n_demos=n_demos,
        )
        if selected is None:
            selected = _find_demo_cover(
                wrong,
                candidates,
                n_demos=n_demos,
            )
        if selected is None:
            continue

        chosen = list(selected)
        chosen_inputs = {demo_input for demo_input, _output in chosen}
        padding_offset = query_offset % len(candidates)
        padding_candidates = (
            candidates[padding_offset:] + candidates[:padding_offset]
        )
        for example in padding_candidates:
            if len(chosen) == n_demos:
                break
            if example[0] not in chosen_inputs:
                chosen.append(example)
                chosen_inputs.add(example[0])
        if len(chosen) != n_demos:
            continue

        demos = dict(sorted(chosen))
        outputs = version_space_outputs(
            demos,
            query,
            vocab_size=vocab_size,
        )
        if outputs != {gold}:
            msg = (
                "internal c23 selection error: selected demonstrations "
                f"leave outputs {sorted(outputs)!r} for {rule_type} k={k} "
                f"query {query!r}, expected only {gold!r}"
            )
            raise UpstreamError(msg)
        if _public_prompt_identity(demos, query) in excluded_public_identities:
            duplicate_selections += 1
            continue
        return demos, query, gold

    msg = (
        f"cannot select exactly {n_demos} demonstrations and a nontrivial "  # noqa: S608
        "determinate held-out query from "
        f"{len(full_demos)} characteristic-sample pairs for "
        f"{rule_type} k={k}, seed={seed}, instance_index={instance_index} "
        f"with max_query_len={max_query_len}; "
        f"{len(firing_queries)} firing queries were available and the "
        f"supported hypothesis class contains {len(hypotheses)} rules; "
        f"{duplicate_selections} otherwise valid public prompt selections "
        "were already emitted"
    )
    raise UpstreamError(msg)


def _validate_generate_raw_config(
    *,
    rule_type: str,
    k: int,
    vocab_size: int,
    seed: int,
    num_instances: int,
    sample_size_times: int,
    max_query_len: int,
    n_demos: int,
) -> None:
    """Validate the public generation configuration before vendor calls."""
    if rule_type not in RULE_TYPES:
        msg = f"unknown rule_type {rule_type!r} (expected one of {RULE_TYPES})"
        raise UpstreamError(msg)
    _require_int("k", k)
    _require_int("vocab_size", vocab_size)
    _require_int("seed", seed)
    _require_int("num_instances", num_instances)
    _require_int("sample_size_times", sample_size_times)
    _require_int("max_query_len", max_query_len)
    _require_int("n_demos", n_demos)
    if (rule_type, k) not in SUPPORTED_RULE_CONFIGURATIONS:
        msg = (
            f"unsupported rule configuration {(rule_type, k)!r} "
            f"(expected one of {SUPPORTED_RULE_CONFIGURATIONS!r})"
        )
        raise UpstreamError(msg)
    if num_instances < 1:
        msg = f"num_instances must be positive, got {num_instances}"
        raise UpstreamError(msg)
    if sample_size_times < 1:
        msg = f"sample_size_times must be positive, got {sample_size_times}"
        raise UpstreamError(msg)
    if max_query_len < MIN_QUERY_LEN:
        msg = (
            f"max_query_len must be at least {MIN_QUERY_LEN}, "
            f"got {max_query_len}"
        )
        raise UpstreamError(msg)
    if n_demos < 0:
        msg = f"n_demos must be non-negative, got {n_demos}"
        raise UpstreamError(msg)


def generate_raw(
    *,
    rule_type: str,
    k: int,
    vocab_size: int,
    seed: int,
    num_instances: int,
    sample_size_times: int,
    max_query_len: int,
    n_demos: int,
    excluded_public_identities: Set[PublicPromptIdentity] = frozenset(),
) -> list[RawInstance]:
    """Generate ``num_instances`` single-rule instances for one stratum.

    Reseeds the vendored generator once at ``seed`` (patch 3/4), sets the
    vendored ``config.vocab`` under a lock for the duration, and for each
    generated rule jointly selects an exact-size demonstration mapping and a
    nontrivial held-out query whose output is determinate over every
    demo-consistent supported hypothesis.

    Determinism: with the vendor patches the whole draw is a pure
    function of ``(rule_type, k, vocab_size, seed, num_instances,
    sample_size_times, max_query_len, n_demos)`` -- verified
    byte-identical across runs under a randomized ``PYTHONHASHSEED``.
    """
    _validate_generate_raw_config(
        rule_type=rule_type,
        k=k,
        vocab_size=vocab_size,
        seed=seed,
        num_instances=num_instances,
        sample_size_times=sample_size_times,
        max_query_len=max_query_len,
        n_demos=n_demos,
    )
    vocab = vocab_for(vocab_size)
    args = _make_args(
        rule_type=rule_type,
        k=k,
        vocab_size=vocab_size,
        number_of_rules=1,
        num_of_datapoints=num_instances,
        sample_size_times=sample_size_times,
    )

    with _VOCAB_LOCK:
        saved = config.vocab
        config.vocab = vocab
        try:
            rules = _sdg.generate_rules(args, seed=seed)
            data = _sdg.generate_data(args, rules, seed=seed)
        finally:
            config.vocab = saved

    if len(rules) != num_instances or len(data) != num_instances:
        msg = (
            f"vendored generator produced {len(rules)} rules / {len(data)} "
            f"datapoints for {rule_type} k={k}, expected {num_instances}"
        )
        raise UpstreamError(msg)

    selected_public_identities = set(excluded_public_identities)
    out: list[RawInstance] = []
    for idx in range(num_instances):
        rule = rules[idx]
        sample_dataset, _prompt = data[idx]
        full_demos = {str(i): str(o) for i, o in sample_dataset.items()}
        demos, query, gold = _select_demos_and_query(
            full_demos,
            rule_type=rule_type,
            k=k,
            rule=rule,
            vocab_size=vocab_size,
            n_demos=n_demos,
            max_query_len=max_query_len,
            query_offset=idx,
            excluded_public_identities=selected_public_identities,
            seed=seed,
            instance_index=idx,
        )
        selected_public_identities.add(
            _public_prompt_identity(demos, query),
        )
        out.append(
            RawInstance(
                rule_type=rule_type,
                k=k,
                rule={str(kk): str(vv) for kk, vv in rule.items()},
                demos=demos,
                query=query,
                gold=gold,
            ),
        )
    return out
