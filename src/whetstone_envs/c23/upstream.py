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

Two upstream facts from the repos review drive the design:

* the vendored generator reads the alphabet from a module-global
  ``config.vocab`` (upstream global mutable state). We set it explicitly
  per call, guarded by a lock and try/finally-restored, rather than
  mutating a shared global that outlives the call.
* the vendored generator is deterministic given ``(seed, args)`` only after
  the four vendor patches (the ``sorted(...)`` determinism fix); the
  boundary threads a real ``seed`` into both ``generate_rules`` and
  ``generate_data`` (patch 3/4).

The oracle re-application (:func:`apply_rule`) calls the vendored
``apply_ISL_rule`` / ``apply_L_OSL_rule`` / ``apply_R_OSL_rule``
**unmodified** -- the c23 oracle never reimplements rule application.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import config  # vendored stub (whetstone_envs.c23 put it on sys.path)
import synthetic_data_generation as _sdg  # vendored InductionBench module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The three transducer families the ISL/OSL path exposes (spec Section 1
# secondary axis). ``L_OSL`` / ``R_OSL`` use the vendored ``_OSL`` spelling
# the generator's ``--type`` argument expects.
ISL = "ISL"
L_OSL = "L_OSL"
R_OSL = "R_OSL"
RULE_TYPES: tuple[str, ...] = (ISL, L_OSL, R_OSL)

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


def vocab_for(vocab_size: int) -> list[str]:
    """Return the vendored alphabet of ``vocab_size`` symbols (``a``, ``b``...)."""
    if not 1 <= vocab_size <= len(_ALPHABET):
        msg = f"vocab_size must be in 1..{len(_ALPHABET)}, got {vocab_size}"
        raise UpstreamError(msg)
    return list(_ALPHABET[:vocab_size])


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


def _held_out_query(
    rng_pick: Sequence[str],
    demos: Mapping[str, str],
    *,
    rule_type: str,
    k: int,
    rule: Mapping[str, str],
) -> tuple[str, str]:
    """Return a ``(query, gold)`` whose input is absent from ``demos``.

    ``rng_pick`` is a pre-sampled, deterministic list of candidate query
    strings (built by the caller from the seeded RNG). Among the candidates
    absent from ``demos``, a query on which the rule **actually fires**
    (gold != input) is preferred, so the held-out query is not trivially the
    identity; falling back to the first held-out candidate if none of them
    trigger the rule. Its gold is the vendored transducer applied to it.
    Raises if every candidate collided with the demos (the caller sizes the
    candidate list generously so this cannot happen in practice).
    """
    first_held_out: tuple[str, str] | None = None
    for candidate in rng_pick:
        if candidate in demos:
            continue
        gold = apply_rule(rule_type, k, rule, candidate)
        if first_held_out is None:
            first_held_out = (candidate, gold)
        if gold != candidate:
            return candidate, gold
    if first_held_out is not None:
        return first_held_out
    msg = (
        f"could not find a held-out query outside {len(demos)} demos for "
        f"a {rule_type} k={k} rule"
    )
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
) -> list[RawInstance]:
    """Generate ``num_instances`` single-rule instances for one stratum.

    Reseeds the vendored generator once at ``seed`` (patch 3/4), sets the
    vendored ``config.vocab`` under a lock for the duration, and for each
    generated rule builds the demonstration mapping plus a held-out query
    (an input absent from the demos) whose gold is the vendored transducer
    applied to it.

    Determinism: with the four vendor patches the whole draw is a pure
    function of ``(rule_type, k, vocab_size, seed, num_instances,
    sample_size_times, query_pool_size, max_query_len)`` -- verified
    byte-identical across runs under a randomized ``PYTHONHASHSEED``.
    """
    if rule_type not in RULE_TYPES:
        msg = f"unknown rule_type {rule_type!r} (expected one of {RULE_TYPES})"
        raise UpstreamError(msg)
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
            # The held-out query candidates are drawn from the SAME seeded
            # RNG stream, immediately after generate_data, so the whole
            # instance (demos + query) is reproducible from `seed` alone.
            candidates = _draw_query_candidates(
                vocab,
                count=num_instances,
                max_len=max_query_len,
            )
        finally:
            config.vocab = saved

    if len(rules) != num_instances or len(data) != num_instances:
        msg = (
            f"vendored generator produced {len(rules)} rules / {len(data)} "
            f"datapoints for {rule_type} k={k}, expected {num_instances}"
        )
        raise UpstreamError(msg)

    out: list[RawInstance] = []
    for idx in range(num_instances):
        rule = rules[idx]
        sample_dataset, _prompt = data[idx]
        full_demos = {str(i): str(o) for i, o in sample_dataset.items()}
        query, gold = _held_out_query(
            candidates[idx],
            full_demos,
            rule_type=rule_type,
            k=k,
            rule=rule,
        )
        demos = _subsample_demos(full_demos, n_demos=n_demos)
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


def _draw_query_candidates(
    vocab: Sequence[str],
    *,
    count: int,
    max_len: int,
) -> list[list[str]]:
    """Draw ``count`` deterministic lists of candidate query strings.

    One list per instance; each list holds several random strings over
    ``vocab`` (lengths 2..``max_len``) so the caller can pick the first that
    is not already a demonstration input. Uses the module-global ``random``
    (already reseeded by ``generate_data``) so the draw stays on the same
    reproducible stream.
    """
    import random  # noqa: PLC0415 - the vendored global RNG, mid-stream

    per_instance = 32
    out: list[list[str]] = []
    for _ in range(count):
        picks: list[str] = []
        for _ in range(per_instance):
            length = random.randint(2, max_len)  # noqa: S311 - repro, not crypto
            picks.append(
                "".join(random.choice(vocab) for _ in range(length)),  # noqa: S311
            )
        out.append(picks)
    return out


def _subsample_demos(
    full_demos: Mapping[str, str],
    *,
    n_demos: int,
) -> dict[str, str]:
    """Pick a small, canonical, rule-revealing demo subset.

    The vendored characteristic sample is large (often >100 pairs, far more
    than the spec's few-shot block). Down-sample to ``n_demos`` pairs *purely
    deterministically* -- no RNG, so this is safe to call outside the vocab
    lock -- preferring pairs where the rule **fires** (output != input) so
    the transformation is actually demonstrated, then filling with identity
    pairs. Within each group, pairs are taken in sorted-key order so the
    subset is reproducible and independent of dict iteration order.
    """
    if n_demos <= 0 or n_demos >= len(full_demos):
        return dict(sorted(full_demos.items()))
    firing = sorted((i, o) for i, o in full_demos.items() if o != i)
    identity = sorted((i, o) for i, o in full_demos.items() if o == i)
    chosen = firing[:n_demos]
    if len(chosen) < n_demos:
        chosen = chosen + identity[: n_demos - len(chosen)]
    return dict(sorted(chosen))
