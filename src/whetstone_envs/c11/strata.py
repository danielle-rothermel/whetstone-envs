"""Per-stratum adversarial instance builders for c11.

Each stratum from the baseline spec's Section 1 table stresses one JCS
sub-rule. A builder is a pure function of a seeded ``random.Random``
returning the *messy* input JSON string the model will see. The messy
string is deliberately non-canonical (spaces after ``:`` and ``,``,
insertion order that is not UTF-16-sorted) so that its canonical form
genuinely differs -- the generator rejects any candidate whose messy and
canonical forms coincide, since then the task would be trivial for that
instance (spec Section 1: "S2 instances are rejected if their keys are
already sorted").

The builders never compute the canonical form themselves; that is the
oracle's job. They only produce inputs and declare, per stratum, the
adversarial predicate an instance must satisfy to actually exercise its
sub-rule.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from collections.abc import Callable

# Stratum labels (spec Section 1 table).
S1_FLAT = "S1_flat"
S2_KEYSORT = "S2_keysort"
S3_NUMBER = "S3_number"
S4_UNICODE = "S4_unicode"
S5_MIXED = "S5_mixed"

STRATA: tuple[str, ...] = (
    S1_FLAT,
    S2_KEYSORT,
    S3_NUMBER,
    S4_UNICODE,
    S5_MIXED,
)

# A messy separator style: a space after every ``:`` and ``,`` and, for
# objects, spaces inside the braces. json.dumps with these separators plus
# a manual brace pad gives an input that is never already canonical.
_MESSY_SEPARATORS = (", ", ": ")

# Small fixed vocabularies drawn from deterministically. Kept ASCII for
# S1/S2 keys so only the targeted sub-rule (sort/number) is exercised
# there; S4/S5 introduce the escaping and unicode content.
_ASCII_KEYS: tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "omega",
)
_ASCII_WORDS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "amber",
    "cyan",
    "coral",
    "ivory",
    "slate",
)
# Strings that force escaping or non-ASCII handling (S4). Each contains a
# non-ASCII character that the messy serializer escapes but rfc8785 emits
# as literal UTF-8. Some also cover exact escaping of controls, quotes, and
# backslashes.
_ESCAPE_STRINGS: tuple[str, ...] = (
    'quote " inside-é',
    "back\\slash-π",
    "tab\tsep-ü",
    "new\nline-雪",
    "carriage\rreturn-é",
    "ctrl\x01char-λ",
    "unicode-é-accent",
    "emoji-\U0001f600-astral",
    "mix\tü\\end",
)

# Generator contract: these exact JSON number tokens enter persisted public
# inputs. They are literals, never float-derived, so their exponent notation
# survives until the oracle parses it. The manifest and c11 regression tests
# pin this tuple's effect on generated identity.
S3_NONCANONICAL_EXPONENT_LEXEMES: tuple[str, ...] = (
    "1e2",
    "1E+2",
    "1.00e+2",
    "100e-2",
    "15e+1",
)


def _messy_dumps(obj: object, *, escape_non_ascii: bool = False) -> str:
    """Serialize ``obj`` to a deliberately non-canonical JSON string.

    Uses padded separators so the result always carries insignificant
    whitespace -- guaranteeing it differs from the zero-whitespace
    canonical form. S4/S5 additionally escape non-ASCII characters in the
    input so their escape policy genuinely differs from JCS's literal
    UTF-8 output.
    """
    return json.dumps(
        obj,
        separators=_MESSY_SEPARATORS,
        ensure_ascii=escape_non_ascii,
    )


def _unsorted_object(rng: random.Random, keys: list[str]) -> str:
    """Build an object string whose key insertion order is not sorted.

    Picks values for the given keys and emits them in a shuffled order
    that is asserted (by the caller's adversarial predicate) to differ
    from UTF-16-sorted order.
    """
    ordered = list(keys)
    rng.shuffle(ordered)
    if ordered == sorted(ordered):
        ordered[0], ordered[1] = ordered[1], ordered[0]
    pairs = {k: rng.choice(_ASCII_WORDS) for k in ordered}
    # dict preserves insertion order; re-key in the shuffled order.
    obj = {k: pairs[k] for k in ordered}
    return _messy_dumps(obj)


def build_s1(rng: random.Random) -> str:
    """S1 flat/shallow: few keys, already-sorted, ASCII, ints only.

    The only non-canonical property is the insignificant whitespace the
    messy separators introduce; there is no sort/number/escape tension,
    so S1 isolates the "zero insignificant whitespace" rule.
    """
    n = rng.randint(2, 4)
    keys = sorted(rng.sample(_ASCII_KEYS, n))
    obj = {k: rng.randint(0, 9) for k in keys}
    return _messy_dumps(obj)


def build_s2(rng: random.Random) -> str:
    """S2 key-sort stress: many keys in non-sorted insertion order."""
    n = rng.randint(4, 7)
    keys = rng.sample(_ASCII_KEYS, n)
    return _unsorted_object(rng, keys)


def build_s3(rng: random.Random) -> str:
    """S3 number canonicalization with guaranteed lexical exponent tension.

    At least one value is emitted from a hand-authored JSON number token
    containing an exponent whose spelling is not its ECMAScript canonical
    spelling. The token is inserted lexically rather than passed through
    :func:`json.dumps`, which would first turn it into a Python float and
    could erase the exponent notation this stratum promises to test.
    """
    keys = sorted(rng.sample(_ASCII_KEYS, rng.randint(2, 4)))
    exponent_key = rng.choice(keys)
    other_lexemes: tuple[str, ...] = (
        "1.0",
        "100.0",
        "-0.0",
        *S3_NONCANONICAL_EXPONENT_LEXEMES,
    )
    members = [
        (
            f"{json.dumps(key)}: "
            + (
                rng.choice(S3_NONCANONICAL_EXPONENT_LEXEMES)
                if key == exponent_key
                else rng.choice(other_lexemes)
            )
        )
        for key in keys
    ]
    return "{" + ", ".join(members) + "}"


def build_s4(rng: random.Random) -> str:
    """S4 unicode/escaping: strings needing exact JCS escape policy."""
    keys = sorted(rng.sample(_ASCII_KEYS, rng.randint(2, 4)))
    obj = {k: rng.choice(_ESCAPE_STRINGS) for k in keys}
    return _messy_dumps(obj, escape_non_ascii=True)


def build_s5(rng: random.Random) -> str:
    """S5 mixed/deep: a nested combination of S2 + S3 + S4 tensions."""
    outer_keys = rng.sample(_ASCII_KEYS, 3)
    # An unsorted top level (S2), one numeric leaf group (S3), one escape
    # leaf group (S4), and a nested sorted-later object.
    nested_keys = sorted(rng.sample(_ASCII_KEYS, 2))
    nested = {k: rng.choice(_ESCAPE_STRINGS) for k in nested_keys}
    numbers = [1.0, -0.0, 1e2, rng.randint(1, 1000)]
    obj = {
        outer_keys[0]: rng.choice(_ESCAPE_STRINGS),
        outer_keys[1]: numbers,
        outer_keys[2]: nested,
    }
    # Re-emit with a deliberately non-sorted outer order.
    shuffled_keys = list(obj)
    rng.shuffle(shuffled_keys)
    if shuffled_keys == sorted(shuffled_keys):
        shuffled_keys[0], shuffled_keys[1] = (
            shuffled_keys[1],
            shuffled_keys[0],
        )
    shuffled = {key: obj[key] for key in shuffled_keys}
    return _messy_dumps(shuffled, escape_non_ascii=True)


BUILDERS: dict[str, Callable[[random.Random], str]] = {
    S1_FLAT: build_s1,
    S2_KEYSORT: build_s2,
    S3_NUMBER: build_s3,
    S4_UNICODE: build_s4,
    S5_MIXED: build_s5,
}
