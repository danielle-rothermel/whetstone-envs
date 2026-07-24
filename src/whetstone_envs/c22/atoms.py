"""Constraint-atom pools and their deterministic kwarg derivation.

The c22 baseline spec (Section 1) crosses two axes: *constraint count*
(``n in {3, 4, 5}``) and *atom-type mix* (``easy-skewed`` vs. ``mixed``,
where mixed pulls at least one atom from the hard pool). This module
pins the concrete IFEval ``instruction_id`` membership of each pool and,
for every atom, a pure ``(rng) -> kwargs`` function that produces the
explicit keyword arguments passed into the vendored
``build_description`` -- so generation never leans on IFEval's
module-global unseeded ``random`` for a value we need to reproduce.

Determinism exclusions (spec Section 1, "deliberately excluded"):

* every atom whose ``check_following`` calls ``langdetect`` (the
  ``change_case:*`` casing atoms and ``language:response_language``) is
  excluded -- ``langdetect`` returns nondeterministically near the
  detection threshold;
* every atom needing the nltk ``punkt`` sentence tokenizer
  (``length_constraints:number_sentences``) is excluded -- it requires a
  pinned runtime download.

The remaining atoms check via pure regex / counting only. The hard-pool
``length_constraints:number_words`` requires exactly N words using
IFEval's ``count_words``, a ``RegexpTokenizer(r"\\w+")`` that needs no
download.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import Random

# --- Nonce vocab for contamination-proofing -------------------------------
# Explicit keyword lists fed into build_description so no atom draws from
# IFEval's built-in WORD_LIST (which overlaps common English and the
# published dataset). These are invented low-frequency tokens; the exact
# strings are part of the pinned generator behavior.
_NONCE_KEYWORDS = (
    "zylthorn",
    "quarnex",
    "vopflim",
    "jaxbryn",
    "wexcorb",
    "plurnyx",
    "gwentar",
    "brimquol",
)
_NONCE_END_PHRASES = (
    "THUS CONCLUDED",
    "END OF ANSWER",
    "FULLY DONE",
    "SIGNED OFF",
)
# Rare letters make "letter should appear less than 1 time" a genuine
# forbidden-letter constraint that is easy to satisfy deliberately but
# easy to violate by accident.
_RARE_LETTERS = ("z", "q", "x", "j", "k")


def _kw_none(_rng: Random) -> dict[str, object]:
    """No explicit kwargs: the atom's description is fully static."""
    return {}


def _kw_keyword_existence(rng: Random) -> dict[str, object]:
    """One nonce keyword the response must include."""
    return {"keywords": [rng.choice(_NONCE_KEYWORDS)]}


def _kw_forbidden_words(rng: Random) -> dict[str, object]:
    """One nonce keyword the response must NOT include."""
    return {"forbidden_words": [rng.choice(_NONCE_KEYWORDS)]}


def _kw_end_checker(rng: Random) -> dict[str, object]:
    """A fixed end phrase the response must end with."""
    return {"end_phrase": rng.choice(_NONCE_END_PHRASES)}


def _kw_number_words(rng: Random) -> dict[str, object]:
    """An exact count: ``Answer with exactly N words.``"""
    return {"num_words": rng.randint(3, 12), "relation": "exactly"}


def _kw_letter_frequency(rng: Random) -> dict[str, object]:
    """A forbidden rare letter: it must appear fewer than 1 time."""
    return {
        "letter": rng.choice(_RARE_LETTERS),
        "let_frequency": 1,
        "let_relation": "less than",
    }


def _kw_placeholders(rng: Random) -> dict[str, object]:
    """A minimum count of ``[bracketed]`` placeholders."""
    return {"num_placeholders": rng.randint(1, 3)}


def _kw_highlights(rng: Random) -> dict[str, object]:
    """A minimum count of ``*markdown*`` highlighted sections."""
    return {"num_highlights": rng.randint(1, 3)}


_POSTSCRIPT_MARKERS = ("P.S.", "P.P.S")


def _kw_postscript(rng: Random) -> dict[str, object]:
    """A required postscript marker at the end of the response."""
    return {"postscript_marker": rng.choice(_POSTSCRIPT_MARKERS)}


@dataclass(frozen=True, slots=True)
class Atom:
    """One constraint-atom template.

    Parameters
    ----------
    instruction_id:
        The vendored IFEval registry key (e.g. ``keywords:existence``).
    derive_kwargs:
        Pure function mapping a seeded :class:`random.Random` to the
        explicit ``build_description`` kwargs. Passing values in keeps
        generation reproducible despite IFEval's module-global RNG.
    """

    instruction_id: str
    derive_kwargs: Callable[[Random], dict[str, object]]


# --- Easy pool ------------------------------------------------------------
# casing/format-wrapper, keyword-presence, start/end-token -- per-atom pass
# ~0.80-0.90 per the spec. All pure regex checks, no langdetect/punkt.
EASY_POOL: tuple[Atom, ...] = (
    Atom("keywords:existence", _kw_keyword_existence),
    Atom("startend:end_checker", _kw_end_checker),
    Atom("detectable_format:title", _kw_none),
    Atom("startend:quotation", _kw_none),
    Atom("punctuation:no_comma", _kw_none),
    Atom("detectable_content:number_placeholders", _kw_placeholders),
    Atom("detectable_content:postscript", _kw_postscript),
    Atom("detectable_format:number_highlighted_sections", _kw_highlights),
)

# --- Hard pool ------------------------------------------------------------
# exact-word-count-equals-N and forbidden-letter across the whole output
# -- per-atom pass ~0.55-0.75 per the spec.
HARD_POOL: tuple[Atom, ...] = (
    Atom("length_constraints:number_words", _kw_number_words),
    Atom("keywords:letter_frequency", _kw_letter_frequency),
    Atom("keywords:forbidden_words", _kw_forbidden_words),
)

_BY_ID: Mapping[str, Atom] = {
    a.instruction_id: a for a in (*EASY_POOL, *HARD_POOL)
}


def atom_for(instruction_id: str) -> Atom:
    """Return the :class:`Atom` for a registry id, or raise ``KeyError``."""
    return _BY_ID[instruction_id]


def all_atom_ids() -> tuple[str, ...]:
    """Every atom id in the easy and hard pools, easy first."""
    return tuple(a.instruction_id for a in (*EASY_POOL, *HARD_POOL))
