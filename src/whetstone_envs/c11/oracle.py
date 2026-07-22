"""The RFC 8785 (JCS) canonicalization oracle for c11.

A pure function of an instance's *public* field -- the messy input JSON
string -- and nothing else. It parses that string with the stdlib
``json`` decoder, then delegates the entire canonicalization to
trailofbits ``rfc8785.dumps`` **strictly unmodified** (spec Section 7.7:
the oracle is used as-is for the plain-JCS baseline). It never consults
the generator's internal state, so it cannot silently become a
re-derivation of how an instance was built (rubric criteria 2 and 8, "no
tautology").

The two entry points differ only in what they return on an input the
canonicalizer cannot express (an integer outside the IEEE-754 safe
domain, ``NaN``/``Infinity``, etc.):

* :func:`canonicalize` raises -- the generator uses it and *wants* the
  loud failure, since it must never mint an instance whose gold cannot
  be produced.
* :func:`score` swallows that failure into a ``0`` score, because a
  model response is graded, not trusted to be canonicalizable.
"""

from __future__ import annotations

import json

import rfc8785

from whetstone_envs.core.probes import normalize


def canonicalize(input_json: str) -> str:
    """Return the RFC 8785 canonical string for a messy ``input_json``.

    The messy string is parsed once with :func:`json.loads`, then handed
    to ``rfc8785.dumps`` unmodified; the resulting canonical bytes are
    decoded as UTF-8 (RFC 8785 output is always valid UTF-8). This is the
    ceiling definition the model reproduces and the ground truth the
    generator freezes as ``Instance.gold``.

    Raises whatever ``json`` or ``rfc8785`` raise: a malformed input, or
    a value outside the JCS-expressible domain (e.g. an integer beyond
    ``2**53 - 1``, which ``rfc8785`` rejects with ``IntegerDomainError``).
    The generator relies on that: an input it cannot canonicalize must
    never become an instance.
    """
    obj = json.loads(input_json)
    return rfc8785.dumps(obj).decode("utf-8")


def score(prediction: str, input_json: str) -> int:
    """Return 1 iff ``prediction`` is the canonical form of ``input_json``.

    Both the model's ``prediction`` and the freshly canonicalized gold
    are passed through the shared
    :func:`whetstone_envs.core.probes.normalize` (strip surrounding
    whitespace and a single wrapping code fence) so scoring differences
    come from the model, not from per-candidate string handling. Whole
    string, exact match, no partial credit (rubric criterion 2).

    An input the oracle cannot canonicalize -- or a prediction that is
    not valid against it -- scores ``0`` rather than raising, since a
    model response is graded, not trusted.
    """
    try:
        gold = canonicalize(input_json)
    except (json.JSONDecodeError, rfc8785.CanonicalizationError):
        return 0
    return int(normalize(prediction) == normalize(gold))


def score_gold(prediction: str, gold: str) -> int:
    """Return the 0/1 score of ``prediction`` against a frozen ``gold``.

    The pool-facing entry point that mirrors the other candidates'
    ``score_gold``: given an instance's already-canonical ``gold`` string
    and a model response, return 0 or 1 by normalized exact match. Use
    :func:`score` instead when scoring straight from the messy input.
    """
    return int(normalize(prediction) == normalize(gold))
