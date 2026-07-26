"""Hand-picked rule fixtures per type (ISL / L-OSL / R-OSL), checklist A.

Each fixture is a hand-constructed ``(rule, held-out query) -> expected
output`` triple whose expected output was derived by *manual rule
application* (the derivation is written out in each test's comment), NOT by
running the generator. Asserting the oracle reproduces the hand-traced
output is what catches an oracle that is silently a re-derivation of
generator internals rather than a true independent apply-the-rule check
(PLAN Verification A, c23 bullet).

The oracle delegates to the vendored ``apply_ISL_rule`` /
``apply_L_OSL_rule`` / ``apply_R_OSL_rule`` transducers UNMODIFIED; a
separate test in this file pins that those functions are byte-identical to
the upstream source, so the fixtures below validate the reference we reuse.
"""

from __future__ import annotations

from whetstone_envs.c23 import oracle, upstream


# --- ISL fixtures --------------------------------------------------------
# ISL: the output char at position i is decided by the last k chars of the
# INPUT up to and including i. rule {'cb': 'd'} rewrites the 'b' to 'd'
# whenever the two-char input suffix is 'cb'.
#
# manual trace, rule {'cb':'d'} on 'acb' (k=2):
#   i=0 'a': suffix 'a' not a rule -> emit 'a'
#   i=1 'c': suffix 'c','ac' not rules -> emit 'c'
#   i=2 'b': suffix 'b' no, suffix 'cb' -> rule -> emit 'd'
#   => 'acd'
def test_isl_rewrites_on_input_suffix() -> None:
    assert oracle.apply_to_query(upstream.ISL, 2, {"cb": "d"}, "acb") == "acd"


# manual trace, rule {'cb':'d'} on 'cbcb' (k=2):
#   'c'; suffix 'cb'->'d'; 'c'; suffix 'cb'->'d' => 'cdcd'
def test_isl_fires_repeatedly() -> None:
    out = oracle.apply_to_query(upstream.ISL, 2, {"cb": "d"}, "cbcb")
    assert out == "cdcd"


# manual trace, rule {'aa':'b'} on 'aaa' (k=2):
#   i=0 'a'; i=1 suffix 'aa'->'b'; i=2 input[:3]='aaa' suffix 'a' no, 'aa'->'b'
#   => 'abb'
def test_isl_context_is_the_raw_input() -> None:
    assert oracle.apply_to_query(upstream.ISL, 2, {"aa": "b"}, "aaa") == "abb"


# --- L-OSL fixtures ------------------------------------------------------
# L-OSL: context is the OUTPUT built so far (left-to-right). After emitting
# each input char, if the output suffix matches a rule's left side, the last
# emitted char is replaced by the rule's right side.
#
# manual trace, rule {'ab':'c'} on 'abab' (k=2):
#   emit 'a' -> 'a'
#   emit 'b' -> 'ab'; output suffix 'ab' -> replace last -> 'ac'
#   emit 'a' -> 'aca'
#   emit 'b' -> 'acab'; suffix 'ab' -> replace last -> 'acac'
#   => 'acac'
def test_losl_rewrites_on_output_suffix() -> None:
    assert (
        oracle.apply_to_query(upstream.L_OSL, 2, {"ab": "c"}, "abab") == "acac"
    )


# manual trace, rule {'cb':'d'} on 'acb' (k=2):
#   'a'; 'ac'; emit 'b'->'acb'; suffix 'cb' -> replace -> 'acd'
#   => 'acd'
def test_losl_single_fire() -> None:
    out = oracle.apply_to_query(upstream.L_OSL, 2, {"cb": "d"}, "acb")
    assert out == "acd"


# --- R-OSL fixtures ------------------------------------------------------
# R-OSL: reverse the input, run the L-OSL-style output-context rewrite on
# the reversed string, then reverse the result. Context is thus the OUTPUT
# succeeding a char in the original orientation.
#
# manual trace, rule {'ab':'d'} on 'abab' (k=2):
#   reverse('abab') = 'baba'
#   L-OSL over 'baba' with rule {'ab':'d'}:
#     'b'; emit 'a'->'ba' (suffix 'ba' no rule); emit 'b'->'bab' (suffix 'ab'
#       -> replace last -> 'bad'); emit 'a'->'bada' (suffix 'da' no)
#     => 'bada'
#   reverse('bada') = 'adab'
#   => 'adab'
def test_rosl_rewrites_on_reversed_output_context() -> None:
    assert (
        oracle.apply_to_query(upstream.R_OSL, 2, {"ab": "d"}, "abab") == "adab"
    )


# manual trace, rule {'cb':'d'} on 'cba' (k=2):
#   reverse('cba')='abc'; L-OSL over 'abc' rule {'cb':'d'}:
#     'a';'ab' (no);'abc' (suffix 'bc' no) => 'abc' unchanged
#   reverse('abc')='cba' => identity here (rule never forms 'cb' in the
#   reversed output), a legitimate non-firing case.
def test_rosl_non_firing_is_identity() -> None:
    out = oracle.apply_to_query(upstream.R_OSL, 2, {"cb": "d"}, "cba")
    assert out == "cba"


# --- deletion rule (right side empty string) -----------------------------
# The generator allows deletion (empty output). rule {'ca':''} on 'bca'
# ISL: 'b'; 'bc' no; suffix 'ca' -> '' (deletion) => 'bc'
def test_isl_deletion_rule() -> None:
    assert oracle.apply_to_query(upstream.ISL, 2, {"ca": ""}, "bca") == "bc"


# --- scoring entry points ------------------------------------------------
def test_score_is_exact_match_against_reapplied_rule() -> None:
    # score() re-derives gold via the vendored transducer and 0/1-compares.
    assert oracle.score("acd", upstream.ISL, 2, {"cb": "d"}, "acb") == 1
    assert oracle.score("acb", upstream.ISL, 2, {"cb": "d"}, "acb") == 0


def test_score_gold_normalizes_both_sides() -> None:
    # A fenced / whitespace-wrapped prediction still matches the bare gold.
    assert oracle.score_gold("```\nacd\n```", "acd") == 1
    assert oracle.score_gold("  acd  ", "acd") == 1
    assert oracle.score_gold("acb", "acd") == 0


# --- Output-line extraction (spec Section 2 decision rule) ---------------
# The ceiling probe (spec Section 2.2) instructs the model to emit "the
# transformed string, on a single line prefixed with 'Output:'". The scorer
# previously exact-matched the WHOLE reply, so every Output:-prefixed reply
# scored 0 (live: ceiling 0/30). These fixtures pin the prescribed
# extraction -- the text after the LAST Output: line -- across the real live
# shapes: Output:-prefixed, bare string (naive), and a multi-Output reply.
def test_extract_output_prefixed_line() -> None:
    assert oracle.extract_output("Output: acd") == "acd"
    # Case-insensitive prefix; the whitespace between the colon and the
    # answer is consumed (the spec strips surrounding whitespace anyway).
    assert oracle.extract_output("OUTPUT:   acd") == "acd"


def test_extract_output_bare_string_has_no_prefix() -> None:
    # Naive-probe bare emission: no Output: line -> whole text unchanged.
    assert oracle.extract_output("acd") == "acd"
    assert oracle.extract_output("VEZ vez fam qor") == "VEZ vez fam qor"


def test_extract_output_takes_the_last_of_several() -> None:
    # A reply that reasons aloud and emits several Output: lines: the LAST
    # is the answer.
    reply = "Output: abc\nOn reflection the rule duplicates.\nOutput: acd"
    assert oracle.extract_output(reply) == "acd"


def test_extract_output_preserves_internal_spaces() -> None:
    # The transformed string may contain spaces (space-separated tokens);
    # extraction must not split or collapse them.
    reply = "Output: VEZ vez fam qor"
    assert oracle.extract_output(reply) == "VEZ vez fam qor"


def test_score_gold_extracts_output_prefixed_reply() -> None:
    # The regression: an Output:-prefixed ceiling reply now scores 1.
    assert oracle.score_gold("Output: acd", "acd") == 1
    assert oracle.score_gold("Output: acb", "acd") == 0


def test_score_gold_multi_output_scores_on_last() -> None:
    reply = "Output: abc\nActually, reconsidering:\nOutput: acd"
    assert oracle.score_gold(reply, "acd") == 1
    assert oracle.score_gold(reply, "abc") == 0


def test_score_gold_bare_naive_emission_still_scores() -> None:
    # No Output: line (naive bare emission) -> whole reply compared.
    assert oracle.score_gold("acd", "acd") == 1


def test_score_gold_fenced_output_prefixed_reply() -> None:
    # An Output: line whose string is code-fenced on the same shape the
    # shared normalize handles: extraction yields the token, normalize
    # strips a wrapping fence around a bare emission.
    assert oracle.score_gold("Output: `acd`", "`acd`") == 1
    assert oracle.score_gold("```\nacd\n```", "acd") == 1


def test_score_gold_space_separated_output_matches() -> None:
    gold = "VEZ vez fam qor"
    assert oracle.score_gold("Output: VEZ vez fam qor", gold) == 1
    assert oracle.score_gold("Output: vez fam qor", gold) == 0


def test_oracle_reuses_the_vendored_transducers_unmodified() -> None:
    # The oracle must call the vendored apply_*_rule functions, not a
    # reimplementation. Assert the boundary dispatches to the exact vendored
    # module objects (import them the way the vendor is imported).
    import synthetic_data_generation as sdg  # vendored, on sys.path

    args_like = type("A", (), {"type": upstream.ISL, "k": 2})()
    # Same result whether we call the oracle or the vendored function raw.
    assert oracle.apply_to_query(
        upstream.ISL,
        2,
        {"cb": "d"},
        "acb",
    ) == sdg.apply_ISL_rule(args_like, {"cb": "d"}, "acb")
