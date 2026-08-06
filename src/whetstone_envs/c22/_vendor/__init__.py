"""Vendored third-party runtime for C22.

This subpackage carries a pinned snapshot of Google Research's IFEval
constraint-checker library (``instruction_following_eval``). C22 uses it to
derive constraint descriptions and evaluate candidate responses. Small,
reviewable patches isolate its imports under this namespace and add C22's
exact-word-count relation.

Provenance and license live in
``instruction_following_eval/PROVENANCE.md`` and
``instruction_following_eval/LICENSE``.
"""
