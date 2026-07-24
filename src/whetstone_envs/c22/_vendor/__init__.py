"""Vendored third-party code for the c22 candidate.

This subpackage carries a pinned snapshot of Google Research's IFEval
constraint-checker library (``instruction_following_eval``), reused as
both the generation-side constraint sampler and the scoring oracle.
Small, reviewable patches isolate its imports under this namespace and
add C22's exact-word-count relation.

Provenance and license live in
``instruction_following_eval/PROVENANCE.md`` and
``instruction_following_eval/LICENSE``.
"""
