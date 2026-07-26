"""Vendored third-party code for the c18 candidate.

This subpackage carries a byte-for-byte copy of the generation-path
source of `asaparov/prontoqa` (the PrOntoQA fictional-ontology
generator), reused unmodified. Unlike c22's vendored checker, this code
is **not imported** into the c18 Python package: it is driven as a
subprocess by :mod:`whetstone_envs.c18.upstream`, which runs the vendored
`run_experiment.py --model-name json` in a throwaway working directory.

Provenance and license live in `prontoqa/PROVENANCE.md` and
`prontoqa/LICENSE`.
"""
