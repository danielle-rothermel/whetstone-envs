"""Task generators, independent oracles, and probe prompts for
whetstone's quick-test optimizer benchmarks.

Each task family (one per candidate: c11 JSON canonicalization, c18
PrOntoQA, c19 Minigrid, c22 stacked IFEval constraints, c23 subregular
rule induction) is a self-contained generator + oracle + probe-prompt
module, kept free of any dependency on whetstone's optimizer or
execution-contract code so it can be developed and tested in isolation.
"""

__all__: list[str] = []
