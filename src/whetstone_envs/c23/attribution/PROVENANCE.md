# InductionBench provenance

C23 adapts the single-rule ISL, left-output-strictly-local, and
right-output-strictly-local generation and reference-transducer path from
[InductionBench](https://github.com/Wenyueh/inductive_reasoning_benchmark) at
commit `e0b839221a8509b351b324dfb247b35a434b7fd5` (Hua et al., 2025).

The adapted work is licensed under Apache-2.0; see `LICENSE`. The modified
derivative files `_inductionbench.py` and `_transducers.py` carry prominent
notices. They retain only the on-path algorithm and replace upstream global
configuration and randomness with explicit immutable configuration and an
injected private `random.Random`. C23's hypothesis selection, prompts, pool
integration, and scoring are first-party code.
