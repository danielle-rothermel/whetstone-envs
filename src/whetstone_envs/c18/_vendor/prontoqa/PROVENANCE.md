# Vendored: asaparov/prontoqa (PrOntoQA generator)

- **Upstream:** https://github.com/asaparov/prontoqa
- **Vendored commit:** `0a6412b6fddf46324a1cb96e066dd7b3d89b87d6`
  ("Simplified the bibtex for the NeurIPS paper.")
- **License:** Apache-2.0 (see `LICENSE` in this directory).
- **Vendored on:** 2026-07-22, for whetstone-envs candidate **c18**
  (PrOntoQA deductive entailment).

## Why vendored

C18 reseeds instances directly from
`asaparov/prontoqa`'s `run_experiment.py --model-name json` (fresh
`--seed` per configured depth stratum, `--ontology fictional` nonce
ontologies) and must never reuse a published instance. Vendoring pins the
exact generator version so a regenerated pool is byte-reproducible against
a frozen manifest, and so the `--model-name json` generation path has no
network or upstream-checkout dependency. The generator is driven as a
subprocess behind `whetstone_envs.c18.upstream` (the boundary), never
imported into the c18 Python package.

## What was and was not changed

Byte-for-byte copies of upstream (unmodified) -- the minimal set the
`--model-name json` generation path imports and reads:

- `run_experiment.py` (entry point / `__main__`)
- `theory.py`, `syntax.py`, `proof.py`, `prompt.py`, `fol.py`
  (the generation-path modules `run_experiment` imports by bare name)
- `bad_patterns.txt` (opened by `run_experiment.py` at import time via a
  relative path)
- `LICENSE`

Not vendored (upstream files that only the *model-evaluation* path uses,
never reached in `--model-name json` mode): `analyze_results.py`,
`make_plots.py`, `gpt3.py`, `opt.py`, `unifiedqa.py`, the bundled
`*.zip` result archives, and the upstream `README.md`.

Added by this vendor (not present upstream):

- `PROVENANCE.md` -- this file.
- `.gitignore` -- ignores the generator's json/log output files, which
  are written to the process cwd. The boundary copies the required runtime
  files into a throwaway temp directory, so output never lands here; the
  ignore is defensive.

No line of upstream generator logic was edited. The two upstream
integration gotchas (the relative `bad_patterns.txt` open and the
cwd-written output filename) are handled entirely in
`whetstone_envs.c18.upstream` by running the subprocess in a temp working
directory, never by editing the vendored source.

## Runtime dependencies

The json-generation path needs only `numpy` and `scipy` (both declared in
the repo's `pyproject.toml`). `scipy` is a hard top-level import of
`run_experiment.py`/`prompt.py` (via `scipy.special.betaincinv` /
`logsumexp`) even though only the unused eval path calls those functions,
so it is a required dependency for generation. `torch` / `transformers` /
`nltk` / `matplotlib` are NOT required (eval-path only) and are not
installed for this path.

## Known upstream caveats

- `--seed` threads into both stdlib `random` and `np.random`; two runs at
  a fixed seed are byte-identical (verified).
- The label is *definitional* (a 50 % negation flag), not an independent
  prover verdict -- c18 re-derives it with an independent forward-chaining
  oracle and asserts agreement at construction.
- `--deduction-rule Composed` with the default `postorder` ordering
  crashes upstream; c18 uses only the default `ModusPonens` path, so this
  is not reached.
- Small concept pools at high hop counts print harmless
  `WARNING: Could not extend ontology...` retry-by-rejection lines to
  stderr; generation stays deterministic. The boundary captures stderr
  and does not propagate it.
