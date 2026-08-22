# Codex audit fixtures

Two real Codex-direct runs, in the shape `optim/audit` reads: `result.json`
beside `runtime.sqlite`.

| Fixture | What it is |
|---|---|
| `codex-completed/` | Two admitted evaluations, the second selected and returned. Every Codex invariant passes. |
| `codex-failed/` | One admitted evaluation, plus an artifact naming a call that was never issued. The Step fails under `codex_selection_unevaluated` and still keeps its one paid evaluation reachable as Tool Evidence. |

They are **committed rather than generated at test time** because the
installed whetstone-ai (0.1.6) carries no Codex-direct surface: the
adapter that produces such a run only exists on 0.1.7. Committing them
lets the audit's read paths be exercised the moment 0.1.7 lands, without
the audit's own tests depending on which tip happens to be installed.

## Regenerating

`generate.py` builds both. It needs a whetstone-ai checkout carrying the
Codex-direct surface **and** `OptimStepResult.proposer_usage` — at the
time of writing that is the merge of `08-22-codex` into the 0.1.6 tip,
not either branch alone (see the note in `test_codex.py`).

    PYTHONPATH=<whetstone-ai>/src python generate.py <whetstone-ai> <out-dir>

The agent is an in-process `CodexRunner` stand-in rather than the
subprocess fake CLI. The admission authority, the evaluating executor,
the effect leases, the ledger, and the adapter's own reconciliation are
the production path either way; the subprocess only changes how the
agent's scripted decisions arrive, and it is macOS-sandbox gated.
