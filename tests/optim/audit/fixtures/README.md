# Codex audit fixtures

Two real Codex-direct runs, in the shape `optim/audit` reads: `result.json`
beside `runtime.sqlite`.

| Fixture | What it is |
|---|---|
| `codex-completed/` | Two admitted evaluations, the second selected and returned. Every Codex invariant passes. |
| `codex-failed/` | One admitted evaluation, plus an artifact naming a call that was never issued. The Step fails under `codex_selection_unevaluated` and still keeps its one paid evaluation reachable as Tool Evidence. |

They are **committed rather than generated at test time** so the audit's
read paths are exercised against a real recorded run without the audit's
own tests depending on a whetstone-ai source checkout being present.

Because they are recorded, they carry a pinned `EvalEvidence` schema
version and must be **regenerated whenever whetstone-ai changes it** —
the audit validates the stored records, so a fixture written under an
older version fails `reported_numbers_resolve` with "cites
whetstone.eval_evidence which is not eval evidence". They currently carry
v6 (whetstone-ai 0.1.13).

## Regenerating

`generate.py` builds both from a whetstone-ai checkout at the version
this package pins:

    PYTHONPATH=<whetstone-ai>/src python generate.py <whetstone-ai> <out-dir>

Copy `result.json` and `runtime.sqlite` from each generated run directory
over the committed one.

The agent is an in-process `CodexRunner` stand-in rather than the
subprocess fake CLI. The admission authority, the evaluating executor,
the effect leases, the ledger, and the adapter's own reconciliation are
the production path either way; the subprocess only changes how the
agent's scripted decisions arrive, and it is macOS-sandbox gated.
