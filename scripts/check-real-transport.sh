#!/usr/bin/env bash
#
# Manual real-transport smoke ladder: one rung per optimizer arm, end to end
# on the REAL OpenRouter transport, on toy c19/c18 splits.
#
# Standing rule (plan note 21): nothing reaches the real experiments having
# been tested only against mocks. Every rung drives the public runner and
# asserts on persisted state -- result.json, runtime.sqlite, a passing
# audit_run, priced run_cost rows, a rendered trajectory report -- because
# persisted state is what a study stage and the audit actually read.
#
# This script SPENDS REAL MONEY. It is never wired into `pre-check.sh` or the
# default CI run; it is invoked deliberately:
#
#     mise exec -- bash scripts/check-real-transport.sh
#     mise exec -- bash scripts/check-real-transport.sh -k rung1
#
# A transcript and a rung table land under
# ~/drotherm/data/whetstone-envs/real-transport/<timestamp>/.

set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"

selector=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)
            selector="${2:-}"
            if [[ -z "${selector}" ]]; then
                echo "-k needs a pytest selector expression" >&2
                exit 2
            fi
            shift 2
            ;;
        -h | --help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# The credential is the one thing this script cannot supply for itself.
# Failing here, before any run directory exists, keeps an un-keyed
# invocation from looking like a passing ladder.
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
OPENROUTER_API_KEY is unset, so no rung could reach the real transport.
The key is available through mise; invoke this script as

    mise exec -- bash scripts/check-real-transport.sh

Never echo the key.
EOF
    exit 2
fi

# The real-Codex opt-in belongs to a different ladder and a different kind of
# spend. Refusing it here means this script can only ever bill the provider.
if [[ -n "${WHETSTONE_ENVS_ALLOW_REAL_CODEX:-}" ]]; then
    echo "WHETSTONE_ENVS_ALLOW_REAL_CODEX is set; the real-transport ladder" \
        "spends on the provider only and must not spawn the billed Codex" \
        "CLI. Unset it and re-run." >&2
    exit 2
fi

# The repo's scripts drive `uv` directly, matching pre-check.sh and CI.
# WHETSTONE_ENVS_UV overrides it so a caller pinning a uv version (for
# example `uvx --from 'uv==0.11.25' uv`) can run this ladder unchanged.
uv_cmd=(${WHETSTONE_ENVS_UV:-uv})

timestamp="$(date +%Y%m%d-%H%M%S)"
out_root="${HOME}/drotherm/data/whetstone-envs/real-transport/${timestamp}"
mkdir -p -- "${out_root}"
transcript="${out_root}/transcript.txt"
rung_table="${out_root}/rungs.md"

echo "real-transport ladder ${timestamp}"
echo "transcript: ${transcript}"

pytest_args=(-v -s tests/real_transport)
if [[ -n "${selector}" ]]; then
    pytest_args+=(-k "${selector}")
fi

set +e
WHETSTONE_ENVS_REAL_TRANSPORT=1 \
    "${uv_cmd[@]}" run --python 3.13 --extra c18 --extra optim \
    pytest "${pytest_args[@]}" \
    2>&1 | tee -- "${transcript}"
status="${PIPESTATUS[0]}"
set -e

# The rung table is projected from the transcript's ledger lines, so it
# reports what the runs actually recorded rather than a restatement.
"${uv_cmd[@]}" run --python 3.13 --extra c18 --extra optim \
    python - "${transcript}" "${rung_table}" \
    <<'PYTHON'
import json
import sys
from pathlib import Path

transcript, destination = (Path(argument) for argument in sys.argv[1:3])

ledgers = []
for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
    marker = "RUNG_LEDGER "
    index = line.find(marker)
    if index == -1:
        continue
    try:
        ledgers.append(json.loads(line[index + len(marker) :]))
    except json.JSONDecodeError:
        continue

lines = [
    "# Real-transport smoke rungs",
    "",
    "| rung | run id | calls | priced | in tok | out tok | USD |",
    "|------|--------|-------|--------|--------|---------|-----|",
]
total = 0.0
for ledger in ledgers:
    spend = ledger.get("spend", [])
    calls = sum(record["calls"] for record in spend)
    priced = sum(record["priced_calls"] for record in spend)
    input_tokens = sum(record["input_tokens"] for record in spend)
    output_tokens = sum(record["output_tokens"] for record in spend)
    usd = ledger.get("usd") or 0.0
    total += usd
    lines.append(
        f"| {ledger['rung']} | {ledger.get('run_id', '-')} | {calls} | "
        f"{priced} | {input_tokens} | {output_tokens} | {usd:.6f} |"
    )
lines += ["", f"**Total ledgered USD: {total:.6f}** across {len(ledgers)} rungs."]
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PYTHON

echo
echo "rung table: ${rung_table}"
exit "${status}"
