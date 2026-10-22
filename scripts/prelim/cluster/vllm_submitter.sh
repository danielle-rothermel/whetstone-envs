#!/bin/bash
set -uo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: $0 ACCOUNT PARTITION QROOT [POLL_S]" >&2
    exit 2
fi
ACCOUNT=$1
PARTITION=$2
QROOT=$3
POLL_S=${4:-120}
POOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$POOL_DIR/env.sh"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
mkdir -p "$QROOT/pool-logs"
ts() { date '+%Y-%m-%dT%H:%M:%S'; }
while :; do
    active=$(squeue -h -u "$USER" -p "$PARTITION" -o '%j' | grep -c "^$POOL_PREFIX")
    remaining=0
    while IFS=$'\t' read -r slug _; do
        pending=$(find "$QROOT/$slug/pending" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
        (( pending == 0 )) && continue
        remaining=$((remaining + 1))
        squeue -h -u "$USER" -n "$POOL_PREFIX$slug" -o '%i' | grep -q . && continue
        attempts=$(cat "$QROOT/$slug/attempts" 2>/dev/null || echo 0)
        (( attempts >= MAX_ATTEMPTS )) && continue
        (( active >= CILVR_MAX_POOLS )) && break
        jid=$(sbatch --parsable --account "$ACCOUNT" --partition "$PARTITION" \
            --job-name "$POOL_PREFIX$slug" --output "$QROOT/pool-logs/$POOL_PREFIX$slug-%j.out" \
            --export "ALL,QROOT=$QROOT,MODEL_SLUG=$slug,POOL_DIR=$POOL_DIR" "$POOL_DIR/vllm_pool.sbatch")
        echo $((attempts + 1)) > "$QROOT/$slug/attempts"
        echo "[$(ts)] submitted $slug job=$jid attempt=$((attempts + 1))"
        active=$((active + 1))
    done < "$QROOT/models.tsv"
    if (( remaining == 0 )); then
        echo "[$(ts)] no pending jobs; exiting"
        exit 0
    fi
    [[ -f "$QROOT/STOP" ]] && { echo "[$(ts)] STOP file; exiting"; exit 0; }
    sleep "$POLL_S"
done
