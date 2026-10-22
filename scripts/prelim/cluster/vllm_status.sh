#!/bin/bash
set -uo pipefail
QROOT=$1
POOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$POOL_DIR/env.sh"
squeue -u "$USER" -o "%.10i %.32j %.8T %.10M %.12P %R" | grep -E "JOBID|$POOL_PREFIX"
printf '%-30s %7s %7s %7s %7s\n' model pending running done failed
while IFS=$'\t' read -r slug _; do
    c=()
    for s in pending running done failed; do c+=("$(find "$QROOT/$slug/$s" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"); done
    printf '%-30s %7s %7s %7s %7s\n' "$slug" "${c[@]}"
done < "$QROOT/models.tsv"
