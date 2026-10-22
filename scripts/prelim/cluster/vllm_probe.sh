#!/bin/bash
set -uo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: $0 QROOT SLUG FAMILY [runner args...]" >&2
    exit 2
fi
QROOT=$1
SLUG=$2
FAMILY=$3
shift 3
POOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$POOL_DIR/env.sh"
source "$QROOT/$SLUG/server.env"
export UV_PYTHON=3.13 UV_NO_SYNC=1 DUMMY_KEY=none
cd "$CLONE"
exec srun --jobid "$JOB" --overlap --ntasks 1 \
    uvx --from uv==0.12.3 uv run --no-sync python scripts/prelim/gepa_family.py \
    --family "$FAMILY" --arm naive --task-model "$SLUG" --base-url "$BASE_URL" --api-key-env DUMMY_KEY \
    --temperature 0 --max-tokens 512 --split-seed 0 --band-low 0 --band-high 1 --probe "$@"
