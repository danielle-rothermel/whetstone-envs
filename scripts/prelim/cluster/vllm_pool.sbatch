#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -uo pipefail
: "${QROOT:?QROOT must be set}"
: "${MODEL_SLUG:?MODEL_SLUG must be set}"
: "${POOL_DIR:?POOL_DIR must be set}"
source "$POOL_DIR/env.sh"
WORKERS=${WORKERS:-8}
IDLE_EXIT_S=${IDLE_EXIT_S:-900}
row=$(awk -F'\t' -v s="$MODEL_SLUG" '$1 == s' "$QROOT/models.tsv")
if [[ -z "$row" ]]; then
    echo "model $MODEL_SLUG not in $QROOT/models.tsv" >&2
    exit 2
fi
IFS=$'\t' read -r slug hf revision kind port maxlen template thinking_off gpu_util <<< "$row"
QUEUE=$QROOT/$slug
LOGDIR=$QROOT/$slug/server
mkdir -p "$LOGDIR" "$QUEUE/pending" "$QUEUE/running" "$QUEUE/done" "$QUEUE/failed" "$QUEUE/logs"
ts() { date '+%Y-%m-%dT%H:%M:%S'; }
echo "[$(ts)] host=$(hostname) job=$SLURM_JOB_ID slug=$slug hf=$hf revision=$revision port=$port maxlen=$maxlen clone=$(git -C "$CLONE" describe --tags --always)"
set +x
set -a
source "$SECRETS"
set +a
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing" >&2; exit 2; }
export VLLM_USE_FLASHINFER_SAMPLER=0
export UV_PYTHON=3.13
export UV_NO_SYNC=1
export DUMMY_KEY=none
SERVE_ARGS=(--served-model-name "$slug" --dtype bfloat16 --max-model-len "$maxlen" --port "$port" --host 127.0.0.1
    --gpu-memory-utilization "$gpu_util" --override-generation-config '{"temperature": 0.0}')
[[ "$revision" != "-" ]] && SERVE_ARGS+=(--revision "$revision")
[[ "$template" != "-" ]] && SERVE_ARGS+=(--chat-template "$TEMPLATES/$template")
[[ "$thinking_off" == "1" ]] && SERVE_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
SERVER_PID=""
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    for claimed in "$QUEUE"/running/*.json; do
        [[ -e "$claimed" ]] || continue
        name=$(basename "$claimed" .json)
        out=$(grep -A1 '"--out"' "$claimed" | tail -1 | tr -d ' ",')
        [[ -n "$out" && -d "$out" ]] && mv "$out" "$out.partial-$SLURM_JOB_ID"
        echo "[$(ts)] requeue $name"
        mv "$claimed" "$QUEUE/pending/$name.json"
    done
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; sleep 5; kill -9 "$SERVER_PID" 2>/dev/null; }
    rm -f "$QUEUE/server.env"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
t0=$(date +%s)
"$VLLM_VENV/bin/vllm" serve "$hf" "${SERVE_ARGS[@]}" > "$LOGDIR/server-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
ok=0
for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null && { ok=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 5
done
echo "[$(ts)] serve ready=$ok seconds=$(( $(date +%s) - t0 ))"
if [[ $ok -ne 1 ]]; then
    tail -60 "$LOGDIR/server-$SLURM_JOB_ID.log"
    exit 12
fi
printf 'JOB=%s\nHOST=%s\nBASE_URL=http://127.0.0.1:%s/v1\n' "$SLURM_JOB_ID" "$(hostname)" "$port" > "$QUEUE/server.env"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
cd "$CLONE"
for i in $(seq 1 "$WORKERS"); do
    uvx --from uv==0.12.3 uv run --no-sync python scripts/prelim/queue_worker.py \
        --queue "$QUEUE" --idle-exit-s "$IDLE_EXIT_S" > "$LOGDIR/worker-$SLURM_JOB_ID-$i.log" 2>&1 &
    WORKER_PIDS+=($!)
    sleep 2
done
while :; do
    alive=0
    for pid in "${WORKER_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ $alive -eq 0 ]] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(ts)] server died" >&2
        exit 13
    fi
    sleep 30
done
grep -E "Avg prompt throughput" "$LOGDIR/server-$SLURM_JOB_ID.log" | tail -2
echo "[$(ts)] workers idle; exiting"
