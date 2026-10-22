export SCRATCH=/scratch/ddr8143
restore_nounset=$(set +o | grep nounset)
set +u
source "$SCRATCH/.config/shell/cluster_env.sh"
eval "$restore_nounset"
export PATH=/home/ddr8143/.local/bin:$PATH
export HOME=/scratch/ddr8143
CLONE=$SCRATCH/drotherm/repos/whetstone-envs
VLLM_VENV=$SCRATCH/venvs/vllm
SECRETS=$SCRATCH/.config/secrets/openrouter.env
TEMPLATES=$SCRATCH/drotherm/scripts/templates
POOL_PREFIX=wsv-
CILVR_MAX_POOLS=${CILVR_MAX_POOLS:-4}
