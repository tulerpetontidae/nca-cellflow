#!/bin/bash
# Resubmit an IMPA job, resuming from latest checkpoint
# Usage: bash scripts/resubmit_impa.sh [config_name] [iterations]
#   e.g. bash scripts/resubmit_impa.sh impa-bbbc021-lb
#   e.g. bash scripts/resubmit_impa.sh impa-bbbc021-lb 160000

set -e

CONFIG_NAME="${1:-impa-bbbc021-lb}"
ITERATIONS="${2:-}"
PROJECT_DIR="/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow"
CONFIG_FILE="${PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/${CONFIG_NAME}"
LOG_DIR="${PROJECT_DIR}/logs"
JOB_NAME="impa-${CONFIG_NAME}"

# Find latest checkpoint
LATEST_CKPT=$(ssh sherlock "ls -t ${CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -1")

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in ${CHECKPOINT_DIR}, submitting fresh"
    bash scripts/submit_impa.sh "$CONFIG_NAME"
    exit 0
fi

echo "Resuming from: ${LATEST_CKPT}"

ssh sherlock "mkdir -p ${LOG_DIR}"

ssh sherlock "cd ${PROJECT_DIR} && sbatch --job-name=${JOB_NAME} --output=${LOG_DIR}/%x-%j.out <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -C 'GPU_SKU:L40S|GPU_SKU:H100_SXM5'

module purge
source /home/groups/ccurtis2/alomakin/miniconda3/etc/profile.d/conda.sh
conda activate void-tracer

cd ${PROJECT_DIR}

echo \"=== Job: ${JOB_NAME} (resumed) ==\"
echo \"Config: ${CONFIG_FILE}\"
echo \"Resume: ${LATEST_CKPT}\"
echo \"Start: \$(date)\"

python scripts/train_impa.py \\
    --config ${CONFIG_FILE} \\
    --resume ${LATEST_CKPT} \\
    --wandb \\
    --wandb_name ${JOB_NAME} \\
    --checkpoint_dir ${CHECKPOINT_DIR} \\
    ${ITERATIONS:+--iterations ${ITERATIONS}}

echo \"End: \$(date)\"
SBATCH_EOF
"

echo "Resubmitted job: ${JOB_NAME} (resuming from ${LATEST_CKPT})"
