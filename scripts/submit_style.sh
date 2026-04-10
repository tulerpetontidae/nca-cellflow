#!/bin/bash
# Submit a style-NCA training job to Sherlock
# Usage: bash scripts/submit_style.sh [config_name]
#   e.g. bash scripts/submit_style.sh style-hidden12-ema-steps30-ss008-tanh-bigD-base-lb

set -e

CONFIG_NAME="${1:-style-hidden12-ema-steps30-ss008-tanh-bigD-base-lb}"
PROJECT_DIR="/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow"
CONFIG_FILE="${PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/${CONFIG_NAME}"
LOG_DIR="${PROJECT_DIR}/logs"
JOB_NAME="snca-${CONFIG_NAME}"

# Create directories on cluster
ssh sherlock "mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}"

# Submit job
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

echo \"=== Job: ${JOB_NAME} ==\"
echo \"Config: ${CONFIG_FILE}\"
echo \"Checkpoints: ${CHECKPOINT_DIR}\"
echo \"Start: \$(date)\"

python scripts/train_style_nca.py \\
    --config ${CONFIG_FILE} \\
    --wandb \\
    --wandb_name ${JOB_NAME} \\
    --checkpoint_dir ${CHECKPOINT_DIR}

echo \"End: \$(date)\"
SBATCH_EOF
"

echo "Submitted job: ${JOB_NAME}"
