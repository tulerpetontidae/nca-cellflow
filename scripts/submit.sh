#!/bin/bash
# Submit a training job to Sherlock
# Usage: bash scripts/submit.sh [config_name]
#   e.g. bash scripts/submit.sh baseline

set -e

CONFIG_NAME="${1:-baseline}"
PROJECT_DIR="/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow"
CONFIG_FILE="${PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/${CONFIG_NAME}"
LOG_DIR="${PROJECT_DIR}/logs"
JOB_NAME="nca-${CONFIG_NAME}"

# Create directories on cluster
ssh sherlock "mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}"

# Submit job
ssh sherlock "cd ${PROJECT_DIR} && sbatch --job-name=${JOB_NAME} --output=${LOG_DIR}/%x-%j.out <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH -C \"GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB\"

module purge
source /home/groups/ccurtis2/alomakin/miniconda3/etc/profile.d/conda.sh
conda activate void-tracer

cd ${PROJECT_DIR}
# pip install -e . --quiet

echo \"=== Job: ${JOB_NAME} ===\"
echo \"Config: ${CONFIG_FILE}\"
echo \"Checkpoints: ${CHECKPOINT_DIR}\"
echo \"Start: \$(date)\"

python scripts/train.py \\
    --config ${CONFIG_FILE} \\
    --wandb \\
    --wandb_name ${JOB_NAME} \\
    --checkpoint_dir ${CHECKPOINT_DIR}

echo \"End: \$(date)\"
SBATCH_EOF
"

echo "Submitted job: ${JOB_NAME}"
