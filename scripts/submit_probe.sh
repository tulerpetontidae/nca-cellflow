#!/bin/bash
# Submit a single capacity-probe job to Sherlock.
# Usage: bash scripts/submit_probe.sh <probe_config_name>
#   e.g. bash scripts/submit_probe.sh probe-d-bigD
#
# Looks up configs/probe/<name>.yaml and runs scripts/probe_classifier.py.
# Logs to the `nca-cellflow-probe` W&B project (set in the YAML).

set -e

CONFIG_NAME="${1:-probe-d-default}"
PROJECT_DIR="/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow"
CONFIG_FILE="${PROJECT_DIR}/configs/probe/${CONFIG_NAME}.yaml"
CHECKPOINT_DIR="${PROJECT_DIR}/probe_checkpoints/${CONFIG_NAME}"
LOG_DIR="${PROJECT_DIR}/logs"
JOB_NAME="probe-${CONFIG_NAME}"

ssh sherlock "mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}"

ssh sherlock "cd ${PROJECT_DIR} && sbatch --job-name=${JOB_NAME} --output=${LOG_DIR}/%x-%j.out <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -C 'GPU_SKU:RTX_3090|GPU_SKU:L40S|GPU_SKU:H100_SXM5'

module purge
source /home/groups/ccurtis2/alomakin/miniconda3/etc/profile.d/conda.sh
conda activate void-tracer

cd ${PROJECT_DIR}

echo \"=== Job: ${JOB_NAME} ===\"
echo \"Config: ${CONFIG_FILE}\"
echo \"Checkpoints: ${CHECKPOINT_DIR}\"
echo \"Start: \$(date)\"

python scripts/probe_classifier.py \\
    --config ${CONFIG_FILE} \\
    --wandb \\
    --wandb_name ${JOB_NAME} \\
    --checkpoint_dir ${CHECKPOINT_DIR}

echo \"End: \$(date)\"
SBATCH_EOF
"

echo "Submitted job: ${JOB_NAME}"
