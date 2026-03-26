#!/bin/bash
# =============================================================================
# Benchmark CellFlux vs NCA-GAN on BBBC021
#
# This script:
# 1. Clones & sets up CellFlux
# 2. Trains CellFlux on your BBBC021 data (or uses pretrained)
# 3. Generates samples from both models
# 4. Runs unified FID comparison
#
# Prerequisites:
#   - NCA-GAN checkpoint already trained
#   - BBBC021 data at DATA_DIR
#
# Usage:
#   bash scripts/benchmark_cellflux.sh
# =============================================================================

set -euo pipefail

# ---- Configuration (edit these) ----
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow/data/bbbc021_six"
METADATA_CSV="${DATA_DIR}/metadata/bbbc021_df_all.csv"
NCA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/noise-hidden6-ema-steps30-v2-ss001-plate-lb/step_80000.pt"

CELLFLUX_DIR="${PROJECT_ROOT}/../CellFlux"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/benchmark"
NUM_SAMPLES=5120

# ---- Step 1: Clone CellFlux if needed ----
echo "=== Step 1: Setting up CellFlux ==="
if [ ! -d "${CELLFLUX_DIR}" ]; then
    echo "Cloning CellFlux..."
    git clone https://github.com/yuhui-zh15/CellFlux.git "${CELLFLUX_DIR}"
else
    echo "CellFlux already cloned at ${CELLFLUX_DIR}"
fi

# ---- Step 2: Create CellFlux conda env if needed ----
echo ""
echo "=== Step 2: CellFlux environment ==="
if ! conda env list | grep -q cellflux; then
    echo "Creating cellflux conda env..."
    cd "${CELLFLUX_DIR}"
    conda env create -f environment.yml
    echo "Done. Activate with: conda activate cellflux"
else
    echo "cellflux conda env already exists"
fi

# ---- Step 3: Update CellFlux config with your data paths ----
echo ""
echo "=== Step 3: Updating CellFlux config ==="
CELLFLUX_CONFIG="${CELLFLUX_DIR}/configs/bbbc021_all.yaml"
if [ -f "${CELLFLUX_CONFIG}" ]; then
    # Backup original
    cp "${CELLFLUX_CONFIG}" "${CELLFLUX_CONFIG}.bak" 2>/dev/null || true

    # Update paths using python for safe YAML editing
    python3 -c "
import yaml
with open('${CELLFLUX_CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['image_path'] = '${DATA_DIR}/'
cfg['data_index_path'] = 'bbbc021_df_all.csv'
cfg['embedding_path'] = 'emb_fp.csv'
with open('${CELLFLUX_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Updated CellFlux config with local paths')
"
fi

# ---- Step 4: Generate NCA-GAN samples ----
echo ""
echo "=== Step 4: Generating NCA-GAN samples ==="
NCA_SAMPLES="${OUTPUT_DIR}/nca_samples"
if [ ! -d "${NCA_SAMPLES}" ] || [ -z "$(ls -A ${NCA_SAMPLES} 2>/dev/null)" ]; then
    conda run -n nca-gan python "${PROJECT_ROOT}/scripts/generate_samples.py" \
        --checkpoint "${NCA_CHECKPOINT}" \
        --metadata_csv "${METADATA_CSV}" \
        --image_dir "${DATA_DIR}" \
        --output_dir "${NCA_SAMPLES}" \
        --num_samples "${NUM_SAMPLES}" \
        --image_size 48
else
    echo "NCA samples already exist at ${NCA_SAMPLES}"
fi

# ---- Step 5: Train CellFlux (or use pretrained) ----
echo ""
echo "=== Step 5: CellFlux training ==="
echo "Option A: Train from scratch (slow, ~days on GPU):"
echo "  conda activate cellflux"
echo "  cd ${CELLFLUX_DIR}"
echo "  python train.py \\"
echo "      --dataset=bbbc021 --config=bbbc021_all \\"
echo "      --batch_size=32 --accum_iter=2 \\"
echo "      --epochs=3000 --eval_frequency=50 \\"
echo "      --class_drop_prob=0.2 --cfg_scale=0.2 \\"
echo "      --ode_method heun2 --ode_options '{\"nfe\": 50}' \\"
echo "      --use_ema --edm_schedule --skewed_timesteps \\"
echo "      --use_initial=2 --noise_level=0.5 \\"
echo "      --compute_fid --fid_samples=${NUM_SAMPLES} \\"
echo "      --output_dir=${OUTPUT_DIR}/cellflux_training"
echo ""
echo "Option B: Download pretrained from HuggingFace (if available):"
echo "  Check: https://huggingface.co/yuhui-zh15"
echo ""

# ---- Step 6: Generate CellFlux samples ----
echo "=== Step 6: Generate CellFlux samples ==="
CELLFLUX_SAMPLES="${OUTPUT_DIR}/cellflux_samples"
echo "After training, generate samples with:"
echo "  conda activate cellflux"
echo "  cd ${CELLFLUX_DIR}"
echo "  python train.py \\"
echo "      --dataset=bbbc021 --config=bbbc021_all \\"
echo "      --batch_size=32 --eval_only \\"
echo "      --cfg_scale=0.2 --ode_method heun2 \\"
echo "      --ode_options '{\"nfe\": 50}' \\"
echo "      --use_ema --edm_schedule \\"
echo "      --use_initial=2 --noise_level=1.0 \\"
echo "      --save_fid_samples --fid_samples=${NUM_SAMPLES} \\"
echo "      --resume=<CHECKPOINT_PATH> \\"
echo "      --output_dir=${CELLFLUX_SAMPLES}"
echo ""
echo "Then copy the fid_samples/{compound}/*.png dirs to ${CELLFLUX_SAMPLES}/"

# ---- Step 7: Run unified FID benchmark ----
echo ""
echo "=== Step 7: Unified FID comparison ==="
echo "Once both sample sets exist, run:"
echo ""
echo "  conda activate nca-gan"
echo "  python ${PROJECT_ROOT}/scripts/benchmark_fid.py \\"
echo "      --image_root ${NCA_SAMPLES} ${CELLFLUX_SAMPLES} \\"
echo "      --model_name nca-gan cellflux \\"
echo "      --metadata_csv ${METADATA_CSV} \\"
echo "      --image_dir ${DATA_DIR} \\"
echo "      --output_json ${OUTPUT_DIR}/fid_comparison.json"
echo ""
echo "=== Setup complete ==="
