# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal PyTorch dataset loader for the IMPA (Image-based Morphological Profiling Assay) project using BBBC021 data. The project implements a specialized paired sampling strategy for cellular image analysis with control (DMSO) and treated conditions.

## Environment

```bash
conda activate nca-gan
```

## Commands

### Testing and Development

```bash
# Test the dataloader functionality
python scripts/test_dataloader.py

# Generate dataset statistics and summaries
python scripts/dataset_summary.py

# Optional: pass custom metadata path
python scripts/dataset_summary.py --metadata data/bbbc021_six/metadata/bbbc021_df_all.csv
```

### Training

```bash
# Basic training (plots shown with plt.show)
python scripts/train.py --batch_size 32 --iterations 20000

# With wandb logging
python scripts/train.py --wandb --wandb_project nca-cellflow

# Resume from checkpoint
python scripts/train.py --resume checkpoints/step_10000.pt

# Train with hidden channels (for future experiments)
python scripts/train.py --hidden_channels 4
```

#### Training Details

The training script (`scripts/train.py`) implements a single-passage NCA-GAN: the NCA transforms control (DMSO) images toward treated distributions, conditioned on compound identity.

**GAN formulation:**
- Relativistic discriminator loss with `softplus(-rel)`
- Zero-centered gradient penalty on both real and fake samples
- Generator gradient clipping (default max_norm=1.0)

**Logged metrics** (stored in checkpoint `logs` dict):
- `loss/D_total`, `loss/D_adv`, `loss/D_reg` — discriminator losses
- `penalty/gp_real_mean`, `penalty/gp_fake_mean` — gradient penalty magnitudes
- `logits/D_real_mean`, `logits/D_fake_mean` — discriminator output statistics
- `loss/G` — generator loss

**Model setup:**
- `num_classes` is auto-detected from dataset compound count
- Discriminator sees only RGB channels (3), conditioned on compound via embedding
- `--hidden_channels 0` by default (RGB only); increase with `--hidden_channels N`
- `fire_rate=1.0` (no stochastic masking by default)

**Checkpoints** saved to `checkpoints/` include: G, D, both optimizers, all logs, and hyperparameters. Use `--resume` to continue training.

**Config files:** YAML configs in `configs/` can be loaded with `--config configs/baseline.yaml`. CLI args override config values.

### Cluster (Sherlock)

```bash
# Deploy code to cluster
bash scripts/deploy.sh

# Submit a job (uses configs/<name>.yaml)
bash scripts/submit.sh baseline
```

Remote project path: `sherlock:/oak/stanford/groups/ccurtis2/users/alomakin/projects/nca-cellflow`

### Using the Dataset in Code

```python
from dataset import IMPADataset
from torch.utils.data import DataLoader

# Initialize dataset
dataset = IMPADataset(
    metadata_csv="data/bbbc021_six/metadata/bbbc021_df_all.csv",
    image_dir="data/bbbc021_six",
    split="train"  # or "test"
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through batches
for img_ctrl, img_trt, cpd_id in loader:
    # img_ctrl: control images (B, C, H, W)
    # img_trt: treated images (B, C, H, W)
    # cpd_id: compound IDs (B,)
    pass
```

## Architecture

### Core Components

**dataset.py - IMPADataset**
The main dataset class implementing a unique sampling strategy:
- Maintains two independent pools: control (DMSO, STATE=0) and treated (STATE=1)
- Dataset length equals the number of control images
- Deterministic control sampling by index
- Random uniform treated sampling from the entire pool (no plate matching)
- Returns: (control_image, treated_image, compound_id)

### Data Structure

```
data/bbbc021_six/
├── metadata/
│   └── bbbc021_df_all.csv    # Metadata with SPLIT, STATE, SAMPLE_KEY, CPD_NAME, etc.
├── Week1/
│   └── {PLATE}/
│       └── {TABLE_NUMBER}_{IMAGE_NUMBER}_{OBJECT_NUMBER}.npy
├── Week2/
├── ...
└── Week10/
```

### SAMPLE_KEY Format

Sample keys follow the pattern: `{BATCH}_{PLATE}_{TABLE_NUMBER}_{IMAGE_NUMBER}_{OBJECT_NUMBER}`

Example: `Week1_22123_1_11_3.0`
- BATCH: Week1
- PLATE: 22123
- Remaining parts: 1_11_3.0

The dataset's `_load()` method parses this to construct the file path:
`data/bbbc021_six/Week1/22123/1_11_3.0.npy`

### Image Loading Pipeline

1. Load .npy file containing uint8 image data (shape: 96x96x3)
2. Convert to float32
3. Permute to PyTorch format (C, H, W)
4. Apply dithering: add uniform random noise [0, 1)
5. Normalize to [0, 1]: divide by 255
6. Scale to [-1, 1]: multiply by 2 and subtract 1

### Metadata Schema

Key columns in bbbc021_df_all.csv:
- **SAMPLE_KEY**: Unique identifier used to locate .npy files
- **BATCH**: Week identifier (Week1-Week10)
- **PLATE**: Plate number (e.g., 22123)
- **CPD_NAME**: Compound name (for treated samples)
- **STATE**: 0 = control (DMSO), 1 = treated
- **SPLIT**: "train" or "test"
- **SMILES**: Chemical structure notation
- **DOSE**: Treatment dosage
- **ANNOT**: Annotation/category (e.g., "Actin disruptors")

### Sampling Strategy

This dataset implements a **cross-plate, asymmetric sampling strategy**:

1. Control images are sampled deterministically by epoch index
2. Treated images are sampled uniformly at random from ALL treated samples
3. No plate or batch matching between control and treated pairs
4. This strategy maximizes diversity and prevents overfitting to plate-specific artifacts
5. Each epoch iterates through all control images exactly once

## Development Notes

- Images are stored as NumPy arrays (.npy files) with uint8 dtype, shape (96, 96, 3)
- The dataset applies dithering during loading to reduce quantization artifacts
- Compound names are mapped to integer IDs for efficient indexing
- All file paths are constructed dynamically from SAMPLE_KEY strings
