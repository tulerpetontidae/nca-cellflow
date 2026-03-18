# NCA-CellFlow

Neural Cellular Automata for cellular morphology generation using BBBC021 data.

## Setup

```bash
# Create conda environment
conda create -n nca-gan python=3.11
conda activate nca-gan

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# or for CPU only:
# conda install pytorch torchvision cpuonly -c pytorch

# Install other dependencies
conda install pandas numpy

# Install package in dev mode
pip install -e .
```

## Usage

```python
from nca_cellflow import IMPADataset
from nca_cellflow.models import BaseNCA, Discriminator

# Dataset
dataset = IMPADataset(
    metadata_csv="data/bbbc021_six/metadata/bbbc021_df_all.csv",
    image_dir="data/bbbc021_six",
    split="train"
)

# Models
nca = BaseNCA(
    channel_n=3,
    hidden_dim=128,
    num_classes=len(dataset.cpd2id),
    cond_dim=64,
)

disc = Discriminator(
    widths=[64, 128, 256],
    cardinalities=[8, 16, 32],
    blocks_per_stage=[2, 2, 2],
    expansion=2,
    num_classes=len(dataset.cpd2id),
    embed_dim=128,
    in_channels=3,
)
```

## Tests

```bash
python tests/test_models.py
```
