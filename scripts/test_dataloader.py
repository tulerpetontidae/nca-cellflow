"""
Test script to verify the IMPADataset loader works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import IMPADataset
import torch


def test_dataloader():
    """Test basic dataloader functionality."""

    print("Testing IMPADataset...")

    # Initialize dataset
    dataset = IMPADataset(
        metadata_csv="data/bbbc021_six/metadata/bbbc021_df_all.csv",
        image_dir="data/bbbc021_six",
        split="train"
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of compounds: {len(dataset.cpd2id)}")
    print(f"Number of control samples: {len(dataset.ctrl_keys)}")
    print(f"Number of treated samples: {len(dataset.trt_keys)}")

    # Test loading a few samples
    print("\nLoading test samples...")
    for i in range(3):
        img_ctrl, img_trt, cpd_id = dataset[i]

        print(f"\nSample {i}:")
        print(f"  Control image shape: {img_ctrl.shape}")
        print(f"  Treated image shape: {img_trt.shape}")
        print(f"  Compound ID: {cpd_id}")
        print(f"  Control value range: [{img_ctrl.min():.3f}, {img_ctrl.max():.3f}]")
        print(f"  Treated value range: [{img_trt.min():.3f}, {img_trt.max():.3f}]")

    # Test with DataLoader
    print("\n" + "="*70)
    print("Testing with PyTorch DataLoader...")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    img_ctrl, img_trt, cpd_id = batch

    print(f"\nBatch shapes:")
    print(f"  Control: {img_ctrl.shape}")
    print(f"  Treated: {img_trt.shape}")
    print(f"  Compound IDs: {cpd_id}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dataloader()
