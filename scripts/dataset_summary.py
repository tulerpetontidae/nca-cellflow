"""
Dataset summary statistics for BBBC021 data.

Outputs statistics about plates, batches, compounds, and samples
across train/test splits.
"""

import pandas as pd
import argparse
from pathlib import Path


def print_summary(metadata_csv: str):
    """Print comprehensive dataset statistics."""

    df = pd.read_csv(metadata_csv, index_col=0)

    print("=" * 70)
    print("BBBC021 Dataset Summary")
    print("=" * 70)

    # Overall statistics
    print(f"\nTotal samples: {len(df):,}")
    print(f"Unique plates: {df['PLATE'].nunique()}")
    print(f"Unique batches: {df['BATCH'].nunique()}")
    print(f"Unique compounds: {df['CPD_NAME'].nunique()}")

    # Split statistics
    print("\n" + "-" * 70)
    print("Split Distribution")
    print("-" * 70)

    for split in sorted(df["SPLIT"].unique()):
        split_df = df[df["SPLIT"] == split]
        ctrl = split_df[split_df["STATE"] == 0]
        trt = split_df[split_df["STATE"] == 1]

        print(f"\n{split.upper()}:")
        print(f"  Total samples: {len(split_df):,}")
        print(f"  Control (DMSO): {len(ctrl):,}")
        print(f"  Treated: {len(trt):,}")
        print(f"  Plates: {split_df['PLATE'].nunique()}")
        print(f"  Batches: {split_df['BATCH'].nunique()}")
        print(f"  Compounds: {trt['CPD_NAME'].nunique()}")

    # Compound statistics
    print("\n" + "-" * 70)
    print("Compound Statistics")
    print("-" * 70)

    trt_df = df[df["STATE"] == 1]
    cpd_counts = trt_df.groupby("CPD_NAME").size()

    print(f"\nTotal compounds: {len(cpd_counts)}")
    print(f"Samples per compound (mean): {cpd_counts.mean():.1f}")
    print(f"Samples per compound (median): {cpd_counts.median():.1f}")
    print(f"Samples per compound (min): {cpd_counts.min()}")
    print(f"Samples per compound (max): {cpd_counts.max()}")

    # Top compounds
    print("\nTop 10 compounds by sample count:")
    for cpd, count in cpd_counts.nlargest(10).items():
        print(f"  {cpd}: {count:,} samples")

    # Plate statistics
    print("\n" + "-" * 70)
    print("Plate Statistics")
    print("-" * 70)

    plate_counts = df.groupby("PLATE").size()
    print(f"\nTotal plates: {len(plate_counts)}")
    print(f"Samples per plate (mean): {plate_counts.mean():.1f}")
    print(f"Samples per plate (median): {plate_counts.median():.1f}")
    print(f"Samples per plate (min): {plate_counts.min()}")
    print(f"Samples per plate (max): {plate_counts.max()}")

    # Batch statistics
    print("\n" + "-" * 70)
    print("Batch Statistics")
    print("-" * 70)

    batch_counts = df.groupby("BATCH").size()
    print(f"\nTotal batches: {len(batch_counts)}")
    print(f"Samples per batch (mean): {batch_counts.mean():.1f}")
    print(f"Samples per batch (median): {batch_counts.median():.1f}")

    print("\nBatch breakdown:")
    for batch, count in batch_counts.items():
        batch_df = df[df["BATCH"] == batch]
        print(f"  {batch}: {count:,} samples, {batch_df['PLATE'].nunique()} plates")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print dataset summary statistics"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/bbbc021_six/metadata/bbbc021_df_all.csv",
        help="Path to metadata CSV file"
    )

    args = parser.parse_args()
    print_summary(args.metadata)
