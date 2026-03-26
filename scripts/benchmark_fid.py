"""
Unified FID benchmark for NCA-GAN and CellFlux (or any model).

Computes overall and per-compound FID between generated PNGs and real test images.
Generated images should be organized as: {image_root}/{compound_name}/{id}.png

Usage:
    # Evaluate NCA samples
    python scripts/benchmark_fid.py \
        --image_root outputs/nca_samples \
        --metadata_csv data/bbbc021_six/metadata/bbbc021_df_all.csv \
        --image_dir data/bbbc021_six \
        --model_name nca-gan

    # Evaluate CellFlux samples (same format)
    python scripts/benchmark_fid.py \
        --image_root outputs/cellflux_samples \
        --metadata_csv data/bbbc021_six/metadata/bbbc021_df_all.csv \
        --image_dir data/bbbc021_six \
        --model_name cellflux

    # Compare multiple models at once
    python scripts/benchmark_fid.py \
        --image_root outputs/nca_samples outputs/cellflux_samples \
        --metadata_csv data/bbbc021_six/metadata/bbbc021_df_all.csv \
        --image_dir data/bbbc021_six \
        --model_name nca-gan cellflux
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_real_test_images(metadata_csv, image_dir, image_size=None):
    """Load all real test treated images, organized by compound.

    Returns:
        dict[str, list[Tensor]]: compound_name -> list of [3,H,W] tensors in [0,1]
    """
    df = pd.read_csv(metadata_csv, index_col=0)
    df = df[(df["SPLIT"] == "test") & (df["STATE"] == 1)]

    image_dir = Path(image_dir)
    per_cpd = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading real images"):
        key = row["SAMPLE_KEY"]
        cpd = row["CPD_NAME"]

        parts = key.split("_")
        path = image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
        if not path.exists():
            continue

        img = np.load(path).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        img = img / 255.0  # [0, 1], no dither for eval

        if image_size is not None and image_size != img.shape[-1]:
            img = F.interpolate(
                img.unsqueeze(0), size=image_size,
                mode="bilinear", antialias=True,
            ).squeeze(0)

        per_cpd[cpd].append(img)

    print(f"Loaded {sum(len(v) for v in per_cpd.values())} real images "
          f"across {len(per_cpd)} compounds")
    return per_cpd


def load_generated_images(image_root):
    """Load generated PNGs from {image_root}/{compound_name}/*.png.

    Returns:
        dict[str, list[Tensor]]: compound_name -> list of [3,H,W] tensors in [0,1]
    """
    image_root = Path(image_root)
    per_cpd = defaultdict(list)

    for cpd_dir in sorted(image_root.iterdir()):
        if not cpd_dir.is_dir():
            continue
        cpd_name = cpd_dir.name
        for img_path in sorted(cpd_dir.glob("*.png")):
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            per_cpd[cpd_name].append(img)

    print(f"Loaded {sum(len(v) for v in per_cpd.values())} generated images "
          f"across {len(per_cpd)} compounds from {image_root}")
    return per_cpd


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid_for_model(real_per_cpd, fake_per_cpd, device, max_samples=None):
    """Compute overall and per-compound FID.

    Args:
        real_per_cpd: dict[str, list[Tensor]] — real images per compound
        fake_per_cpd: dict[str, list[Tensor]] — generated images per compound
        device: torch device
        max_samples: cap total samples (None = use all)

    Returns:
        dict with 'overall_fid', 'per_cpd_fid', 'mean_per_cpd_fid', 'num_real', 'num_fake'
    """
    # Compounds present in both real and fake
    shared_cpds = sorted(set(real_per_cpd.keys()) & set(fake_per_cpd.keys()))
    if not shared_cpds:
        print("[warn] No overlapping compounds between real and generated!")
        return None

    missing = set(real_per_cpd.keys()) - set(fake_per_cpd.keys())
    if missing:
        print(f"[warn] Compounds in real but not generated: {missing}")

    # --- Overall FID ---
    fid_overall = FrechetInceptionDistance(normalize=True).to(device)
    total_real = 0
    total_fake = 0

    for cpd in tqdm(shared_cpds, desc="Overall FID"):
        real_imgs = real_per_cpd[cpd]
        fake_imgs = fake_per_cpd[cpd]

        if max_samples:
            cap_per_cpd = max_samples // len(shared_cpds)
            real_imgs = real_imgs[:cap_per_cpd]
            fake_imgs = fake_imgs[:cap_per_cpd]

        # Process in batches to avoid OOM
        batch_size = 128
        for i in range(0, len(real_imgs), batch_size):
            batch = torch.stack(real_imgs[i:i + batch_size]).to(device)
            fid_overall.update(batch, real=True)
            total_real += batch.shape[0]

        for i in range(0, len(fake_imgs), batch_size):
            batch = torch.stack(fake_imgs[i:i + batch_size]).to(device)
            fid_overall.update(batch, real=False)
            total_fake += batch.shape[0]

    overall_fid = fid_overall.compute().item()

    # --- Per-compound FID ---
    per_cpd_fid = {}
    fid_per_cpd = FrechetInceptionDistance(normalize=True).to(device)

    for cpd in tqdm(shared_cpds, desc="Per-compound FID"):
        real_imgs = real_per_cpd[cpd]
        fake_imgs = fake_per_cpd[cpd]

        if len(real_imgs) < 2 or len(fake_imgs) < 2:
            print(f"  [skip] {cpd}: {len(real_imgs)} real, {len(fake_imgs)} fake (need >=2)")
            continue

        fid_per_cpd.reset()

        for i in range(0, len(real_imgs), 128):
            batch = torch.stack(real_imgs[i:i + 128]).to(device)
            fid_per_cpd.update(batch, real=True)

        for i in range(0, len(fake_imgs), 128):
            batch = torch.stack(fake_imgs[i:i + 128]).to(device)
            fid_per_cpd.update(batch, real=False)

        try:
            per_cpd_fid[cpd] = fid_per_cpd.compute().item()
        except Exception as e:
            print(f"  [skip] {cpd}: FID failed ({e})")

    mean_per_cpd = np.mean(list(per_cpd_fid.values())) if per_cpd_fid else float("nan")

    return {
        "overall_fid": overall_fid,
        "mean_per_cpd_fid": mean_per_cpd,
        "per_cpd_fid": per_cpd_fid,
        "num_real": total_real,
        "num_fake": total_fake,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified FID benchmark")
    parser.add_argument("--image_root", type=str, nargs="+", required=True,
                        help="Path(s) to generated image directories")
    parser.add_argument("--model_name", type=str, nargs="+", default=None,
                        help="Name(s) for each model (defaults to directory names)")
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to real image dataset")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Resize real images to match generated (None = native)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples per model")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = [Path(p).name for p in args.image_root]
    assert len(args.model_name) == len(args.image_root), \
        "Number of model names must match number of image roots"

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load real images (shared across all models)
    # Detect image_size from first generated image if not specified
    if args.image_size is None:
        first_root = Path(args.image_root[0])
        for cpd_dir in first_root.iterdir():
            if cpd_dir.is_dir():
                pngs = list(cpd_dir.glob("*.png"))
                if pngs:
                    sample = Image.open(pngs[0])
                    args.image_size = sample.size[0]  # assume square
                    print(f"Auto-detected image size: {args.image_size}")
                    break

    real_per_cpd = load_real_test_images(
        args.metadata_csv, args.image_dir, image_size=args.image_size,
    )

    # Evaluate each model
    all_results = {}
    for name, root in zip(args.model_name, args.image_root):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        fake_per_cpd = load_generated_images(root)
        result = compute_fid_for_model(
            real_per_cpd, fake_per_cpd, device, max_samples=args.max_samples,
        )

        if result is None:
            continue

        all_results[name] = result

        print(f"\n  Overall FID:       {result['overall_fid']:.2f}")
        print(f"  Mean per-cpd FID:  {result['mean_per_cpd_fid']:.2f}")
        print(f"  Samples: {result['num_real']} real, {result['num_fake']} fake")
        print(f"  Per-compound FID:")
        for cpd, fid in sorted(result["per_cpd_fid"].items()):
            print(f"    {cpd:30s}  {fid:.2f}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':30s}  {'Overall FID':>12s}  {'Mean/cpd FID':>12s}  {'Samples':>8s}")
        print("-" * 70)
        for name, result in all_results.items():
            print(f"{name:30s}  {result['overall_fid']:12.2f}  "
                  f"{result['mean_per_cpd_fid']:12.2f}  {result['num_fake']:8d}")

    # Save
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        # Convert numpy types for JSON
        serializable = {}
        for name, result in all_results.items():
            serializable[name] = {
                "overall_fid": float(result["overall_fid"]),
                "mean_per_cpd_fid": float(result["mean_per_cpd_fid"]),
                "per_cpd_fid": {k: float(v) for k, v in result["per_cpd_fid"].items()},
                "num_real": result["num_real"],
                "num_fake": result["num_fake"],
            }
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
