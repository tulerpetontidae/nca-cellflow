"""Pack all .npy single-cell images into a single pickle for fast loading.

Usage:
    python scripts/prepack_images.py [--image_dir data/bbbc021_six] [--output data/bbbc021_six/images.pkl]

Loading 97K individual .npy files takes ~33s. A single pickle loads in ~1.6s (20x faster).
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    image_dir = Path(args.image_dir)
    output = Path(args.output) if args.output else image_dir / "images.pkl"

    npy_files = sorted(image_dir.glob("**/*.npy"))
    print(f"Found {len(npy_files)} .npy files in {image_dir}")

    from tqdm import tqdm
    data = {}
    for f in tqdm(npy_files, desc="Loading"):
        batch = f.parent.parent.name
        plate = f.parent.name
        sample_key = f"{batch}_{plate}_{f.stem}"
        data[sample_key] = np.load(f)

    print(f"Loaded {len(data)} images, writing to {output}...")
    with open(output, "wb") as fout:
        pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = output.stat().st_size / 1e9
    print(f"Done: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
