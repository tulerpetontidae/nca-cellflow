"""Prepare a large BBBC021-style single image for NCA inference.

Loads three per-channel TIFFs (Actin / Tubulin / DAPI) from `data/single_image/`,
stacks them in the channel convention **Ch0=Actin, Ch1=Tubulin, Ch2=DAPI**,
performs per-channel percentile normalisation to match the uint8 distribution of
the training .npy crops, then saves the float32 [-1, 1] tensor + a uint8 preview
to disk.

The training .npy crops have per-image channel means roughly:
    Ch0 (Actin):   ~127 ± 26  (range 19-178)
    Ch1 (Tubulin): ~125 ± 27  (range 19-173)
    Ch2 (DAPI):     ~68 ± 20  (range 12-127)

The script computes per-channel (p1, p99) from each TIFF, rescales so p1→0 and
p99→255, clips, casts to uint8, then converts to float32 in [-1, 1] — matching
`IMPADataset._load` (non-augmented path).

Files (matched by UUID substring):
    `8FFC4C9ED7C8` → Tubulin (Ch1)
    `3EA429CAB4CE` → Actin   (Ch0)
    `E6A954BD1174` → DAPI    (Ch2)

Usage:
    python scripts/prepare_single_image.py
    python scripts/prepare_single_image.py --input_dir data/single_image --output_dir data/single_image --lo 1.0 --hi 99.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as e:
    raise SystemExit(
        "tifffile is required: `pip install tifffile` (already in the nca-gan env)."
    ) from e


# UUID → semantic channel. Order matches the user's memory entry feedback_channels.md.
CHANNEL_FILE_MARKERS: dict[str, str] = {
    "actin":   "3EA429CAB4CE",  # Ch0
    "tubulin": "8FFC4C9ED7C8",  # Ch1
    "dapi":    "E6A954BD1174",  # Ch2
}
CHANNEL_ORDER = ["actin", "tubulin", "dapi"]  # produces [Ch0, Ch1, Ch2]


def find_tiff(input_dir: Path, marker: str) -> Path:
    """Find a single TIFF in input_dir whose name contains `marker`."""
    matches = [p for p in input_dir.iterdir()
               if p.is_file() and p.suffix.lower() in (".tif", ".tiff") and marker in p.name]
    if not matches:
        raise FileNotFoundError(
            f"No TIFF containing marker '{marker}' in {input_dir}. "
            f"Directory contents: {sorted(p.name for p in input_dir.iterdir())}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple TIFFs match marker '{marker}' in {input_dir}: {[p.name for p in matches]}"
        )
    return matches[0]


def percentile_normalize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[np.ndarray, tuple[float, float]]:
    """Percentile-normalize a 2D array to uint8.

    p_lo → 0, p_hi → 255, clip, cast. Returns the uint8 array and the (p_lo, p_hi) used.
    """
    p_lo = float(np.percentile(arr, lo))
    p_hi = float(np.percentile(arr, hi))
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0  # degenerate guard
    x = (arr.astype(np.float32) - p_lo) / (p_hi - p_lo)
    x = np.clip(x, 0.0, 1.0) * 255.0
    return x.astype(np.uint8), (p_lo, p_hi)


def to_model_input(stack_uint8: np.ndarray) -> np.ndarray:
    """Convert a uint8 [H, W, 3] image to the float32 [-1, 1] format the model expects.

    Matches `IMPADataset._load(augment=False)`: `(arr + 0.5) / 255.0 * 2 - 1`.
    """
    f = (stack_uint8.astype(np.float32) + 0.5) / 255.0
    return f * 2.0 - 1.0  # [-1, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("data/single_image"),
                    help="Directory containing the 3 per-channel TIFFs")
    ap.add_argument("--output_dir", type=Path, default=Path("data/single_image"),
                    help="Where to write prepared.npy and preview files")
    ap.add_argument("--lo", type=float, default=1.0,
                    help="Lower percentile for normalization (default 1.0)")
    ap.add_argument("--hi", type=float, default=99.0,
                    help="Upper percentile for normalization (default 99.0)")
    ap.add_argument("--output_name", type=str, default="prepared",
                    help="Output basename (will save <name>.npy and <name>_preview.png)")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep] input_dir  = {input_dir}")
    print(f"[prep] output_dir = {output_dir}")
    print(f"[prep] percentile range = ({args.lo}, {args.hi})")
    print()

    # --- 1. Load each TIFF ---
    raw_channels: dict[str, np.ndarray] = {}
    for name in CHANNEL_ORDER:
        marker = CHANNEL_FILE_MARKERS[name]
        path = find_tiff(input_dir, marker)
        arr = tifffile.imread(path)
        if arr.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D grayscale TIFF, got shape {arr.shape}")
        raw_channels[name] = arr
        print(f"[load] {name:>8s}  <- {path.name}")
        print(f"       shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}  "
              f"p1={np.percentile(arr, 1):.1f}  p99={np.percentile(arr, 99):.1f}")

    # Sanity: all three channels the same spatial size
    shapes = {name: a.shape for name, a in raw_channels.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Channel shape mismatch: {shapes}")
    H, W = next(iter(shapes.values()))
    print(f"\n[prep] common size = {H} x {W}\n")

    # --- 2. Per-channel percentile normalise to uint8 ---
    print("[prep] per-channel percentile normalisation → uint8")
    uint8_channels: list[np.ndarray] = []
    for name in CHANNEL_ORDER:
        u8, (p_lo, p_hi) = percentile_normalize(raw_channels[name], args.lo, args.hi)
        uint8_channels.append(u8)
        print(f"       {name:>8s}: p_lo={p_lo:.1f} p_hi={p_hi:.1f} "
              f"-> mean={u8.mean():.1f} std={u8.std():.1f} "
              f"(target training mean: "
              f"{'127' if name == 'actin' else '125' if name == 'tubulin' else '68'})")

    stack_uint8 = np.stack(uint8_channels, axis=-1)  # [H, W, 3], Ch0=Actin, Ch1=Tubulin, Ch2=DAPI
    assert stack_uint8.shape == (H, W, 3), stack_uint8.shape
    assert stack_uint8.dtype == np.uint8

    # --- 3. Convert to model input format [-1, 1] float32 ---
    model_input = to_model_input(stack_uint8)  # [H, W, 3] float32 in [-1, 1]
    # Training tensors are CHW
    model_input_chw = np.transpose(model_input, (2, 0, 1)).astype(np.float32)  # [3, H, W]

    print(f"\n[prep] model input: shape={model_input_chw.shape}  dtype={model_input_chw.dtype}")
    print(f"       range=[{model_input_chw.min():.3f}, {model_input_chw.max():.3f}]")
    print(f"       per-channel mean (in [-1,1]):")
    for c, name in enumerate(CHANNEL_ORDER):
        print(f"         Ch{c} ({name}): {model_input_chw[c].mean():+.3f}  std={model_input_chw[c].std():.3f}")

    # --- 4. Save ---
    out_npy = output_dir / f"{args.output_name}.npy"
    np.save(out_npy, model_input_chw)
    print(f"\n[save] {out_npy}  ({out_npy.stat().st_size / 1024:.0f} KB)")

    # Also save the uint8 HWC version for easy inspection
    out_uint8 = output_dir / f"{args.output_name}_uint8_hwc.npy"
    np.save(out_uint8, stack_uint8)
    print(f"[save] {out_uint8}  (uint8 [H,W,3] for preview)")

    # Attempt a PNG preview (requires matplotlib or PIL).
    try:
        from PIL import Image
        preview_path = output_dir / f"{args.output_name}_preview.png"
        Image.fromarray(stack_uint8).save(preview_path)
        print(f"[save] {preview_path}")
    except Exception as e:
        print(f"[warn] preview PNG skipped: {e}")

    print("\n[done] use with:  img = np.load('data/single_image/prepared.npy')"
          "  # [3, H, W] float32, already in [-1, 1]")


if __name__ == "__main__":
    main()
