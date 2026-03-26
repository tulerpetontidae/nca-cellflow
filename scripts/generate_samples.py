"""
Generate images from an NCA checkpoint for FID evaluation.

Saves PNG images organized as: {output_dir}/{compound_name}/{index}.png
This format is compatible with CellFlux's eval_fid.py.

Usage:
    python scripts/generate_samples.py \
        --checkpoint checkpoints/some_run/step_80000.pt \
        --metadata_csv data/bbbc021_six/metadata/bbbc021_df_all.csv \
        --image_dir data/bbbc021_six \
        --output_dir outputs/nca_samples \
        --num_samples 5120
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from nca_cellflow.models import BaseNCA, NoiseNCA


# ---------------------------------------------------------------------------
# Dataset for deterministic test-set generation (no augmentation, no dither)
# ---------------------------------------------------------------------------

class TestDataset(torch.utils.data.Dataset):
    """Iterate over all test ctrl/trt pairs deterministically (no augmentation)."""

    def __init__(self, metadata_csv, image_dir, image_size=96):
        import pandas as pd
        from pathlib import Path

        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == "test"]

        ctrl = df[df["STATE"] == 0]
        trt = df[df["STATE"] == 1]

        self.ctrl_keys = ctrl["SAMPLE_KEY"].values
        self.trt_keys = trt["SAMPLE_KEY"].values
        self.trt_cpd = trt["CPD_NAME"].values

        cpds = sorted(set(self.trt_cpd))
        self.cpd2id = {c: i for i, c in enumerate(cpds)}
        self.id2cpd = {i: c for c, i in self.cpd2id.items()}

        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.trt_keys)

    def __getitem__(self, idx):
        # Pair each treated image with a random ctrl (deterministic via idx seed)
        rng = np.random.RandomState(idx)
        ctrl_idx = rng.randint(len(self.ctrl_keys))

        img_ctrl = self._load(self.ctrl_keys[ctrl_idx])
        img_trt = self._load(self.trt_keys[idx])
        cpd_id = self.cpd2id[self.trt_cpd[idx]]
        return img_ctrl, img_trt, cpd_id

    def _load(self, key):
        parts = key.split("_")
        path = self.image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
        img = np.load(path).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        # Deterministic normalization (no dither, no flips)
        img = img / 255.0
        if self.image_size != img.shape[-1]:
            img = F.interpolate(
                img.unsqueeze(0), size=self.image_size,
                mode="bilinear", antialias=True,
            ).squeeze(0)
        img = img * 2.0 - 1.0  # [-1, 1]
        return img


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generator(checkpoint_path, device):
    """Reconstruct generator from checkpoint extra dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extra = ckpt["extra"]

    channel_n = extra["channel_n"]
    nca_type = extra.get("nca_type", "base")
    step_size = extra.get("step_size", 0.1)
    use_alive_mask = extra.get("use_alive_mask", False)
    alive_threshold = extra.get("alive_threshold", 0.05)
    num_compounds = extra["num_compounds"]
    nca_steps = extra.get("nca_steps", 60)
    hidden_channels = extra.get("hidden_channels", 0)
    cond_type = extra.get("cond_type", "id")

    img_channels = 4 if use_alive_mask else 3

    nca_kwargs = dict(
        channel_n=channel_n,
        hidden_dim=128,
        num_classes=num_compounds,
        cond_dim=32,
        cond_type=cond_type,
        fire_rate=1.0,
        step_size=step_size,
        use_alive_mask=use_alive_mask,
        alive_threshold=alive_threshold,
    )

    if nca_type == "noise":
        noise_channels = extra.get("noise_channels", 1)
        G = NoiseNCA(noise_channels=noise_channels, **nca_kwargs)
    else:
        G = BaseNCA(**nca_kwargs)

    # Load EMA weights if available, otherwise main weights
    if "G_ema_state" in ckpt:
        # EMA state dict has 'module.' prefix from AveragedModel
        ema_state = ckpt["G_ema_state"]
        clean_state = {}
        for k, v in ema_state.items():
            clean_key = k.replace("module.", "")
            clean_state[clean_key] = v
        G.load_state_dict(clean_state)
        print("[ckpt] loaded EMA weights")
    else:
        G.load_state_dict(ckpt["G_state"])
        print("[ckpt] loaded main weights")

    G = G.to(device).eval()
    return G, extra


def tensor_to_pil(img_tensor):
    """Convert [-1,1] CHW tensor to PIL Image."""
    img = img_tensor.detach().cpu().float().clamp(-1, 1)
    img = ((img + 1.0) / 2.0 * 255.0).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser(description="Generate NCA samples for FID evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5120,
                        help="Max number of samples to generate (0 = all test treated images)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image size (default: read from checkpoint)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    G, extra = load_generator(args.checkpoint, device)
    nca_steps = extra.get("nca_steps", 60)
    hidden_channels = extra.get("hidden_channels", 0)
    use_alive_mask = extra.get("use_alive_mask", False)
    alive_threshold = extra.get("alive_threshold", 0.05)
    img_channels = 4 if use_alive_mask else 3
    image_size = args.image_size or 48  # default for lb configs

    print(f"NCA type={extra.get('nca_type')}, steps={nca_steps}, "
          f"hidden={hidden_channels}, alive={use_alive_mask}")

    # Dataset
    dataset = TestDataset(args.metadata_csv, args.image_dir, image_size=image_size)
    id2cpd = dataset.id2cpd
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    print(f"Test set: {len(dataset)} treated images, generating {num_samples}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Generate
    os.makedirs(args.output_dir, exist_ok=True)
    for cpd_name in id2cpd.values():
        os.makedirs(os.path.join(args.output_dir, cpd_name), exist_ok=True)

    cpd_counters = {name: 0 for name in id2cpd.values()}
    total_generated = 0

    with torch.no_grad():
        for img_ctrl, img_trt, cpd_id in tqdm(loader, desc="Generating"):
            if total_generated >= num_samples:
                break

            img_ctrl = img_ctrl.to(device)
            cpd_id = cpd_id.to(device)
            B = img_ctrl.shape[0]

            # Build NCA input
            if use_alive_mask:
                ctrl_01 = (img_ctrl + 1) / 2
                alive = (ctrl_01.max(dim=1, keepdim=True).values > alive_threshold).float()
                nca_in = torch.cat([img_ctrl, alive], dim=1)
            else:
                nca_in = img_ctrl

            if hidden_channels > 0:
                pad = torch.zeros(B, hidden_channels, nca_in.shape[2], nca_in.shape[3], device=device)
                nca_in = torch.cat([nca_in, pad], dim=1)

            # Run NCA
            fake_full = G(nca_in, cpd_id, n_steps=nca_steps)
            fake_rgb = fake_full[:, :3]

            # Save as PNGs
            for i in range(B):
                if total_generated >= num_samples:
                    break
                cpd_name = id2cpd[cpd_id[i].item()]
                idx = cpd_counters[cpd_name]
                cpd_counters[cpd_name] += 1

                pil_img = tensor_to_pil(fake_rgb[i])
                save_path = os.path.join(args.output_dir, cpd_name, f"{idx}.png")
                pil_img.save(save_path)
                total_generated += 1

    print(f"\nGenerated {total_generated} images to {args.output_dir}")
    for cpd_name, count in sorted(cpd_counters.items()):
        print(f"  {cpd_name}: {count}")


if __name__ == "__main__":
    main()
