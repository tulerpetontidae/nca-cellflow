"""Export petridish LatentNCA to ONNX + JSON assets for the web demo.

Writes to web/ (or --output_dir):
    nca_step.onnx   - single NCA step, inputs (x, cond_z), output new_x
    config.json     - compounds, doses, MOA, step_size, dims, embed/dose_proj weights
    seeds.json      - N DMSO control cell crops [N, 3, 48, 48] as float arrays

cond_z (shape [B, cond_dim + dose_dim + z_dim]) is computed in JS from
compound id + dose + z using the exported embed / dose_proj weights.

Usage:
    python scripts/export_petridish_web.py \
        --checkpoint checkpoints/petridish/petridish-homeo1_petridish_step_118000.pt \
        --output_dir web
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca_cellflow import LabeledImageBank
from nca_cellflow.models import LatentNCA


# ---------------------------------------------------------------------------
# Export wrapper: a single NCA step taking precomputed cond_z
# ---------------------------------------------------------------------------


class PetridishStepExport(nn.Module):
    """Single-step wrapper matching LatentNCA.step for homeo1 config.

    homeo1 uses: use_tanh=True, fire_rate=1.0, use_alive_mask=False.
    So this mirrors LatentNCA.step exactly (no fire mask, no alive mask).
    """

    def __init__(self, nca: LatentNCA):
        super().__init__()
        assert nca.use_tanh, "export assumes use_tanh=True"
        assert nca.fire_rate == 1.0, "export assumes fire_rate=1.0"
        assert not nca.use_alive_mask, "export assumes use_alive_mask=False"

        self.sensor = nca.sensor
        self.fc1 = nca.fc1
        self.fc2 = nca.fc2
        self.fc3 = nca.fc3
        self.film1 = nca.film1
        self.film2 = nca.film2
        self.step_size = float(nca.step_size)

    def forward(self, x, cond_z):
        g1, b1 = self.film1(cond_z).chunk(2, dim=1)
        g1 = g1[:, None, None, :]
        b1 = b1[:, None, None, :]
        g2, b2 = self.film2(cond_z).chunk(2, dim=1)
        g2 = g2[:, None, None, :]
        b2 = b2[:, None, None, :]

        features = self.sensor(x)  # [B, 3C, H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, 3C]

        h = F.relu(self.fc1(features))
        h = g1 * h + b1
        h = F.relu(self.fc2(h))
        h = g2 * h + b2
        dx = self.fc3(h).permute(0, 3, 1, 2)  # [B, C, H, W]

        dx = torch.tanh(dx)
        return x + dx * self.step_size


# ---------------------------------------------------------------------------
# Checkpoint -> LatentNCA (EMA weights)
# ---------------------------------------------------------------------------


def _clean_ema_sd(sd):
    out = {}
    for k, v in sd.items():
        if k == "n_averaged":
            continue
        k = k.replace("module.", "", 1) if k.startswith("module.") else k
        out[k] = v
    return out


def load_petridish(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = state.get("extra", {}) or {}

    # Infer num_classes from embed.weight shape (prefer this over extra dict)
    ema_sd_raw = state.get("G_ema_state") or state.get("G_state")
    ema_sd = _clean_ema_sd(ema_sd_raw)
    num_classes = ema_sd["embed.weight"].shape[0]
    cond_dim = ema_sd["embed.weight"].shape[1]
    dose_dim = ema_sd.get("dose_proj.weight", torch.zeros(0, 1)).shape[0]
    film_in = ema_sd["film1.weight"].shape[1]
    z_dim = film_in - cond_dim - dose_dim
    channel_n = ema_sd["fc3.weight"].shape[0]
    hidden_dim = ema_sd["fc1.weight"].shape[0]
    img_channels = 3
    hidden_channels = channel_n - img_channels

    step_size = float(extra.get("step_size", 0.027))
    use_tanh = bool(extra.get("use_tanh", True))
    fire_rate = float(extra.get("fire_rate", 1.0))
    image_size = int(extra.get("image_size", 48))

    print(f"Architecture:")
    print(f"  num_classes     = {num_classes}")
    print(f"  cond_dim        = {cond_dim}")
    print(f"  dose_dim        = {dose_dim}")
    print(f"  z_dim           = {z_dim}")
    print(f"  channel_n       = {channel_n}  ({img_channels} img + {hidden_channels} hidden)")
    print(f"  hidden_dim      = {hidden_dim}")
    print(f"  step_size       = {step_size}")
    print(f"  use_tanh        = {use_tanh}")
    print(f"  image_size      = {image_size}")

    nca = LatentNCA(
        channel_n=channel_n,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        cond_dim=cond_dim,
        fire_rate=fire_rate,
        step_size=step_size,
        use_tanh=use_tanh,
        dose_dim=dose_dim,
    )
    missing, unexpected = nca.load_state_dict(ema_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    nca.eval()

    return nca, {
        "num_classes": num_classes,
        "cond_dim": cond_dim,
        "dose_dim": dose_dim,
        "z_dim": z_dim,
        "channel_n": channel_n,
        "img_channels": img_channels,
        "hidden_channels": hidden_channels,
        "hidden_dim": hidden_dim,
        "step_size": step_size,
        "image_size": image_size,
        "metadata_csv": extra.get("metadata_csv", "data/bbbc021_six/metadata/bbbc021_df_all.csv"),
        "image_dir": extra.get("image_dir", "data/bbbc021_six"),
    }


# ---------------------------------------------------------------------------
# Compound metadata + mid-dose selection
# ---------------------------------------------------------------------------


def build_compound_metadata(bank: LabeledImageBank, metadata_csv: str):
    """Return a list of dicts, one per compound (id 1..num_compounds).

    Each entry: {id, name, annot, doses, mid_dose}
    """
    import pandas as pd
    df = pd.read_csv(metadata_csv, index_col=0)
    trt = df[(df["STATE"] == 1) & (df["SPLIT"] == "train")]

    # Map compound name -> MOA annotation (pick the first row)
    cpd2annot = {}
    for _, row in trt.iterrows():
        name = row["CPD_NAME"]
        if name not in cpd2annot:
            cpd2annot[name] = str(row.get("ANNOT", "unknown"))

    entries = []
    for cpd_id in sorted(bank.id2cpd.keys()):
        name = bank.id2cpd[cpd_id]
        # Doses available in the image bank for this compound
        doses = sorted([d for d in bank._index[cpd_id].keys()])
        if not doses:
            continue
        mid_dose = float(doses[len(doses) // 2])  # matches vis_petridish
        entries.append({
            "id": int(cpd_id),
            "name": str(name),
            "annot": cpd2annot.get(name, "unknown"),
            "doses": [float(d) for d in doses],
            "mid_dose": mid_dose,
        })
    return entries


# ---------------------------------------------------------------------------
# Seed cell extraction
# ---------------------------------------------------------------------------


def sample_seed_cells(bank: LabeledImageBank, n: int, img_channels: int, image_size: int):
    """Sample N DMSO control cells and return raw uint8 crops [N, 3, H, W].

    We save uint8 so the .bin file is ~4x smaller. In JS the values are
    normalized to [-1, 1] via  v = ((u8 + 0.5) / 255.0) * 2 - 1, matching
    LabeledImageBank._load_cached's non-augmented path.
    """
    out = np.zeros((n, img_channels, image_size, image_size), dtype=np.uint8)
    for i in range(n):
        # Pull the raw cached tensor (float32 in [0, 255] before normalization)
        # sample_one applies _load_cached which normalizes; reach into _cache directly
        # to get the raw uint8.
        plate = None
        keys = bank._flat_index[(0, 0.0)]
        key = keys[np.random.randint(len(keys))]
        raw = bank._cache[key]
        if isinstance(raw, np.ndarray):  # [H, W, C] uint8
            arr = raw
        else:  # torch.Tensor in [C, H, W] float32 with resized cache
            arr = raw.permute(1, 2, 0).cpu().numpy()
        if arr.shape[0] != image_size:
            # Resize via torch interpolate to match the image size
            import torch.nn.functional as F
            t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=image_size, mode="bilinear", antialias=True)
            arr = t.squeeze(0).permute(1, 2, 0).numpy()
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)  # [H, W, C]
        out[i] = arr.transpose(2, 0, 1)  # [C, H, W]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default="checkpoints/petridish/petridish-homeo1_petridish_step_118000.pt")
    ap.add_argument("--output_dir", type=str, default="web")
    ap.add_argument("--n_seeds", type=int, default=200,
                    help="Number of DMSO seed cells to bundle")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading {args.checkpoint} ...")
    nca, info = load_petridish(args.checkpoint)

    # Build image bank (preload=True uses the pkl; much faster than disk sampling)
    print(f"Building image bank from {info['metadata_csv']} ...")
    bank = LabeledImageBank(
        info["metadata_csv"], info["image_dir"], split="train",
        image_size=info["image_size"], preload=True,
    )

    # ----- Compound metadata -----
    compounds = build_compound_metadata(bank, info["metadata_csv"])
    print(f"Compounds: {len(compounds)}")

    # ----- Export ONNX -----
    export_model = PetridishStepExport(nca).eval()
    B = 4
    dummy_x = torch.randn(B, info["channel_n"], info["image_size"], info["image_size"])
    dummy_cond_z = torch.randn(B, info["cond_dim"] + info["dose_dim"] + info["z_dim"])

    onnx_path = out_dir / "nca_step.onnx"
    # Use legacy TorchScript-based exporter (dynamo=False) to avoid the onnxscript dependency.
    torch.onnx.export(
        export_model,
        (dummy_x, dummy_cond_z),
        str(onnx_path),
        input_names=["x", "cond_z"],
        output_names=["new_x"],
        dynamic_axes={
            "x": {0: "batch"},
            "cond_z": {0: "batch"},
            "new_x": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Saved ONNX -> {onnx_path}")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_ort = sess.run(None, {
            "x": dummy_x.numpy(),
            "cond_z": dummy_cond_z.numpy(),
        })[0]
        with torch.no_grad():
            out_torch = export_model(dummy_x, dummy_cond_z).numpy()
        diff = float(np.abs(out_ort - out_torch).max())
        print(f"ONNX vs Torch max abs diff: {diff:.3e}")
    except Exception as e:
        print(f"[warn] ONNX verification skipped: {e}")

    # ----- Config JSON (weights + metadata) -----
    embed_w = nca.embed.weight.detach().cpu().numpy().tolist()           # [num_classes, cond_dim]
    dose_w = nca.dose_proj.weight.detach().cpu().numpy().tolist()        # [dose_dim, 1]
    dose_b = nca.dose_proj.bias.detach().cpu().numpy().tolist()          # [dose_dim]

    config = {
        "image_size": info["image_size"],
        "channel_n": info["channel_n"],
        "img_channels": info["img_channels"],
        "hidden_channels": info["hidden_channels"],
        "num_classes": info["num_classes"],
        "cond_dim": info["cond_dim"],
        "dose_dim": info["dose_dim"],
        "z_dim": info["z_dim"],
        "step_size": info["step_size"],
        "embed_weight": embed_w,   # used for compound embedding lookup in JS
        "dose_proj_weight": dose_w,  # dose_proj(log10(dose))
        "dose_proj_bias": dose_b,
        "dmso_id": 0,
        "compounds": compounds,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Saved config -> {config_path}")

    # ----- Seed cells -----
    print(f"Sampling {args.n_seeds} DMSO seed cells ...")
    seeds = sample_seed_cells(bank, args.n_seeds, info["img_channels"], info["image_size"])
    # Save as raw uint8 .bin (NCHW order). JS side converts to [-1, 1].
    seeds_path = out_dir / "seeds.bin"
    seeds.tofile(str(seeds_path))
    config["seeds"] = {
        "file": "seeds.bin",
        "shape": list(seeds.shape),
        "dtype": "uint8",
    }
    # Re-save config with the seeds entry included
    with open(config_path, "w") as f:
        json.dump(config, f)
    sz_kb = seeds_path.stat().st_size / 1024
    print(f"Saved seeds -> {seeds_path}  ({sz_kb:.0f} KB)")

    print("Done.")


if __name__ == "__main__":
    main()
