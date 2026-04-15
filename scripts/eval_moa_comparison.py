"""
Evaluate MoA classification accuracy for all 4 generative models.

Uses the SAME deterministic ctrl-trt pairings (EvalDataset) across all models.
Reports accuracy/F1 split by in-distribution vs OOD compounds.

Usage:
    python scripts/eval_moa_comparison.py \
        --nca_ckpt checkpoints/nca/step_160000.pt \
        --nca_inter_ckpt checkpoints/nca-inter/step_160000.pt \
        --impa_ckpt checkpoints/impa/step_80000.pt \
        --cellflux_ckpt checkpoints/cellflux/step_80000.pt \
        --moa_ckpt checkpoints/moa_classifier_48px/best.pt \
        --image_size 48
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from nca_cellflow import EvalDataset
from nca_cellflow.models import (
    LatentNCA, IMPAGenerator, IMPAMappingNetwork, CellFluxUNet, MOAClassifier,
)
from nca_cellflow.models.cellflux_unet import ode_sample_heun, edm_time_grid

OOD_COMPOUNDS = {
    "docetaxel", "AZ841", "cytochalasin D", "simvastatin",
    "cyclohexamide", "latrunculin B", "epothilone B", "lactacystin",
}


def load_moa_classifier(moa_ckpt_path, device):
    ckpt = torch.load(moa_ckpt_path, map_location="cpu", weights_only=False)
    model = MOAClassifier(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict_head(ckpt["classifier_state"])
    model.eval()
    return model, ckpt["moa2id"], ckpt["id2moa"]


def build_fp_matrix(cpd2id, fp_path, device):
    id2cpd = {v: k for k, v in cpd2id.items()}
    fp_df = pd.read_csv(fp_path, index_col=0)
    fp_vecs = [fp_df.loc[id2cpd[i]].values.astype(np.float32)
               for i in range(len(cpd2id))]
    return torch.tensor(np.stack(fp_vecs)).to(device)


# ---------------------------------------------------------------------------
# Model loaders — each returns a generate_fn(img_ctrl, cpd_id) -> fake_rgb
# ---------------------------------------------------------------------------

def load_nca(ckpt_path, fp_matrix, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = ckpt["extra"]
    G = LatentNCA(
        channel_n=extra["channel_n"],
        hidden_dim=extra.get("nca_hidden_dim", 256),
        num_classes=extra["num_compounds"],
        cond_dim=extra.get("nca_cond_dim", 32),
        cond_type="fingerprint", fp_dim=fp_matrix.shape[1],
        z_dim=extra["z_dim"],
        fire_rate=1.0,
        step_size=extra["step_size"],
        use_tanh=extra.get("use_tanh", True),
        use_alive_mask=extra.get("use_alive_mask", False),
        alive_threshold=extra.get("alive_threshold", 0.05),
    ).to(device)

    if "G_ema_state" in ckpt:
        state = {k.replace("module.", ""): v
                 for k, v in ckpt["G_ema_state"].items() if k != "n_averaged"}
    else:
        state = ckpt["G_state"]
    G.load_state_dict(state)
    G.eval()

    hc = extra.get("hidden_channels", 0)
    nca_steps = extra["nca_steps"]

    def generate(img_ctrl, cpd_id):
        if hc > 0:
            pad = torch.zeros(img_ctrl.shape[0], hc,
                              img_ctrl.shape[2], img_ctrl.shape[3],
                              device=device)
            x = torch.cat([img_ctrl, pad], dim=1)
        else:
            x = img_ctrl
        cond = fp_matrix[cpd_id]
        fake = G(x, cond, n_steps=nca_steps)
        return fake[:, :3]

    return generate, extra


def load_impa(ckpt_path, fp_matrix, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = ckpt["extra"]

    G = IMPAGenerator(
        img_size=extra["image_size"],
        style_dim=extra["style_dim"],
        max_conv_dim=extra["max_conv_dim"],
    ).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    M = IMPAMappingNetwork(
        latent_dim=fp_matrix.shape[1] + extra["z_dimension"],
        style_dim=extra["style_dim"],
        num_layers=extra["num_layers_mapping_net"],
    ).to(device)
    M.load_state_dict(ckpt["M_state"])
    M.eval()

    z_dim = extra["z_dimension"]

    def generate(img_ctrl, cpd_id):
        fp = fp_matrix[cpd_id]
        z = torch.randn(img_ctrl.shape[0], z_dim, device=device)
        s = M(torch.cat([fp, z], dim=1))
        _, fake = G(img_ctrl, s)
        return fake[:, :3]

    return generate, extra


def load_cellflux(ckpt_path, fp_matrix, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = ckpt["extra"]

    model = CellFluxUNet(
        in_channels=3,
        model_channels=extra["model_channels"],
        num_res_blocks=extra["num_res_blocks"],
        dropout=0.3,
        channel_mult=tuple(extra["channel_mult"]),
        attention_resolutions=extra["attention_resolutions"],
        use_scale_shift_norm=True,
        use_new_attention_order=True,
        condition_dim=extra["condition_dim"],
    )
    # EMA is stored as shadow dict
    if "ema_state" in ckpt and isinstance(ckpt["ema_state"], dict) and "shadow" in ckpt["ema_state"]:
        shadow = ckpt["ema_state"]["shadow"]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.data.copy_(shadow[name])
    else:
        model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    ode_nfe = extra.get("ode_nfe", 50)
    cfg_scale = extra.get("cfg_scale", 0.2)
    time_grid = edm_time_grid(ode_nfe).to(device)

    def generate(img_ctrl, cpd_id):
        cond = fp_matrix[cpd_id]
        fake = ode_sample_heun(model, img_ctrl, time_grid,
                               cond=cond, cfg_scale=cfg_scale)
        return fake[:, :3]

    return generate, extra


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name, generate_fn, loader, id2cpd, cpd_to_moa,
                   moa_model, moa2id, id2moa, device):
    """Run generation + MoA classification, return per-sample results."""
    preds_id, labels_id = [], []
    preds_ood, labels_ood = [], []
    n = 0

    with torch.no_grad():
        for img_ctrl, img_trt, cpd_id, dose in loader:
            img_ctrl = img_ctrl.to(device)

            fake_rgb = generate_fn(img_ctrl, cpd_id)
            fake_01 = (fake_rgb.clamp(-1, 1) + 1) / 2

            logits = moa_model(fake_01)
            preds = logits.argmax(dim=1).cpu()

            for i in range(cpd_id.shape[0]):
                cpd_name = id2cpd[cpd_id[i].item()]
                moa = cpd_to_moa.get(cpd_name)
                if moa and moa in moa2id:
                    p, l = preds[i].item(), moa2id[moa]
                    if cpd_name in OOD_COMPOUNDS:
                        preds_ood.append(p); labels_ood.append(l)
                    else:
                        preds_id.append(p); labels_id.append(l)

            n += cpd_id.shape[0]
            print(f"\r  {name}: {n}/{len(loader.dataset)}", end="", flush=True)

    print()
    results = {}
    for split, p, l in [("id", preds_id, labels_id),
                         ("ood", preds_ood, labels_ood),
                         ("all", preds_id + preds_ood, labels_id + labels_ood)]:
        p, l = np.array(p), np.array(l)
        if len(p) == 0:
            continue
        results[split] = {
            "accuracy": float((p == l).mean()),
            "macro_f1": float(f1_score(l, p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(l, p, average="weighted", zero_division=0)),
            "n": len(p),
        }
    return results


def main():
    p = argparse.ArgumentParser(description="MoA classification comparison")
    p.add_argument("--metadata_csv", type=str,
                   default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--fp_path", type=str,
                   default="data/bbbc021_six/metadata/emb_morgan_fp.csv")
    p.add_argument("--image_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Checkpoint paths (leave empty to skip a model)
    p.add_argument("--nca_ckpt", type=str, default=None)
    p.add_argument("--nca_inter_ckpt", type=str, default=None)
    p.add_argument("--impa_ckpt", type=str, default=None)
    p.add_argument("--cellflux_ckpt", type=str, default=None)
    p.add_argument("--moa_ckpt", type=str, required=True)

    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

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

    # --- MoA classifier ---
    moa_model, moa2id, id2moa = load_moa_classifier(args.moa_ckpt, device)
    print(f"MoA classifier: {len(moa2id)} classes")

    # --- Eval dataset (deterministic, shared across all models) ---
    eval_ds = EvalDataset(
        args.metadata_csv, args.image_dir,
        split="test", image_size=args.image_size,
    )
    id2cpd = {v: k for k, v in eval_ds.cpd2id.items()}
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Eval dataset: {len(eval_ds)} test images, {len(eval_ds.cpd2id)} compounds")

    # --- FP matrix ---
    fp_matrix = build_fp_matrix(eval_ds.cpd2id, args.fp_path, device)

    # --- MoA ground truth ---
    meta = pd.read_csv(args.metadata_csv, index_col=0)
    cpd_to_moa = meta[meta["STATE"] == 1].groupby("CPD_NAME")["ANNOT"].first().to_dict()

    # --- Also eval on real treated images as ceiling ---
    print("\n--- Real treated images (ceiling) ---")
    preds_id, labels_id, preds_ood, labels_ood = [], [], [], []
    with torch.no_grad():
        for img_ctrl, img_trt, cpd_id, dose in loader:
            real_01 = (img_trt.clamp(-1, 1) + 1) / 2
            logits = moa_model(real_01.to(device))
            preds = logits.argmax(dim=1).cpu()
            for i in range(cpd_id.shape[0]):
                cpd_name = id2cpd[cpd_id[i].item()]
                moa = cpd_to_moa.get(cpd_name)
                if moa and moa in moa2id:
                    p, l = preds[i].item(), moa2id[moa]
                    if cpd_name in OOD_COMPOUNDS:
                        preds_ood.append(p); labels_ood.append(l)
                    else:
                        preds_id.append(p); labels_id.append(l)
    real_results = {}
    for split, pr, lr in [("id", preds_id, labels_id), ("ood", preds_ood, labels_ood),
                           ("all", preds_id + preds_ood, labels_id + labels_ood)]:
        pr, lr = np.array(pr), np.array(lr)
        real_results[split] = {
            "accuracy": float((pr == lr).mean()),
            "macro_f1": float(f1_score(lr, pr, average="macro", zero_division=0)),
            "n": len(pr),
        }

    # --- Evaluate each model ---
    models = [
        ("NCA", args.nca_ckpt, load_nca),
        ("NCA-inter", args.nca_inter_ckpt, load_nca),
        ("IMPA", args.impa_ckpt, load_impa),
        ("CellFlux", args.cellflux_ckpt, load_cellflux),
    ]

    all_results = {"Real": real_results}

    for name, ckpt_path, loader_fn in models:
        if ckpt_path is None:
            print(f"\n--- {name}: SKIPPED (no checkpoint) ---")
            continue
        print(f"\n--- {name} ---")
        generate_fn, extra = loader_fn(ckpt_path, fp_matrix, device)
        results = evaluate_model(
            name, generate_fn, loader, id2cpd, cpd_to_moa,
            moa_model, moa2id, id2moa, device,
        )
        all_results[name] = results

    # --- Summary table ---
    print("\n" + "=" * 80)
    print(f"{'Model':<14} {'Split':<6} {'Accuracy':>8} {'Macro F1':>9} {'Weighted F1':>11} {'n':>6}")
    print("-" * 80)
    for model_name in ["Real", "NCA", "NCA-inter", "IMPA", "CellFlux"]:
        if model_name not in all_results:
            continue
        for split in ["all", "id", "ood"]:
            if split not in all_results[model_name]:
                continue
            r = all_results[model_name][split]
            wf1 = r.get("weighted_f1", "")
            wf1_str = f"{wf1:>11.4f}" if wf1 else f"{'':>11}"
            print(f"{model_name:<14} {split:<6} {r['accuracy']:>8.4f} {r['macro_f1']:>9.4f} {wf1_str} {r['n']:>6}")
        print()


if __name__ == "__main__":
    main()
