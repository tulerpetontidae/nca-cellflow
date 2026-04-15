"""
Evaluate MoA classification accuracy for a single generative model.

Runs independently per model so fast models (NCA, IMPA) don't wait for slow
ones (CellFlux). Saves results to a JSON file and optionally logs to wandb.

Usage:
    python scripts/eval_moa.py --model nca \
        --ckpt checkpoints/nca/step_160000.pt \
        --moa_ckpt checkpoints/moa_classifier_48px/best.pt \
        --image_size 48 --wandb

    python scripts/eval_moa.py --model cellflux \
        --ckpt checkpoints/cellflux/step_80000.pt \
        --moa_ckpt checkpoints/moa_classifier_48px/best.pt \
        --image_size 48 --wandb

    # Ground truth (real treated images, no generative model needed)
    python scripts/eval_moa.py --model real \
        --moa_ckpt checkpoints/moa_classifier_48px/best.pt \
        --image_size 48
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from nca_cellflow import EvalDataset
from nca_cellflow.models import MOAClassifier

OOD_COMPOUNDS = {
    "docetaxel", "AZ841", "cytochalasin D", "simvastatin",
    "cyclohexamide", "latrunculin B", "epothilone B", "lactacystin",
}

DEFAULT_WANDB_PROJECT = "nca-cellflow-eval"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def build_fp_matrix(cpd2id, fp_path, device):
    id2cpd = {v: k for k, v in cpd2id.items()}
    fp_df = pd.read_csv(fp_path, index_col=0)
    fp_vecs = [fp_df.loc[id2cpd[i]].values.astype(np.float32)
               for i in range(len(cpd2id))]
    return torch.tensor(np.stack(fp_vecs)).to(device)


def load_nca(ckpt_path, fp_matrix, device):
    from nca_cellflow.models import LatentNCA

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
    from nca_cellflow.models import IMPAGenerator, IMPAMappingNetwork

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
    from nca_cellflow.models import CellFluxUNet
    from nca_cellflow.models.cellflux_unet import ode_sample_heun, edm_time_grid

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


MODEL_LOADERS = {
    "nca": load_nca,
    "nca-inter": load_nca,
    "impa": load_impa,
    "cellflux": load_cellflux,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels):
    p, l = np.array(preds), np.array(labels)
    if len(p) == 0:
        return None
    return {
        "accuracy": float((p == l).mean()),
        "macro_f1": float(f1_score(l, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(l, p, average="weighted", zero_division=0)),
        "n": len(p),
    }


def evaluate(generate_fn, loader, id2cpd, cpd_to_moa,
             moa_model, moa2id, device, model_name):
    preds_id, labels_id = [], []
    preds_ood, labels_ood = [], []
    n = 0

    with torch.no_grad():
        for img_ctrl, img_trt, cpd_id, dose in loader:
            img_ctrl = img_ctrl.to(device)

            if generate_fn is not None:
                fake_rgb = generate_fn(img_ctrl, cpd_id)
            else:
                # Real treated images
                fake_rgb = img_trt.to(device)

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
            print(f"\r  {model_name}: {n}/{len(loader.dataset)}", end="", flush=True)

    print()
    results = {}
    for split, p, l in [("id", preds_id, labels_id),
                         ("ood", preds_ood, labels_ood),
                         ("all", preds_id + preds_ood, labels_id + labels_ood)]:
        m = compute_metrics(p, l)
        if m is not None:
            results[split] = m
    return results


def main():
    p = argparse.ArgumentParser(description="MoA eval for a single model")
    p.add_argument("--model", type=str, required=True,
                   choices=["nca", "nca-inter", "impa", "cellflux", "real"],
                   help="Model to evaluate")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Model checkpoint (not needed for --model real)")
    p.add_argument("--metadata_csv", type=str,
                   default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--fp_path", type=str,
                   default="data/bbbc021_six/metadata/emb_morgan_fp.csv")
    p.add_argument("--image_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--moa_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="figures/moa_eval")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Override wandb project name")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.model != "real" and args.ckpt is None:
        p.error("--ckpt is required for model types other than 'real'")

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

    # MoA classifier
    moa_model, moa2id, id2moa = load_moa_classifier(args.moa_ckpt, device)
    print(f"MoA classifier: {len(moa2id)} classes")

    # Eval dataset
    eval_ds = EvalDataset(
        args.metadata_csv, args.image_dir,
        split="test", image_size=args.image_size,
    )
    id2cpd = {v: k for k, v in eval_ds.cpd2id.items()}
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Eval dataset: {len(eval_ds)} test images, {len(eval_ds.cpd2id)} compounds")

    # MoA ground truth mapping
    meta = pd.read_csv(args.metadata_csv, index_col=0)
    cpd_to_moa = meta[meta["STATE"] == 1].groupby("CPD_NAME")["ANNOT"].first().to_dict()

    # Load generative model
    if args.model == "real":
        generate_fn = None
        extra = {}
    else:
        fp_matrix = build_fp_matrix(eval_ds.cpd2id, args.fp_path, device)
        loader_fn = MODEL_LOADERS[args.model]
        generate_fn, extra = loader_fn(args.ckpt, fp_matrix, device)

    # Evaluate
    print(f"\n--- {args.model} ---")
    results = evaluate(
        generate_fn, loader, id2cpd, cpd_to_moa,
        moa_model, moa2id, device, args.model,
    )

    # Print
    print(f"\n{'Split':<6} {'Accuracy':>8} {'Macro F1':>9} {'Weighted F1':>11} {'n':>6}")
    print("-" * 45)
    for split in ["all", "id", "ood"]:
        if split not in results:
            continue
        r = results[split]
        print(f"{split:<6} {r['accuracy']:>8.4f} {r['macro_f1']:>9.4f} "
              f"{r['weighted_f1']:>11.4f} {r['n']:>6}")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"moa_{args.model}_{args.image_size}px.json")
    save_data = {
        "model": args.model,
        "image_size": args.image_size,
        "ckpt": args.ckpt,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Wandb
    if args.wandb:
        import wandb
        project = args.wandb_project or DEFAULT_WANDB_PROJECT
        run_name = args.wandb_name or f"moa-eval-{args.model}-{args.image_size}px"
        wandb.init(project=project, name=run_name,
                   config={"model": args.model, "image_size": args.image_size,
                           "ckpt": args.ckpt, **extra})
        for split, r in results.items():
            for metric, val in r.items():
                wandb.log({f"moa/{split}/{metric}": val})
        wandb.finish()
        print(f"Logged to wandb project={project}")


def load_moa_classifier(moa_ckpt_path, device):
    ckpt = torch.load(moa_ckpt_path, map_location="cpu", weights_only=False)
    model = MOAClassifier(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict_head(ckpt["classifier_state"])
    model.eval()
    return model, ckpt["moa2id"], ckpt["id2moa"]


if __name__ == "__main__":
    main()
