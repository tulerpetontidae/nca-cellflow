"""
Capacity probe: train a Discriminator or StyleEncoder backbone as a plain
multi-class compound classifier on real BBBC021 treated images.

This is a decoupled calibration step (no GAN, no NCA) used to measure the
representational ceiling of each candidate backbone before we commit it to
GAN training. See VISION.md and the `configs/probe/*.yaml` sweep.

Usage:
    python scripts/probe_classifier.py --config configs/probe/probe-d-bigD.yaml
    python scripts/probe_classifier.py --config configs/probe/probe-d-bigD.yaml --iterations 200 --num_workers 0
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed")

from nca_cellflow import ClassificationDataset
from nca_cellflow.models import DiscriminatorClassifier, StyleEncoderClassifier


# ---------------------------------------------------------------------------
# Argparse + YAML config loader (pattern copied from scripts/train.py:659)
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser()

    # What to probe
    p.add_argument("--probe_type", type=str, default="discriminator",
                   choices=["discriminator", "style_encoder"])

    # Data
    p.add_argument("--metadata_csv", type=str, required=False)
    p.add_argument("--image_dir", type=str, required=False)
    p.add_argument("--image_size", type=int, default=48)
    p.add_argument("--balanced_cpd", action="store_true", default=True)
    p.add_argument("--in_channels", type=int, default=3)

    # Discriminator arch (when probe_type=discriminator)
    p.add_argument("--d_stages", type=int, default=3,
                   help="Number of downsampling stages. Total widths = stages+1.")
    p.add_argument("--d_base_channels", type=int, default=32)
    p.add_argument("--d_blocks", type=int, default=2)
    p.add_argument("--d_cardinality", type=int, default=4)
    p.add_argument("--d_expansion", type=int, default=2)
    p.add_argument("--d_kernel_size", type=int, default=3)

    # StyleEncoder arch (when probe_type=style_encoder)
    p.add_argument("--s_base_channels", type=int, default=64)
    p.add_argument("--s_num_downsamples", type=int, default=4)
    p.add_argument("--s_max_channels", type=int, default=512)

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)

    # Eval
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=50)

    # Misc
    p.add_argument("--wandb_project", type=str, default="nca-cellflow-probe")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="probe_checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI overrides)")
    p.add_argument("--seed", type=int, default=0)

    return p


def load_config_into_args(args):
    """Load YAML config and use as defaults; CLI args take priority.

    Mirrors scripts/train.py:659. CLI args that equal the argparse default are
    overwritten by the YAML; any CLI value the user explicitly passed is kept.
    """
    if args.config is None:
        return args
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    parser = make_parser()
    type_map = {}
    for action in parser._actions:
        for opt in action.option_strings:
            name = opt.lstrip("-").replace("-", "_")
            if action.type is not None:
                type_map[name] = action.type
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
            if k in type_map and not isinstance(v, type_map[k]):
                v = type_map[k](v)
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(args, num_classes: int) -> torch.nn.Module:
    if args.probe_type == "discriminator":
        widths = [args.d_base_channels] * (args.d_stages + 1)
        blocks = [args.d_blocks] * (args.d_stages + 1)
        cards = [args.d_cardinality] * (args.d_stages + 1)
        return DiscriminatorClassifier(
            widths=widths,
            cardinalities=cards,
            blocks_per_stage=blocks,
            expansion=args.d_expansion,
            kernel_size=args.d_kernel_size,
            in_channels=args.in_channels,
            num_classes=num_classes,
        )
    elif args.probe_type == "style_encoder":
        return StyleEncoderClassifier(
            in_channels=args.in_channels,
            base_channels=args.s_base_channels,
            num_downsamples=args.s_num_downsamples,
            num_classes=num_classes,
            max_channels=args.s_max_channels,
        )
    else:
        raise ValueError(f"unknown probe_type: {args.probe_type}")


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_lambda(step: int, warmup: int, total: int, min_mult: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_mult * lr`."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_mult + (1.0 - min_mult) * cosine


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes: int,
             id2cpd: dict[int, str]) -> tuple[dict, dict, np.ndarray]:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0

    per_cpd_correct = np.zeros(num_classes, dtype=np.int64)
    per_cpd_total = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for img, cpd_id in val_loader:
        img = img.to(device, non_blocking=True)
        cpd_id = cpd_id.to(device, non_blocking=True)
        logits = model(img)
        loss = F.cross_entropy(logits, cpd_id, reduction="sum")
        loss_sum += loss.item()
        total += cpd_id.size(0)

        pred_top1 = logits.argmax(dim=1)
        top1 += (pred_top1 == cpd_id).sum().item()

        _, pred_top5 = logits.topk(5, dim=1)
        top5 += (pred_top5 == cpd_id[:, None]).any(dim=1).sum().item()

        # per-class stats (on CPU, small loop over batch is fine)
        pred_np = pred_top1.cpu().numpy()
        gt_np = cpd_id.cpu().numpy()
        for i in range(gt_np.shape[0]):
            g = gt_np[i]; p = pred_np[i]
            per_cpd_total[g] += 1
            if p == g:
                per_cpd_correct[g] += 1
            confusion[g, p] += 1

    metrics = {
        "val_loss": loss_sum / max(1, total),
        "val_acc1": top1 / max(1, total),
        "val_acc5": top5 / max(1, total),
    }
    per_cpd_acc = {}
    for cid in range(num_classes):
        if per_cpd_total[cid] > 0:
            per_cpd_acc[id2cpd[cid]] = per_cpd_correct[cid] / per_cpd_total[cid]

    return metrics, per_cpd_acc, confusion


def plot_confusion(confusion: np.ndarray, id2cpd: dict[int, str]) -> plt.Figure:
    """Row-normalized confusion matrix, rows = true class."""
    cm = confusion.astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm = cm / row_sums
    n = cm.shape[0]
    labels = [id2cpd[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()
    args = load_config_into_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- data ----
    train_ds = ClassificationDataset(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        split="train",
        image_size=args.image_size,
        balanced_cpd=args.balanced_cpd,
        iter_all=False,
        augment=True,
    )
    val_ds = ClassificationDataset(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        split="test",
        image_size=args.image_size,
        balanced_cpd=False,
        iter_all=True,
        augment=False,
    )
    num_classes = train_ds.num_classes
    assert num_classes == val_ds.num_classes, "train/val label space mismatch"
    print(f"Train: {len(train_ds)} treated images | Val: {len(val_ds)} | "
          f"{num_classes} compounds")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- model ----
    model = build_model(args, num_classes=num_classes).to(device)
    total_params, train_params = count_params(model)
    print(f"{args.probe_type} params: total={total_params:,}  trainable={train_params:,}")

    # ---- optimizer ----
    try:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay, fused=True)
    except Exception:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: lr_lambda(s, args.warmup_steps, args.iterations)
    )

    # ---- AMP ----
    use_amp = device.type == "cuda"
    autocast = lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # ---- wandb ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )
        wandb.log({
            "probe/params_total": total_params,
            "probe/params_trainable": train_params,
            "probe/num_classes": num_classes,
        }, step=0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- infinite batch iterator ----
    data_iter = iter(train_loader)
    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    # ---- training loop ----
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    best_val_acc1 = -1.0

    pbar = tqdm(range(args.iterations), desc=f"probe[{args.probe_type}]")
    for step in pbar:
        model.train()
        img, cpd_id = next_batch()
        img = img.to(device, non_blocking=True)
        cpd_id = cpd_id.to(device, non_blocking=True)

        with autocast():
            logits = model(img)
            loss = F.cross_entropy(logits.float(), cpd_id)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        scheduler.step()

        running_loss += loss.item() * cpd_id.size(0)
        running_correct += (logits.argmax(dim=1) == cpd_id).sum().item()
        running_total += cpd_id.size(0)

        if (step + 1) % args.log_every == 0:
            train_loss = running_loss / running_total
            train_acc1 = running_correct / running_total
            pbar.set_postfix(loss=f"{train_loss:.3f}", acc1=f"{train_acc1:.3f}")
            if use_wandb:
                wandb.log({
                    "probe/train_loss": train_loss,
                    "probe/train_acc1": train_acc1,
                    "probe/lr": scheduler.get_last_lr()[0],
                    "probe/grad_norm": float(grad_norm),
                    "step": step + 1,
                }, step=step + 1)
            running_loss = 0.0
            running_correct = 0
            running_total = 0

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.iterations:
            metrics, per_cpd, confusion = evaluate(
                model, val_loader, device,
                num_classes=num_classes,
                id2cpd=train_ds.id2cpd,
            )
            print(f"[step {step + 1}] "
                  f"val_acc1={metrics['val_acc1']:.4f}  "
                  f"val_acc5={metrics['val_acc5']:.4f}  "
                  f"val_loss={metrics['val_loss']:.4f}")
            if use_wandb:
                log_dict = {
                    "probe/val_loss": metrics["val_loss"],
                    "probe/val_acc1": metrics["val_acc1"],
                    "probe/val_acc5": metrics["val_acc5"],
                    "step": step + 1,
                }
                for cpd_name, acc in per_cpd.items():
                    log_dict[f"probe/per_cpd/{cpd_name}"] = acc
                wandb.log(log_dict, step=step + 1)

            # Save the best model (small — no harm in keeping)
            if metrics["val_acc1"] > best_val_acc1:
                best_val_acc1 = metrics["val_acc1"]
                ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "best_val_acc1": best_val_acc1,
                }, ckpt_path)

            # On final eval, log confusion matrix image
            if (step + 1) == args.iterations:
                fig = plot_confusion(confusion, train_ds.id2cpd)
                if use_wandb:
                    wandb.log({"probe/confusion_matrix": wandb.Image(fig),
                               "step": step + 1}, step=step + 1)
                plt.close(fig)

    print(f"Done. Best val_acc1 = {best_val_acc1:.4f}")
    if use_wandb:
        wandb.log({"probe/best_val_acc1": best_val_acc1}, step=args.iterations)
        wandb.finish()


if __name__ == "__main__":
    main()
