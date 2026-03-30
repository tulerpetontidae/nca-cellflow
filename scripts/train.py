"""
NCA-GAN training script for IMPA cellular morphology generation.

Single-passage GAN: NCA transforms control (DMSO) images toward treated distributions,
conditioned on compound identity. Uses relativistic GAN loss with zero-centered
gradient penalty.
"""

import os
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("[warn] torchmetrics not installed — FID evaluation disabled")

from nca_cellflow import IMPADataset, EvalDataset
from nca_cellflow.models import BaseNCA, NoiseNCA, LatentNCA, NCAStyleEncoder, Discriminator, PatchDiscriminator


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_cond_fn(cond_type: str, fp_matrix: torch.Tensor | None):
    """Return a function that maps cpd_id (int tensor) to conditioning input for G."""
    if cond_type == "fingerprint":
        assert fp_matrix is not None
        def cond_fn(cpd_id):
            return fp_matrix[cpd_id]  # [B, fp_dim]
        return cond_fn
    else:
        return lambda cpd_id: cpd_id  # passthrough


def zero_centered_gradient_penalty(samples: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    # For patch D, logits has spatial dims [B, H', W'] — average over patches
    # before summing over batch so the GP doesn't scale with num_patches.
    if logits.dim() > 1:
        # Patch D: average over spatial dims per sample, then sum over batch
        reduced = logits.reshape(logits.shape[0], -1).mean(dim=1).sum()
    else:
        # Global D: scalar per sample
        reduced = logits.sum()
    (grad,) = torch.autograd.grad(outputs=reduced, inputs=samples, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def relativistic_d_loss(d_real, d_fake):
    """Compute relativistic D adversarial loss + GP for dict or tensor D outputs.

    For PatchDiscriminator (dict with 'patch' and 'global' keys), computes
    separate losses for each scale and sums them.
    For global Discriminator (tensor), computes a single loss.
    """
    if isinstance(d_real, dict):
        loss = 0.0
        for key in d_real:
            rel = d_real[key] - d_fake[key]
            loss = loss + F.softplus(-rel).mean()
        return loss
    else:
        rel = d_real - d_fake
        return F.softplus(-rel).mean()


def _to_float(d_out):
    """Cast D output to float (handles dict or tensor)."""
    if isinstance(d_out, dict):
        return {k: v.float() for k, v in d_out.items()}
    return d_out.float()


def multi_scale_gp(samples, d_out):
    """Compute GP for dict or tensor D outputs.

    For PatchDiscriminator, combines all heads into a single scalar before
    computing gradients so we only call autograd.grad once per input.
    """
    if isinstance(d_out, dict):
        # Combine: average patches per sample for patch head, keep global as-is, sum all
        combined = 0.0
        for key in d_out:
            v = d_out[key]
            if v.dim() > 1:
                combined = combined + v.reshape(v.shape[0], -1).mean(dim=1).sum()
            else:
                combined = combined + v.sum()
        (grad,) = torch.autograd.grad(outputs=combined, inputs=samples, create_graph=True)
        return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)
    else:
        return zero_centered_gradient_penalty(samples, d_out)


def d_logit_mean(d_out):
    """Mean logit value for logging."""
    if isinstance(d_out, dict):
        return sum(v.mean().item() for v in d_out.values()) / len(d_out)
    return d_out.mean().item()


def set_requires_grad(model: torch.nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_rgb_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] CHW tensor to [0,1] HWC numpy for plotting."""
    img = img.detach().cpu().float().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, G, D, G_opt, D_opt, logs, extra=None, G_ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_opt_state": G_opt.state_dict(),
        "D_opt_state": D_opt.state_dict(),
        "logs": dict(logs),
        "extra": extra,
    }
    if G_ema is not None:
        state["G_ema_state"] = G_ema.state_dict()
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path, G, D, G_opt=None, D_opt=None, G_ema=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    if G_opt is not None and "G_opt_state" in ckpt:
        G_opt.load_state_dict(ckpt["G_opt_state"])
    if D_opt is not None and "D_opt_state" in ckpt:
        D_opt.load_state_dict(ckpt["D_opt_state"])
    if G_ema is not None and "G_ema_state" in ckpt:
        G_ema.load_state_dict(ckpt["G_ema_state"])
    logs = defaultdict(list, ckpt.get("logs", {}))
    step = ckpt.get("step", 0)
    extra = ckpt.get("extra", None)
    print(f"[ckpt] loaded {path} at step {step}")
    return step, logs, extra


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectories(G, dataset, device, args, id2cpd, img_channels, step, cond_fn=lambda x: x):
    """
    NCA trajectories: one row per compound, 5 columns = intermediate states.
    Uses first control image as input for all conditions.
    """
    G.eval()
    n_cols = 5
    n_steps = args.nca_steps
    output_steps = sorted(set([0] + [int(round(i * n_steps / (n_cols - 1))) for i in range(n_cols)]))

    img_ctrl = dataset._load(dataset.ctrl_keys[0]).unsqueeze(0).to(device)
    if args.use_alive_mask:
        ctrl_01 = (img_ctrl + 1) / 2
        alive = (ctrl_01.max(dim=1, keepdim=True).values > args.alive_threshold).float()
        img_ctrl = torch.cat([img_ctrl, alive], dim=1)
    if args.hidden_channels > 0:
        pad = torch.zeros(1, args.hidden_channels, img_ctrl.shape[2], img_ctrl.shape[3], device=device)
        nca_input = torch.cat([img_ctrl, pad], dim=1)
    else:
        nca_input = img_ctrl

    num_compounds = len(id2cpd)
    fig, axes = plt.subplots(num_compounds, len(output_steps),
                             figsize=(3 * len(output_steps), 3 * num_compounds))
    if num_compounds == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for cpd_idx in range(num_compounds):
            cond = torch.tensor([cpd_idx], device=device)
            trajectory = G.sample(nca_input, cond_fn(cond), n_steps=n_steps, output_steps=output_steps)
            for col, (t_step, state) in enumerate(zip(output_steps, trajectory)):
                rgb = to_rgb_numpy(state[0, :3])
                ax = axes[cpd_idx, col]
                ax.imshow(rgb)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if cpd_idx == 0:
                    ax.set_title(f"t={t_step}", fontsize=40)
            axes[cpd_idx, 0].set_ylabel(id2cpd[cpd_idx], fontsize=36,
                                        rotation=90, labelpad=60, va="center")

    fig.suptitle(f"Trajectories (step {step})", fontsize=48)
    plt.tight_layout()
    return fig


def plot_image_grid(images, titles, suptitle, max_images=16):
    """Grid of images with per-image titles. images: list of CHW tensors in [-1,1]."""
    n = min(len(images), max_images)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        if i < n:
            ax.imshow(to_rgb_numpy(images[i]))
            ax.set_title(titles[i], fontsize=16)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=24)
    plt.tight_layout()
    return fig


def log_visualizations(G, dataset, device, args, id2cpd, img_channels,
                       fake_img, img_ctrl, img_trt, cpd_id, step, use_wandb,
                       cond_fn=lambda x: x):
    """Generate and display/log all visualization figures.

    Args:
        fake_img: Already-computed NCA output (visible channels only) from this training step.
        img_ctrl: Control images that were fed into the NCA this step.
    """
    G.eval()

    # 1. Trajectory plot
    fig_traj = plot_trajectories(G, dataset, device, args, id2cpd, img_channels, step, cond_fn=cond_fn)
    if use_wandb:
        wandb.log({"vis/trajectories": wandb.Image(fig_traj)}, step=step)
        plt.close(fig_traj)
    else:
        plt.show()

    n = min(img_trt.shape[0], 16)
    titles = [id2cpd[cpd_id[i].item()] for i in range(n)]

    # 2. Real control samples (NCA input) — always show RGB only
    fig_ctrl = plot_image_grid(
        [img_ctrl[i, :3] for i in range(n)],
        ["DMSO"] * n,
        f"Control / NCA input (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/control_samples": wandb.Image(fig_ctrl)}, step=step)
        plt.close(fig_ctrl)
    else:
        plt.show()

    # 3. Generated samples (NCA outputs from this training step) — RGB only
    fig_fake = plot_image_grid(
        [fake_img[i, :3].detach() for i in range(n)],
        titles,
        f"Generated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/generated_samples": wandb.Image(fig_fake)}, step=step)
        plt.close(fig_fake)
    else:
        plt.show()

    # 4. Real treated samples (target distribution) — RGB only
    fig_real = plot_image_grid(
        [img_trt[i, :3] for i in range(n)],
        titles,
        f"Real treated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/real_samples": wandb.Image(fig_real)}, step=step)
        plt.close(fig_real)
    else:
        plt.show()

    G.train()


# ---------------------------------------------------------------------------
# FID Evaluation
# ---------------------------------------------------------------------------

_global_fid = None
_cpd_fid = None


@torch.no_grad()
def compute_fid(G, eval_dataset, device, args, id2cpd, img_channels, cond_fn=lambda x: x):
    """Compute global and per-compound FID using torchmetrics.

    Uses EvalDataset: deterministic iteration over all treated test images,
    deterministic same-plate ctrl pairing, no augmentation.
    Matches CellFlux evaluation protocol.
    """
    global _global_fid, _cpd_fid
    if not FID_AVAILABLE:
        print("[warn] torchmetrics not available, skipping FID")
        return None, None

    try:
        if _global_fid is None:
            _global_fid = FrechetInceptionDistance(normalize=True).to(device)
            _cpd_fid = FrechetInceptionDistance(normalize=True).to(device)

        loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False,
        )

        _global_fid.reset()
        per_cpd_real = defaultdict(list)
        per_cpd_fake = defaultdict(list)

        G.eval()
        for img_ctrl, img_trt, cpd_id in loader:
            img_ctrl = img_ctrl.to(device)
            img_trt = img_trt.to(device)
            cpd_id = cpd_id.to(device)

            if args.use_alive_mask:
                ctrl_01 = (img_ctrl + 1) / 2
                alive_ctrl = (ctrl_01.max(dim=1, keepdim=True).values > args.alive_threshold).float()
                img_ctrl_full = torch.cat([img_ctrl, alive_ctrl], dim=1)
            else:
                img_ctrl_full = img_ctrl

            if args.hidden_channels > 0:
                pad = torch.zeros(
                    img_ctrl_full.shape[0], args.hidden_channels,
                    img_ctrl_full.shape[2], img_ctrl_full.shape[3], device=device,
                )
                nca_input = torch.cat([img_ctrl_full, pad], dim=1)
            else:
                nca_input = img_ctrl_full

            fake_full = G(nca_input, cond_fn(cpd_id), n_steps=args.nca_steps)
            fake_rgb = fake_full[:, :3].contiguous()

            # [-1, 1] -> [0, 1], quantize to uint8 like CellFlux eval
            real_01 = (img_trt.clamp(-1, 1) + 1) / 2
            real_01 = torch.floor(real_01 * 255).float() / 255.0
            fake_01 = (fake_rgb.clamp(-1, 1) + 1) / 2
            fake_01 = torch.floor(fake_01 * 255).float() / 255.0

            _global_fid.update(real_01, real=True)
            _global_fid.update(fake_01, real=False)

            for i in range(cpd_id.shape[0]):
                cid = cpd_id[i].item()
                per_cpd_real[cid].append(real_01[i].cpu())
                per_cpd_fake[cid].append(fake_01[i].cpu())

        fid_global = _global_fid.compute().item()

        # Per-compound FID
        fid_per_cpd = {}
        for cid in sorted(per_cpd_real.keys()):
            real_stack = torch.stack(per_cpd_real[cid]).to(device)
            fake_stack = torch.stack(per_cpd_fake[cid]).to(device)
            if len(real_stack) < 2 or len(fake_stack) < 2:
                continue
            _cpd_fid.reset()
            _cpd_fid.update(real_stack, real=True)
            _cpd_fid.update(fake_stack, real=False)
            try:
                fid_per_cpd[id2cpd[cid]] = _cpd_fid.compute().item()
            except Exception:
                continue

        G.train()
        return fid_global, fid_per_cpd

    except Exception as e:
        print(f"[warn] FID computation failed: {e}")
        G.train()
        torch.cuda.empty_cache()
        return None, None


_traj_fid = None


def compute_fid_trajectory(G, eval_dataset, device, args, img_channels, cond_fn=lambda x: x):
    """Compute FID at sampled NCA steps against all real cells (ctrl + trt).

    Memory-efficient: only stores one step's worth of fakes at a time.
    Samples ~6 evenly-spaced steps to avoid storing all 30 trajectories.

    Returns dict {step: fid_value} or None on failure.
    """
    global _traj_fid
    if not FID_AVAILABLE:
        return None

    try:
        if _traj_fid is None:
            _traj_fid = FrechetInceptionDistance(normalize=True).to(device)

        # Sample ~6 evenly-spaced steps (always include first and last)
        T = args.nca_steps
        if T <= 6:
            eval_steps = list(range(1, T + 1))
        else:
            step_size = max(1, T // 5)
            eval_steps = list(range(step_size, T, step_size))
            if T not in eval_steps:
                eval_steps.append(T)
            if 1 not in eval_steps:
                eval_steps.insert(0, 1)

        # Use smaller batch size to reduce peak memory
        traj_bs = min(args.batch_size, 64)
        loader = DataLoader(
            eval_dataset, batch_size=traj_bs, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False,
        )

        G.eval()

        # Step 1: compute real stats once, feeding directly to FID (no storage)
        _traj_fid.reset()
        for img_ctrl, img_trt, cpd_id in loader:
            for img in [img_ctrl, img_trt]:
                img_01 = (img.clamp(-1, 1) + 1) / 2
                img_01 = torch.floor(img_01 * 255).float() / 255.0
                _traj_fid.update(img_01.to(device), real=True)
        real_mu = _traj_fid.real_features_sum.clone()
        real_cov = _traj_fid.real_features_cov_sum.clone()
        real_n = _traj_fid.real_features_num_samples.clone()
        torch.cuda.empty_cache()

        # Step 2: for each eval step, generate fakes and compute FID
        fid_traj = {}
        for t_eval in eval_steps:
            _traj_fid.reset()
            _traj_fid.real_features_sum.copy_(real_mu)
            _traj_fid.real_features_cov_sum.copy_(real_cov)
            _traj_fid.real_features_num_samples.copy_(real_n)

            for img_ctrl, img_trt, cpd_id in loader:
                img_ctrl = img_ctrl.to(device)
                cpd_id = cpd_id.to(device)

                # Build NCA input
                if args.use_alive_mask:
                    ctrl_01 = (img_ctrl + 1) / 2
                    alive_ctrl = (ctrl_01.max(dim=1, keepdim=True).values > args.alive_threshold).float()
                    img_ctrl_full = torch.cat([img_ctrl, alive_ctrl], dim=1)
                else:
                    img_ctrl_full = img_ctrl
                if args.hidden_channels > 0:
                    pad = torch.zeros(
                        img_ctrl_full.shape[0], args.hidden_channels,
                        img_ctrl_full.shape[2], img_ctrl_full.shape[3], device=device,
                    )
                    nca_input = torch.cat([img_ctrl_full, pad], dim=1)
                else:
                    nca_input = img_ctrl_full

                with torch.no_grad():
                    cond = cond_fn(cpd_id)
                    out = G(nca_input, cond, n_steps=t_eval)
                    rgb = out[:, :3].clamp(-1, 1)
                    rgb_01 = (rgb + 1) / 2
                    rgb_01 = torch.floor(rgb_01 * 255).float() / 255.0
                    _traj_fid.update(rgb_01, real=False)

            try:
                fid_traj[t_eval] = _traj_fid.compute().item()
            except Exception:
                pass
            torch.cuda.empty_cache()

        G.train()
        return fid_traj

    except Exception as e:
        print(f"[warn] FID trajectory failed: {e}")
        G.train()
        torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NCA-GAN on IMPA dataset")

    # data
    p.add_argument("--metadata_csv", type=str, default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96,
                   help="Resize images to this size (default 96, native resolution)")
    p.add_argument("--plate_match", action="store_true",
                   help="Sample ctrl and trt from the same plate")
    p.add_argument("--balanced_cpd", action="store_true",
                   help="Sample compounds uniformly (then image from that compound)")
    p.add_argument("--iter_trt", action="store_true",
                   help="Iterate over treated images, randomly sample ctrl (CellFlux paper style)")

    # model
    p.add_argument("--nca_type", type=str, default="base",
                   choices=["base", "noise", "latent"],
                   help="NCA variant: 'base', 'noise' (per-step noise), or 'latent' (single z via FiLM)")
    p.add_argument("--nca_hidden_dim", type=int, default=128)
    p.add_argument("--nca_cond_dim", type=int, default=64)
    p.add_argument("--cond_type", type=str, default="id", choices=["id", "fingerprint"],
                   help="Conditioning type: 'id' for learned embedding, 'fingerprint' for Morgan FP projection")
    p.add_argument("--fp_path", type=str, default=None,
                   help="Path to Morgan fingerprint CSV (required when cond_type=fingerprint)")
    p.add_argument("--hidden_channels", type=int, default=0,
                   help="Extra hidden channels for NCA state (0 = RGB only)")
    p.add_argument("--noise_channels", type=int, default=1,
                   help="Number of noise channels for NoiseNCA")
    p.add_argument("--z_dim", type=int, default=16,
                   help="Latent noise dimension for LatentNCA")
    p.add_argument("--fire_rate", type=float, default=1.0)
    p.add_argument("--use_alive_mask", action="store_true",
                   help="Add alive/dead channel (index 3) to prevent generation in empty space")
    p.add_argument("--alive_threshold", type=float, default=0.05,
                   help="Threshold for alive mask (in [0,1] space for init, raw for NCA step)")
    p.add_argument("--step_size", type=float, default=0.1,
                   help="Residual update scale: x = x + dx * step_size")
    p.add_argument("--nca_steps", type=int, default=60,
                   help="Number of NCA steps per generation")

    # discriminator
    p.add_argument("--d_type", type=str, default="global", choices=["global", "patch"],
                   help="Discriminator type: 'global' (scalar output) or 'patch' (spatial score map)")
    p.add_argument("--d_stages", type=int, default=3)
    p.add_argument("--d_base_channels", type=int, default=32)
    p.add_argument("--d_blocks", type=int, default=2)
    p.add_argument("--d_cardinality", type=int, default=4)
    p.add_argument("--d_expansion", type=int, default=2)
    p.add_argument("--d_kernel_size", type=int, default=3)
    p.add_argument("--d_embed_dim", type=int, default=32)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gradient penalty weight")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--diversity_weight", type=float, default=0.0,
                   help="Weight for diversity loss (0 = disabled). Penalizes identical outputs from different noise.")
    p.add_argument("--intermediate_weight", type=float, default=0.0,
                   help="Weight for intermediate step regularization (0 = disabled). "
                        "Adds a null class to D and supervises random intermediate NCA states.")
    p.add_argument("--gradual_weight", type=float, default=0.0,
                   help="Weight for gradual intermediate conditioning (0 = disabled). "
                        "At NCA step t, G loss = alpha*target + (1-alpha)*null, alpha=(t+1)/T.")
    p.add_argument("--style_weight", type=float, default=0.0,
                   help="Weight for style reconstruction loss (LatentNCA only, 0 = disabled). "
                        "Trains a StyleEncoder to recover the style from the generated image.")
    p.add_argument("--accumulate_steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * accumulate_steps)")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay for generator (0 = disabled, e.g. 0.999)")
    p.add_argument("--ema_warmup_steps", type=int, default=1000,
                   help="EMA warmup steps (ramp decay from 0 to ema_decay)")

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--vis_every", type=int, default=1000,
                   help="Log visualization plots every N steps")
    p.add_argument("--grad_diag_every", type=int, default=0,
                   help="Log per-step gradient norms every N steps (0 = disabled). "
                        "Measures ||dL/dx_t|| at each NCA step to diagnose gradient attenuation.")
    p.add_argument("--fid_every", type=int, default=0,
                   help="Compute FID every N steps (0 = disabled, uses all test treated images)")
    p.add_argument("--fid_trajectory_every", type=int, default=0,
                   help="Compute per-step FID trajectory every N steps (0 = disabled)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile on generator")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detected if omitted)")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb logging (requires wandb installed)")
    p.add_argument("--wandb_project", type=str, default="nca-cellflow")
    p.add_argument("--wandb_name", type=str, default=None,
                   help="W&B run name (auto-generated if omitted)")

    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (CLI args override config values)")

    return p


def load_config_into_args(args):
    """Load YAML config and use as defaults; CLI args take priority."""
    if args.config is None:
        return args
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Build type map from argparse actions
    parser = make_parser()
    type_map = {}
    for action in parser._actions:
        for opt in action.option_strings:
            name = opt.lstrip("-").replace("-", "_")
            if action.type is not None:
                type_map[name] = action.type
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
            # Cast to the argparse-declared type if needed
            if k in type_map and not isinstance(v, type_map[k]):
                v = type_map[k](v)
            setattr(args, k, v)
    return args


def train(args):
    args = load_config_into_args(args)
    print(f"[config] ema_decay={args.ema_decay}, fid_every={args.fid_every}, "
          f"accumulate_steps={args.accumulate_steps}, nca_type={args.nca_type}, "
          f"diversity_weight={args.diversity_weight}")

    # ---- wandb ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[warn] --wandb requested but wandb not installed, falling back to plt.show")
    if not use_wandb:
        matplotlib.use("TkAgg")  # interactive backend for plt.show

    # ---- device ----
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- dataset ----
    dataset = IMPADataset(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        split="train",
        image_size=args.image_size,
        plate_match=args.plate_match,
        balanced_cpd=args.balanced_cpd,
        iter_trt=getattr(args, 'iter_trt', False),
    )
    num_compounds = len(dataset.cpd2id)
    id2cpd = {v: k for k, v in dataset.cpd2id.items()}
    print(f"Dataset: {len(dataset)} control images, {len(dataset.trt_keys)} treated images, "
          f"{num_compounds} compounds")

    # ---- eval dataset for FID (deterministic, no augmentation, test split) ----
    eval_dataset = None
    if args.fid_every > 0:
        eval_dataset = EvalDataset(
            metadata_csv=args.metadata_csv,
            image_dir=args.image_dir,
            split="test",
            image_size=args.image_size,
        )
        print(f"FID eval dataset: {len(eval_dataset)} treated test images")

    # ---- fingerprint conditioning ----
    fp_matrix = None  # [num_compounds, fp_dim] tensor on device, or None
    fp_dim = 1024
    if args.cond_type == "fingerprint":
        assert args.fp_path is not None, "--fp_path required when cond_type=fingerprint"
        fp_df = pd.read_csv(args.fp_path, index_col=0)
        fp_vecs = []
        for cid in range(num_compounds):
            cpd_name = id2cpd[cid]
            assert cpd_name in fp_df.index, f"Compound '{cpd_name}' not found in {args.fp_path}"
            fp_vecs.append(fp_df.loc[cpd_name].values.astype(np.float32))
        fp_matrix = torch.tensor(np.stack(fp_vecs)).to(device)  # [num_compounds, fp_dim]
        fp_dim = fp_matrix.shape[1]
        print(f"Loaded fingerprints: {fp_matrix.shape} from {args.fp_path}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(loader)

    # ---- visible channels = 3 RGB + optional alive + hidden ----
    img_channels = 4 if args.use_alive_mask else 3
    channel_n = img_channels + args.hidden_channels

    # ---- models ----
    nca_kwargs = dict(
        channel_n=channel_n,
        hidden_dim=args.nca_hidden_dim,
        num_classes=num_compounds,
        cond_dim=args.nca_cond_dim,
        cond_type=args.cond_type,
        fp_dim=fp_dim,
        fire_rate=args.fire_rate,
        step_size=args.step_size,
        use_alive_mask=args.use_alive_mask,
        alive_threshold=args.alive_threshold,
    )
    if args.nca_type == "noise":
        G = NoiseNCA(noise_channels=args.noise_channels, **nca_kwargs).to(device)
    elif args.nca_type == "latent":
        G = LatentNCA(z_dim=args.z_dim, **nca_kwargs).to(device)
    else:
        G = BaseNCA(**nca_kwargs).to(device)

    # ---- EMA ----
    use_ema = args.ema_decay > 0
    if use_ema:
        ema_decay = args.ema_decay
        ema_warmup = args.ema_warmup_steps

        def _ema_avg_fn(avg_param, model_param, num_averaged):
            decay = min(ema_decay, (1 + num_averaged) / (ema_warmup + num_averaged))
            return avg_param + (1 - decay) * (model_param - avg_param)

        G_ema = torch.optim.swa_utils.AveragedModel(G, avg_fn=_ema_avg_fn)
        print(f"EMA enabled (decay={ema_decay}, warmup={ema_warmup})")
    else:
        G_ema = None

    # Conditioning function: maps cpd_id -> G conditioning input
    cond_fn = make_cond_fn(args.cond_type, fp_matrix)

    if args.compile:
        G = torch.compile(G)

    # Discriminator: extra null class when intermediate or gradual regularization is on
    use_intermediate = args.intermediate_weight > 0
    use_gradual = args.gradual_weight > 0
    needs_null_class = use_intermediate or use_gradual
    d_num_classes = num_compounds + 1 if needs_null_class else num_compounds
    null_class_id = num_compounds  # index of the null/cell class

    d_widths = [args.d_base_channels] * (args.d_stages + 1)
    d_blocks = [args.d_blocks] * (args.d_stages + 1)
    d_cards = [args.d_cardinality] * (args.d_stages + 1)

    D_cls = PatchDiscriminator if args.d_type == "patch" else Discriminator
    D = D_cls(
        widths=d_widths,
        cardinalities=d_cards,
        blocks_per_stage=d_blocks,
        expansion=args.d_expansion,
        num_classes=d_num_classes,
        embed_dim=args.d_embed_dim,
        kernel_size=args.d_kernel_size,
        in_channels=img_channels,
    ).to(device)

    # ---- StyleEncoder (for LatentNCA style reconstruction) ----
    use_style = args.style_weight > 0 and args.nca_type == "latent"
    S = None
    if use_style:
        S = NCAStyleEncoder(
            in_channels=img_channels, style_dim=args.nca_cond_dim + args.z_dim,
            base_channels=64,
        ).to(device)

    g_total, g_train = count_parameters(G)
    d_total, d_train = count_parameters(D)
    print(f"Generator  params: total={g_total:,}  trainable={g_train:,}")
    print(f"Discriminator params: total={d_total:,}  trainable={d_train:,}")
    if S is not None:
        s_total, s_train = count_parameters(S)
        print(f"StyleEncoder params: total={s_total:,}  trainable={s_train:,}")

    # ---- optimizers ----
    adam_kwargs = dict(betas=(args.beta1, args.beta2), eps=1e-8)
    # G optimizer includes StyleEncoder params when style loss is on
    g_params = list(G.parameters()) + (list(S.parameters()) if S is not None else [])
    try:
        G_opt = torch.optim.Adam(g_params, lr=args.lr_g, fused=True, **adam_kwargs)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, fused=True, **adam_kwargs)
    except Exception:
        G_opt = torch.optim.Adam(g_params, lr=args.lr_g, **adam_kwargs)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, **adam_kwargs)

    # ---- AMP ----
    use_amp = device.type == "cuda"
    autocast = lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # ---- logging / resume ----
    logs = defaultdict(list)
    start_step = 0

    if args.resume:
        start_step, logs, extra = load_checkpoint(
            args.resume, G, D, G_opt, D_opt, G_ema=G_ema, map_location=device
        )
        if S is not None and extra and extra.get("S_state"):
            S.load_state_dict(extra["S_state"])
        print(f"Resuming from step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)


    # ---- wandb init ----
    wandb_run_id = None
    if args.resume and extra and isinstance(extra, dict):
        wandb_run_id = extra.get("wandb_run_id", None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=wandb_run_id,
            config=vars(args),
            resume="allow" if args.resume else None,
        )
        wandb.watch(G, log="gradients", log_freq=500)
        wandb.watch(D, log="gradients", log_freq=500)

    # ---- helper to get a batch (infinite iterator) ----
    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ---- training loop ----
    accum = args.accumulate_steps
    pbar = tqdm(range(start_step, args.iterations), desc="Training", initial=start_step, total=args.iterations)
    for step in pbar:
        G.train()
        D.train()

        # ============== Sample shared batches for D and G ==============
        shared_batches = []
        for _ in range(accum):
            img_ctrl, img_trt, cpd_id = next_batch()
            img_ctrl = img_ctrl.to(device)
            img_trt = img_trt.to(device)
            cpd_id = cpd_id.to(device)

            if args.use_alive_mask:
                # Alive channel: 1 where any RGB > threshold in [0,1] space, 0 otherwise
                ctrl_01 = (img_ctrl + 1) / 2
                alive_ctrl = (ctrl_01.max(dim=1, keepdim=True).values > args.alive_threshold).float()
                img_ctrl = torch.cat([img_ctrl, alive_ctrl], dim=1)  # [B, 4, H, W]
                trt_01 = (img_trt + 1) / 2
                alive_trt = (trt_01.max(dim=1, keepdim=True).values > args.alive_threshold).float()
                img_trt = torch.cat([img_trt, alive_trt], dim=1)  # [B, 4, H, W]

            if args.hidden_channels > 0:
                pad = torch.zeros(
                    img_ctrl.shape[0], args.hidden_channels,
                    img_ctrl.shape[2], img_ctrl.shape[3],
                    device=device,
                )
                nca_input = torch.cat([img_ctrl, pad], dim=1)
            else:
                nca_input = img_ctrl

            shared_batches.append((nca_input, img_trt, cpd_id))

        # ============== Discriminator step ==============
        set_requires_grad(G, False)
        set_requires_grad(D, True)
        D_opt.zero_grad(set_to_none=True)

        accum_d_loss = 0.0
        accum_adv_d = 0.0
        accum_reg = 0.0
        accum_gp_real = 0.0
        accum_gp_fake = 0.0
        accum_d_real = 0.0
        accum_d_fake = 0.0

        accum_d_inter = 0.0

        for nca_input, img_trt, cpd_id in shared_batches:
            with torch.no_grad():
                if use_intermediate or use_gradual:
                    # Sample intermediate step: [0, T-2] to avoid endpoint
                    t_inter_d = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full, inter_full = G.forward_with_intermediate(
                        nca_input, cond_fn(cpd_id), n_steps=args.nca_steps, t_intermediate=t_inter_d,
                    )
                    inter_img = inter_full[:, :img_channels].contiguous()
                else:
                    fake_full = G(nca_input, cond_fn(cpd_id), n_steps=args.nca_steps)
                fake_img = fake_full[:, :img_channels].contiguous()

            # --- Standard endpoint loss (compound-conditioned) ---
            real_req = img_trt.detach().requires_grad_(True)
            fake_req = fake_img.detach().requires_grad_(True)

            with autocast():
                d_real = D(real_req, cpd_id)
                d_fake = D(fake_req, cpd_id)
                adv_d = relativistic_d_loss(d_real, d_fake)

            gp_real = multi_scale_gp(real_req.float(), _to_float(d_real))
            gp_fake = multi_scale_gp(fake_req.float(), _to_float(d_fake))
            reg = 0.5 * args.gamma * (gp_real.mean() + gp_fake.mean())

            d_loss = (adv_d + reg) / accum
            d_loss.backward()

            # --- Null class D on actual intermediate (for intermediate and/or gradual) ---
            if use_intermediate or use_gradual:
                real_null = torch.cat([nca_input[:, :img_channels], img_trt], dim=0)
                null_ids = torch.full(
                    (real_null.shape[0],), null_class_id, device=device, dtype=torch.long,
                )
                inter_dup = inter_img.repeat(2, 1, 1, 1)
                inter_null_ids = torch.full(
                    (inter_dup.shape[0],), null_class_id, device=device, dtype=torch.long,
                )

                real_null_req = real_null.detach().requires_grad_(True)
                inter_req = inter_dup.detach().requires_grad_(True)

                with autocast():
                    d_real_null = D(real_null_req, null_ids)
                    d_inter = D(inter_req, inter_null_ids)
                    adv_inter = relativistic_d_loss(d_real_null, d_inter)

                gp_real_null = multi_scale_gp(real_null_req.float(), _to_float(d_real_null))
                gp_inter = multi_scale_gp(inter_req.float(), _to_float(d_inter))
                reg_inter = 0.5 * args.gamma * (gp_real_null.mean() + gp_inter.mean())

                d_inter_weight = args.intermediate_weight if use_intermediate else args.gradual_weight
                d_inter_loss = d_inter_weight * (adv_inter + reg_inter) / accum
                d_inter_loss.backward()
                accum_d_inter += d_inter_loss.item()

            accum_d_loss += d_loss.item()
            accum_adv_d += adv_d.item() / accum
            accum_reg += reg.item() / accum
            accum_gp_real += gp_real.mean().item() / accum
            accum_gp_fake += gp_fake.mean().item() / accum
            accum_d_real += d_logit_mean(d_real) / accum
            accum_d_fake += d_logit_mean(d_fake) / accum

        D_opt.step()

        # ============== Generator step ==============
        set_requires_grad(G, True)
        set_requires_grad(D, False)
        G_opt.zero_grad(set_to_none=True)

        accum_g_loss = 0.0
        accum_div_loss = 0.0
        accum_g_inter = 0.0
        accum_g_gradual = 0.0
        accum_style_loss = 0.0

        for nca_input, img_trt, cpd_id in shared_batches:
            # Pre-compute style + z for LatentNCA (so we can pass z and get style for loss)
            style_target = None
            z_sample = None
            if use_style:
                style_target, z_sample = G._prepare_cond(cond_fn(cpd_id))

            # Extra kwargs for LatentNCA (pass z to reuse same sample)
            z_kwargs = {"z": z_sample} if z_sample is not None else {}

            with autocast():
                if use_intermediate or use_gradual:
                    t_inter_g = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full, inter_full_g = G.forward_with_intermediate(
                        nca_input, cond_fn(cpd_id), n_steps=args.nca_steps, t_intermediate=t_inter_g,
                        **z_kwargs,
                    )
                    inter_img = inter_full_g[:, :img_channels].contiguous()
                else:
                    fake_full = G(nca_input, cond_fn(cpd_id), n_steps=args.nca_steps, **z_kwargs)
                fake_img = fake_full[:, :img_channels].contiguous()

                d_real = D(img_trt.detach(), cpd_id)
                d_fake = D(fake_img, cpd_id)

            # --- Standard endpoint G loss (reversed direction for G) ---
            if isinstance(d_fake, dict):
                g_loss = 0.0
                for key in d_fake:
                    rel = d_fake[key] - d_real[key]
                    g_loss = g_loss + F.softplus(-rel).mean()
                g_loss = g_loss / accum
            else:
                rel = d_fake - d_real
                g_loss = F.softplus(-rel).mean() / accum

            # --- Intermediate G loss (null class) ---
            g_inter_loss_val = 0.0
            if use_intermediate:
                real_null = torch.cat([nca_input[:, :img_channels], img_trt], dim=0)
                null_ids = torch.full(
                    (real_null.shape[0],), null_class_id, device=device, dtype=torch.long,
                )
                inter_dup = inter_img.repeat(2, 1, 1, 1)
                inter_null_ids = torch.full(
                    (inter_dup.shape[0],), null_class_id, device=device, dtype=torch.long,
                )

                with autocast():
                    d_real_null = D(real_null.detach(), null_ids)
                    d_inter = D(inter_dup, inter_null_ids)
                if isinstance(d_inter, dict):
                    g_inter_adv = 0.0
                    for key in d_inter:
                        rel_inter = d_inter[key] - d_real_null[key]
                        g_inter_adv = g_inter_adv + F.softplus(-rel_inter).mean()
                else:
                    rel_inter = d_inter - d_real_null
                    g_inter_adv = F.softplus(-rel_inter).mean()
                g_inter_loss = args.intermediate_weight * g_inter_adv / accum
                g_inter_loss_val = g_inter_loss.item()

            # --- Gradual intermediate G loss (weighted target + null) ---
            g_gradual_loss_val = 0.0
            g_gradual_loss = 0.0
            if use_gradual:
                # alpha = (t+1)/T: step 0 in loop = 1st NCA step → alpha=1/T
                alpha = (t_inter_g + 1) / args.nca_steps

                # Target class: intermediate vs trt
                # Null class: intermediate vs all cells (ctrl + trt)
                real_all = torch.cat([nca_input[:, :img_channels], img_trt], dim=0)
                null_ids = torch.full(
                    (real_all.shape[0],), null_class_id, device=device, dtype=torch.long,
                )
                inter_dup = inter_img.repeat(2, 1, 1, 1)
                null_ids_fake = torch.full(
                    (inter_dup.shape[0],), null_class_id, device=device, dtype=torch.long,
                )

                with autocast():
                    d_real_tgt = D(img_trt.detach(), cpd_id)
                    d_fake_tgt = D(inter_img, cpd_id)
                    d_real_null = D(real_all.detach(), null_ids)
                    d_fake_null = D(inter_dup, null_ids_fake)

                def _rel_g(d_fake, d_real):
                    if isinstance(d_fake, dict):
                        loss = 0.0
                        for key in d_fake:
                            loss = loss + F.softplus(-(d_fake[key] - d_real[key])).mean()
                        return loss
                    return F.softplus(-(d_fake - d_real)).mean()

                g_gradual_adv = alpha * _rel_g(d_fake_tgt, d_real_tgt) \
                              + (1 - alpha) * _rel_g(d_fake_null, d_real_null)
                g_gradual_loss = args.gradual_weight * g_gradual_adv / accum
                g_gradual_loss_val = g_gradual_loss.item()

            # --- Diversity loss ---
            total_loss = g_loss
            if use_intermediate:
                total_loss = total_loss + g_inter_loss
            if use_gradual:
                total_loss = total_loss + g_gradual_loss
            if args.diversity_weight > 0:
                with autocast():
                    fake_full2 = G(nca_input, cond_fn(cpd_id), n_steps=args.nca_steps)
                    fake_img2 = fake_full2[:, :img_channels].contiguous()
                div_loss = -torch.mean(torch.abs(fake_img - fake_img2)) / accum
                total_loss = total_loss + args.diversity_weight * div_loss
                accum_div_loss += div_loss.item()

            # --- Style reconstruction loss (LatentNCA only) ---
            if use_style:
                with autocast():
                    style_hat = S(fake_img)
                sty_loss = F.l1_loss(style_hat, style_target.detach()) / accum
                total_loss = total_loss + args.style_weight * sty_loss
                accum_style_loss += sty_loss.item()

            total_loss.backward()

            accum_g_loss += g_loss.item()
            accum_g_inter += g_inter_loss_val
            accum_g_gradual += g_gradual_loss_val

        nn_utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
        G_opt.step()

        if use_ema:
            G_ema.update_parameters(G)

        # ============== Gradient Diagnostics ==============
        if args.grad_diag_every > 0 and (step + 1) % args.grad_diag_every == 0:
            set_requires_grad(G, False)  # no G param grads needed
            set_requires_grad(D, False)
            nca_input_diag, img_trt_diag, cpd_id_diag = shared_batches[-1]

            # Unroll NCA with hooks to capture per-step gradient norms.
            # We register x_t at each step via register_hook on intermediate tensors.
            # The trick: use x.register_hook to capture ||dL/dx_t|| during backward.
            x_t = nca_input_diag.detach().requires_grad_(True)
            hook_grads = {}

            def make_hook(t_idx):
                def hook(grad):
                    hook_grads[t_idx] = grad.norm().item()
                return hook

            # Pre-compute conditioning (handles LatentNCA's embed + z concat)
            diag_cond = cond_fn(cpd_id_diag)
            if hasattr(G, '_prepare_cond'):
                diag_cond, _ = G._prepare_cond(diag_cond)

            x_t.register_hook(make_hook(0))
            for t in range(args.nca_steps):
                x_t = G.step(x_t, diag_cond)
                if t < args.nca_steps - 1:
                    x_t.register_hook(make_hook(t + 1))

            # Compute G loss on final state
            fake_img_diag = x_t[:, :img_channels].contiguous()
            with torch.no_grad():
                d_real_diag = D(img_trt_diag, cpd_id_diag)
            d_fake_diag = D(fake_img_diag, cpd_id_diag)
            if isinstance(d_fake_diag, dict):
                g_loss_diag = 0.0
                for key in d_fake_diag:
                    g_loss_diag = g_loss_diag + F.softplus(-(d_fake_diag[key] - d_real_diag[key])).mean()
            else:
                g_loss_diag = F.softplus(-(d_fake_diag - d_real_diag)).mean()
            g_loss_diag.backward()

            grad_norms = {f"grad_norm/step_{t}": v for t, v in hook_grads.items()}
            if use_wandb:
                wandb.log(grad_norms, step=step + 1)
            first = hook_grads.get(0, 0.0)
            last = hook_grads.get(args.nca_steps - 1, 0.0)
            print(f"[GradDiag] step {step+1}: "
                  f"t=0: {first:.6f}  "
                  f"t={args.nca_steps-1}: {last:.6f}  "
                  f"ratio: {last / max(first, 1e-10):.1f}x")

            set_requires_grad(G, True)

        # ============== Logging ==============
        log_dict = {
            "loss/D_total": accum_d_loss,
            "loss/D_adv": accum_adv_d,
            "loss/D_reg": accum_reg,
            "penalty/gp_real_mean": accum_gp_real,
            "penalty/gp_fake_mean": accum_gp_fake,
            "logits/D_real_mean": accum_d_real,
            "logits/D_fake_mean": accum_d_fake,
            "loss/G": accum_g_loss,
        }
        if args.diversity_weight > 0:
            log_dict["loss/diversity"] = accum_div_loss
        if use_style:
            log_dict["loss/style_recon"] = accum_style_loss
        if use_intermediate or use_gradual:
            log_dict["loss/D_inter"] = accum_d_inter
        if use_intermediate:
            log_dict["loss/G_inter"] = accum_g_inter
        if use_gradual:
            log_dict["loss/G_gradual"] = accum_g_gradual
        for k, v in log_dict.items():
            logs[k].append(v)

        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix({
                "d": f"{accum_d_loss:.4f}",
                "g": f"{accum_g_loss:.4f}",
                "gp_r": f"{accum_gp_real:.2f}",
            })

        # ============== Visualizations ==============
        if (step + 1) % args.vis_every == 0:
            vis_G = G_ema.module if use_ema else G
            # Use last shared batch for visualization
            vis_nca_input, vis_img_trt, vis_cpd_id = shared_batches[-1]
            log_visualizations(
                vis_G, dataset, device, args, id2cpd, img_channels,
                fake_img, vis_nca_input, vis_img_trt, vis_cpd_id, step + 1, use_wandb,
                cond_fn=cond_fn,
            )

        # ============== FID (test split) ==============
        if args.fid_every > 0 and (step + 1) % args.fid_every == 0 and eval_dataset is not None:
            fid_G = G_ema.module if use_ema else G
            fid_global, fid_per_cpd = compute_fid(
                fid_G, eval_dataset, device, args, id2cpd, img_channels,
                cond_fn=cond_fn,
            )
            if fid_global is not None:
                fid_log = {"fid/global": fid_global}
                fid_vals = []
                for cpd_name, fid_val in fid_per_cpd.items():
                    fid_log[f"fid/cpd/{cpd_name}"] = fid_val
                    fid_vals.append(fid_val)
                if fid_vals:
                    fid_log["fid/mean_per_cpd"] = np.mean(fid_vals)
                print(f"[FID] global={fid_global:.2f}  mean_per_cpd={np.mean(fid_vals):.2f}")
                if use_wandb:
                    wandb.log(fid_log, step=step + 1)
                logs["fid/global"].append(fid_global)

        # ============== FID trajectory (per-step FID vs all real cells) ==============
        if args.fid_trajectory_every > 0 and (step + 1) % args.fid_trajectory_every == 0 and eval_dataset is not None:
            traj_G = G_ema.module if use_ema else G
            fid_traj = compute_fid_trajectory(
                traj_G, eval_dataset, device, args, img_channels, cond_fn=cond_fn,
            )
            if fid_traj is not None:
                traj_log = {f"fid_traj/step_{t}": v for t, v in fid_traj.items()}
                print(f"[FID traj] " + "  ".join(f"t={t}:{v:.1f}" for t, v in sorted(fid_traj.items())))
                if use_wandb:
                    wandb.log(traj_log, step=step + 1)

        # ============== Checkpointing ==============
        is_save_step = (step + 1) % args.save_every == 0
        is_last_step = step == args.iterations - 1
        if is_save_step or is_last_step:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step + 1}.pt")
            save_checkpoint(
                ckpt_path,
                step=step + 1,
                G=G, D=D,
                G_opt=G_opt, D_opt=D_opt,
                logs=logs,
                G_ema=G_ema,
                extra={
                    "gamma": args.gamma,
                    "lr_g": args.lr_g,
                    "lr_d": args.lr_d,
                    "nca_type": args.nca_type,
                    "nca_steps": args.nca_steps,
                    "hidden_channels": args.hidden_channels,
                    "noise_channels": args.noise_channels,
                    "z_dim": args.z_dim,
                    "channel_n": channel_n,
                    "num_compounds": num_compounds,
                    "ema_decay": args.ema_decay,
                    "ema_warmup_steps": args.ema_warmup_steps,
                    "diversity_weight": args.diversity_weight,
                    "step_size": args.step_size,
                    "use_alive_mask": args.use_alive_mask,
                    "alive_threshold": args.alive_threshold,
                    "cond_type": args.cond_type,
                    "fp_path": args.fp_path,
                    "d_type": args.d_type,
                    "intermediate_weight": args.intermediate_weight,
                    "gradual_weight": args.gradual_weight,
                    "style_weight": args.style_weight,
                    "S_state": S.state_dict() if S is not None else None,
                    "wandb_run_id": wandb.run.id if use_wandb else None,
                },
            )

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    train(args)
