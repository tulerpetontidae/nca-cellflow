"""
CellFlux (flow matching + UNet) training script using our data pipeline.

Drop-in comparison against NCA-GAN: same dataset, same wandb logging,
same visualization, same FID evaluation. Uses CellFlux's UNet architecture
with conditional optimal transport flow matching.

Usage:
    python scripts/train_cellflux.py --config configs/cellflux-bbbc021-lb.yaml
    python scripts/train_cellflux.py --config configs/cellflux-bbbc021-lb.yaml --batch_size 64
"""

import os
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from nca_cellflow.models.cellflux_unet import (
    CellFluxUNet, ode_sample_heun, ode_sample_midpoint, ode_sample_euler,
    edm_time_grid, skewed_timestep_sample,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_rgb_numpy(img):
    img = img.detach().cpu().float().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Simple EMA wrapper that stores shadow params separately."""

    def __init__(self, model, decay=0.999, warmup=1000):
        self.model = model
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    def update(self):
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (self.warmup + self.step_count))
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(d).add_(p.data, alpha=1 - d)

    def apply_shadow(self):
        """Copy shadow params into model (for eval)."""
        self.backup = {name: p.data.clone() for name, p in self.model.named_parameters()
                       if p.requires_grad and name in self.shadow}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original params (after eval)."""
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "step_count": self.step_count}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.step_count = state["step_count"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, model, optimizer, logs, ema=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "logs": dict(logs),
        "extra": extra,
    }
    if ema is not None:
        state["ema_state"] = ema.state_dict()
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path, model, optimizer=None, ema=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    if ema is not None and "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])
    logs = defaultdict(list, ckpt.get("logs", {}))
    step = ckpt.get("step", 0)
    print(f"[ckpt] loaded {path} at step {step}")
    return step, logs


# ---------------------------------------------------------------------------
# Visualization (mirrors NCA training)
# ---------------------------------------------------------------------------

def plot_image_grid(images, titles, suptitle, max_images=16):
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


@torch.no_grad()
def plot_flow_trajectories(model, dataset, device, args, id2cpd, fp_matrix,
                           step, ode_fn, time_grid, cfg_scale):
    """
    Flow trajectories: one row per compound, columns = intermediate ODE states.
    Mirrors the NCA trajectory plot.

    Uses the same ctrl image and same noise seed across all compounds so
    differences between rows reflect conditioning only.  Runs a coarse
    Heun ODE (vis_nfe steps) to keep wall-clock reasonable — the column
    headers show actual t values so quality differences vs full sampling
    are transparent.
    """
    model.eval()
    n_cols = 5
    vis_nfe = min(20, len(time_grid) - 1)  # coarser grid for speed

    # Always use uniform time grid for visualization so columns show
    # evenly-spaced t values (EDM schedule concentrates steps near t=0,
    # making trajectories look like "noise → final image" with no transition)
    vis_time_grid = torch.linspace(0, 1, vis_nfe + 1, device=device)

    n_vis_steps = len(vis_time_grid) - 1
    step_indices = sorted(set(
        [0] + [int(round(i * n_vis_steps / (n_cols - 1))) for i in range(n_cols)]
    ))

    # Load ctrl image and fix noise so every compound starts from the same x_0
    img_ctrl = dataset._load(dataset.ctrl_keys[0]).unsqueeze(0).to(device)
    noise = torch.randn_like(img_ctrl) * args.noise_level
    x_0_shared = img_ctrl + noise  # same init for all compounds

    num_compounds = len(id2cpd)
    fig, axes = plt.subplots(num_compounds, len(step_indices),
                             figsize=(3 * len(step_indices), 3 * num_compounds))
    if num_compounds == 1:
        axes = axes[None, :]

    for cpd_idx in range(num_compounds):
        cond = fp_matrix[cpd_idx:cpd_idx+1].to(device)  # [1, fp_dim]

        # Heun ODE with intermediate capture
        x = x_0_shared.clone()
        states = [x.clone()]
        for i in range(n_vis_steps):
            t_cur = vis_time_grid[i]
            t_next = vis_time_grid[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full((x.shape[0],), t_cur, device=device)

            v1 = _model_forward(model, x, t_batch, cond, cfg_scale, device)
            x_euler = x + v1 * dt

            t_batch_next = torch.full((x.shape[0],), t_next, device=device)
            v2 = _model_forward(model, x_euler, t_batch_next, cond, cfg_scale, device)

            x = x + 0.5 * dt * (v1 + v2)
            states.append(x.clone())

        for col, si in enumerate(step_indices):
            rgb = to_rgb_numpy(states[si][0, :3])
            ax = axes[cpd_idx, col]
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if cpd_idx == 0:
                t_val = vis_time_grid[si].item() if si < len(vis_time_grid) else 1.0
                ax.set_title(f"t={t_val:.2f}", fontsize=40)
        axes[cpd_idx, 0].set_ylabel(id2cpd[cpd_idx], fontsize=36,
                                     rotation=90, labelpad=60, va="center")

    fig.suptitle(f"Flow trajectories (step {step})", fontsize=48)
    plt.tight_layout()
    model.train()
    return fig


def _model_forward(model, x, t, cond, cfg_scale, device):
    """Single velocity eval with optional CFG — used by trajectory vis."""
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type):
        v_cond = model(x, t, cond=cond)
    if cfg_scale > 0.0 and cond is not None:
        with torch.amp.autocast(device_type=device_type):
            v_uncond = model(x, t, cond=None)
        return ((1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond).float()
    return v_cond.float()


def log_visualizations(model, dataset, device, args, id2cpd, fp_matrix,
                       fake_img, img_ctrl, img_trt, cpd_id, step, use_wandb,
                       ode_fn, time_grid, cfg_scale):
    model.eval()

    # 1. Trajectory plot
    fig_traj = plot_flow_trajectories(
        model, dataset, device, args, id2cpd, fp_matrix,
        step, ode_fn, time_grid, cfg_scale,
    )
    if use_wandb:
        wandb.log({"vis/trajectories": wandb.Image(fig_traj)}, step=step)
        plt.close(fig_traj)
    else:
        plt.show()

    n = min(img_trt.shape[0], 16)
    titles = [id2cpd[cpd_id[i].item()] for i in range(n)]

    # 2. Control samples
    fig_ctrl = plot_image_grid(
        [img_ctrl[i, :3] for i in range(n)], ["DMSO"] * n,
        f"Control / Flow input (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/control_samples": wandb.Image(fig_ctrl)}, step=step)
        plt.close(fig_ctrl)
    else:
        plt.show()

    # 3. Generated samples
    fig_fake = plot_image_grid(
        [fake_img[i, :3].detach() for i in range(n)], titles,
        f"Generated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/generated_samples": wandb.Image(fig_fake)}, step=step)
        plt.close(fig_fake)
    else:
        plt.show()

    # 4. Real treated
    fig_real = plot_image_grid(
        [img_trt[i, :3] for i in range(n)], titles,
        f"Real treated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/real_samples": wandb.Image(fig_real)}, step=step)
        plt.close(fig_real)
    else:
        plt.show()

    model.train()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

_global_fid = None
_cpd_fid = None
_traj_fid = None


@torch.no_grad()
def compute_fid_trajectory(model, eval_dataset, device, args, fp_matrix,
                           cfg_scale):
    """Compute FID at equidistant time points along the ODE trajectory.

    Uses a uniform time grid t in [0, 1] with ~6 evaluation points.
    At each eval time t_eval, runs a Heun ODE from t=0 to t=t_eval and
    computes FID of the intermediate state against all real cells (ctrl + trt).

    Returns dict {t_value: fid_value} or None on failure.
    """
    global _traj_fid
    if not FID_AVAILABLE:
        return None

    try:
        if _traj_fid is None:
            _traj_fid = FrechetInceptionDistance(normalize=True).to(device)

        # ~6 equidistant eval points in t, always include 0 and 1
        n_eval = 6
        eval_times = torch.linspace(0.0, 1.0, n_eval + 1)[1:]  # skip t=0 (just noise)
        eval_times = eval_times.tolist()  # [0.167, 0.333, 0.5, 0.667, 0.833, 1.0]

        # Fine Heun grid for the full ODE (we stop at each eval point)
        n_ode_steps = 50
        full_time_grid = torch.linspace(0.0, 1.0, n_ode_steps + 1, device=device)

        traj_bs = min(args.batch_size, 64)
        loader = DataLoader(
            eval_dataset, batch_size=traj_bs, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False,
        )

        model.eval()

        # Step 1: compute real stats once (ctrl + trt pooled)
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

        # Precompute which ODE grid indices correspond to eval times
        eval_grid_indices = []
        for t_eval in eval_times:
            # Find the closest grid point at or just past t_eval
            idx = int(round(t_eval * n_ode_steps))
            idx = max(1, min(idx, n_ode_steps))
            eval_grid_indices.append(idx)

        # Step 2: for each batch, run ODE and capture intermediates
        # Collect fake features at each eval point
        fid_traj = {}
        for t_idx, (t_eval, grid_idx) in enumerate(zip(eval_times, eval_grid_indices)):
            _traj_fid.reset()
            _traj_fid.real_features_sum.copy_(real_mu)
            _traj_fid.real_features_cov_sum.copy_(real_cov)
            _traj_fid.real_features_num_samples.copy_(real_n)

            # ODE sub-grid from 0 to this eval point
            sub_grid = full_time_grid[:grid_idx + 1]

            for img_ctrl, img_trt, cpd_id in loader:
                img_ctrl = img_ctrl.to(device)
                cpd_id = cpd_id.to(device)
                cond = fp_matrix[cpd_id].to(device)

                x = img_ctrl + torch.randn_like(img_ctrl) * args.noise_level

                # Heun ODE up to this time point
                for i in range(len(sub_grid) - 1):
                    t_cur = sub_grid[i]
                    t_next = sub_grid[i + 1]
                    dt = t_next - t_cur
                    t_batch = torch.full((x.shape[0],), t_cur.item(), device=device)

                    v1 = _eval_velocity_for_traj(model, x, t_batch, cond, cfg_scale)
                    x_euler = x + v1 * dt
                    t_batch_next = torch.full((x.shape[0],), t_next.item(), device=device)
                    v2 = _eval_velocity_for_traj(model, x_euler, t_batch_next, cond, cfg_scale)
                    x = x + 0.5 * dt * (v1 + v2)

                rgb = x[:, :3].clamp(-1, 1)
                rgb_01 = (rgb + 1) / 2
                rgb_01 = torch.floor(rgb_01 * 255).float() / 255.0
                _traj_fid.update(rgb_01, real=False)

            try:
                fid_traj[round(t_eval, 3)] = _traj_fid.compute().item()
            except Exception:
                pass
            torch.cuda.empty_cache()

        model.train()
        return fid_traj

    except Exception as e:
        print(f"[warn] FID trajectory failed: {e}")
        model.train()
        torch.cuda.empty_cache()
        return None


def _eval_velocity_for_traj(model, x, t, cond, cfg_scale):
    """Velocity eval for trajectory FID — mirrors _eval_velocity in cellflux_unet."""
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type):
        v_cond = model(x, t, cond=cond)
    if cfg_scale > 0.0 and cond is not None:
        with torch.amp.autocast(device_type=device_type):
            v_uncond = model(x, t, cond=None)
        return ((1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond).float()
    return v_cond.float()


@torch.no_grad()
def compute_fid(model, eval_dataset, device, args, id2cpd, fp_matrix,
                ode_fn, time_grid, cfg_scale):
    """Compute global and per-compound FID using torchmetrics.

    Uses EvalDataset: deterministic iteration over all treated test images,
    deterministic same-plate ctrl pairing, no augmentation.
    Matches NCA-GAN evaluation protocol.
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

        model.eval()
        for img_ctrl, img_trt, cpd_id in loader:
            img_ctrl = img_ctrl.to(device)
            img_trt = img_trt.to(device)
            cpd_id = cpd_id.to(device)

            # Build conditioning
            cond = fp_matrix[cpd_id].to(device)

            # Initial state: ctrl + noise
            x_0 = img_ctrl + torch.randn_like(img_ctrl) * args.noise_level

            # ODE solve
            fake = ode_fn(model, x_0, time_grid.to(device), cond=cond, cfg_scale=cfg_scale)
            fake_rgb = fake[:, :3].contiguous()

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

        model.train()
        return fid_global, fid_per_cpd

    except Exception as e:
        print(f"[warn] FID computation failed: {e}")
        model.train()
        torch.cuda.empty_cache()
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="Train CellFlux (flow matching + UNet) on IMPA dataset")

    # data
    p.add_argument("--metadata_csv", type=str, default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--plate_match", action="store_true")
    p.add_argument("--balanced_cpd", action="store_true")
    p.add_argument("--iter_trt", action="store_true",
                   help="Iterate over treated images, randomly sample ctrl (CellFlux paper style)")
    p.add_argument("--fp_path", type=str, default=None,
                   help="Path to Morgan fingerprint CSV (required)")

    # model (CellFlux UNet)
    p.add_argument("--model_channels", type=int, default=128)
    p.add_argument("--num_res_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--attention_resolutions", type=int, nargs="+", default=[2])
    p.add_argument("--condition_dim", type=int, default=1024,
                   help="Fingerprint embedding dimension")

    # flow matching
    p.add_argument("--noise_level", type=float, default=0.5,
                   help="Noise scale added to ctrl images for x_0")
    p.add_argument("--noise_prob", type=float, default=0.5,
                   help="Probability of adding noise (vs clean ctrl) during training")
    p.add_argument("--class_drop_prob", type=float, default=0.2,
                   help="Probability of dropping conditioning (for CFG)")
    p.add_argument("--cfg_scale", type=float, default=0.2,
                   help="Classifier-free guidance scale at inference")
    p.add_argument("--skewed_timesteps", action="store_true",
                   help="Use EDM-style skewed timestep sampling")
    p.add_argument("--ode_method", type=str, default="heun",
                   choices=["euler", "midpoint", "heun"])
    p.add_argument("--ode_nfe", type=int, default=50,
                   help="Number of function evaluations for ODE sampling")
    p.add_argument("--edm_schedule", action="store_true",
                   help="Use EDM time discretization schedule")

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=80000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_warmup_steps", type=int, default=1000)

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--vis_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--fid_every", type=int, default=0,
                   help="Compute FID every N steps (0 = disabled, uses all test treated images)")
    p.add_argument("--fid_trajectory_every", type=int, default=0,
                   help="Compute FID trajectory every N steps (0 = disabled)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="nca-cellflow")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--config", type=str, default=None)

    return p


def load_config_into_args(args):
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
            if k in type_map and not isinstance(v, (list, type_map[k])):
                v = type_map[k](v)
            setattr(args, k, v)
    return args


def train(args):
    args = load_config_into_args(args)
    print(f"[config] CellFlux UNet: model_channels={args.model_channels}, "
          f"channel_mult={args.channel_mult}, noise_level={args.noise_level}, "
          f"cfg_scale={args.cfg_scale}, skewed={args.skewed_timesteps}")

    # ---- wandb ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if not use_wandb:
        matplotlib.use("TkAgg")

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
    print(f"Dataset: {len(dataset)} control images, {len(dataset.trt_keys)} treated, "
          f"{num_compounds} compounds")

    # ---- fingerprint embeddings ----
    assert args.fp_path is not None, "--fp_path required for CellFlux conditioning"
    fp_df = pd.read_csv(args.fp_path, index_col=0)
    fp_vecs = []
    for cid in range(num_compounds):
        cpd_name = id2cpd[cid]
        assert cpd_name in fp_df.index, f"Compound '{cpd_name}' not found in {args.fp_path}"
        fp_vecs.append(fp_df.loc[cpd_name].values.astype(np.float32))
    fp_matrix = torch.tensor(np.stack(fp_vecs)).to(device)  # [num_compounds, fp_dim]
    fp_dim = fp_matrix.shape[1]
    print(f"Loaded fingerprints: {fp_matrix.shape} from {args.fp_path}")

    # ---- eval dataset for FID (deterministic, no augmentation, test split) ----
    eval_dataset = None
    if args.fid_every > 0 or getattr(args, 'fid_trajectory_every', 0) > 0:
        eval_dataset = EvalDataset(
            metadata_csv=args.metadata_csv,
            image_dir=args.image_dir,
            split="test",
            image_size=args.image_size,
        )
        print(f"FID eval dataset: {len(eval_dataset)} treated test images")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    data_iter = iter(loader)

    # ---- model ----
    model = CellFluxUNet(
        in_channels=3,
        model_channels=args.model_channels,
        out_channels=3,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        dropout=args.dropout,
        channel_mult=tuple(args.channel_mult),
        use_scale_shift_norm=True,
        use_new_attention_order=True,
        condition_dim=fp_dim,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"CellFlux UNet params: total={total:,}  trainable={trainable:,}")

    # ---- EMA ----
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay, warmup=args.ema_warmup_steps)
        print(f"EMA enabled (decay={args.ema_decay}, warmup={args.ema_warmup_steps})")

    # ---- ODE solver setup ----
    ode_fns = {"euler": ode_sample_euler, "midpoint": ode_sample_midpoint, "heun": ode_sample_heun}
    ode_fn = ode_fns[args.ode_method]

    if args.edm_schedule:
        time_grid = edm_time_grid(args.ode_nfe).to(device)
    else:
        time_grid = torch.linspace(0, 1, args.ode_nfe + 1).to(device)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
    )

    # ---- AMP ----
    use_amp = device.type == "cuda"
    autocast = lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # ---- logging / resume ----
    logs = defaultdict(list)
    start_step = 0

    if args.resume:
        start_step, logs = load_checkpoint(
            args.resume, model, optimizer, ema=ema, map_location=device,
        )
        print(f"Resuming from step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- wandb init ----
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"cellflux-{args.model_channels}ch",
            config=vars(args),
            resume="allow" if args.resume else None,
        )

    # ---- batch helper ----
    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ---- training loop ----
    pbar = tqdm(range(start_step, args.iterations), desc="Training",
                initial=start_step, total=args.iterations)

    for step in pbar:
        model.train()

        img_ctrl, img_trt, cpd_id = next_batch()
        img_ctrl = img_ctrl.to(device)
        img_trt = img_trt.to(device)
        cpd_id = cpd_id.to(device)

        # ---- Build conditioning (with CFG dropout) ----
        fp_cond = fp_matrix[cpd_id]  # [B, fp_dim]

        # CFG: drop conditioning with probability class_drop_prob
        if torch.rand(1).item() < args.class_drop_prob:
            fp_cond = None

        # ---- Conditional OT flow matching ----
        B = img_ctrl.shape[0]

        # Sample timesteps
        if args.skewed_timesteps:
            t = skewed_timestep_sample(B, device=device)
        else:
            t = torch.rand(B, device=device)

        # Source: ctrl image + optional noise
        if torch.rand(1).item() > args.noise_prob:
            x_0 = img_ctrl
        else:
            x_0 = img_ctrl + torch.randn_like(img_ctrl) * args.noise_level

        # Target: treated image
        x_1 = img_trt

        # OT path: x_t = (1-t)*x_0 + t*x_1, velocity = x_1 - x_0
        t_expand = t[:, None, None, None]
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0  # target velocity

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            v_pred = model(x_t, t, cond=fp_cond)
            loss = torch.pow(v_pred - u_t, 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update()

        # ---- Logging ----
        loss_val = loss.item()
        logs["loss/flow_matching"].append(loss_val)

        log_dict = {"loss/flow_matching": loss_val}
        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # ---- Visualizations ----
        if (step + 1) % args.vis_every == 0:
            if ema is not None:
                ema.apply_shadow()
            model.eval()

            # Generate a batch for visualization using full ODE solver
            with torch.no_grad():
                vis_cond = fp_matrix[cpd_id]
                x_0_vis = img_ctrl + torch.randn_like(img_ctrl) * args.noise_level
                fake = ode_fn(model, x_0_vis, time_grid, cond=vis_cond, cfg_scale=args.cfg_scale)
                fake_rgb = fake[:, :3]

            log_visualizations(
                model, dataset, device, args, id2cpd, fp_matrix,
                fake_rgb, img_ctrl, img_trt, cpd_id, step + 1, use_wandb,
                ode_fn, time_grid, args.cfg_scale,
            )

            model.train()
            if ema is not None:
                ema.restore()

        # ---- FID (test split) ----
        if args.fid_every > 0 and (step + 1) % args.fid_every == 0 and eval_dataset is not None:
            if ema is not None:
                ema.apply_shadow()

            fid_global, fid_per_cpd = compute_fid(
                model, eval_dataset, device, args, id2cpd, fp_matrix,
                ode_fn, time_grid, args.cfg_scale,
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

            if ema is not None:
                ema.restore()

        # ---- FID trajectory ----
        fid_traj_every = getattr(args, 'fid_trajectory_every', 0)
        if fid_traj_every > 0 and (step + 1) % fid_traj_every == 0 and eval_dataset is not None:
            if ema is not None:
                ema.apply_shadow()

            fid_traj = compute_fid_trajectory(
                model, eval_dataset, device, args, fp_matrix, args.cfg_scale,
            )
            if fid_traj is not None:
                traj_log = {}
                for t_val, fid_val in sorted(fid_traj.items()):
                    traj_log[f"fid_traj/t_{t_val:.3f}"] = fid_val
                    print(f"  [FID traj] t={t_val:.3f} -> {fid_val:.2f}")
                if use_wandb:
                    wandb.log(traj_log, step=step + 1)

            if ema is not None:
                ema.restore()

        # ---- Checkpointing ----
        is_save_step = (step + 1) % args.save_every == 0
        is_last_step = step == args.iterations - 1
        if is_save_step or is_last_step:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step + 1}.pt")
            save_checkpoint(
                ckpt_path, step=step + 1, model=model, optimizer=optimizer,
                logs=logs, ema=ema,
                extra={
                    "model_channels": args.model_channels,
                    "channel_mult": args.channel_mult,
                    "num_res_blocks": args.num_res_blocks,
                    "attention_resolutions": args.attention_resolutions,
                    "condition_dim": fp_dim,
                    "noise_level": args.noise_level,
                    "cfg_scale": args.cfg_scale,
                    "ode_method": args.ode_method,
                    "ode_nfe": args.ode_nfe,
                    "num_compounds": num_compounds,
                    "image_size": args.image_size,
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
