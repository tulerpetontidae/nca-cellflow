"""
Self-supervised NCA-GAN with style encoder (homeostasis).

z ~ N(0, I) is sampled randomly and injected into the NCA via FiLM.
The NCA must embed z into the generated image so that a style encoder E
can recover it: E(NCA(input, z)) ≈ z.  The discriminator is unconditional
— it only checks "does this look like a real cell?"

This forces the NCA to learn meaningful image modifications (controlled by
z) that stay on the cell manifold (constrained by D).  z axes should
self-organise into interpretable morphological dimensions.

With pool mode, the NCA must maintain z-specified phenotype stably over
multiple 30-step cycles ("homeostasis test").

Key design:
  - z ~ N(0, I): random, NOT from encoding.  This prevents identity collapse.
  - LatentNCA with dummy single class: compound embedding is a learned constant.
  - ResBlkStyleEncoder: Q-network predicting z from generated images.
  - Style reconstruction loss: ||E(fake) - z|| (InfoGAN / Q-head).
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
    print("[warn] torchmetrics not installed -- FID evaluation disabled")

import pandas as pd
from pathlib import Path

from nca_cellflow import CtrlPairDataset, ReplayPool
from nca_cellflow.dataset import _load_image
from nca_cellflow.models import LatentNCA, ResBlkStyleEncoder, Discriminator


# ---------------------------------------------------------------------------
# GAN utilities (matching train.py formulation)
# ---------------------------------------------------------------------------

def zero_centered_gradient_penalty(samples, logits):
    if logits.dim() > 1:
        reduced = logits.reshape(logits.shape[0], -1).mean(dim=1).sum()
    else:
        reduced = logits.sum()
    (grad,) = torch.autograd.grad(outputs=reduced, inputs=samples, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def relativistic_d_loss(d_real, d_fake):
    return F.softplus(-(d_real - d_fake)).mean()


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_rgb_numpy(img):
    img = img.detach().cpu().float().clamp(-1, 1)
    return ((img + 1.0) / 2.0).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Anti-steganography: augment image before style encoder
# ---------------------------------------------------------------------------

def _gaussian_blur(x, kernel_size=5, sigma=1.0):
    k = kernel_size
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


def augment_for_encoder(x, noise_sigma=0.0, blur_sigma=0.0, downsample=0):
    """Augment generated image before style encoder to prevent steganography.

    Applied in order: blur -> downsample -> noise.  Each is optional (0 = off).
    """
    if blur_sigma > 0:
        x = _gaussian_blur(x, kernel_size=5, sigma=blur_sigma)
    if downsample > 0:
        x = F.avg_pool2d(x, downsample)
    if noise_sigma > 0:
        x = x + noise_sigma * torch.randn_like(x)
    return x


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, G, D, E, G_opt, D_opt, logs, extra=None, G_ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "E_state": E.state_dict(),
        "G_opt_state": G_opt.state_dict(),
        "D_opt_state": D_opt.state_dict(),
        "logs": dict(logs),
        "extra": extra,
    }
    if G_ema is not None:
        state["G_ema_state"] = G_ema.state_dict()
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path, G, D, E, G_opt=None, D_opt=None, G_ema=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    E.load_state_dict(ckpt["E_state"])
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

def plot_random_z_trajectories(G, ctrl_bank, device, args, step):
    """Fixed input, different random z -> trajectory grid."""
    G.eval()
    n_rows = 6
    T = args.nca_steps
    n_traj = 5  # trajectory columns after input
    n_cols = 1 + n_traj
    output_steps = sorted(set(int(round((i + 1) * T / n_traj)) for i in range(n_traj)))

    raw = ctrl_bank[0].float()
    img_input = ((raw + 0.5) / 255.0 * 2.0 - 1.0).unsqueeze(0).to(device)
    if args.hidden_channels > 0:
        pad = torch.zeros(1, args.hidden_channels, img_input.shape[2], img_input.shape[3], device=device)
        nca_input = torch.cat([img_input, pad], dim=1)
    else:
        nca_input = img_input

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    dummy_cond = torch.zeros(1, dtype=torch.long, device=device)

    # Use fixed seed for reproducible visualisation across steps
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    with torch.no_grad():
        for row in range(n_rows):
            z = torch.randn(1, args.z_dim, device=device, generator=rng)
            trajectory = G.sample(nca_input, dummy_cond, n_steps=T,
                                  output_steps=output_steps, z=z)
            # Column 0: input image
            axes[row, 0].imshow(to_rgb_numpy(img_input[0, :3]))
            if row == 0:
                axes[row, 0].set_title("input", fontsize=28)
            axes[row, 0].set_ylabel(f"z[{row}]", fontsize=20)
            for col, (t_step, state) in enumerate(zip(output_steps, trajectory)):
                ax = axes[row, col + 1] if col + 1 < n_cols else None
                if ax is None:
                    break
                ax.imshow(to_rgb_numpy(state[0, :3]))
                if row == 0:
                    ax.set_title(f"t={t_step}", fontsize=28)
            for c in range(n_cols):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
                for spine in axes[row, c].spines.values():
                    spine.set_visible(False)

    fig.suptitle(f"Random z trajectories (step {step})", fontsize=36)
    plt.tight_layout()
    G.train()
    return fig


def plot_cycle_trajectories(G, ctrl_bank, device, args, step):
    """10 images x 6 cycles, each cycle = nca_steps with independent z."""
    G.eval()
    n_images = 10
    n_cycles = 6
    T = args.nca_steps
    N = ctrl_bank.shape[0]
    dummy_cond = torch.zeros(1, dtype=torch.long, device=device)

    fig, axes = plt.subplots(n_images, n_cycles + 1, figsize=(3 * (n_cycles + 1), 3 * n_images))

    with torch.no_grad():
        for row in range(n_images):
            idx = row * max(1, N // n_images)
            raw = ctrl_bank[idx].float()
            img = (raw + 0.5) / 255.0 * 2.0 - 1.0
            state = img.unsqueeze(0).to(device)
            if args.hidden_channels > 0:
                pad = torch.zeros(1, args.hidden_channels,
                                  state.shape[2], state.shape[3], device=device)
                state = torch.cat([state, pad], dim=1)

            axes[row, 0].imshow(to_rgb_numpy(img))
            if row == 0:
                axes[row, 0].set_title("input", fontsize=28)

            for cycle in range(n_cycles):
                z = torch.randn(1, args.z_dim, device=device)
                state = G(state, dummy_cond, n_steps=T, z=z)
                axes[row, cycle + 1].imshow(to_rgb_numpy(state[0, :3]))
                if row == 0:
                    axes[row, cycle + 1].set_title(f"cycle {cycle+1}", fontsize=28)

            for c in range(n_cycles + 1):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
                for spine in axes[row, c].spines.values():
                    spine.set_visible(False)

    fig.suptitle(f"6-cycle trajectories, independent z (step {step})", fontsize=36)
    plt.tight_layout()
    G.train()
    return fig


def plot_image_grid(images, suptitle, max_images=16):
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
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=24)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FID (per-plate, test ctrl only)
# ---------------------------------------------------------------------------

_fid_metric = None


@torch.no_grad()
def compute_fid(G, metadata_csv, image_dir, device, args):
    """FID: NCA(test_ctrl, random_z) vs matched-size sample of train ctrl.

    Real distribution: random sample of train ctrl images (same count as test).
    Fake distribution: NCA applied to test ctrl images with random z.
    """
    global _fid_metric
    if not FID_AVAILABLE:
        print("[warn] torchmetrics not installed — skipping FID")
        return None
    try:
        G.eval()
        if _fid_metric is None:
            _fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        _fid_metric.reset()

        df = pd.read_csv(metadata_csv, index_col=0)
        df_ctrl = df[df["STATE"] == 0]
        test_keys = df_ctrl[df_ctrl["SPLIT"] == "test"]["SAMPLE_KEY"].values
        train_keys = df_ctrl[df_ctrl["SPLIT"] == "train"]["SAMPLE_KEY"].values

        # Sample train ctrl to match test count
        n = len(test_keys)
        rng = np.random.RandomState(0)
        real_keys = rng.choice(train_keys, size=min(n, len(train_keys)), replace=False)

        image_path = Path(image_dir)
        dummy_cond = torch.zeros(args.batch_size, dtype=torch.long, device=device)

        # Real: train ctrl sample
        for i in range(0, len(real_keys), args.batch_size):
            batch_keys = real_keys[i:i + args.batch_size]
            imgs = torch.stack([
                _load_image(image_path, k, args.image_size, augment=False)
                for k in batch_keys
            ]).to(device)
            real_01 = torch.floor(((imgs.clamp(-1, 1) + 1) / 2) * 255).float() / 255.0
            _fid_metric.update(real_01, real=True)

        # Fake: NCA(test ctrl, random z)
        for i in range(0, len(test_keys), args.batch_size):
            batch_keys = test_keys[i:i + args.batch_size]
            B = len(batch_keys)
            imgs = torch.stack([
                _load_image(image_path, k, args.image_size, augment=False)
                for k in batch_keys
            ]).to(device)

            z = torch.randn(B, args.z_dim, device=device)
            if args.hidden_channels > 0:
                pad = torch.zeros(B, args.hidden_channels,
                                  args.image_size, args.image_size, device=device)
                nca_in = torch.cat([imgs, pad], dim=1)
            else:
                nca_in = imgs
            fake = G(nca_in, dummy_cond[:B], n_steps=args.nca_steps, z=z)
            fake_rgb = fake[:, :3].clamp(-1, 1)
            fake_01 = torch.floor(((fake_rgb + 1) / 2) * 255).float() / 255.0
            _fid_metric.update(fake_01, real=False)

        fid = _fid_metric.compute().item()
        G.train()
        return fid

    except Exception as e:
        print(f"[warn] FID failed: {e}")
        import traceback
        traceback.print_exc()
        G.train()
        torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def populate_pool(pool, ctrl_bank, device, img_channels):
    """Fill pool with ctrl images from preloaded bank.  z is random from init."""
    N = ctrl_bank.shape[0]
    for i in range(pool.pool_size):
        idx = np.random.randint(N)
        img = ctrl_bank[idx].float()
        img = (img + 0.5) / 255.0 * 2.0 - 1.0  # center-of-bin, no augment
        pool.states[i] = 0.0
        pool.states[i, :img_channels] = img.to(device)
    pool.iters.zero_()


def recycle_pool_slots(pool, threshold, ctrl_bank, device, img_channels):
    """Replace old pool slots with fresh ctrl + fresh random z."""
    mask = pool.iters > threshold
    if not mask.any():
        return 0
    N = ctrl_bank.shape[0]
    idxs = mask.nonzero(as_tuple=True)[0]
    for idx in idxs:
        i = idx.item()
        j = np.random.randint(N)
        img = ctrl_bank[j].float()
        img = (img + 0.5) / 255.0 * 2.0 - 1.0
        pool.states[i] = 0.0
        pool.states[i, :img_channels] = img.to(device)
        pool.z[i].normal_()
        pool.iters[i] = 0
    return len(idxs)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="Self-supervised NCA-GAN with style encoder (homeostasis)")

    # data
    p.add_argument("--metadata_csv", type=str, default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=48)

    # NCA (LatentNCA with dummy class)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--nca_hidden_dim", type=int, default=128)
    p.add_argument("--nca_cond_dim", type=int, default=4,
                   help="Dummy compound embedding dim (constant bias, kept small)")
    p.add_argument("--hidden_channels", type=int, default=12)
    p.add_argument("--fire_rate", type=float, default=1.0)
    p.add_argument("--nca_steps", type=int, default=30)
    p.add_argument("--step_size", type=float, default=0.08)
    p.add_argument("--use_tanh", action="store_true")

    # style encoder (Q-network, ResBlk-based)
    p.add_argument("--s_base_channels", type=int, default=128)
    p.add_argument("--s_num_downsamples", type=int, default=4)
    p.add_argument("--s_max_channels", type=int, default=512)

    # discriminator (unconditional, bigD)
    p.add_argument("--d_stages", type=int, default=3)
    p.add_argument("--d_base_channels", type=int, default=48)
    p.add_argument("--d_blocks", type=int, default=3)
    p.add_argument("--d_cardinality", type=int, default=4)
    p.add_argument("--d_expansion", type=int, default=2)
    p.add_argument("--d_kernel_size", type=int, default=3)

    # style reconstruction
    p.add_argument("--style_weight", type=float, default=1.0)

    # anti-steganography (augment before style encoder)
    p.add_argument("--encoder_noise", type=float, default=0.0,
                   help="Gaussian noise sigma added before E (0 = off)")
    p.add_argument("--encoder_blur_sigma", type=float, default=0.0,
                   help="Gaussian blur sigma before E (0 = off)")
    p.add_argument("--encoder_downsample", type=int, default=0,
                   help="Avg-pool factor before E, e.g. 4 means 48->12 (0 = off)")

    # regularisation
    p.add_argument("--intermediate_weight", type=float, default=0.0,
                   help="Intermediate NCA step D loss (manifold regularisation)")
    p.add_argument("--diversity_weight", type=float, default=0.0,
                   help="Diversity loss: different z -> different downsampled output (0 = off)")
    p.add_argument("--diversity_decay_steps", type=int, default=50000,
                   help="Linearly decay diversity_weight to 0 over this many steps")

    # pool
    p.add_argument("--use_pool", action="store_true",
                   help="Replay pool for long-term homeostasis")
    p.add_argument("--pool_size", type=int, default=2024)
    p.add_argument("--pool_recycle_every", type=int, default=50,
                   help="Check for stale pool slots every N training steps")
    p.add_argument("--pool_recycle_threshold", type=int, default=4,
                   help="Recycle after this many NCA cycles (each = nca_steps)")

    # training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--iterations", type=int, default=160000)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--gamma", type=float, default=1.0, help="GP weight")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_warmup_steps", type=int, default=1000)

    # misc
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--vis_every", type=int, default=1000)
    p.add_argument("--fid_every", type=int, default=5000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="nca-cellflow-homeostasis")
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
            if k in type_map and not isinstance(v, type_map[k]):
                v = type_map[k](v)
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    args = load_config_into_args(args)
    print(f"[config] z_dim={args.z_dim}, style_weight={args.style_weight}, "
          f"intermediate_weight={args.intermediate_weight}, use_pool={args.use_pool}, "
          f"enc_noise={args.encoder_noise}, enc_blur={args.encoder_blur_sigma}, "
          f"enc_down={args.encoder_downsample}")

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[warn] --wandb requested but wandb not installed")
    if not use_wandb:
        matplotlib.use("TkAgg")

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Preload all ctrl images into a single tensor (uint8 on CPU, augment on GPU)
    dataset = CtrlPairDataset(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        split="train",
        image_size=args.image_size,
        augment=False,  # we'll augment ourselves after moving to GPU
    )
    # Stack all cached images into a contiguous tensor for fast indexing
    _all_imgs = []
    for key in dataset.ctrl_keys:
        raw = dataset._cache[key]
        if isinstance(raw, np.ndarray):
            _all_imgs.append(torch.from_numpy(raw.astype(np.float32)).permute(2, 0, 1))
        else:
            _all_imgs.append(raw)
    ctrl_bank = torch.stack(_all_imgs).to(device)  # [N, 3, H, W] on GPU
    del _all_imgs, dataset._cache
    N_ctrl = ctrl_bank.shape[0]
    print(f"Ctrl bank: {N_ctrl} images on {device}, "
          f"{ctrl_bank.element_size() * ctrl_bank.nelement() / 1e6:.0f} MB")

    # Channels
    img_channels = 3
    channel_n = img_channels + args.hidden_channels

    # --- Generator (LatentNCA, dummy single class) ---
    G = LatentNCA(
        channel_n=channel_n,
        z_dim=args.z_dim,
        hidden_dim=args.nca_hidden_dim,
        num_classes=1,
        cond_dim=args.nca_cond_dim,
        cond_type="id",
        fire_rate=args.fire_rate,
        step_size=args.step_size,
        use_tanh=args.use_tanh,
    ).to(device)

    # --- Style Encoder (Q-network: image -> z_hat) ---
    # Auto-reduce downsamples when E sees downsampled input
    s_num_ds = args.s_num_downsamples
    if args.encoder_downsample > 0:
        eff_size = args.image_size // args.encoder_downsample
        max_ds = max(1, int(np.log2(eff_size)) - 1)
        s_num_ds = min(s_num_ds, max_ds)
        print(f"[E] input downsampled {args.image_size}->{eff_size}, "
              f"using {s_num_ds} downsamples (was {args.s_num_downsamples})")
    E = ResBlkStyleEncoder(
        in_channels=img_channels,
        base_channels=args.s_base_channels,
        num_downsamples=s_num_ds,
        max_channels=args.s_max_channels,
        z_dim=args.z_dim,
    ).to(device)

    # --- Discriminator (unconditional) ---
    d_widths = [args.d_base_channels] * (args.d_stages + 1)
    d_blocks = [args.d_blocks] * (args.d_stages + 1)
    d_cards = [args.d_cardinality] * (args.d_stages + 1)
    D = Discriminator(
        widths=d_widths,
        cardinalities=d_cards,
        blocks_per_stage=d_blocks,
        expansion=args.d_expansion,
        num_classes=None,
        embed_dim=0,
        kernel_size=args.d_kernel_size,
        in_channels=img_channels,
    ).to(device)

    # --- EMA ---
    use_ema = args.ema_decay > 0
    if use_ema:
        ema_decay = args.ema_decay
        ema_warmup = args.ema_warmup_steps

        def _ema_avg_fn(avg_param, model_param, num_averaged):
            decay = min(ema_decay, (1 + num_averaged) / (ema_warmup + num_averaged))
            return avg_param + (1 - decay) * (model_param - avg_param)

        G_ema = torch.optim.swa_utils.AveragedModel(G, avg_fn=_ema_avg_fn)
    else:
        G_ema = None

    g_total, _ = count_parameters(G)
    e_total, _ = count_parameters(E)
    d_total, _ = count_parameters(D)
    print(f"Generator (LatentNCA): {g_total:,} params")
    print(f"StyleEncoder (Q-net):  {e_total:,} params")
    print(f"Discriminator:         {d_total:,} params")

    # --- Optimisers (E trained with G) ---
    adam_kw = dict(betas=(args.beta1, args.beta2), eps=1e-8)
    g_params = list(G.parameters()) + list(E.parameters())
    try:
        G_opt = torch.optim.Adam(g_params, lr=args.lr_g, fused=True, **adam_kw)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, fused=True, **adam_kw)
    except Exception:
        G_opt = torch.optim.Adam(g_params, lr=args.lr_g, **adam_kw)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, **adam_kw)

    # --- AMP ---
    use_amp = device.type == "cuda"
    autocast = lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # --- Pool ---
    use_pool = args.use_pool
    pool = None
    if use_pool:
        pool = ReplayPool(
            pool_size=args.pool_size,
            channel_n=channel_n,
            H=args.image_size,
            W=args.image_size,
            z_dim=args.z_dim,
            device=device,
        )
        print(f"Pool: {args.pool_size} slots, {args.nca_steps} steps/cycle, "
              f"recycle after {args.pool_recycle_threshold} cycles "
              f"({args.pool_recycle_threshold * args.nca_steps} total steps)")

    # --- Logging / resume ---
    logs = defaultdict(list)
    start_step = 0
    extra = None
    if args.resume:
        start_step, logs, extra = load_checkpoint(
            args.resume, G, D, E, G_opt, D_opt, G_ema=G_ema, map_location=device)
        if use_pool and extra and "pool_state" in extra:
            pool.load_state_dict(extra["pool_state"], device=device)
            print(f"[pool] restored {args.pool_size} slots from checkpoint")
        print(f"Resuming from step {start_step}")

    # Populate pool from scratch if not resumed
    if use_pool and (not args.resume or not extra or "pool_state" not in extra):
        populate_pool(pool, ctrl_bank, device, img_channels)
        print(f"[pool] populated {args.pool_size} slots with fresh ctrl images")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb_run_id = None
    if args.resume and extra and isinstance(extra, dict):
        wandb_run_id = extra.get("wandb_run_id", None)
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   id=wandb_run_id, config=vars(args),
                   resume="allow" if args.resume else None)

    # --- Helpers ---
    use_intermediate = args.intermediate_weight > 0
    dummy_cond = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    def sample_ctrl_batch(B):
        """Sample B random ctrl images from the GPU-resident bank, augment in-place."""
        idx = torch.randint(N_ctrl, (B,), device=device)
        imgs = ctrl_bank[idx].clone()  # [B, 3, H, W] already on GPU
        # Dither + normalize to [-1, 1]
        imgs = (imgs + torch.rand_like(imgs)) / 255.0
        imgs = imgs * 2.0 - 1.0
        # Random flips
        if torch.rand(1).item() < 0.5:
            imgs = imgs.flip(-1)
        if torch.rand(1).item() < 0.5:
            imgs = imgs.flip(-2)
        return imgs

    # --- Training loop ---
    pbar = tqdm(range(start_step, args.iterations), desc="Training",
                initial=start_step, total=args.iterations)
    for step in pbar:
        G.train()
        D.train()
        E.train()

        # ================ Sample batch ================
        B = args.batch_size
        img_input = sample_ctrl_batch(B)
        img_ref = sample_ctrl_batch(B)  # independent ctrl images for D real
        cond = dummy_cond[:B]

        # ================ D step ================
        set_requires_grad(G, False)
        set_requires_grad(D, True)
        D_opt.zero_grad(set_to_none=True)

        # z is always fresh random — never stored in pool
        z = torch.randn(B, args.z_dim, device=device)

        with torch.no_grad():
            if use_pool:
                p_idx, p_states, _, _, _, _ = pool.get_batch(B)
                if use_intermediate:
                    t_inter_d = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full, inter_full = G.forward_with_intermediate(
                        p_states, cond, n_steps=args.nca_steps,
                        t_intermediate=t_inter_d, z=z)
                else:
                    fake_full = G(p_states, cond, n_steps=args.nca_steps, z=z)
            else:
                if args.hidden_channels > 0:
                    pad = torch.zeros(B, args.hidden_channels,
                                      img_input.shape[2], img_input.shape[3], device=device)
                    nca_input = torch.cat([img_input, pad], dim=1)
                else:
                    nca_input = img_input
                if use_intermediate:
                    t_inter_d = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full, inter_full = G.forward_with_intermediate(
                        nca_input, cond, n_steps=args.nca_steps,
                        t_intermediate=t_inter_d, z=z)
                else:
                    fake_full = G(nca_input, cond, n_steps=args.nca_steps, z=z)
            fake_img = fake_full[:, :img_channels].contiguous()

        # Endpoint D loss
        real_req = img_ref.detach().requires_grad_(True)
        fake_req = fake_img.detach().requires_grad_(True)
        with autocast():
            d_real = D(real_req)
            d_fake = D(fake_req)
            adv_d = relativistic_d_loss(d_real, d_fake)
        gp_real = zero_centered_gradient_penalty(real_req.float(), d_real.float())
        gp_fake = zero_centered_gradient_penalty(fake_req.float(), d_fake.float())
        reg = 0.5 * args.gamma * (gp_real.mean() + gp_fake.mean())
        d_loss = adv_d + reg
        d_loss.backward()

        # Intermediate D loss
        accum_d_inter = 0.0
        if use_intermediate:
            inter_img = inter_full[:, :img_channels].contiguous()
            real_inter_req = img_ref.detach().requires_grad_(True)
            inter_req = inter_img.detach().requires_grad_(True)
            with autocast():
                d_real_i = D(real_inter_req)
                d_inter = D(inter_req)
                adv_inter = relativistic_d_loss(d_real_i, d_inter)
            gp_ri = zero_centered_gradient_penalty(real_inter_req.float(), d_real_i.float())
            gp_fi = zero_centered_gradient_penalty(inter_req.float(), d_inter.float())
            reg_i = 0.5 * args.gamma * (gp_ri.mean() + gp_fi.mean())
            d_inter_loss = args.intermediate_weight * (adv_inter + reg_i)
            d_inter_loss.backward()
            accum_d_inter = d_inter_loss.item()

        D_opt.step()

        # ================ G step ================
        set_requires_grad(G, True)
        set_requires_grad(D, False)
        G_opt.zero_grad(set_to_none=True)

        z_g = z  # same random z as D step
        if use_pool:
            with autocast():
                if use_intermediate:
                    t_inter_g = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full_g, inter_full_g = G.forward_with_intermediate(
                        p_states.detach(), cond, n_steps=args.nca_steps,
                        t_intermediate=t_inter_g, z=z_g)
                else:
                    fake_full_g = G(p_states.detach(), cond,
                                    n_steps=args.nca_steps, z=z_g)
        else:
            if args.hidden_channels > 0:
                pad = torch.zeros(B, args.hidden_channels,
                                  img_input.shape[2], img_input.shape[3], device=device)
                nca_input = torch.cat([img_input, pad], dim=1)
            else:
                nca_input = img_input
            with autocast():
                if use_intermediate:
                    t_inter_g = torch.randint(0, args.nca_steps - 1, (1,)).item()
                    fake_full_g, inter_full_g = G.forward_with_intermediate(
                        nca_input, cond, n_steps=args.nca_steps,
                        t_intermediate=t_inter_g, z=z_g)
                else:
                    fake_full_g = G(nca_input, cond, n_steps=args.nca_steps, z=z_g)

        fake_img_g = fake_full_g[:, :img_channels].contiguous()

        # Relativistic G loss
        with autocast():
            d_real_g = D(img_ref.detach())
            d_fake_g = D(fake_img_g)
        g_loss = F.softplus(-(d_fake_g - d_real_g)).mean()
        total_loss = g_loss

        # Intermediate G loss
        accum_g_inter = 0.0
        if use_intermediate:
            inter_img_g = inter_full_g[:, :img_channels].contiguous()
            with autocast():
                d_real_inter_g = D(img_ref.detach())
                d_inter_g = D(inter_img_g)
            g_inter_adv = F.softplus(-(d_inter_g - d_real_inter_g)).mean()
            total_loss = total_loss + args.intermediate_weight * g_inter_adv
            accum_g_inter = g_inter_adv.item()

        # Style reconstruction: E(augmented fake) should recover the random z
        accum_style = 0.0
        if args.style_weight > 0:
            aug_fake = augment_for_encoder(
                fake_img_g, noise_sigma=args.encoder_noise,
                blur_sigma=args.encoder_blur_sigma,
                downsample=args.encoder_downsample)
            with autocast():
                z_hat = E(aug_fake)
            sty_loss = F.mse_loss(z_hat, z_g.detach())
            total_loss = total_loss + args.style_weight * sty_loss
            accum_style = sty_loss.item()

        # Diversity loss: different z -> different output (measured on downsampled images)
        accum_div = 0.0
        if args.diversity_weight > 0:
            decay = max(0.0, 1.0 - step / args.diversity_decay_steps)
            div_w = args.diversity_weight * decay
            if div_w > 1e-6:
                z2 = torch.randn(B, args.z_dim, device=device)
                with torch.no_grad():
                    if use_pool:
                        fake2 = G(p_states.detach(), cond, n_steps=args.nca_steps, z=z2)
                    else:
                        fake2 = G(nca_input, cond, n_steps=args.nca_steps, z=z2)
                    fake2_rgb = fake2[:, :img_channels].contiguous()
                # Downsample to 12x12 to measure structural diversity, not pixel noise
                fake1_down = F.avg_pool2d(fake_img_g.float(), 4)
                fake2_down = F.avg_pool2d(fake2_rgb.float(), 4)
                div_loss = -(fake1_down - fake2_down.detach()).abs().mean()
                total_loss = total_loss + div_w * div_loss
                accum_div = div_loss.item()

        total_loss.backward()
        nn_utils.clip_grad_norm_(g_params, max_norm=args.grad_clip)
        G_opt.step()

        if use_ema:
            G_ema.update_parameters(G)

        # ================ Pool update ================
        if use_pool:
            pool.update(p_idx, fake_full_g.detach(),
                        pool.labels[p_idx], pool.doses[p_idx],
                        [pool.plates[i.item()] for i in p_idx],
                        pool.z[p_idx])  # z not used, just pass existing
            if (step + 1) % args.pool_recycle_every == 0:
                n_recycled = recycle_pool_slots(
                    pool, args.pool_recycle_threshold,
                    ctrl_bank, device, img_channels)
                if n_recycled > 0:
                    print(f"[pool] recycled {n_recycled} slots at step {step+1}")

        # ================ Logging ================
        log_dict = {
            "loss/D_total": d_loss.item(),
            "loss/D_adv": adv_d.item(),
            "loss/D_reg": reg.item(),
            "penalty/gp_real": gp_real.mean().item(),
            "penalty/gp_fake": gp_fake.mean().item(),
            "logits/D_real": d_real.mean().item(),
            "logits/D_fake": d_fake.mean().item(),
            "loss/G": g_loss.item(),
        }
        if args.style_weight > 0:
            log_dict["loss/style_recon"] = accum_style
        if use_intermediate:
            log_dict["loss/D_inter"] = accum_d_inter
            log_dict["loss/G_inter"] = accum_g_inter
        if args.diversity_weight > 0:
            log_dict["loss/diversity"] = accum_div
            log_dict["diversity_weight_eff"] = args.diversity_weight * max(0.0, 1.0 - step / args.diversity_decay_steps)

        for k, v in log_dict.items():
            logs[k].append(v)
        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix(d=f"{d_loss.item():.4f}", g=f"{g_loss.item():.4f}",
                             sty=f"{accum_style:.3f}")

        # ================ Visualisations ================
        if (step + 1) % args.vis_every == 0:
            vis_G = G_ema.module if use_ema else G
            fig_traj = plot_random_z_trajectories(vis_G, ctrl_bank, device, args, step + 1)
            fig_cyc = plot_cycle_trajectories(vis_G, ctrl_bank, device, args, step + 1)
            fig_gen = plot_image_grid(
                [fake_img_g[i, :3].detach() for i in range(min(B, 16))],
                f"Generated (step {step+1})")
            fig_real = plot_image_grid(
                [img_ref[i].detach() for i in range(min(B, 16))],
                f"Real ctrl (step {step+1})")
            if use_wandb:
                wandb.log({
                    "vis/z_trajectories": wandb.Image(fig_traj),
                    "vis/cycle_trajectories": wandb.Image(fig_cyc),
                    "vis/generated": wandb.Image(fig_gen),
                    "vis/real": wandb.Image(fig_real),
                }, step=step + 1)
                plt.close(fig_traj)
                plt.close(fig_cyc)
                plt.close(fig_gen)
                plt.close(fig_real)
            else:
                plt.show()

        # ================ FID ================
        if args.fid_every > 0 and (step + 1) % args.fid_every == 0:
            eval_G = G_ema.module if use_ema else G
            fid = compute_fid(eval_G, args.metadata_csv, args.image_dir, device, args)
            if fid is not None:
                logs["fid/global"].append(fid)
                print(f"[FID] step {step+1}: {fid:.2f}")
                if use_wandb:
                    wandb.log({"fid/global": fid}, step=step + 1)

        # ================ Checkpointing ================
        is_save = (step + 1) % args.save_every == 0
        is_last = step == args.iterations - 1
        if is_save or is_last:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step + 1}.pt")
            ckpt_extra = {
                "z_dim": args.z_dim,
                "nca_steps": args.nca_steps,
                "hidden_channels": args.hidden_channels,
                "channel_n": channel_n,
                "step_size": args.step_size,
                "use_tanh": args.use_tanh,
                "nca_cond_dim": args.nca_cond_dim,
                "nca_hidden_dim": args.nca_hidden_dim,
                "s_base_channels": args.s_base_channels,
                "s_num_downsamples": args.s_num_downsamples,
                "s_max_channels": args.s_max_channels,
                "style_weight": args.style_weight,
                "encoder_noise": args.encoder_noise,
                "encoder_blur_sigma": args.encoder_blur_sigma,
                "encoder_downsample": args.encoder_downsample,
                "intermediate_weight": args.intermediate_weight,
                "diversity_weight": args.diversity_weight,
                "diversity_decay_steps": args.diversity_decay_steps,
                "use_pool": args.use_pool,
                "ema_decay": args.ema_decay,
                "wandb_run_id": wandb.run.id if use_wandb else None,
            }
            if use_pool:
                ckpt_extra["pool_state"] = pool.state_dict()
            save_checkpoint(ckpt_path, step + 1, G, D, E, G_opt, D_opt,
                            logs, extra=ckpt_extra, G_ema=G_ema)

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    train(args)
