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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from nca_cellflow import IMPADataset
from nca_cellflow.models import BaseNCA, NoiseNCA, Discriminator


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def zero_centered_gradient_penalty(samples: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    (grad,) = torch.autograd.grad(outputs=logits.sum(), inputs=samples, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


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

def save_checkpoint(path, step, G, D, G_opt, D_opt, logs, extra=None):
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
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path, G, D, G_opt=None, D_opt=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    if G_opt is not None and "G_opt_state" in ckpt:
        G_opt.load_state_dict(ckpt["G_opt_state"])
    if D_opt is not None and "D_opt_state" in ckpt:
        D_opt.load_state_dict(ckpt["D_opt_state"])
    logs = defaultdict(list, ckpt.get("logs", {}))
    step = ckpt.get("step", 0)
    extra = ckpt.get("extra", None)
    print(f"[ckpt] loaded {path} at step {step}")
    return step, logs, extra


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectories(G, dataset, device, args, id2cpd, img_channels, step):
    """
    NCA trajectories: one row per compound, 5 columns = intermediate states.
    Uses first control image as input for all conditions.
    """
    G.eval()
    n_cols = 5
    n_steps = args.nca_steps
    output_steps = sorted(set([0] + [int(round(i * n_steps / (n_cols - 1))) for i in range(n_cols)]))

    img_ctrl = dataset._load(dataset.ctrl_keys[0]).unsqueeze(0).to(device)
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
            trajectory = G.sample(nca_input, cond, n_steps=n_steps, output_steps=output_steps)
            for col, (t_step, state) in enumerate(zip(output_steps, trajectory)):
                rgb = to_rgb_numpy(state[0, :img_channels])
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
                       fake_img, img_ctrl, img_trt, cpd_id, step, use_wandb):
    """Generate and display/log all visualization figures.

    Args:
        fake_img: Already-computed NCA output (visible channels only) from this training step.
        img_ctrl: Control images that were fed into the NCA this step.
    """
    G.eval()

    # 1. Trajectory plot
    fig_traj = plot_trajectories(G, dataset, device, args, id2cpd, img_channels, step)
    if use_wandb:
        wandb.log({"vis/trajectories": wandb.Image(fig_traj)}, step=step)
        plt.close(fig_traj)
    else:
        plt.show()

    n = min(img_trt.shape[0], 16)
    titles = [id2cpd[cpd_id[i].item()] for i in range(n)]

    # 2. Real control samples (NCA input)
    fig_ctrl = plot_image_grid(
        [img_ctrl[i, :img_channels] for i in range(n)],
        ["DMSO"] * n,
        f"Control / NCA input (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/control_samples": wandb.Image(fig_ctrl)}, step=step)
        plt.close(fig_ctrl)
    else:
        plt.show()

    # 3. Generated samples (NCA outputs from this training step)
    fig_fake = plot_image_grid(
        [fake_img[i].detach() for i in range(n)],
        titles,
        f"Generated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/generated_samples": wandb.Image(fig_fake)}, step=step)
        plt.close(fig_fake)
    else:
        plt.show()

    # 4. Real treated samples (target distribution)
    fig_real = plot_image_grid(
        [img_trt[i, :img_channels] for i in range(n)],
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
# Main
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NCA-GAN on IMPA dataset")

    # data
    p.add_argument("--metadata_csv", type=str, default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96,
                   help="Resize images to this size (default 96, native resolution)")

    # model
    p.add_argument("--nca_type", type=str, default="base",
                   choices=["base", "noise"],
                   help="NCA variant: 'base' or 'noise' (with noise injection)")
    p.add_argument("--nca_hidden_dim", type=int, default=128)
    p.add_argument("--nca_cond_dim", type=int, default=64)
    p.add_argument("--hidden_channels", type=int, default=0,
                   help="Extra hidden channels for NCA state (0 = RGB only)")
    p.add_argument("--noise_channels", type=int, default=1,
                   help="Number of noise channels for NoiseNCA")
    p.add_argument("--fire_rate", type=float, default=1.0)
    p.add_argument("--nca_steps", type=int, default=60,
                   help="Number of NCA steps per generation")

    # discriminator
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

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--vis_every", type=int, default=1000,
                   help="Log visualization plots every N steps")
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
    )
    num_compounds = len(dataset.cpd2id)
    id2cpd = {v: k for k, v in dataset.cpd2id.items()}
    print(f"Dataset: {len(dataset)} control images, {len(dataset.trt_keys)} treated images, "
          f"{num_compounds} compounds")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(loader)

    # ---- visible channels = 3 RGB + optional hidden ----
    img_channels = 3
    channel_n = img_channels + args.hidden_channels

    # ---- models ----
    if args.nca_type == "noise":
        G = NoiseNCA(
            channel_n=channel_n,
            noise_channels=args.noise_channels,
            hidden_dim=args.nca_hidden_dim,
            num_classes=num_compounds,
            cond_dim=args.nca_cond_dim,
            fire_rate=args.fire_rate,
        ).to(device)
    else:
        G = BaseNCA(
            channel_n=channel_n,
            hidden_dim=args.nca_hidden_dim,
            num_classes=num_compounds,
            cond_dim=args.nca_cond_dim,
            fire_rate=args.fire_rate,
        ).to(device)

    if args.compile:
        G = torch.compile(G)

    # Discriminator only sees visible (RGB) channels
    d_widths = [args.d_base_channels] * (args.d_stages + 1)
    d_blocks = [args.d_blocks] * (args.d_stages + 1)
    d_cards = [args.d_cardinality] * (args.d_stages + 1)

    D = Discriminator(
        widths=d_widths,
        cardinalities=d_cards,
        blocks_per_stage=d_blocks,
        expansion=args.d_expansion,
        num_classes=num_compounds,
        embed_dim=args.d_embed_dim,
        kernel_size=args.d_kernel_size,
        in_channels=img_channels,
    ).to(device)

    g_total, g_train = count_parameters(G)
    d_total, d_train = count_parameters(D)
    print(f"Generator  params: total={g_total:,}  trainable={g_train:,}")
    print(f"Discriminator params: total={d_total:,}  trainable={d_train:,}")

    # ---- optimizers ----
    adam_kwargs = dict(betas=(args.beta1, args.beta2), eps=1e-8)
    try:
        G_opt = torch.optim.Adam(G.parameters(), lr=args.lr_g, fused=True, **adam_kwargs)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, fused=True, **adam_kwargs)
    except Exception:
        G_opt = torch.optim.Adam(G.parameters(), lr=args.lr_g, **adam_kwargs)
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, **adam_kwargs)

    # ---- AMP ----
    use_amp = device.type == "cuda"
    autocast = lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # ---- logging / resume ----
    logs = defaultdict(list)
    start_step = 0

    if args.resume:
        start_step, logs, extra = load_checkpoint(
            args.resume, G, D, G_opt, D_opt, map_location=device
        )
        print(f"Resuming from step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- wandb init ----
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
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
    pbar = tqdm(range(start_step, args.iterations), desc="Training", initial=start_step, total=args.iterations)
    for step in pbar:
        img_ctrl, img_trt, cpd_id = next_batch()
        img_ctrl = img_ctrl.to(device)
        img_trt = img_trt.to(device)
        cpd_id = cpd_id.to(device)

        # If using hidden channels, pad ctrl with zeros
        if args.hidden_channels > 0:
            pad = torch.zeros(
                img_ctrl.shape[0], args.hidden_channels,
                img_ctrl.shape[2], img_ctrl.shape[3],
                device=device,
            )
            nca_input = torch.cat([img_ctrl, pad], dim=1)
        else:
            nca_input = img_ctrl

        G.train()
        D.train()

        # ============== Discriminator step ==============
        set_requires_grad(G, False)
        set_requires_grad(D, True)
        D_opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake_full = G(nca_input, cpd_id, n_steps=args.nca_steps)
            fake_img = fake_full[:, :img_channels].contiguous()

        real_req = img_trt.detach().requires_grad_(True)
        fake_req = fake_img.detach().requires_grad_(True)

        with autocast():
            d_real = D(real_req, cpd_id)
            d_fake = D(fake_req, cpd_id)
            rel = d_real - d_fake
            adv_d = F.softplus(-rel).mean()

        # Gradient penalty in fp32 for stability
        gp_real = zero_centered_gradient_penalty(real_req.float(), d_real.float())
        gp_fake = zero_centered_gradient_penalty(fake_req.float(), d_fake.float())
        reg = 0.5 * args.gamma * (gp_real.mean() + gp_fake.mean())

        d_loss = adv_d + reg
        d_loss.backward()
        D_opt.step()

        # ============== Generator step ==============
        set_requires_grad(G, True)
        set_requires_grad(D, False)
        G_opt.zero_grad(set_to_none=True)

        with autocast():
            fake_full = G(nca_input, cpd_id, n_steps=args.nca_steps)
            fake_img = fake_full[:, :img_channels].contiguous()

            d_real = D(img_trt.detach(), cpd_id)
            d_fake = D(fake_img, cpd_id)

        rel = d_fake - d_real
        g_loss = F.softplus(-rel).mean()

        g_loss.backward()
        nn_utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
        G_opt.step()

        # ============== Logging ==============
        log_dict = {
            "loss/D_total": d_loss.item(),
            "loss/D_adv": adv_d.item(),
            "loss/D_reg": reg.item(),
            "penalty/gp_real_mean": gp_real.mean().item(),
            "penalty/gp_fake_mean": gp_fake.mean().item(),
            "logits/D_real_mean": d_real.mean().item(),
            "logits/D_fake_mean": d_fake.mean().item(),
            "loss/G": g_loss.item(),
        }
        for k, v in log_dict.items():
            logs[k].append(v)

        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix({
                "d": f"{d_loss.item():.4f}",
                "g": f"{g_loss.item():.4f}",
                "gp_r": f"{gp_real.mean().item():.2f}",
            })

        # ============== Visualizations ==============
        if (step + 1) % args.vis_every == 0:
            log_visualizations(
                G, dataset, device, args, id2cpd, img_channels,
                fake_img, img_ctrl, img_trt, cpd_id, step + 1, use_wandb,
            )

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
                extra={
                    "gamma": args.gamma,
                    "lr_g": args.lr_g,
                    "lr_d": args.lr_d,
                    "nca_type": args.nca_type,
                    "nca_steps": args.nca_steps,
                    "hidden_channels": args.hidden_channels,
                    "noise_channels": args.noise_channels,
                    "channel_n": channel_n,
                    "num_compounds": num_compounds,
                },
            )

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    train(args)
