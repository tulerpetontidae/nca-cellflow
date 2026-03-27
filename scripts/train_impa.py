"""
IMPA (StarGAN-v2 style) training script using our data pipeline.

Drop-in comparison against NCA-GAN and CellFlux: same dataset, same wandb
logging, same visualization, same FID evaluation.

Implements the IMPA paper's architecture and training procedure:
  - Generator: encoder-decoder with AdaIN conditioning
  - MappingNetwork: fingerprint + noise → style
  - StyleEncoder: image → style (for cycle consistency)
  - Discriminator: multi-task BCE GAN with R1 gradient penalty
  - Losses: adversarial + style recon + cycle consistency + diversity (decaying)

Reference: https://github.com/theislab/IMPA

Usage:
    python scripts/train_impa.py --config configs/impa-bbbc021-lb.yaml
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
from nca_cellflow.models.impa import (
    IMPAGenerator, IMPAMappingNetwork, IMPAStyleEncoder, IMPADiscriminator, he_init,
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


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, G, D, M, S, G_opt, D_opt, MS_opt, logs,
                    extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "M_state": M.state_dict(),
        "S_state": S.state_dict(),
        "G_opt_state": G_opt.state_dict(),
        "D_opt_state": D_opt.state_dict(),
        "MS_opt_state": MS_opt.state_dict(),
        "logs": dict(logs),
        "extra": extra,
    }
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path, G, D, M, S, G_opt=None, D_opt=None, MS_opt=None,
                    map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    M.load_state_dict(ckpt["M_state"])
    S.load_state_dict(ckpt["S_state"])
    if G_opt is not None and "G_opt_state" in ckpt:
        G_opt.load_state_dict(ckpt["G_opt_state"])
    if D_opt is not None and "D_opt_state" in ckpt:
        D_opt.load_state_dict(ckpt["D_opt_state"])
    if MS_opt is not None and "MS_opt_state" in ckpt:
        MS_opt.load_state_dict(ckpt["MS_opt_state"])
    logs = defaultdict(list, ckpt.get("logs", {}))
    step = ckpt.get("step", 0)
    extra = ckpt.get("extra", None)
    print(f"[ckpt] loaded {path} at step {step}")
    return step, logs, extra


# ---------------------------------------------------------------------------
# Visualization
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
def plot_per_compound(G, M, dataset, device, args, id2cpd, fp_matrix, step):
    """Per-compound generation: same ctrl → different compound styles.

    One row per compound, columns: ctrl input, generated, generated (different noise).
    Shows conditioning effect and stochasticity.
    """
    G.eval()
    M.eval()
    img_ctrl = dataset._load(dataset.ctrl_keys[0]).unsqueeze(0).to(device)
    num_compounds = len(id2cpd)

    fig, axes = plt.subplots(num_compounds, 3, figsize=(9, 3 * num_compounds))
    if num_compounds == 1:
        axes = axes[None, :]

    for cpd_idx in range(num_compounds):
        fp = fp_matrix[cpd_idx:cpd_idx+1].to(device)
        z1 = torch.randn(1, args.z_dimension, device=device)
        z2 = torch.randn(1, args.z_dimension, device=device)
        s1 = M(torch.cat([fp, z1], dim=1))
        s2 = M(torch.cat([fp, z2], dim=1))
        _, fake1 = G(img_ctrl, s1)
        _, fake2 = G(img_ctrl, s2)

        for col, (img, title) in enumerate([
            (img_ctrl[0, :3], "ctrl"),
            (fake1[0, :3], "gen (z₁)"),
            (fake2[0, :3], "gen (z₂)"),
        ]):
            ax = axes[cpd_idx, col]
            ax.imshow(to_rgb_numpy(img))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if cpd_idx == 0:
                ax.set_title(title, fontsize=40)
        axes[cpd_idx, 0].set_ylabel(id2cpd[cpd_idx], fontsize=36,
                                     rotation=90, labelpad=60, va="center")

    fig.suptitle(f"Per-compound generation (step {step})", fontsize=48)
    plt.tight_layout()
    G.train()
    M.train()
    return fig


def log_visualizations(G, M, dataset, device, args, id2cpd, fp_matrix,
                       fake_img, img_ctrl, img_trt, cpd_id, step, use_wandb):
    G.eval()
    M.eval()

    # 1. Per-compound generation grid (ctrl → gen_z1, gen_z2 for each compound)
    fig_cpd = plot_per_compound(G, M, dataset, device, args, id2cpd,
                                fp_matrix, step)
    if use_wandb:
        wandb.log({"vis/per_compound": wandb.Image(fig_cpd)}, step=step)
        plt.close(fig_cpd)
    else:
        plt.show()

    n = min(img_trt.shape[0], 16)
    titles = [id2cpd[cpd_id[i].item()] for i in range(n)]

    # 2. Control samples
    fig_ctrl = plot_image_grid(
        [img_ctrl[i, :3] for i in range(n)], ["DMSO"] * n,
        f"Control / IMPA input (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/control_samples": wandb.Image(fig_ctrl)}, step=step)
        plt.close(fig_ctrl)
    else:
        plt.show()

    # 3. Generated samples (from training step)
    fig_fake = plot_image_grid(
        [fake_img[i, :3].detach() for i in range(n)], titles,
        f"Generated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/generated_samples": wandb.Image(fig_fake)}, step=step)
        plt.close(fig_fake)
    else:
        plt.show()

    # 4. Real treated samples
    fig_real = plot_image_grid(
        [img_trt[i, :3] for i in range(n)], titles,
        f"Real treated (step {step})",
    )
    if use_wandb:
        wandb.log({"vis/real_samples": wandb.Image(fig_real)}, step=step)
        plt.close(fig_real)
    else:
        plt.show()

    G.train()
    M.train()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

_global_fid = None
_cpd_fid = None


@torch.no_grad()
def compute_fid(G, M, eval_dataset, device, args, id2cpd, fp_matrix):
    """Compute FID using EvalDataset (deterministic, test split, no augment)."""
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
        M.eval()
        for img_ctrl, img_trt, cpd_id in loader:
            img_ctrl = img_ctrl.to(device)
            img_trt = img_trt.to(device)
            cpd_id = cpd_id.to(device)

            # Style conditioning
            fp = fp_matrix[cpd_id]
            z = torch.randn(img_ctrl.shape[0], args.z_dimension, device=device)
            s = M(torch.cat([fp, z], dim=1))

            _, fake = G(img_ctrl, s)
            fake_rgb = fake[:, :3].contiguous()

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

        G.train()
        M.train()
        return fid_global, fid_per_cpd

    except Exception as e:
        print(f"[warn] FID computation failed: {e}")
        G.train()
        M.train()
        torch.cuda.empty_cache()
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="Train IMPA on IMPA dataset")

    # data
    p.add_argument("--metadata_csv", type=str,
                   default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--plate_match", action="store_true")
    p.add_argument("--balanced_cpd", action="store_true")
    p.add_argument("--iter_trt", action="store_true",
                   help="Iterate over treated images, randomly sample ctrl")
    p.add_argument("--fp_path", type=str, default=None,
                   help="Path to Morgan fingerprint CSV (required)")

    # model (IMPA paper defaults for bbbc021_six)
    p.add_argument("--style_dim", type=int, default=64)
    p.add_argument("--z_dimension", type=int, default=16)
    p.add_argument("--dim_in", type=int, default=64)
    p.add_argument("--max_conv_dim", type=int, default=512)
    p.add_argument("--num_layers_mapping_net", type=int, default=1)

    # losses
    p.add_argument("--lambda_reg", type=float, default=1.0,
                   help="R1 gradient penalty weight")
    p.add_argument("--lambda_cyc", type=float, default=1.0,
                   help="Cycle consistency loss weight")
    p.add_argument("--lambda_sty", type=float, default=1.0,
                   help="Style reconstruction loss weight")
    p.add_argument("--lambda_ds", type=float, default=1.0,
                   help="Diversity loss weight (decays linearly to 0)")
    p.add_argument("--ds_iter", type=int, default=100000,
                   help="Number of iterations over which diversity loss decays to 0")

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=80000)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR for G, D, style_encoder")
    p.add_argument("--f_lr", type=float, default=1e-4,
                   help="LR for mapping_network")
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="Gradient clipping (0 = disabled, IMPA paper does not clip)")

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--vis_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--fid_every", type=int, default=0,
                   help="Compute FID every N steps (0 = disabled)")
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
    print(f"[config] IMPA: img_size={args.image_size}, style_dim={args.style_dim}, "
          f"z_dim={args.z_dimension}, lambda_ds={args.lambda_ds}, ds_iter={args.ds_iter}")

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
    print(f"Dataset: {len(dataset)} images, {len(dataset.trt_keys)} treated, "
          f"{num_compounds} compounds")

    # ---- eval dataset for FID ----
    eval_dataset = None
    if args.fid_every > 0:
        eval_dataset = EvalDataset(
            metadata_csv=args.metadata_csv,
            image_dir=args.image_dir,
            split="test",
            image_size=args.image_size,
        )
        print(f"FID eval dataset: {len(eval_dataset)} treated test images")

    # ---- fingerprint embeddings ----
    assert args.fp_path is not None, "--fp_path required for IMPA conditioning"
    fp_df = pd.read_csv(args.fp_path, index_col=0)
    fp_vecs = []
    for cid in range(num_compounds):
        cpd_name = id2cpd[cid]
        assert cpd_name in fp_df.index, f"Compound '{cpd_name}' not found in {args.fp_path}"
        fp_vecs.append(fp_df.loc[cpd_name].values.astype(np.float32))
    fp_matrix = torch.tensor(np.stack(fp_vecs)).to(device)
    fp_dim = fp_matrix.shape[1]
    print(f"Loaded fingerprints: {fp_matrix.shape} from {args.fp_path}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    data_iter = iter(loader)

    # ---- models ----
    G = IMPAGenerator(
        img_size=args.image_size, style_dim=args.style_dim,
        max_conv_dim=args.max_conv_dim, dim_in=args.dim_in,
    ).to(device)

    D = IMPADiscriminator(
        img_size=args.image_size, num_domains=num_compounds,
        max_conv_dim=args.max_conv_dim, dim_in=args.dim_in,
    ).to(device)

    mapping_input_dim = fp_dim + args.z_dimension
    M = IMPAMappingNetwork(
        latent_dim=mapping_input_dim, style_dim=args.style_dim,
        hidden_dim=512, num_layers=args.num_layers_mapping_net,
    ).to(device)

    S = IMPAStyleEncoder(
        img_size=args.image_size, style_dim=args.style_dim,
        max_conv_dim=args.max_conv_dim, dim_in=args.dim_in,
    ).to(device)

    # He init (matching IMPA paper)
    for net in [G, D, M, S]:
        net.apply(he_init)

    for name, net in [("Generator", G), ("Discriminator", D),
                      ("MappingNetwork", M), ("StyleEncoder", S)]:
        t, tr = count_parameters(net)
        print(f"{name} params: total={t:,}  trainable={tr:,}")

    # ---- optimizers ----
    adam_kwargs = dict(betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr, **adam_kwargs)
    G_opt = torch.optim.Adam(G.parameters(), lr=args.lr, **adam_kwargs)
    # Mapping network + style encoder share one optimizer step (as in IMPA)
    MS_opt = torch.optim.Adam(
        list(M.parameters()) + list(S.parameters()),
        lr=args.f_lr, **adam_kwargs,
    )

    # ---- logging / resume ----
    logs = defaultdict(list)
    start_step = 0
    lambda_ds = args.lambda_ds
    initial_lambda_ds = args.lambda_ds

    if args.resume:
        start_step, logs, extra = load_checkpoint(
            args.resume, G, D, M, S, G_opt, D_opt, MS_opt,
            map_location=device,
        )
        if extra and "lambda_ds" in extra:
            lambda_ds = extra["lambda_ds"]
        print(f"Resuming from step {start_step}, lambda_ds={lambda_ds:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- wandb init ----
    if use_wandb:
        wandb_run_id = None
        if args.resume and extra and isinstance(extra, dict):
            wandb_run_id = extra.get("wandb_run_id", None)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"impa-{args.image_size}px",
            id=wandb_run_id,
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
        G.train(); D.train(); M.train(); S.train()

        img_ctrl, img_trt, cpd_id = next_batch()
        img_ctrl = img_ctrl.to(device)
        img_trt = img_trt.to(device)
        cpd_id = cpd_id.to(device)

        B = img_ctrl.shape[0]

        # ---- Encode label: fp + noise → style ----
        fp = fp_matrix[cpd_id]  # [B, fp_dim]
        z1 = torch.randn(B, args.z_dimension, device=device)
        z2 = torch.randn(B, args.z_dimension, device=device)
        s_trg1 = M(torch.cat([fp, z1], dim=1))
        s_trg2 = M(torch.cat([fp, z2], dim=1))

        # ============== Discriminator step ==============
        D_opt.zero_grad(set_to_none=True)

        img_trt.requires_grad_(True)
        d_real = D(img_trt, cpd_id)
        loss_real = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real))

        # R1 gradient penalty (on real images only, matching IMPA)
        grad_real = torch.autograd.grad(
            outputs=d_real.sum(), inputs=img_trt,
            create_graph=True, retain_graph=True,
        )[0]
        loss_reg = 0.5 * grad_real.pow(2).reshape(B, -1).sum(1).mean()

        with torch.no_grad():
            _, x_fake = G(img_ctrl, s_trg1)
        d_fake = D(x_fake, cpd_id)
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake))

        d_loss = loss_real + loss_fake + args.lambda_reg * loss_reg
        d_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
        D_opt.step()
        img_trt.requires_grad_(False)

        # ============== Generator + MappingNetwork + StyleEncoder step ==============
        G_opt.zero_grad(set_to_none=True)
        MS_opt.zero_grad(set_to_none=True)

        # Adversarial loss
        _, x_fake = G(img_ctrl, s_trg1)
        d_out = D(x_fake, cpd_id)
        loss_adv = F.binary_cross_entropy_with_logits(
            d_out, torch.ones_like(d_out))

        # Style reconstruction loss
        s_pred = S(x_fake)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg1))

        # Diversity loss
        _, x_fake2 = G(img_ctrl, s_trg2)
        x_fake2 = x_fake2.detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # Cycle consistency loss
        s_org = S(img_ctrl)
        _, x_rec = G(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - img_ctrl))

        g_loss = (loss_adv
                  + args.lambda_sty * loss_sty
                  - lambda_ds * loss_ds
                  + args.lambda_cyc * loss_cyc)
        g_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(M.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(S.parameters(), args.grad_clip)
        G_opt.step()
        MS_opt.step()

        # Decay diversity loss weight
        if lambda_ds > 0:
            lambda_ds -= initial_lambda_ds / args.ds_iter
            lambda_ds = max(lambda_ds, 0.0)

        # ============== Logging ==============
        log_dict = {
            "loss/D_total": d_loss.item(),
            "loss/D_real": loss_real.item(),
            "loss/D_fake": loss_fake.item(),
            "loss/D_reg": loss_reg.item(),
            "loss/G_total": g_loss.item(),
            "loss/G_adv": loss_adv.item(),
            "loss/G_sty": loss_sty.item(),
            "loss/G_ds": loss_ds.item(),
            "loss/G_cyc": loss_cyc.item(),
            "misc/lambda_ds": lambda_ds,
        }
        for k, v in log_dict.items():
            logs[k].append(v)

        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix({
                "d": f"{d_loss.item():.4f}",
                "g": f"{g_loss.item():.4f}",
                "ds_w": f"{lambda_ds:.3f}",
            })

        # ============== Visualizations ==============
        if (step + 1) % args.vis_every == 0:
            log_visualizations(
                G, M, dataset, device, args, id2cpd, fp_matrix,
                x_fake[:, :3], img_ctrl, img_trt, cpd_id, step + 1, use_wandb,
            )

        # ============== FID (test split) ==============
        if args.fid_every > 0 and (step + 1) % args.fid_every == 0 and eval_dataset is not None:
            fid_global, fid_per_cpd = compute_fid(
                G, M, eval_dataset, device, args, id2cpd, fp_matrix,
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

        # ============== Checkpointing ==============
        is_save_step = (step + 1) % args.save_every == 0
        is_last_step = step == args.iterations - 1
        if is_save_step or is_last_step:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step + 1}.pt")
            save_checkpoint(
                ckpt_path, step=step + 1,
                G=G, D=D, M=M, S=S,
                G_opt=G_opt, D_opt=D_opt, MS_opt=MS_opt,
                logs=logs,
                extra={
                    "image_size": args.image_size,
                    "style_dim": args.style_dim,
                    "z_dimension": args.z_dimension,
                    "dim_in": args.dim_in,
                    "max_conv_dim": args.max_conv_dim,
                    "num_layers_mapping_net": args.num_layers_mapping_net,
                    "num_compounds": num_compounds,
                    "lambda_ds": lambda_ds,
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
