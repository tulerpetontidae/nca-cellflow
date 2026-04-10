"""
Petri Dish NCA training: pool-based training with homeostasis and transitions.

States persist in a GPU replay pool across iterations. The NCA learns to:
  - Maintain DMSO (control) cells that drift organically via z (homeostasis)
  - Transform cells when a drug is applied (DMSO → compound)
  - Recover cells when a drug is removed (compound → DMSO)

Uses relativistic GAN loss with zero-centered GP, same as train.py.
Logs to wandb project 'nca-petridish'.
"""

import os
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
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
    # torch-fidelity is required at runtime for InceptionV3 weights
    FrechetInceptionDistance(normalize=True)
    FID_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    FID_AVAILABLE = False
    print(f"[warn] FID disabled: {e}")

from nca_cellflow import LabeledImageBank, ReplayPool, EvalDataset
from nca_cellflow.models import LatentNCA, NCAStyleEncoder, Discriminator, PatchDiscriminator
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Utilities (same as train.py)
# ---------------------------------------------------------------------------

def zero_centered_gradient_penalty(samples, logits):
    if logits.dim() > 1:
        reduced = logits.reshape(logits.shape[0], -1).mean(dim=1).sum()
    else:
        reduced = logits.sum()
    (grad,) = torch.autograd.grad(outputs=reduced, inputs=samples, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def relativistic_d_loss(d_real, d_fake):
    if isinstance(d_real, dict):
        loss = 0.0
        for key in d_real:
            rel = d_real[key] - d_fake[key]
            loss = loss + F.softplus(-rel).mean()
        return loss
    rel = d_real - d_fake
    return F.softplus(-rel).mean()


def _to_float(d_out):
    if isinstance(d_out, dict):
        return {k: v.float() for k, v in d_out.items()}
    return d_out.float()


def multi_scale_gp(samples, d_out):
    if isinstance(d_out, dict):
        combined = 0.0
        for key in d_out:
            v = d_out[key]
            if v.dim() > 1:
                combined = combined + v.reshape(v.shape[0], -1).mean(dim=1).sum()
            else:
                combined = combined + v.sum()
        (grad,) = torch.autograd.grad(outputs=combined, inputs=samples, create_graph=True)
        return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)
    return zero_centered_gradient_penalty(samples, d_out)


def d_logit_mean(d_out):
    if isinstance(d_out, dict):
        return sum(v.mean().item() for v in d_out.values()) / len(d_out)
    return d_out.mean().item()


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
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, G, D, G_opt, D_opt, logs, extra=None,
                    G_ema=None, S=None, S_opt=None, pool=None):
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
    if S is not None:
        state["S_state"] = S.state_dict()
    if S_opt is not None:
        state["S_opt_state"] = S_opt.state_dict()
    if pool is not None:
        state["pool"] = pool.state_dict()
    torch.save(state, path)
    print(f"[ckpt] saved {path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def vis_petridish(G_eval, image_bank, id2label, device, step,
                  n_steps, img_channels, hidden_channels,
                  n_cells=4, n_rounds=4):
    """Generate 3 evaluation figures: homeostasis, transitions, recovery.

    Returns dict of {name: matplotlib figure}.
    """
    G_eval.eval()
    cpd_targets = {}
    for cpd_id, dose in image_bank.available_targets:
        if cpd_id not in cpd_targets:
            cpd_targets[cpd_id] = []
        cpd_targets[cpd_id].append(dose)
    # Pick middle dose per compound
    cpd_dose = {cid: sorted(ds)[len(ds) // 2] for cid, ds in cpd_targets.items()}
    cpd_ids = sorted(cpd_dose.keys())
    n_cpds = len(cpd_ids)

    def _prep(imgs):
        x = imgs.to(device)
        if hidden_channels > 0:
            pad = torch.zeros(x.shape[0], hidden_channels, x.shape[2], x.shape[3], device=device)
            x = torch.cat([x, pad], dim=1)
        return x

    def _rgb(state):
        return ((state[:, :img_channels].detach().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()

    # Sample control images (shared across all figures)
    ctrl_imgs = []
    for _ in range(n_cells):
        img, _ = image_bank.sample_one(0, dose=0.0)
        ctrl_imgs.append(img)
    ctrl_imgs = torch.stack(ctrl_imgs)
    ctrl_rgb = _rgb(ctrl_imgs.unsqueeze(0).squeeze(0) if ctrl_imgs.dim() == 3 else ctrl_imgs)
    # fix: ctrl_imgs is [N,C,H,W]
    ctrl_rgb = ((ctrl_imgs.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()

    # --- Figure 1: Homeostasis ---
    # rows = cells, cols = [input, round1..n_rounds]
    state = _prep(ctrl_imgs)
    homeo_snaps = [ctrl_rgb.copy()]
    cond = torch.zeros(n_cells, dtype=torch.long, device=device)
    dose_t = torch.zeros(n_cells, device=device)
    for _ in range(n_rounds):
        state = G_eval(state, cond, n_steps=n_steps, dose=dose_t)
        homeo_snaps.append(_rgb(state))

    fig_h, axes = plt.subplots(n_cells, n_rounds + 1, figsize=(2 * (n_rounds + 1), 2 * n_cells))
    if n_cells == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_cells):
        for r in range(n_rounds + 1):
            axes[i, r].imshow(homeo_snaps[r][i])
            axes[i, r].axis("off")
            if i == 0:
                axes[0, r].set_title("Input" if r == 0 else f"R{r}", fontsize=10)
    fig_h.suptitle(f"Homeostasis (step {step})", fontsize=14, y=1.01)
    fig_h.tight_layout()

    # --- Figure 2: Transitions (DMSO -> each compound, 4 rounds) ---
    # rows = compounds, cols = [input, round1..n_rounds, real]
    ncols_t = n_rounds + 2  # input + rounds + real
    fig_t, axes = plt.subplots(n_cpds, ncols_t * n_cells,
                                figsize=(1.5 * ncols_t * n_cells, 1.8 * n_cpds))
    if n_cpds == 1:
        axes = axes[np.newaxis, :]

    for row, cid in enumerate(cpd_ids):
        dose_val = cpd_dose[cid]
        cond = torch.full((n_cells,), cid, dtype=torch.long, device=device)
        dose_t = torch.full((n_cells,), dose_val, device=device)
        state = _prep(ctrl_imgs)
        snaps = [ctrl_rgb.copy()]
        for _ in range(n_rounds):
            state = G_eval(state, cond, n_steps=n_steps, dose=dose_t)
            snaps.append(_rgb(state))
        # Sample real treated
        real_imgs = []
        for _ in range(n_cells):
            img, _ = image_bank.sample_one(cid, dose=dose_val)
            real_imgs.append(img)
        real_rgb = ((torch.stack(real_imgs).clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
        snaps.append(real_rgb)

        for ci in range(n_cells):
            for c in range(ncols_t):
                col = ci * ncols_t + c
                axes[row, col].imshow(snaps[c][ci])
                axes[row, col].axis("off")
                if row == 0 and ci == 0:
                    if c == 0:
                        axes[0, col].set_title("In", fontsize=8)
                    elif c < ncols_t - 1:
                        axes[0, col].set_title(f"R{c}", fontsize=8)
                    else:
                        axes[0, col].set_title("Real", fontsize=8)
        axes[row, 0].set_ylabel(f"{id2label.get(cid, '?')}", fontsize=8,
                                 rotation=0, labelpad=35, va="center")
    fig_t.suptitle(f"Transitions: DMSO -> drug (step {step})", fontsize=14, y=1.01)
    fig_t.tight_layout()

    # --- Figure 3: Recovery (1 round drug -> 3 rounds DMSO) ---
    n_rec = min(n_rounds - 1, 3)
    ncols_r = 1 + 1 + n_rec  # input + drug + recovery rounds
    fig_r, axes = plt.subplots(n_cpds, ncols_r * n_cells,
                                figsize=(1.5 * ncols_r * n_cells, 1.8 * n_cpds))
    if n_cpds == 1:
        axes = axes[np.newaxis, :]

    for row, cid in enumerate(cpd_ids):
        dose_val = cpd_dose[cid]
        state = _prep(ctrl_imgs)
        snaps = [ctrl_rgb.copy()]

        # 1 round with drug
        cond = torch.full((n_cells,), cid, dtype=torch.long, device=device)
        dose_t = torch.full((n_cells,), dose_val, device=device)
        state = G_eval(state, cond, n_steps=n_steps, dose=dose_t)
        snaps.append(_rgb(state))

        # n_rec rounds recovery (DMSO)
        cond_dmso = torch.zeros(n_cells, dtype=torch.long, device=device)
        dose_dmso = torch.zeros(n_cells, device=device)
        for _ in range(n_rec):
            state = G_eval(state, cond_dmso, n_steps=n_steps, dose=dose_dmso)
            snaps.append(_rgb(state))

        for ci in range(n_cells):
            for c in range(ncols_r):
                col = ci * ncols_r + c
                axes[row, col].imshow(snaps[c][ci])
                axes[row, col].axis("off")
                if row == 0 and ci == 0:
                    if c == 0:
                        axes[0, col].set_title("In", fontsize=8)
                    elif c == 1:
                        axes[0, col].set_title("+Drug", fontsize=8)
                    else:
                        axes[0, col].set_title(f"Rec{c-1}", fontsize=8)
        axes[row, 0].set_ylabel(f"{id2label.get(cid, '?')}", fontsize=8,
                                 rotation=0, labelpad=35, va="center")
    fig_r.suptitle(f"Recovery: drug -> DMSO (step {step})", fontsize=14, y=1.01)
    fig_r.tight_layout()

    return {"vis/homeostasis": fig_h, "vis/transitions": fig_t, "vis/recovery": fig_r}


# ---------------------------------------------------------------------------
# FID Evaluation
# ---------------------------------------------------------------------------

_fid_metric = None


@torch.no_grad()
def compute_fid_global(G, eval_dataset, device, args, img_channels):
    """Compute global FID on test set. Returns float or None on failure."""
    global _fid_metric
    if not FID_AVAILABLE:
        return None
    try:
        if _fid_metric is None:
            _fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        _fid_metric.reset()

        loader = DataLoader(eval_dataset, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)
        G.eval()
        for img_ctrl, img_trt, cpd_id, dose in loader:
            img_ctrl = img_ctrl.to(device)
            img_trt = img_trt.to(device)
            cpd_id = cpd_id.to(device)
            dose = dose.to(device)

            if args.hidden_channels > 0:
                pad = torch.zeros(img_ctrl.shape[0], args.hidden_channels,
                                  img_ctrl.shape[2], img_ctrl.shape[3], device=device)
                nca_input = torch.cat([img_ctrl, pad], dim=1)
            else:
                nca_input = img_ctrl

            # Shift cpd_id by +1 (EvalDataset uses 0-indexed, petri dish uses 1-indexed with 0=DMSO)
            fake_full = G(nca_input, cpd_id + 1, n_steps=args.nca_steps, dose=dose)
            fake_rgb = fake_full[:, :img_channels].contiguous()

            real_01 = (img_trt[:, :img_channels].clamp(-1, 1) + 1) / 2
            real_01 = torch.floor(real_01 * 255).float() / 255.0
            fake_01 = (fake_rgb.clamp(-1, 1) + 1) / 2
            fake_01 = torch.floor(fake_01 * 255).float() / 255.0

            _fid_metric.update(real_01, real=True)
            _fid_metric.update(fake_01, real=False)

        fid = _fid_metric.compute().item()
        G.train()
        return fid
    except Exception as e:
        import traceback
        print(f"[warn] FID failed: {e}")
        traceback.print_exc()
        G.train()
        return None


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

def build_transition_table(image_bank, homeo_weight=3.0):
    """Build plate-aware transition tables.

    Each plate has its own set of available (compound, dose) targets.
    DMSO on plate X can only transition to compounds on plate X.

    Returns:
        plate_targets: {plate -> {(cpd_id, dose) -> [(target_cpd, target_dose), ...]}}
        plate_weights: {plate -> {(cpd_id, dose) -> np array of weights}}
    """
    plate_targets = {}
    plate_weights = {}

    # Build per-plate available targets
    plate_to_targets = defaultdict(list)
    for cpd_id, doses_dict in image_bank._index.items():
        if cpd_id == 0:
            continue  # skip DMSO
        for dose, plates_dict in doses_dict.items():
            for plate in plates_dict:
                plate_to_targets[plate].append((cpd_id, dose))

    # Also include plates that only have DMSO
    for plate in image_bank._index[0][0.0]:
        if plate not in plate_to_targets:
            plate_to_targets[plate] = []

    for plate, available in plate_to_targets.items():
        tbl_targets = {}
        tbl_weights = {}

        # DMSO transitions: homeo + forward to plate-local compounds
        dmso_tgts = [(0, 0.0)]  # homeostasis
        dmso_ws = [homeo_weight]
        for cpd_id, dose in available:
            dmso_tgts.append((cpd_id, dose))
            dmso_ws.append(1.0)
        w = np.array(dmso_ws)
        tbl_targets[(0, 0.0)] = dmso_tgts
        tbl_weights[(0, 0.0)] = w / w.sum()

        # Compound transitions: homeo + recovery to DMSO
        for cpd_id, dose in available:
            tgts = [(cpd_id, dose), (0, 0.0)]
            ws = np.array([homeo_weight, 1.0])
            tbl_targets[(cpd_id, dose)] = tgts
            tbl_weights[(cpd_id, dose)] = ws / ws.sum()

        plate_targets[plate] = tbl_targets
        plate_weights[plate] = tbl_weights

    return plate_targets, plate_weights


def sample_targets(current_labels, current_doses, plates,
                   plate_targets, plate_weights):
    """Sample target (cpd_id, dose) per item using plate-aware transition table.

    Returns:
        target_labels: [B] long tensor
        target_doses: [B] float tensor
        is_homeo: [B] bool tensor
    """
    B = current_labels.shape[0]
    target_labels = torch.zeros(B, dtype=torch.long, device=current_labels.device)
    target_doses = torch.zeros(B, dtype=torch.float32, device=current_labels.device)

    for i in range(B):
        plate = plates[i]
        key = (current_labels[i].item(), current_doses[i].item())

        tbl = plate_targets.get(plate, {})
        if key not in tbl:
            key = (0, 0.0)
        if key not in tbl:
            # Plate has no transition table (shouldn't happen)
            target_labels[i] = 0
            target_doses[i] = 0.0
            continue

        tgts = tbl[key]
        ws = plate_weights[plate][key]
        idx = np.random.choice(len(tgts), p=ws)
        target_labels[i] = tgts[idx][0]
        target_doses[i] = tgts[idx][1]

    is_homeo = (target_labels == current_labels) & (torch.abs(target_doses - current_doses) < 1e-6)
    return target_labels, target_doses, is_homeo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="Train Petri Dish NCA")

    # data
    p.add_argument("--metadata_csv", type=str, default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96)

    # model
    p.add_argument("--nca_hidden_dim", type=int, default=128)
    p.add_argument("--nca_cond_dim", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--dose_dim", type=int, default=8)
    p.add_argument("--hidden_channels", type=int, default=6)
    p.add_argument("--fire_rate", type=float, default=1.0)
    p.add_argument("--step_size", type=float, default=0.027)
    p.add_argument("--use_tanh", action="store_true", default=True)
    p.add_argument("--nca_steps", type=int, default=55)

    # discriminator
    p.add_argument("--d_type", type=str, default="global", choices=["global", "patch"])
    p.add_argument("--d_stages", type=int, default=3)
    p.add_argument("--d_base_channels", type=int, default=32)
    p.add_argument("--d_blocks", type=int, default=2)
    p.add_argument("--d_cardinality", type=int, default=4)
    p.add_argument("--d_expansion", type=int, default=2)
    p.add_argument("--d_kernel_size", type=int, default=3)
    p.add_argument("--d_embed_dim", type=int, default=32)

    # pool
    p.add_argument("--pool_size", type=int, default=2048)
    p.add_argument("--pool_recycle_iter", type=int, default=4)

    # transitions
    p.add_argument("--homeo_weight", type=float, default=3.0)
    p.add_argument("--homeo_perturb_steps", type=int, default=0,
                   help="Wrong-label NCA steps on homeostasis states (0=off, nca-clock default is off)")
    p.add_argument("--z_drift", type=float, default=0.0,
                   help="Z drift beta during training (0=off, drift is normally inference-only)")

    # checkpoint sampling
    p.add_argument("--ckpt_window", type=int, default=10,
                   help="Evaluate D at random step from [T-ckpt_window, T]")

    # losses
    p.add_argument("--gamma", type=float, default=1.0, help="Gradient penalty weight")
    p.add_argument("--intermediate_weight", type=float, default=1.0,
                   help="Weight for intermediate step regularization")
    p.add_argument("--style_weight", type=float, default=1.0,
                   help="Weight for style reconstruction loss")

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=40000)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_warmup_steps", type=int, default=1000)

    # misc
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--vis_every", type=int, default=500)
    p.add_argument("--fid_every", type=int, default=0,
                   help="Compute global FID every N steps (0 = disabled)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="nca-petridish")
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


def main():
    matplotlib.use("Agg")
    args = make_parser().parse_args()
    args = load_config_into_args(args)

    # Auto-derive checkpoint subdir from config name to avoid collisions
    if args.config and args.checkpoint_dir == "checkpoints":
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.checkpoint_dir = os.path.join("checkpoints", config_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    # ---- Image bank ----
    image_bank = LabeledImageBank(
        args.metadata_csv, args.image_dir, split="train", image_size=args.image_size,
    )
    num_compounds = image_bank.num_compounds  # excludes DMSO
    num_classes = image_bank.num_classes       # includes DMSO (index 0)
    print(f"Compounds: {num_compounds} + DMSO, {len(image_bank.available_targets)} (cpd, dose) targets")

    # Label map: cpd_id -> display name
    id2label = {0: "DMSO"}
    id2label.update(image_bank.id2cpd)

    # ---- Eval dataset for FID ----
    eval_dataset = None
    print(f"[fid] fid_every={args.fid_every}, FID_AVAILABLE={FID_AVAILABLE}")
    if args.fid_every > 0:
        eval_dataset = EvalDataset(args.metadata_csv, args.image_dir,
                                   split="test", image_size=args.image_size)
        print(f"[fid] Loaded eval dataset: {len(eval_dataset)} samples")
    else:
        print("[fid] FID disabled (fid_every=0)")

    # ---- Transition table ----
    plate_targets, plate_weights = build_transition_table(image_bank, args.homeo_weight)

    # ---- Models ----
    img_channels = 3
    channel_n = img_channels + args.hidden_channels

    # D gets extra null class for intermediate regularization
    use_intermediate = args.intermediate_weight > 0
    d_num_classes = num_classes + 1 if use_intermediate else num_classes
    null_class_id = num_classes  # index for null/any-cell class

    G = LatentNCA(
        channel_n=channel_n,
        z_dim=args.z_dim,
        hidden_dim=args.nca_hidden_dim,
        num_classes=num_classes,
        cond_dim=args.nca_cond_dim,
        fire_rate=args.fire_rate,
        step_size=args.step_size,
        use_tanh=args.use_tanh,
        dose_dim=args.dose_dim,
    ).to(device)

    d_widths = [args.d_base_channels] * (args.d_stages + 1)
    d_blocks = [args.d_blocks] * (args.d_stages + 1)
    d_cards = [args.d_cardinality] * (args.d_stages + 1)

    D_cls = PatchDiscriminator if args.d_type == "patch" else Discriminator
    D = D_cls(
        widths=d_widths, cardinalities=d_cards, blocks_per_stage=d_blocks,
        expansion=args.d_expansion, num_classes=d_num_classes,
        embed_dim=args.d_embed_dim, kernel_size=args.d_kernel_size,
        in_channels=img_channels,
    ).to(device)

    # Style encoder
    S = None
    S_opt = None
    use_style = args.style_weight > 0
    style_dim = args.nca_cond_dim + args.dose_dim + args.z_dim
    if use_style:
        S = NCAStyleEncoder(in_channels=img_channels, style_dim=style_dim).to(device)

    print(f"G params: {count_parameters(G)}")
    print(f"D params: {count_parameters(D)}")
    if S:
        print(f"S params: {count_parameters(S)}")

    # ---- EMA ----
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

    # ---- Optimizers ----
    G_opt = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    if S is not None:
        S_opt = torch.optim.Adam(S.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    # ---- Pool ----
    pool = ReplayPool(
        pool_size=args.pool_size, channel_n=channel_n,
        H=args.image_size, W=args.image_size,
        z_dim=args.z_dim, device=device,
    )
    print("Populating pool with control images...")
    pool.populate(image_bank, img_channels=img_channels, hidden_channels=args.hidden_channels)

    # ---- Resume ----
    logs = defaultdict(list)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        G_opt.load_state_dict(ckpt["G_opt_state"])
        D_opt.load_state_dict(ckpt["D_opt_state"])
        if G_ema is not None and "G_ema_state" in ckpt:
            G_ema.load_state_dict(ckpt["G_ema_state"])
        if S is not None and "S_state" in ckpt:
            S.load_state_dict(ckpt["S_state"])
        if S_opt is not None and "S_opt_state" in ckpt:
            S_opt.load_state_dict(ckpt["S_opt_state"])
        if "pool" in ckpt:
            pool.load_state_dict(ckpt["pool"], device=device)
        logs = defaultdict(list, ckpt.get("logs", {}))
        start_step = ckpt.get("step", 0) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    # ---- Wandb ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   config=vars(args), resume="allow")

    # ---- Checkpoint steps for D eval ----
    T = args.nca_steps
    ckpt_start = max(1, T - args.ckpt_window)
    ckpt_set = set(range(ckpt_start, T + 1))  # steps to save during forward

    # ---- Training loop ----
    pbar = tqdm(range(start_step, args.iterations), desc="Training",
                initial=start_step, total=args.iterations)

    for step in pbar:
        G.train()
        D.train()
        if S is not None:
            S.train()

        # ============ 1. Sample from pool ============
        (batch_idx, pool_states, pool_labels, pool_doses,
         pool_plates, pool_z) = pool.get_batch(args.batch_size)

        # ============ 2. Sample targets ============
        target_labels, target_doses, is_homeo = sample_targets(
            pool_labels, pool_doses, pool_plates,
            plate_targets, plate_weights)
        target_labels = target_labels.to(device)
        target_doses = target_doses.to(device)

        # ============ 3. Homeostasis perturbation ============
        # Run wrong-label NCA on homeostasis states to prevent identity mapping
        n_homeo = is_homeo.sum().item()
        if n_homeo > 0 and args.homeo_perturb_steps > 0:
            with torch.no_grad():
                homeo_mask = is_homeo.to(device)
                # Pick random wrong labels
                wrong_labels = (pool_labels[homeo_mask] +
                                torch.randint(1, num_classes, (n_homeo,), device=device)
                                ) % num_classes
                wrong_doses = torch.zeros(n_homeo, device=device)
                # Pick random wrong doses from available targets
                for i in range(n_homeo):
                    avail = image_bank.available_targets
                    rand_cpd, rand_dose = avail[np.random.randint(len(avail))]
                    wrong_labels[i] = rand_cpd
                    wrong_doses[i] = rand_dose

                wrong_z = pool_z[homeo_mask]
                for _ in range(args.homeo_perturb_steps):
                    cond_z, _ = G._prepare_cond(wrong_labels, wrong_z, wrong_doses)
                    pool_states[homeo_mask] = G.step(pool_states[homeo_mask], cond_z)

        # ============ 4. Z: keep fixed per slot (drift is inference-only) ============
        # z stays fixed for each pool slot — FiLM entanglement + style recon
        # force the model to use z. Brownian drift applied at eval time only.

        # ============ 5. Forward NCA with checkpoint sampling ============
        # Single no-grad unroll: produces ckpt frames AND (if enabled) the
        # intermediate snapshot from the SAME trajectory. Mirrors the pattern
        # in train.py where forward_with_intermediate returns both endpoints.
        if use_intermediate:
            t_inter_d = torch.randint(0, max(1, T // 2), (1,)).item()
            inter_step_d = t_inter_d + 1  # sample() saves frame at index t+1
            output_steps_d = ckpt_set | {inter_step_d}
        else:
            output_steps_d = ckpt_set

        with torch.no_grad():
            saved = G.sample(
                pool_states, target_labels, n_steps=T,
                output_steps=output_steps_d, z=pool_z, dose=target_doses,
            )
            sorted_d = sorted(output_steps_d)
            if use_intermediate:
                inter_full = saved[sorted_d.index(inter_step_d)]
                ckpt_frames = [saved[i] for i, s in enumerate(sorted_d) if s in ckpt_set]
            else:
                ckpt_frames = saved
            all_ckpts = torch.stack(ckpt_frames)  # [num_ckpts, B, C, H, W]
            final_state = all_ckpts[-1]  # last checkpoint = step T

            # Per-sample random checkpoint for D evaluation
            ckpt_indices = torch.randint(0, all_ckpts.shape[0], (args.batch_size,))
            batch_arange = torch.arange(args.batch_size)
            fake_batch = all_ckpts[ckpt_indices, batch_arange]  # [B, C, H, W]

        fake_img = fake_batch[:, :img_channels].contiguous()

        # ============ 6. Get real images for D ============
        real_img = image_bank.sample_batch(
            target_labels, target_doses, pool_plates).to(device)

        # Real for null class ("any cell"): half DMSO + half random treated.
        # Mirrors train.py:1134 (real_null = ctrl + trt). Reused by G phase below.
        real_null = None
        if use_intermediate:
            half_b = args.batch_size // 2
            null_dmso = image_bank.sample_batch(
                torch.zeros(half_b, dtype=torch.long, device=device),
                torch.zeros(half_b, device=device),
            ).to(device)
            trt_targets = image_bank.available_targets  # excludes DMSO
            trt_idx = np.random.randint(len(trt_targets), size=args.batch_size - half_b)
            null_trt_cpd = torch.tensor(
                [trt_targets[i][0] for i in trt_idx], dtype=torch.long, device=device)
            null_trt_dose = torch.tensor(
                [trt_targets[i][1] for i in trt_idx], device=device)
            null_trt = image_bank.sample_batch(null_trt_cpd, null_trt_dose).to(device)
            real_null = torch.cat([null_dmso, null_trt], dim=0)

        # ============ 7. Discriminator step ============
        set_requires_grad(G, False)
        set_requires_grad(D, True)
        D_opt.zero_grad(set_to_none=True)

        real_req = real_img.requires_grad_(True)
        fake_req = fake_img.detach().requires_grad_(True)

        d_real = _to_float(D(real_req, target_labels))
        d_fake = _to_float(D(fake_req, target_labels))

        adv_d = relativistic_d_loss(d_real, d_fake)
        gp_real = multi_scale_gp(real_req, d_real)
        gp_fake = multi_scale_gp(fake_req, d_fake)
        reg = 0.5 * args.gamma * (gp_real.mean() + gp_fake.mean())
        d_loss = adv_d + reg

        # Intermediate regularization for D (uses inter_full from the single unroll above)
        d_inter_loss = torch.tensor(0.0, device=device)
        if use_intermediate:
            inter_img = inter_full[:, :img_channels].contiguous()

            null_ids = torch.full((args.batch_size,), null_class_id,
                                  dtype=torch.long, device=device)
            real_null_req = real_null.requires_grad_(True)
            inter_req = inter_img.detach().requires_grad_(True)

            d_real_null = _to_float(D(real_null_req, null_ids))
            d_inter = _to_float(D(inter_req, null_ids))

            d_inter_loss = relativistic_d_loss(d_real_null, d_inter)
            gp_null_real = multi_scale_gp(real_null_req, d_real_null)
            gp_null_fake = multi_scale_gp(inter_req, d_inter)
            d_inter_loss = d_inter_loss + 0.5 * args.gamma * (gp_null_real.mean() + gp_null_fake.mean())
            d_loss = d_loss + args.intermediate_weight * d_inter_loss

        d_loss.backward()
        nn_utils.clip_grad_norm_(D.parameters(), args.grad_clip)
        D_opt.step()

        # ============ 8. Generator step ============
        set_requires_grad(G, True)
        set_requires_grad(D, False)
        G_opt.zero_grad(set_to_none=True)
        if S_opt is not None:
            S_opt.zero_grad(set_to_none=True)

        # Pre-compute cond_z for style loss target. This recomputes the same
        # value sample() uses internally; used only as a detached target.
        cond_z_g = None
        if use_style:
            cond_z_g, _ = G._prepare_cond(target_labels, pool_z, target_doses)

        # Single grad unroll: produces final state AND (if enabled) intermediate
        # snapshot from one trajectory. Halves BPTT depth vs prior 2-unroll layout.
        inter_full_g = None
        if use_intermediate:
            t_inter_g = torch.randint(0, max(1, T // 2), (1,)).item()
            inter_step_g = t_inter_g + 1
            output_steps_g = {T, inter_step_g}
            sorted_g = sorted(output_steps_g)
            samples_g = G.sample(
                pool_states, target_labels, n_steps=T,
                output_steps=output_steps_g, z=pool_z, dose=target_doses,
            )
            fake_full_g = samples_g[sorted_g.index(T)]
            inter_full_g = samples_g[sorted_g.index(inter_step_g)]
        else:
            fake_full_g = G(pool_states, target_labels, n_steps=T, z=pool_z, dose=target_doses)

        fake_img_g = fake_full_g[:, :img_channels].contiguous()

        d_fake_g = _to_float(D(fake_img_g, target_labels))
        d_real_g = _to_float(D(real_img.detach(), target_labels))
        g_loss = F.softplus(-(d_fake_g if not isinstance(d_fake_g, dict)
                              else sum(v.mean() for v in d_fake_g.values()) -
                              sum(v.mean() for v in d_real_g.values()))).mean() \
            if not isinstance(d_fake_g, dict) else \
            sum(F.softplus(-(d_fake_g[k] - d_real_g[k])).mean() for k in d_fake_g)
        total_g = g_loss

        # Intermediate regularization for G (reuses inter_full_g and real_null from above)
        g_inter_loss = torch.tensor(0.0, device=device)
        if use_intermediate:
            inter_img_g = inter_full_g[:, :img_channels].contiguous()
            null_ids_g = torch.full((args.batch_size,), null_class_id,
                                    dtype=torch.long, device=device)
            d_inter_g = _to_float(D(inter_img_g, null_ids_g))
            d_real_null_g = _to_float(D(real_null.detach(), null_ids_g))

            if isinstance(d_inter_g, dict):
                g_inter_loss = sum(F.softplus(-(d_inter_g[k] - d_real_null_g[k])).mean() for k in d_inter_g)
            else:
                g_inter_loss = F.softplus(-(d_inter_g - d_real_null_g)).mean()
            total_g = total_g + args.intermediate_weight * g_inter_loss

        # Style reconstruction loss
        style_loss = torch.tensor(0.0, device=device)
        if use_style and S is not None:
            style_hat = S(fake_img_g)
            style_loss = F.l1_loss(style_hat, cond_z_g.detach())
            total_g = total_g + args.style_weight * style_loss

        total_g.backward()
        nn_utils.clip_grad_norm_(G.parameters(), args.grad_clip)
        G_opt.step()
        if S_opt is not None:
            S_opt.step()

        # ============ 9. EMA update ============
        if G_ema is not None:
            G_ema.update_parameters(G)

        # ============ 10. Update pool ============
        with torch.no_grad():
            # Use the final state from the no-grad forward (step 5)
            pool.update(
                batch_idx, final_state, target_labels, target_doses,
                pool_plates, pool_z,
            )

        # ============ 11. Recycle old states ============
        n_recycled = pool.recycle(
            args.pool_recycle_iter, image_bank,
            img_channels=img_channels, hidden_channels=args.hidden_channels,
        )

        # ============ Logging (single wandb.log per step) ============
        wandb_dict = {}

        if step % args.log_every == 0:
            log_dict = {
                "loss/D_total": d_loss.item(),
                "loss/D_adv": adv_d.item(),
                "loss/D_reg": reg.item(),
                "loss/G": g_loss.item(),
                "loss/G_total": total_g.item(),
                "logits/D_real_mean": d_logit_mean(d_real),
                "logits/D_fake_mean": d_logit_mean(d_fake),
                "pool/recycled": n_recycled,
                "pool/mean_iter": pool.iters.float().mean().item(),
                "pool/homeo_frac": is_homeo.float().mean().item(),
            }
            if use_intermediate:
                log_dict["loss/D_inter"] = d_inter_loss.item()
                log_dict["loss/G_inter"] = g_inter_loss.item()
            if use_style:
                log_dict["loss/style"] = style_loss.item()

            for k, v in log_dict.items():
                logs[k].append(v)

            pbar.set_postfix({
                "D": f"{adv_d.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "rec": n_recycled,
            })
            wandb_dict.update(log_dict)

        # ============ Visualization ============
        if step % args.vis_every == 0 and step > 0:
            vis_model = G_ema.module if G_ema is not None else G
            vis_figs = vis_petridish(
                vis_model, image_bank, id2label, device, step,
                n_steps=T, img_channels=img_channels,
                hidden_channels=args.hidden_channels,
            )
            for k, fig in vis_figs.items():
                wandb_dict[k] = wandb.Image(fig) if use_wandb else None
            plt.close("all")
            G.train()

        # ============ FID ============
        if args.fid_every > 0 and step > 0 and step % args.fid_every == 0 and eval_dataset is not None:
            fid_model = G_ema.module if G_ema is not None else G
            fid_val = compute_fid_global(fid_model, eval_dataset, device, args, img_channels)
            if fid_val is not None:
                wandb_dict["fid/global"] = fid_val
                print(f"[FID] global={fid_val:.1f}")

        # Single wandb.log call per step
        if use_wandb and wandb_dict:
            wandb.log({k: v for k, v in wandb_dict.items() if v is not None}, step=step)

        # ============ Save checkpoint ============
        if step % args.save_every == 0 and step > 0:
            extra = {k: getattr(args, k) for k in vars(args)}
            path = os.path.join(args.checkpoint_dir, f"petridish_step_{step}.pt")
            save_checkpoint(
                path, step, G, D, G_opt, D_opt, logs, extra=extra,
                G_ema=G_ema, S=S, S_opt=S_opt, pool=pool,
            )

    # Final save
    extra = {k: getattr(args, k) for k in vars(args)}
    path = os.path.join(args.checkpoint_dir, f"petridish_final.pt")
    save_checkpoint(path, args.iterations, G, D, G_opt, D_opt, logs, extra=extra,
                    G_ema=G_ema, S=S, S_opt=S_opt, pool=pool)
    print("Training complete.")


if __name__ == "__main__":
    main()
