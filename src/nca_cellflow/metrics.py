"""Evaluation metrics shared across training scripts.

`compute_texture_stats` was originally defined inline in `scripts/train.py`.
Moved here so `scripts/train_impa.py` and `scripts/train_cellflux.py` can report
the same texture quality numbers without code duplication, making the three
families of runs directly comparable on the same wandb keys.

Metrics computed (per call, on one paired batch of real + fake):
    texture/spectrum_ratio_high  — P_fake/P_real for top-quarter radial freqs (<1 = blurry)
    texture/spectrum_ratio_low   — P_fake/P_real for bottom-quarter radial freqs
    texture/hf_energy_real       — high-frequency energy fraction of real batch
    texture/hf_energy_fake       — high-frequency energy fraction of fake batch
    texture/hf_energy_ratio      — hf_fake / hf_real (<1 = missing high-freq content)
    texture/laplacian_var_real   — mean Laplacian variance of real (sharpness proxy)
    texture/laplacian_var_fake   — mean Laplacian variance of fake
    texture/laplacian_ratio      — lap_fake / lap_real (<1 = blurry)

Input images must be paired real/fake batches of the same shape [B, C, H, W].
No per-compound breakdown — this is a cheap single-batch statistic intended to
run alongside FID eval without adding a full dataset pass. The training scripts
call it on the last training batch to avoid an extra forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _radial_profile(mag: torch.Tensor) -> torch.Tensor:
    """Azimuthally average a 2D magnitude tensor [H, W] → [max_radius]."""
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    y, x = torch.meshgrid(torch.arange(H, device=mag.device),
                          torch.arange(W, device=mag.device), indexing="ij")
    r = torch.sqrt((y - cy).float() ** 2 + (x - cx).float() ** 2).long().clamp(max=max_r - 1)
    profile = torch.zeros(max_r, device=mag.device)
    counts = torch.zeros(max_r, device=mag.device)
    profile.scatter_add_(0, r.reshape(-1), mag.reshape(-1))
    counts.scatter_add_(0, r.reshape(-1), torch.ones_like(mag.reshape(-1)))
    return profile / counts.clamp(min=1)


@torch.no_grad()
def compute_texture_stats(real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> dict:
    """
    Compute texture quality metrics comparing real and fake image batches.

    Args:
        real_imgs: [B, C, H, W] real images (any range).
        fake_imgs: [B, C, H, W] fake images, same shape as real_imgs.

    Returns:
        Dict with `texture/*` keys (see module docstring for the full list).
    """
    device = real_imgs.device

    # --- Radial power spectrum ---
    # Use full 2D FFT, shift DC to center, average across batch & channels
    def mean_spectrum(imgs):
        f = torch.fft.fft2(imgs)
        f = torch.fft.fftshift(f, dim=(-2, -1))
        mag = torch.log1p(f.abs())  # log-magnitude
        return mag.mean(dim=(0, 1))  # average over batch and channels → [H, W]

    spec_real = mean_spectrum(real_imgs)
    spec_fake = mean_spectrum(fake_imgs)
    profile_real = _radial_profile(spec_real)
    profile_fake = _radial_profile(spec_fake)

    max_r = profile_real.shape[0]
    quarter = max_r // 4
    ratio = profile_fake / profile_real.clamp(min=1e-6)
    spectrum_ratio_low = ratio[:quarter].mean().item()
    spectrum_ratio_high = ratio[-quarter:].mean().item()

    # --- High-frequency energy fraction ---
    def hf_energy_frac(imgs):
        f = torch.fft.rfft2(imgs)
        power = (f.abs() ** 2).mean(dim=(0, 1))  # [H, W//2+1]
        H = power.shape[0]
        total = power.sum()
        # High freq = outer half of frequency space
        hf_mask = torch.zeros_like(power, dtype=torch.bool)
        cy = H // 2
        cx = power.shape[1]  # rfft: only positive freqs
        y = torch.arange(H, device=device)
        x = torch.arange(cx, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Wrap y freqs: shift so DC is at center
        yy_shift = torch.where(yy > cy, H - yy, yy)
        r = torch.sqrt(yy_shift.float() ** 2 + xx.float() ** 2)
        cutoff = min(cy, cx) / 2
        hf_mask = r > cutoff
        return (power[hf_mask].sum() / total.clamp(min=1e-8)).item()

    hf_real = hf_energy_frac(real_imgs)
    hf_fake = hf_energy_frac(fake_imgs)

    # --- Laplacian variance (sharpness) ---
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
    ).reshape(1, 1, 3, 3).expand(real_imgs.shape[1], -1, -1, -1)

    def lap_var(imgs):
        lap = F.conv2d(imgs, laplacian_kernel, padding=1, groups=imgs.shape[1])
        # Per-image variance, then mean across batch
        return lap.var(dim=(-2, -1)).mean().item()

    lap_real = lap_var(real_imgs)
    lap_fake = lap_var(fake_imgs)

    return {
        "texture/spectrum_ratio_high": spectrum_ratio_high,
        "texture/spectrum_ratio_low": spectrum_ratio_low,
        "texture/hf_energy_real": hf_real,
        "texture/hf_energy_fake": hf_fake,
        "texture/hf_energy_ratio": hf_fake / max(hf_real, 1e-8),
        "texture/laplacian_var_real": lap_real,
        "texture/laplacian_var_fake": lap_fake,
        "texture/laplacian_ratio": lap_fake / max(lap_real, 1e-8),
    }
