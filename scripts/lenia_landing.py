"""
Continuous cellular automata — landing slide visualizations.

Four rules share the same rendering/encoding pipeline:

    lenia        — classic Lenia, permissive Hydrogeminium soup
    lenia_chaos  — Lenia with tight-sigma, edge-of-chaos growth
    flow_lenia   — mass-conserving Flow Lenia (semi-Lagrangian advection)
    gs           — Gray-Scott reaction-diffusion (3 independent chemistries)

Usage:
    python scripts/lenia_landing.py --rule flow_lenia
    python scripts/lenia_landing.py --rule gs --steps 600   # quick test
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Lenia kernel and growth helpers
# ---------------------------------------------------------------------------


def exp_core(u, alpha=4.0):
    """Bert Chan's exponential bell, peaks at u=0.5 with value 1."""
    eps = 1e-9
    u_safe = u.clamp(eps, 1.0 - eps)
    val = torch.exp(alpha - alpha / (4.0 * u_safe * (1.0 - u_safe)))
    inside = (u > 0) & (u < 1)
    return torch.where(inside, val, torch.zeros_like(val))


def make_kernel(R, beta, device):
    """Multi-peak concentric-ring kernel of radius R. Returns (2R+1, 2R+1), sum=1."""
    size = 2 * R + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - R
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    D = torch.sqrt(x * x + y * y) / R
    mask = (D < 1).float()

    nb = len(beta)
    Br = nb * D
    i = Br.long().clamp(0, nb - 1)
    local = Br - i.float()

    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    amp = beta_t[i]

    K = amp * exp_core(local) * mask
    K = K / K.sum()
    return K


def prepare_kernel_fft(K, H, W):
    """Zero-pad K to (H, W) and shift so its center lands at (0, 0), then FFT."""
    kH, kW = K.shape
    padded = torch.zeros(H, W, device=K.device, dtype=K.dtype)
    padded[:kH, :kW] = K
    padded = torch.roll(padded, shifts=(-(kH // 2), -(kW // 2)), dims=(0, 1))
    return torch.fft.rfft2(padded)


def build_lenia_ctx(species, H, W, device):
    """Return (K_fft, mu_g, sigma_g) for a list of (R, beta, mu, sigma) species."""
    C = len(species)
    K_fft_list = []
    mu_g = torch.zeros(C, 1, 1, device=device)
    sigma_g = torch.zeros(C, 1, 1, device=device)
    for c, (R, beta, m, s) in enumerate(species):
        Kc = make_kernel(R, beta, device)
        K_fft_list.append(prepare_kernel_fft(Kc, H, W))
        mu_g[c, 0, 0] = m
        sigma_g[c, 0, 0] = s
    return torch.stack(K_fft_list, dim=0), mu_g, sigma_g


# ---------------------------------------------------------------------------
# Flow Lenia: periodic semi-Lagrangian advection
# ---------------------------------------------------------------------------


def make_advect_grid(H, W, device):
    """Base (y, x) pixel coordinate grid, shape (H, W) each."""
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return yy, xx


def semi_lagrangian_advect(A, vy, vx, yy, xx):
    """Backward-trace advection with periodic boundaries.

    A: (C, H, W), vy/vx: (C, H, W) velocities in pixels per step.
    Returns new A of the same shape.
    """
    C, H, W = A.shape
    src_y = (yy.unsqueeze(0) - vy) % H
    src_x = (xx.unsqueeze(0) - vx) % W
    grid_y = 2.0 * src_y / (H - 1) - 1.0
    grid_x = 2.0 * src_x / (W - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (C, H, W, 2)
    A_in = A.unsqueeze(1)  # (C, 1, H, W)
    warped = F.grid_sample(
        A_in, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return warped.squeeze(1)


# ---------------------------------------------------------------------------
# Gray-Scott reaction-diffusion
# ---------------------------------------------------------------------------


def make_laplacian_kernel(device):
    """9-point Laplacian stencil (unit spatial step)."""
    k = torch.tensor(
        [[0.05, 0.20, 0.05],
         [0.20, -1.00, 0.20],
         [0.05, 0.20, 0.05]],
        device=device, dtype=torch.float32,
    )
    return k.view(1, 1, 3, 3)


def laplacian(field, kernel):
    """Per-channel 2D Laplacian with periodic boundaries. field: (C, H, W)."""
    C = field.shape[0]
    x = field.unsqueeze(1)  # (C, 1, H, W)
    x = F.pad(x, (1, 1, 1, 1), mode="circular")
    return F.conv2d(x, kernel).squeeze(1)


def init_gray_scott(C, H, W, device, seed):
    """U≈1, V≈0 background with random seed patches where V is activated."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.ones(C, H, W, dtype=torch.float32)
    V = torch.zeros(C, H, W, dtype=torch.float32)
    n_patches = 120
    for c in range(C):
        for _ in range(n_patches):
            cy = int(torch.randint(0, H, (1,), generator=g).item())
            cx = int(torch.randint(0, W, (1,), generator=g).item())
            r = int(torch.randint(6, 18, (1,), generator=g).item())
            y0, y1 = max(0, cy - r), min(H, cy + r)
            x0, x1 = max(0, cx - r), min(W, cx + r)
            V[c, y0:y1, x0:x1] = 0.5
            U[c, y0:y1, x0:x1] = 0.25
    # jitter to break symmetry
    V = V + 0.02 * torch.rand(C, H, W, generator=g)
    U = U + 0.02 * torch.rand(C, H, W, generator=g)
    return U.to(device), V.clamp(0, 1).to(device)


# ---------------------------------------------------------------------------
# Initialization and rendering
# ---------------------------------------------------------------------------


def init_soup(n_channels, H, W, scale, device, seed=0,
              value_range=(0.0, 1.0), sparsity=0.0, env_scale=None):
    """Low-frequency smoothed noise: random grid upsampled to (H, W).

    value_range rescales the output so the soup sits closer to a target
    band — useful when growth sigma is narrow and full-range noise would
    put most cells outside the survival band.

    sparsity (0..1) creates actual empty regions using a SECOND noise
    tensor at a coarser scale (env_scale) as a spatial envelope. Inside
    the envelope patches, the fine-scale value noise is preserved in the
    full value_range so regions still hit the growth band naturally;
    outside the envelope everything is zero. At sparsity=0 the envelope
    is disabled; at 0.6 roughly 40% of the canvas has initial structure.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    # ---- fine-scale value noise ----
    sh, sw = max(H // scale, 2), max(W // scale, 2)
    val = torch.rand(n_channels, 1, sh, sw, generator=g)
    val_full = F.interpolate(val, size=(H, W), mode="bilinear", align_corners=False)
    lo, hi = value_range
    val_full = lo + (hi - lo) * val_full
    val_full = val_full[:, 0]  # (C, H, W)

    if sparsity > 0.0:
        # ---- coarse-scale spatial envelope ----
        # env_scale controls the size of surviving patches; must be
        # comparable to (or larger than) the NCA kernel so patches have
        # enough internal support to hit the growth band after convolution.
        es = env_scale if env_scale is not None else max(scale * 5, 48)
        esh, esw = max(H // es, 2), max(W // es, 2)
        env = torch.rand(n_channels, 1, esh, esw, generator=g)
        env_full = F.interpolate(env, size=(H, W), mode="bilinear", align_corners=False)
        env_full = env_full[:, 0]  # (C, H, W) in [0, 1]
        # Narrow fade zone: most cells are either 0 or ~1, so surviving
        # patches keep their full fine-scale value noise (same statistics
        # as the dense baseline). Wide fade dilutes values below the
        # growth band and kills everything.
        fade = 0.08
        env_mask = ((env_full - sparsity) / fade).clamp(0.0, 1.0)
        val_full = val_full * env_mask

    return val_full.to(device)


def colorize(field, colors, scale=1.0, gamma=0.85, mode="softmax", sharp=2.5):
    """Map a (C, H, W) field to an RGB frame.

    mode:
        sum     — straight additive; overlap saturates to white.
        max     — dominant channel wins outright; hard mono-color regions.
        mix     — intensity-weighted color average; overlap blends toward a
                  new color, intensity comes from the summed mass (capped).
        softmax — softmax-weighted color average with temperature `sharp`.
                  Low sharp (~2) ≈ mix, high sharp (~8) ≈ max. Intensity
                  comes from the summed mass so overlap is visibly brighter.
    """
    x = (field * scale).clamp(0, 1)  # (C, H, W)
    if mode == "sum":
        rgb = torch.einsum("chw,cr->hwr", x, colors)
    elif mode == "max":
        intensity, idx = x.max(dim=0)
        rgb = colors[idx] * intensity.unsqueeze(-1)
    elif mode == "mix":
        weights = x / (x.sum(dim=0, keepdim=True) + 1e-6)  # (C, H, W)
        avg_color = torch.einsum("chw,cr->hwr", weights, colors)
        intensity = x.sum(dim=0).clamp(0, 1)
        rgb = avg_color * intensity.unsqueeze(-1)
    elif mode == "softmax":
        w = torch.softmax(x * sharp, dim=0)  # (C, H, W)
        intensity = x.sum(dim=0).clamp(0, 1)  # (H, W) — summed brightness
        rgb = torch.einsum("chw,cr->hwr", w, colors) * intensity.unsqueeze(-1)
    else:
        raise ValueError(f"unknown blend mode: {mode}")
    rgb = rgb.clamp(0, 1).pow(gamma)
    return (rgb * 255).byte().contiguous().cpu().numpy()


def open_ffmpeg_writer(path, w, h, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "4.0",
        "-crf", "18",
        "-preset", "medium",
        "-tune", "animation",
        "-bf", "0",
        "-g", str(fps * 2),
        "-keyint_min", str(fps),
        "-r", str(fps),
        "-vsync", "cfr",
        "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------


LENIA_SPECIES = {
    # Permissive soup: wider sigma, mu slightly off stable point.
    "lenia": [
        (60, [1.0, 5.0 / 12.0, 2.0 / 3.0], 0.160, 0.0290),
        (56, [1.0, 2.0 / 3.0],              0.162, 0.0305),
        (64, [3.0 / 4.0, 1.0, 2.0 / 3.0],   0.158, 0.0285),
    ],
    # Edge-of-chaos: very tight sigma forces the growth band into a narrow
    # slice, so mass that drifts out of the band dies hard. Asymmetric
    # multi-peak beta breaks radial symmetry and lets traveling structures
    # form. Soup is initialized in a narrower band (see init_range_chaos).
    "lenia_chaos": [
        (60, [1.0, 0.30, 0.70], 0.156, 0.0120),
        (56, [0.70, 1.0, 0.30], 0.160, 0.0125),
        (64, [0.50, 1.0, 0.80], 0.152, 0.0115),
    ],
    # Flow Lenia v1: medium Hydrogeminium creatures (R~30), stable growth
    # band — the advection + curl rotation is what makes them move.
    "flow_lenia": [
        (32, [1.0, 5.0 / 12.0, 2.0 / 3.0], 0.156, 0.0224),
        (28, [1.0, 2.0 / 3.0],              0.158, 0.0230),
        (36, [1.0, 5.0 / 12.0, 2.0 / 3.0], 0.154, 0.0220),
    ],
    # Flow Lenia v2 (alt): smaller creatures (R~22), single-peak or
    # skewed kernels, a lower-mu growth band. More eel/filament-like
    # structures rather than the multi-ring blobs of v1.
    "flow_lenia_alt": [
        (22, [1.0],          0.148, 0.0180),
        (26, [0.6, 1.0],     0.150, 0.0195),
        (24, [1.0, 0.4],     0.144, 0.0175),
    ],
}


# Gray-Scott (F, k) regimes per channel — each channel evolves a distinct
# chemistry, producing three visually different pattern families that overlap.
GS_PARAMS = [
    (0.0367, 0.0649),  # "mitosis" — dividing spots
    (0.0620, 0.0609),  # "u-skate / solitons" — moving spots and worms
    (0.0220, 0.0510),  # "spirals / holes" — rotating waves
]
GS_DU = 0.16
GS_DV = 0.08


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=str, default="lenia",
                        choices=["lenia", "lenia_chaos", "flow_lenia",
                                 "flow_lenia_alt", "gs"],
                        help="which continuous CA rule to simulate")
    parser.add_argument("--out", type=str, default=None,
                        help="output path (default: figures/landing_{rule}.mov)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.30,
                        help="Lenia-family integration step")
    parser.add_argument("--dt_gs", type=float, default=1.0,
                        help="Gray-Scott integration step")
    parser.add_argument("--sim_per_frame", type=int, default=6,
                        help="sim updates per rendered frame (>1 = more dynamic motion)")
    parser.add_argument("--soup_scale", type=int, default=None,
                        help="downsample factor for noise soup init (larger = chunkier "
                             "blobs). Default: 12 for flow rules, 8 elsewhere.")
    parser.add_argument("--sparsity", type=float, default=None,
                        help="fraction of soup to zero out (sparse init with empty "
                             "regions between patches). Default: 0.6 for flow rules, "
                             "0.0 elsewhere.")
    parser.add_argument("--cross_weight", type=float, default=0.2,
                        help="mix fraction of mean-channel U into each channel's U (Lenia family)")
    parser.add_argument("--flow_scale", type=float, default=80.0,
                        help="Flow Lenia velocity multiplier (pixels per unit gradient)")
    parser.add_argument("--rot_weight", type=float, default=0.4,
                        help="Flow Lenia rotational component (curl/tangential) weight")
    parser.add_argument("--flow_growth", type=float, default=0.03,
                        help="Flow Lenia residual growth coefficient "
                             "(0 = pure flow, ~0.03 keeps complexity alive, "
                             ">0.10 causes flicker)")
    parser.add_argument("--u_scale", type=float, default=1.8,
                        help="display scale for the colorized field")
    parser.add_argument("--blend", type=str, default="softmax",
                        choices=["sum", "max", "mix", "softmax"],
                        help="how to composite the 3 species into RGB")
    parser.add_argument("--blend_sharp", type=float, default=2.5,
                        help="softmax temperature for blend=softmax (low=mix, high=max)")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    W, H = args.width, args.height
    device = args.device
    rule = args.rule

    if args.out is None:
        args.out = f"figures/landing_{rule}.mov"

    print(f"device={device} canvas={W}x{H} rule={rule} steps={args.steps}")

    # Palette: teal, amber, rose (shared across all rules)
    colors = torch.tensor([
        [0.20, 0.95, 0.85],
        [1.00, 0.75, 0.30],
        [1.00, 0.35, 0.55],
    ], device=device)
    C = 3

    # ------ rule-specific setup ------
    is_flow = rule in ("flow_lenia", "flow_lenia_alt")
    if rule in ("lenia", "lenia_chaos") or is_flow:
        species = LENIA_SPECIES[rule]
        assert len(species) == C
        K_fft, mu_g, sigma_g = build_lenia_ctx(species, H, W, device)
        # Soup init range per rule:
        #   - chaos regime: start inside the narrow survival band or the
        #     tight sigma kills everything before dynamics develop.
        #   - flow family: start sparse so patterns have room to move and
        #     don't overlap into a bright mush.
        #   - default lenia: full range, dense soup.
        if rule == "lenia_chaos":
            soup_range = (0.05, 0.30)
        elif is_flow:
            soup_range = (0.0, 0.55)
        else:
            soup_range = (0.0, 1.0)
        # per-rule defaults (CLI overrides if set)
        soup_scale = args.soup_scale if args.soup_scale is not None else 8
        sparsity = args.sparsity if args.sparsity is not None else (0.55 if is_flow else 0.0)
        A = init_soup(C, H, W, soup_scale, device, seed=args.seed,
                      value_range=soup_range, sparsity=sparsity)
        if is_flow:
            yy, xx = make_advect_grid(H, W, device)
        else:
            yy = xx = None
        gs_state = None
    elif rule == "gs":
        lap_k = make_laplacian_kernel(device)
        U_gs, V_gs = init_gray_scott(C, H, W, device, seed=args.seed)
        gs_F = torch.tensor([p[0] for p in GS_PARAMS], device=device).view(C, 1, 1)
        gs_k = torch.tensor([p[1] for p in GS_PARAMS], device=device).view(C, 1, 1)
        gs_state = (U_gs, V_gs)
        A = K_fft = mu_g = sigma_g = yy = xx = None
    else:
        raise ValueError(f"unknown rule {rule}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = open_ffmpeg_writer(out_path, W, H, args.fps)

    t_start = time.time()
    try:
        for step in range(args.steps):
            for _ in range(args.sim_per_frame):
                if rule in ("lenia", "lenia_chaos"):
                    A_fft = torch.fft.rfft2(A)
                    U = torch.fft.irfft2(A_fft * K_fft, s=(H, W))
                    if args.cross_weight > 0:
                        U_mean = U.mean(dim=0, keepdim=True)
                        U = (1.0 - args.cross_weight) * U + args.cross_weight * U_mean
                    G = 2.0 * torch.exp(-((U - mu_g) ** 2) / (2 * sigma_g ** 2)) - 1.0
                    A = (A + args.dt * G).clamp(0, 1)

                elif is_flow:
                    A_fft = torch.fft.rfft2(A)
                    U = torch.fft.irfft2(A_fft * K_fft, s=(H, W))
                    # Velocity field: gradient of perception (attractive
                    # toward kernel-weighted centers of mass) PLUS a
                    # tangential / curl-like component (90°-rotated
                    # gradient) that sends mass spinning around the
                    # perception contours instead of collapsing radially.
                    # The curl component is what breaks the "round blob"
                    # symmetry of pure ∇U flow.
                    dU_dy, dU_dx = torch.gradient(U, dim=(-2, -1))
                    vy = dU_dy - args.rot_weight * dU_dx
                    vx = dU_dx + args.rot_weight * dU_dy
                    vy = vy * args.dt * args.flow_scale
                    vx = vx * args.dt * args.flow_scale
                    A = semi_lagrangian_advect(A, vy, vx, yy, xx)
                    # Optional residual growth (default 0 = pure flow). The
                    # additive growth term is what previously caused
                    # flickering by fighting the advection; keeping it at
                    # zero gives clean, mass-conserving dynamics.
                    if args.flow_growth > 0.0:
                        G = 2.0 * torch.exp(-((U - mu_g) ** 2) / (2 * sigma_g ** 2)) - 1.0
                        A = A + args.flow_growth * args.dt * G
                    A = A.clamp(0, 1)

                elif rule == "gs":
                    U_gs, V_gs = gs_state
                    Lu = laplacian(U_gs, lap_k)
                    Lv = laplacian(V_gs, lap_k)
                    uvv = U_gs * V_gs * V_gs
                    dU = GS_DU * Lu - uvv + gs_F * (1.0 - U_gs)
                    dV = GS_DV * Lv + uvv - (gs_F + gs_k) * V_gs
                    U_gs = (U_gs + args.dt_gs * dU).clamp(0, 1)
                    V_gs = (V_gs + args.dt_gs * dV).clamp(0, 1)
                    gs_state = (U_gs, V_gs)

            if rule == "gs":
                # V is the "active" chemical; scale it up since peaks ~0.4
                frame = colorize(gs_state[1], colors, scale=2.2,
                                 mode=args.blend, sharp=args.blend_sharp)
            else:
                frame = colorize(A, colors, scale=args.u_scale,
                                 mode=args.blend, sharp=args.blend_sharp)
            proc.stdin.write(frame.tobytes())

            if step % 60 == 0:
                elapsed = time.time() - t_start
                rate = (step + 1) / max(elapsed, 1e-6)
                eta = (args.steps - step - 1) / max(rate, 1e-6)
                print(f"step {step:5d}/{args.steps}  {rate:5.1f} steps/s  eta {eta:6.1f}s")
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait()

    print(f"wrote {out_path}  ({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
