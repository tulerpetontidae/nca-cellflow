"""
Physarum slime mold simulation -- landing slide visualization.

Three species of agents on a 1920x1080 canvas, each depositing on its own
trail channel and being slightly repelled by the other species' trails.
Output is rendered to mp4 with custom species colors.

Defaults: ~900k agents, 60s @ 30fps. On a 3090 expect ~1-2 minutes runtime.

Usage:
    python scripts/physarum_landing.py --out figures/physarum_landing.mp4
    python scripts/physarum_landing.py --steps 2700 --n_per_species 400000
"""

import argparse
import math
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


def init_agents(n, w, h, device, mode="ring"):
    if mode == "ring":
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.42 * torch.sqrt(torch.rand(n, device=device))
        theta = torch.rand(n, device=device) * 2 * math.pi
        x = cx + radius * torch.cos(theta)
        y = cy + radius * torch.sin(theta)
        # Face roughly inward with jitter so swarms collapse and bloom
        angle = (theta + math.pi + (torch.rand(n, device=device) - 0.5) * 0.6) % (2 * math.pi)
    else:
        x = torch.rand(n, device=device) * w
        y = torch.rand(n, device=device) * h
        angle = torch.rand(n, device=device) * 2 * math.pi
    return torch.stack([x, y], dim=1), angle


def sense(field, pos, angle, offset, sensor_dist, w, h):
    sx = (pos[:, 0] + sensor_dist * torch.cos(angle + offset)).long() % w
    sy = (pos[:, 1] + sensor_dist * torch.sin(angle + offset)).long() % h
    return field[sy, sx]


def step_species(species, sense_field, w, h):
    pos = species["pos"]
    angle = species["angle"]
    sd = species["sensor_dist"]
    sa = species["sensor_angle"]
    ra = species["rot_angle"]
    ms = species["move_speed"]

    f = sense(sense_field, pos, angle, 0.0, sd, w, h)
    l = sense(sense_field, pos, angle, sa, sd, w, h)
    r = sense(sense_field, pos, angle, -sa, sd, w, h)

    left_max = (l > f) & (l > r)
    right_max = (r > f) & (r > l)
    f_max = (f >= l) & (f >= r)
    equal = ~(left_max | right_max | f_max)

    rand_turn = (torch.rand_like(angle) * 2 - 1) * ra
    turn = torch.zeros_like(angle)
    turn = torch.where(left_max, torch.full_like(angle, ra), turn)
    turn = torch.where(right_max, torch.full_like(angle, -ra), turn)
    turn = torch.where(equal, rand_turn, turn)

    angle = (angle + turn) % (2 * math.pi)
    new_x = (pos[:, 0] + ms * torch.cos(angle)) % w
    new_y = (pos[:, 1] + ms * torch.sin(angle)) % h

    species["pos"] = torch.stack([new_x, new_y], dim=1)
    species["angle"] = angle


def deposit(trail_ch, pos, amount, w, h):
    px = pos[:, 0].long().clamp(0, w - 1)
    py = pos[:, 1].long().clamp(0, h - 1)
    idx = py * w + px
    flat = trail_ch.reshape(-1)
    flat.index_add_(0, idx, torch.full_like(idx, amount, dtype=trail_ch.dtype))


def diffuse_decay(trails, decay):
    # 3x3 mean blur per channel, then multiplicative decay
    blurred = F.avg_pool2d(trails[None], kernel_size=3, stride=1, padding=1)[0]
    return blurred * decay


def colorize(trails, colors, scale=8.0, gamma=0.7):
    # trails: (3, H, W). colors: (3, 3) species -> RGB
    x = trails.clamp(min=0)
    x = torch.log1p(x * 0.6) / scale
    x = x.clamp(0, 1)
    rgb = torch.einsum("chw,cr->hwr", x, colors)
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
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "medium",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figures/physarum_landing.mp4")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_per_species", type=int, default=300_000)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decay", type=float, default=0.94)
    parser.add_argument("--repel", type=float, default=0.5,
                        help="how much agents are repelled by other species' trails")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    W, H = args.width, args.height
    device = args.device
    print(f"device={device} canvas={W}x{H} steps={args.steps} "
          f"n_per_species={args.n_per_species}")

    # Three species with distinct gait
    species_params = [
        dict(sensor_dist=9.0,  sensor_angle=math.pi / 4,  rot_angle=math.pi / 8,  move_speed=1.0, deposit=5.0),
        dict(sensor_dist=14.0, sensor_angle=math.pi / 6,  rot_angle=math.pi / 12, move_speed=1.4, deposit=5.0),
        dict(sensor_dist=6.0,  sensor_angle=math.pi / 3,  rot_angle=math.pi / 6,  move_speed=0.7, deposit=5.0),
    ]
    species = []
    for sp_args in species_params:
        pos, angle = init_agents(args.n_per_species, W, H, device, mode="ring")
        species.append({"pos": pos, "angle": angle, **sp_args})

    trails = torch.zeros(3, H, W, device=device)

    # Cinematic palette: magenta, cyan, gold
    colors = torch.tensor([
        [1.00, 0.30, 0.65],
        [0.25, 0.90, 1.00],
        [1.00, 0.80, 0.35],
    ], device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = open_ffmpeg_writer(out_path, W, H, args.fps)

    t_start = time.time()
    try:
        for step in range(args.steps):
            # Build per-species sense fields: own trail minus a fraction of others
            total = trails.sum(dim=0)
            for ch, sp in enumerate(species):
                own = trails[ch]
                sense_field = own - args.repel * (total - own)
                step_species(sp, sense_field, W, H)
                deposit(trails[ch], sp["pos"], sp["deposit"], W, H)

            trails = diffuse_decay(trails, args.decay)

            frame = colorize(trails, colors)
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
