"""
Benchmark torch.compile on the Discriminator's training-step code path.

Reproduces the exact D step from scripts/train.py (forward on real + fake,
relativistic D loss, zero-centered gradient penalty with create_graph=True,
backward) for the `w96-b3` shape used in the submitted experiments.

Compares uncompiled vs torch.compile(D). Runs on MPS if available, else CPU.

Not a production proxy — MPS ≠ CUDA — but a strong local signal for whether
torch.compile works at all on this D architecture, and an order-of-magnitude
speedup estimate.

Usage:
    python scripts/bench_d_compile.py
    python scripts/bench_d_compile.py --batch 64 --iters 30 --warmup 5
"""

import argparse
import time

import torch
import torch.nn.functional as F

from nca_cellflow.models import Discriminator


# ---------- D loss + GP (copy of relevant bits from train.py) -----------------

def relativistic_d_loss(d_real, d_fake):
    return F.softplus(-(d_real - d_fake.mean())).mean() \
         + F.softplus(d_fake - d_real.mean()).mean()


def zero_centered_gp(samples, logits):
    reduced = logits.sum()
    (grad,) = torch.autograd.grad(
        outputs=reduced, inputs=samples, create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def d_step(D, real, fake, cpd_id, gamma: float = 1.0, *, use_gp: bool = True):
    real_req = real.detach().requires_grad_(True)
    fake_req = fake.detach().requires_grad_(True)

    d_real = D(real_req, cpd_id)
    d_fake = D(fake_req, cpd_id)
    adv = relativistic_d_loss(d_real, d_fake)

    if use_gp:
        gp_real = zero_centered_gp(real_req, d_real)
        gp_fake = zero_centered_gp(fake_req, d_fake)
        reg = 0.5 * gamma * (gp_real.mean() + gp_fake.mean())
        loss = adv + reg
    else:
        loss = adv

    loss.backward()
    return loss.detach()


# ---------- Benchmark ---------------------------------------------------------

def build_d(num_classes: int, device: torch.device) -> Discriminator:
    # Matches configs/.../tanh-w96D-lb.yaml and probe-d-w96-b3.yaml
    stages = 3
    base = 96
    blocks = 3
    card = 4
    widths = [base] * (stages + 1)
    D = Discriminator(
        widths=widths,
        cardinalities=[card] * (stages + 1),
        blocks_per_stage=[blocks] * (stages + 1),
        expansion=2,
        num_classes=num_classes,
        embed_dim=32,
        kernel_size=3,
        in_channels=3,
    ).to(device)
    return D


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(D, real, fake, cpd_id, *, iters: int, warmup: int, device, label: str,
          use_gp: bool = True):
    # Warmup (includes compile if applicable)
    for _ in range(warmup):
        for p in D.parameters():
            if p.grad is not None:
                p.grad = None
        _ = d_step(D, real, fake, cpd_id, use_gp=use_gp)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        for p in D.parameters():
            if p.grad is not None:
                p.grad = None
        _ = d_step(D, real, fake, cpd_id, use_gp=use_gp)
    sync(device)
    elapsed = time.perf_counter() - t0

    per_step_ms = 1000.0 * elapsed / iters
    print(f"  [{label:12s}] {iters} steps in {elapsed:6.2f}s  |  {per_step_ms:7.2f} ms/step")
    return per_step_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size. Local MPS has less memory — don't use 128 blindly.")
    p.add_argument("--image_size", type=int, default=48)
    p.add_argument("--num_classes", type=int, default=35,
                   help="34 compounds + 1 null class (matches inter=True setup)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_compile", action="store_true",
                   help="Skip the compiled variant (debug only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    print(f"shape:  batch={args.batch}, C=3, H=W={args.image_size}, classes={args.num_classes}")
    print()

    # Fixed inputs
    real = torch.randn(args.batch, 3, args.image_size, args.image_size, device=device)
    fake = torch.randn(args.batch, 3, args.image_size, args.image_size, device=device)
    cpd_id = torch.randint(0, args.num_classes, (args.batch,), device=device)

    results: dict[tuple[str, bool], float] = {}

    # --- Four cells: {uncompiled, compiled} × {with GP, without GP} ---
    for compile_it in (False, True):
        if compile_it and args.no_compile:
            continue
        tag = "compiled" if compile_it else "uncompiled"
        print(f"building D ({tag})...")
        D = build_d(args.num_classes, device)
        if compile_it:
            try:
                D = torch.compile(D)
            except Exception as e:
                print(f"  !! torch.compile raised at construction: {e}")
                continue
        if not compile_it:
            print(f"D params: {sum(p.numel() for p in D.parameters()):,}")
        print()

        for use_gp in (True, False):
            label = f"{tag}-{'GP' if use_gp else 'noGP'}"
            print(f"benchmarking {label} D step:")
            try:
                ms = bench(
                    D, real, fake, cpd_id,
                    iters=args.iters, warmup=args.warmup,
                    device=device, label=label, use_gp=use_gp,
                )
                results[(tag, use_gp)] = ms
            except Exception as e:
                print(f"  !! {label} failed: {type(e).__name__}: {str(e)[:200]}")
            print()

    # --- Summary ---
    print("=" * 66)
    print(f"{'':14s}  {'GP (actual train)':>20s}  {'no-GP (hypothetical)':>22s}")
    for tag in ("uncompiled", "compiled"):
        gp_ms = results.get((tag, True))
        nogp_ms = results.get((tag, False))
        gp_str = f"{gp_ms:7.2f} ms/step" if gp_ms is not None else "       failed"
        nogp_str = f"{nogp_ms:7.2f} ms/step" if nogp_ms is not None else "       failed"
        print(f"  {tag:12s}  {gp_str:>20s}  {nogp_str:>22s}")
    print("=" * 66)

    # Relative comparisons the plan actually cares about
    print()
    if (("uncompiled", True) in results) and (("uncompiled", False) in results):
        ratio = results[("uncompiled", True)] / results[("uncompiled", False)]
        pct = 100 * (results[("uncompiled", True)] - results[("uncompiled", False)]) \
              / results[("uncompiled", True)]
        print(f"  GP cost within uncompiled: {ratio:.2f}x "
              f"({pct:+.0f}% of step time spent on GP)")
    if (("compiled", False) in results) and (("uncompiled", False) in results):
        sp = results[("uncompiled", False)] / results[("compiled", False)]
        print(f"  compile speedup on no-GP path: {sp:.2f}x")
    if (("compiled", True) in results) and (("uncompiled", True) in results):
        sp = results[("uncompiled", True)] / results[("compiled", True)]
        print(f"  compile speedup on GP path:   {sp:.2f}x")

    print()
    print("Caveat:", device.type, "≠ CUDA. Speedup magnitudes differ on L40S/H100,")
    print("but the compatibility signal (does it compile at all?) carries over.")


if __name__ == "__main__":
    main()
