"""
Local smoke test for the Fix B inter-path change in scripts/train.py.

Replicates the exact inter D step + G step code paths from the patched
train.py, runs them on fake tensors (no dataset, no NCA unroll), and
verifies:
- forward passes return valid shapes
- relativistic + GP losses compute cleanly
- backward runs without error and produces finite gradients on D params

This does NOT test training dynamics — it's a compile-level correctness
check before resubmitting the inter run to Sherlock.

Usage:
    python scripts/smoke_inter_fixB.py
"""

import torch
import torch.nn.functional as F

from nca_cellflow.models import Discriminator


def relativistic_d_loss(d_real, d_fake):
    return F.softplus(-(d_real - d_fake.mean())).mean() \
         + F.softplus(d_fake - d_real.mean()).mean()


def zero_centered_gp(samples, logits):
    reduced = logits.sum()
    (grad,) = torch.autograd.grad(
        outputs=reduced, inputs=samples, create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def build_d(num_compounds: int, device: torch.device) -> Discriminator:
    """Matches the w96D-inter config: w96 b3 s3, +1 null class."""
    num_classes = num_compounds + 1  # +1 null class since inter is on
    return Discriminator(
        widths=[96] * 4,
        cardinalities=[4] * 4,
        blocks_per_stage=[3] * 4,
        expansion=2,
        num_classes=num_classes,
        embed_dim=32,
        kernel_size=3,
        in_channels=3,
    ).to(device)


def main():
    torch.manual_seed(0)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    # Match train config scale (batch downscaled so MPS can run GP in reasonable time)
    B = 8
    H = W = 48
    num_compounds = 34
    null_class_id = num_compounds
    gamma = 1.0

    D = build_d(num_compounds, device)
    print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

    nca_input_img = torch.randn(B, 3, H, W, device=device)  # ctrl (RGB slice)
    img_trt = torch.randn(B, 3, H, W, device=device)
    cpd_id = torch.randint(0, num_compounds, (B,), device=device)
    # Simulated NCA intermediate (what inter_img = inter_full[:, :3] would be)
    inter_img = torch.randn(B, 3, H, W, device=device)

    # =========================================================================
    # D step: null-class inter branch (fixed version)
    # =========================================================================
    print("\n[D step] null-class inter branch")

    half = B // 2
    real_null = torch.cat([nca_input_img[:half], img_trt[:half]], dim=0)
    null_ids = torch.full((real_null.shape[0],), null_class_id,
                          device=device, dtype=torch.long)
    inter_null_ids = torch.full((inter_img.shape[0],), null_class_id,
                                device=device, dtype=torch.long)
    print(f"  real_null shape: {tuple(real_null.shape)}  (expected: ({B}, 3, {H}, {W}))")
    print(f"  inter_img shape: {tuple(inter_img.shape)}  (expected: ({B}, 3, {H}, {W}))")
    assert real_null.shape[0] == B, f"real_null batch {real_null.shape[0]} != B ({B})"

    real_null_req = real_null.detach().requires_grad_(True)
    inter_req = inter_img.detach().requires_grad_(True)

    d_real_null = D(real_null_req, null_ids)
    d_inter = D(inter_req, inter_null_ids)
    print(f"  d_real_null shape: {tuple(d_real_null.shape)}")
    print(f"  d_inter     shape: {tuple(d_inter.shape)}")
    assert d_real_null.shape[0] == B and d_inter.shape[0] == B

    adv_inter = relativistic_d_loss(d_real_null, d_inter)
    gp_real_null = zero_centered_gp(real_null_req, d_real_null)
    gp_inter = zero_centered_gp(inter_req, d_inter)
    reg_inter = 0.5 * gamma * (gp_real_null.mean() + gp_inter.mean())
    d_inter_loss = 0.5 * (adv_inter + reg_inter)

    print(f"  adv_inter       : {adv_inter.item():.4f}")
    print(f"  gp_real_null    : {gp_real_null.mean().item():.4f}")
    print(f"  gp_inter        : {gp_inter.mean().item():.4f}")
    print(f"  d_inter_loss    : {d_inter_loss.item():.4f}")
    assert torch.isfinite(d_inter_loss), "d_inter_loss is non-finite!"

    for p in D.parameters():
        if p.grad is not None:
            p.grad = None
    d_inter_loss.backward()

    total_grad_norm = sum(
        p.grad.float().norm().item() ** 2 for p in D.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  D grad norm     : {total_grad_norm:.4f}")
    assert total_grad_norm > 0, "D got no gradient from inter loss!"
    assert all(torch.isfinite(p.grad).all() for p in D.parameters() if p.grad is not None), \
        "D has non-finite gradients!"
    print("  [PASS] D step inter branch")

    # =========================================================================
    # G step: intermediate G-loss (null class) branch (fixed version)
    # =========================================================================
    print("\n[G step] intermediate G loss null-class branch")

    # Disable D grads (G step)
    for p in D.parameters():
        p.requires_grad_(False)

    # Simulate an inter_img that requires grad (comes from NCA in real code)
    inter_img_g = torch.randn(B, 3, H, W, device=device, requires_grad=True)

    real_null = torch.cat([nca_input_img[:half], img_trt[:half]], dim=0)
    null_ids = torch.full((real_null.shape[0],), null_class_id,
                          device=device, dtype=torch.long)
    inter_null_ids = torch.full((inter_img_g.shape[0],), null_class_id,
                                device=device, dtype=torch.long)

    d_real_null = D(real_null.detach(), null_ids)
    d_inter = D(inter_img_g, inter_null_ids)
    print(f"  d_real_null shape: {tuple(d_real_null.shape)}")
    print(f"  d_inter     shape: {tuple(d_inter.shape)}")
    assert d_real_null.shape == d_inter.shape, \
        f"G step elementwise subtraction requires matched shapes: " \
        f"{d_real_null.shape} vs {d_inter.shape}"

    rel_inter = d_inter - d_real_null  # elementwise
    g_inter_adv = F.softplus(-rel_inter).mean()
    g_inter_loss = 0.5 * g_inter_adv
    print(f"  g_inter_loss    : {g_inter_loss.item():.4f}")
    assert torch.isfinite(g_inter_loss)

    g_inter_loss.backward()
    grad_on_inter = inter_img_g.grad
    assert grad_on_inter is not None, \
        "G step: inter_img did not receive gradient from inter loss!"
    print(f"  grad on inter_img norm: {grad_on_inter.float().norm().item():.4f}")
    assert torch.isfinite(grad_on_inter).all()
    print("  [PASS] G step inter branch")

    print("\nAll checks passed. Safe to deploy.")


if __name__ == "__main__":
    main()
