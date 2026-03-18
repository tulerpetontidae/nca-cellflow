"""
Simple integration tests for NCA and Discriminator models.
Run with: python tests/test_models.py
"""

import torch
import sys
sys.path.insert(0, "src")

from nca_cellflow.models import BaseNCA, Discriminator


def test_nca_shapes():
    """Test NCA with dataset-like shapes."""
    B, C, H, W = 4, 3, 96, 96
    num_classes = 10

    nca = BaseNCA(
        channel_n=C,
        hidden_dim=64,
        num_classes=num_classes,
        cond_dim=32,
    )

    x = torch.randn(B, C, H, W)
    cond = torch.randint(0, num_classes, (B,))

    # Single step
    out = nca.step(x, cond)
    assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"

    # Forward with multiple steps
    out = nca.forward(x, cond, n_steps=5)
    assert out.shape == (B, C, H, W)

    # Sample trajectory
    traj = nca.sample(x, cond, n_steps=10, output_steps=[0, 5, 10])
    assert len(traj) == 3
    assert all(s.shape == (B, C, H, W) for s in traj)

    print("NCA: OK")


def test_discriminator_shapes():
    """Test discriminator with dataset-like shapes."""
    B, C, H, W = 4, 3, 96, 96
    num_classes = 10

    # Unconditional
    disc = Discriminator(
        widths=[32, 64, 128],
        cardinalities=[4, 8, 16],
        blocks_per_stage=[1, 1, 1],
        expansion=2,
        in_channels=C,
    )

    x = torch.randn(B, C, H, W)
    out = disc(x)
    assert out.shape == (B,), f"Expected {(B,)}, got {out.shape}"
    print("Discriminator (unconditional): OK")

    # Conditional
    disc_cond = Discriminator(
        widths=[32, 64, 128],
        cardinalities=[4, 8, 16],
        blocks_per_stage=[1, 1, 1],
        expansion=2,
        num_classes=num_classes,
        embed_dim=64,
        in_channels=C,
    )

    y = torch.randint(0, num_classes, (B,))
    out = disc_cond(x, y)
    assert out.shape == (B,), f"Expected {(B,)}, got {out.shape}"
    print("Discriminator (conditional): OK")


def test_gan_forward():
    """Test full GAN forward pass: ctrl -> NCA -> fake, disc(fake, cond)."""
    B, C, H, W = 4, 3, 96, 96
    num_classes = 10

    nca = BaseNCA(channel_n=C, hidden_dim=64, num_classes=num_classes, cond_dim=32)
    disc = Discriminator(
        widths=[32, 64, 128],
        cardinalities=[4, 8, 16],
        blocks_per_stage=[1, 1, 1],
        expansion=2,
        num_classes=num_classes,
        embed_dim=64,
        in_channels=C,
    )

    # Simulate dataset batch
    img_ctrl = torch.randn(B, C, H, W)  # control image (start state)
    img_trt = torch.randn(B, C, H, W)  # target treated image
    cpd_id = torch.randint(0, num_classes, (B,))

    # Generator: ctrl -> fake_trt
    fake_trt = nca(img_ctrl, cpd_id, n_steps=10)

    # Discriminator
    score_real = disc(img_trt, cpd_id)
    score_fake = disc(fake_trt, cpd_id)

    assert score_real.shape == (B,)
    assert score_fake.shape == (B,)

    print("GAN forward pass: OK")


def test_gradients():
    """Test that gradients flow through both models."""
    B, C, H, W = 2, 3, 32, 32  # smaller for speed
    num_classes = 5

    nca = BaseNCA(channel_n=C, hidden_dim=32, num_classes=num_classes, cond_dim=16)
    disc = Discriminator(
        widths=[16, 32],
        cardinalities=[4, 8],
        blocks_per_stage=[1, 1],
        expansion=2,
        num_classes=num_classes,
        embed_dim=32,
        in_channels=C,
    )

    x = torch.randn(B, C, H, W, requires_grad=True)
    cond = torch.randint(0, num_classes, (B,))

    # NCA gradients
    fake = nca(x, cond, n_steps=3)
    loss_g = fake.mean()
    loss_g.backward()
    assert x.grad is not None
    print("NCA gradients: OK")

    # Discriminator gradients
    x2 = torch.randn(B, C, H, W, requires_grad=True)
    score = disc(x2, cond)
    loss_d = score.mean()
    loss_d.backward()
    assert x2.grad is not None
    print("Discriminator gradients: OK")


if __name__ == "__main__":
    test_nca_shapes()
    test_discriminator_shapes()
    test_gan_forward()
    test_gradients()
    print("\nAll tests passed!")
