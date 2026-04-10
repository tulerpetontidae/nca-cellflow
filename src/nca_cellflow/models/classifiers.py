"""
Classifier wrappers for capacity probing.

These build the same backbones as the GAN's `Discriminator` and the ResBlk-based
style encoder family (`IMPAStyleEncoder`), but expose a plain multi-class
classification head instead of the GAN-specific class-conditioned projection.
Used by `scripts/probe_classifier.py` to calibrate D / StyleEncoder capacity
as a decoupled 34-class compound classification task on real treated BBBC021
images, prior to re-using the calibrated architecture inside the GAN loop.

Deliberately lives in its own file so the probe experiment is strictly
additive and can be deleted without touching the live GAN training code.
"""

import torch
import torch.nn as nn

from .discriminator import Conv2d, DiscriminatorStage
from .impa import ResBlk


class DiscriminatorClassifier(nn.Module):
    """Plain classifier built from the GAN `Discriminator`'s building blocks.

    Construction mirrors `Discriminator.__init__` (widths, cardinalities,
    blocks_per_stage, expansion, kernel_size) so the probe measures the
    capacity of the exact same backbone the GAN uses.

    Architecture:
        from_rgb (1x1 Conv)
        → DiscriminatorStage × (len(widths) - 1)     # downsampling stages
        → DiscriminatorStage with DiscriminativeBasis # final: adapt-pool + linear
        → [B, num_classes] logits
    """

    def __init__(
        self,
        widths: list[int],
        cardinalities: list[int],
        blocks_per_stage: list[int],
        expansion: int,
        kernel_size: int = 3,
        in_channels: int = 3,
        num_classes: int = 34,
        resample_filter: list[int] = [1, 2, 1],
    ) -> None:
        super().__init__()
        assert len(widths) >= 1
        assert len(widths) == len(cardinalities) == len(blocks_per_stage)

        variance_scale_param = sum(blocks_per_stage)

        self.from_rgb = Conv2d(in_channels, widths[0], kernel_size=1)

        main_layers: list[nn.Module] = []
        for idx in range(len(widths) - 1):
            main_layers.append(
                DiscriminatorStage(
                    widths[idx],
                    widths[idx + 1],
                    cardinalities[idx],
                    blocks_per_stage[idx],
                    expansion,
                    kernel_size,
                    variance_scale_param,
                    resample_filter,
                )
            )
        # Final stage: residual blocks + DiscriminativeBasis (adaptive pool +
        # linear). Setting `resample_filter=None` routes DiscriminatorStage
        # through DiscriminativeBasis, which maps [B, widths[-1], H', W']
        # directly to [B, num_classes].
        main_layers.append(
            DiscriminatorStage(
                widths[-1],
                num_classes,
                cardinalities[-1],
                blocks_per_stage[-1],
                expansion,
                kernel_size,
                variance_scale_param,
                resample_filter=None,
            )
        )
        self.main = nn.ModuleList(main_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        return x  # [B, num_classes]


class StyleEncoderClassifier(nn.Module):
    """Scalable ResBlk-based style encoder with a classification head.

    Uses the same `ResBlk` as `IMPAStyleEncoder` (see models/impa.py), but
    exposes explicit width (`base_channels`) and depth (`num_downsamples`)
    knobs — `IMPAStyleEncoder` auto-derives depth from `img_size`, which is
    too coarse for a capacity sweep.

    Architecture:
        Conv2d(in_channels → base_channels, 3×3)
        → ResBlk(downsample=True) × num_downsamples  # channels double, cap at max_channels
        → LeakyReLU → AdaptiveAvgPool2d(1) → LeakyReLU
        → Linear(last_channels → num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_downsamples: int = 4,
        num_classes: int = 34,
        max_channels: int = 512,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv2d(in_channels, base_channels, 3, 1, 1)]
        dim_in = base_channels
        dim_out = base_channels
        for _ in range(num_downsamples):
            dim_out = min(dim_in * 2, max_channels)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*blocks)
        self.linear = nn.Linear(dim_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        return self.linear(h)
