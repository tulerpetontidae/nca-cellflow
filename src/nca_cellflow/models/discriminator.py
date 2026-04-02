"""
Multi-scale Residual Discriminator for NCA-GAN

Features:
- Progressive downsampling with interpolative resampling
- Residual blocks with grouped convolutions
- MSR (He) initialization for stable training
- Biased activation with learnable per-channel bias
- Adaptive pooling for variable image sizes
- Optional class-conditional discrimination
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Constants for biased activation
LRELU_SLOPE = 0.2
BIASED_ACT_GAIN = math.sqrt(2.0 / (1.0 + LRELU_SLOPE**2))


def make_kernel(weights: list[int]) -> torch.Tensor:
    """Create a 2D convolution kernel from 1D weights via outer product."""
    w = np.array(weights, dtype=np.float32)
    k = np.outer(w, w)
    k = torch.from_numpy(k)
    k = k / k.sum()
    return k


def msr_init(layer: nn.Module, activation_gain: float = 1.0) -> nn.Module:
    """
    MSR (MSRA/He) initialization for convolutional and linear layers.

    Initializes weights with normal distribution scaled by fan-in and activation gain.
    """
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d):
            fan_in = layer.weight.size(1) * layer.weight[0][0].numel()
            nn.init.normal_(layer.weight, mean=0.0, std=activation_gain / math.sqrt(fan_in))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            fan_in = layer.weight.size(1)
            nn.init.normal_(layer.weight, mean=0.0, std=activation_gain / math.sqrt(fan_in))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            # Fallback to default init
            pass
    return layer


class BiasedActivation(nn.Module):
    """
    Leaky ReLU with learnable per-channel bias.

    Applies bias before activation for better gradient flow.
    """

    gain: float = BIASED_ACT_GAIN

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            y = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)
        elif x.dim() == 2:
            y = x + self.bias.to(x.dtype).reshape(1, -1)
        else:
            raise ValueError("Unsupported input rank for BiasedActivation")
        return F.leaky_relu(y, negative_slope=LRELU_SLOPE)


class Conv2d(nn.Module):
    """
    Convolution layer with MSR initialization.

    Wraps nn.Conv2d with automatic padding and initialization.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = False,
        activation_gain: float = 1.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = msr_init(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=bias),
            activation_gain,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class InterpolativeUpsampler(nn.Module):
    """
    Upsampler using nearest neighbor interpolation followed by blur.

    Reduces checkerboard artifacts compared to transposed convolution.
    """

    def __init__(self, filt: list[int]) -> None:
        super().__init__()
        k = make_kernel(filt)
        self.register_buffer("kernel", k)
        self.radius = len(filt) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample by nearest then blur with depthwise conv
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        c = x.shape[1]
        weight = self.kernel.to(x.dtype).reshape(1, 1, self.kernel.shape[0], self.kernel.shape[1]).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, bias=None, stride=1, padding=self.radius, groups=c)


class InterpolativeDownsampler(nn.Module):
    """
    Downsampler using blur followed by strided convolution.

    Reduces aliasing artifacts.
    """

    def __init__(self, filt: list[int]) -> None:
        super().__init__()
        k = make_kernel(filt)
        self.register_buffer("kernel", k)
        self.radius = len(filt) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Blur with depthwise conv then downsample by stride=2
        c = x.shape[1]
        weight = self.kernel.to(x.dtype).reshape(1, 1, self.kernel.shape[0], self.kernel.shape[1]).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, bias=None, stride=2, padding=self.radius, groups=c)


class ResidualBlock(nn.Module):
    """
    Residual block with bottleneck structure and grouped convolutions.

    Architecture: 1x1 expand -> grouped conv -> 1x1 project
    Includes variance scaling for stable deep training.
    """

    def __init__(
        self, in_ch: int, cardinality: int, expansion: int, kernel_size: int, variance_scale_param: float
    ) -> None:
        super().__init__()
        num_linear_layers = 3
        expanded_ch = in_ch * expansion
        # Ensure valid grouping
        groups = max(1, math.gcd(expanded_ch, cardinality))
        act_gain = BiasedActivation.gain * (variance_scale_param ** (-1.0 / (2 * num_linear_layers - 2)))

        self.conv1 = Conv2d(in_ch, expanded_ch, kernel_size=1, activation_gain=act_gain)
        self.act1 = BiasedActivation(expanded_ch)
        self.conv2 = Conv2d(expanded_ch, expanded_ch, kernel_size=kernel_size, groups=groups, activation_gain=act_gain)
        self.act2 = BiasedActivation(expanded_ch)
        self.conv3 = Conv2d(expanded_ch, in_ch, kernel_size=1, activation_gain=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(self.act1(y))
        y = self.conv3(self.act2(y))
        return x + y


class DownsampleLayer(nn.Module):
    """
    Downsampling layer with optional channel matching.

    Uses interpolative downsampler to reduce spatial dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, resample_filter: list[int]) -> None:
        super().__init__()
        self.resampler = InterpolativeDownsampler(resample_filter)
        self.match = None
        if in_ch != out_ch:
            self.match = Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resampler(x)
        x = self.match(x) if self.match is not None else x
        return x


class DiscriminativeBasis(nn.Module):
    """
    Final discriminator layer mapping from any spatial size to scalar logits.

    Uses adaptive average pooling to collapse spatial dims to 1x1, then linear projection.
    Works with both square and non-square images.
    """

    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        # Adaptive pooling to handle any spatial dimensions (square or non-square)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = msr_init(nn.Linear(in_ch, out_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptive_pool(x)  # [B, C, 1, 1]
        x = x.reshape(x.shape[0], -1)  # [B, C]
        return self.linear(x)


class DiscriminatorStage(nn.Module):
    """
    Discriminator stage consisting of residual blocks + downsampling/basis.

    Args:
        resample_filter: If None, uses discriminative basis (final stage).
                        Otherwise, uses downsampling layer.
    """

    def __init__(
        self,
        in_ch: int,
        out_dim: int,
        cardinality: int,
        num_blocks: int,
        expansion: int,
        kernel_size: int,
        variance_scale_param: float,
        resample_filter: list[int] | None = None,
    ) -> None:
        super().__init__()
        if resample_filter is None:
            transition: nn.Module = DiscriminativeBasis(in_ch, out_dim)
        else:
            transition = DownsampleLayer(in_ch, out_dim, resample_filter)
        blocks = [
            ResidualBlock(in_ch, cardinality, expansion, kernel_size, variance_scale_param) for _ in range(num_blocks)
        ]
        blocks += [transition]
        self.layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PatchDiscriminator(nn.Module):
    """
    Hybrid patch + global discriminator.

    Shares a backbone, then branches into two heads:
    - **Patch head**: 1x1 conv → spatial score map [B, H', W'] for local spatial signal
    - **Global head**: adaptive pool → linear → scalar [B] with class-conditioned
      embedding dot-product for strong compound-level signal

    The forward method returns a dict with 'patch' and 'global' keys so the
    training loop can combine them (e.g. average the two losses).

    For 48x48 input with 3 downsampling stages the patch output is ~6x6 = 36 patches.

    Args:
        widths: Channel counts for each stage
        cardinalities: Group counts for each stage
        blocks_per_stage: Number of residual blocks per stage
        expansion: Channel expansion factor in residual blocks
        num_classes: Number of classes for conditional discrimination (None for unconditional)
        embed_dim: Embedding dimension for class conditioning
        kernel_size: Convolution kernel size
        resample_filter: Filter for interpolative resampling
        in_channels: Number of input channels
    """

    def __init__(
        self,
        widths: list[int],
        cardinalities: list[int],
        blocks_per_stage: list[int],
        expansion: int,
        num_classes: int | None = None,
        embed_dim: int = 0,
        kernel_size: int = 3,
        resample_filter: list[int] = [1, 2, 1],
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        assert len(widths) >= 1
        assert len(widths) == len(cardinalities) == len(blocks_per_stage)

        variance_scale_param = sum(blocks_per_stage)

        main_layers: list[nn.Module] = []
        self.from_rgb = Conv2d(in_channels, widths[0], kernel_size=1)

        # All downsampling stages (no final basis stage)
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
        # Final residual blocks without downsampling or basis
        final_blocks = [
            ResidualBlock(widths[-1], cardinalities[-1], expansion, kernel_size, variance_scale_param)
            for _ in range(blocks_per_stage[-1])
        ]
        main_layers.append(nn.Sequential(*final_blocks))
        self.main = nn.ModuleList(main_layers)

        self.use_class_cond = num_classes is not None and embed_dim > 0

        # Patch head: 1x1 conv → spatial score map (unconditional per-patch)
        self.to_patch_score = Conv2d(widths[-1], 1, kernel_size=1)

        # Global head: adaptive pool → class-conditioned scalar
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        if self.use_class_cond:
            self.global_proj = msr_init(nn.Linear(widths[-1], embed_dim, bias=False))
            self.embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.global_proj = msr_init(nn.Linear(widths[-1], 1, bias=False))
            self.embed = None

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        # x: [B, C_last, H', W']

        # Patch scores (unconditional spatial signal)
        patch_score = self.to_patch_score(x).squeeze(1)  # [B, H', W']

        # Global score (class-conditioned)
        g = self.adaptive_pool(x).reshape(x.shape[0], -1)  # [B, C_last]
        g = self.global_proj(g)  # [B, embed_dim] or [B, 1]
        if self.use_class_cond and y is not None:
            e = self.embed(y)  # [B, embed_dim]
            global_score = torch.sum(g * e, dim=1)  # [B]
        else:
            global_score = g.reshape(-1)  # [B]

        return {"patch": patch_score, "global": global_score}

    def forward_with_embed(self, x: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward with pre-computed embedding (for interpolated conditioning)."""
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        patch_score = self.to_patch_score(x).squeeze(1)
        g = self.adaptive_pool(x).reshape(x.shape[0], -1)
        g = self.global_proj(g)
        global_score = torch.sum(g * e, dim=1)
        return {"patch": patch_score, "global": global_score}


class TextureDiscriminator(nn.Module):
    """
    Shallow, high-resolution discriminator head for texture/high-frequency detail.

    Operates at full or near-full resolution (at most 1 downsampling stage) so that
    high-frequency texture information is preserved in the feature maps.  This
    addresses the spectral bias identified by Schwarz et al. (NeurIPS 2021): deep
    discriminators with aggressive downsampling lose high-frequency signal, so G
    never learns to produce sharp textures.

    Uses **dilated convolutions** with exponentially increasing dilation rates
    (1, 2, 4, …) to grow the receptive field without downsampling or extra params.
    With 3 layers of 3×3 kernels at dilations [1, 2, 4], the receptive field is
    15×15 — large enough to capture cell texture patterns (actin fibers, granules).

    Architecture:  from_rgb → [dilated_conv + act] × N → 1×1 patch score
    Optional class conditioning via projection (same as PatchDiscriminator).

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB)
        base_channels: Width of the conv layers
        num_layers: Number of conv layers (no downsampling)
        num_classes: Number of classes for conditional discrimination (None = unconditional)
        embed_dim: Embedding dimension for class conditioning
        kernel_size: Conv kernel size
        downsample: If True, apply one 2× downsampling after the first layer
        dilations: Dilation rates per layer, or None to auto-generate [1, 2, 4, …].
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_layers: int = 3,
        num_classes: int | None = None,
        embed_dim: int = 32,
        kernel_size: int = 3,
        downsample: bool = False,
        resample_filter: list[int] = [1, 2, 1],
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.from_rgb = Conv2d(in_channels, base_channels, kernel_size=1)
        self.act0 = BiasedActivation(base_channels)

        self.downsample_layer = None
        if downsample:
            self.downsample_layer = InterpolativeDownsampler(resample_filter)

        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]  # [1, 2, 4, ...]

        layers: list[nn.Module] = []
        for d in dilations:
            padding = d * (kernel_size - 1) // 2
            conv = msr_init(nn.Conv2d(
                base_channels, base_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                dilation=d, bias=False,
            ))
            layers.append(conv)
            layers.append(BiasedActivation(base_channels))
        self.layers = nn.ModuleList(layers)

        # Patch score head: 1×1 conv → spatial score map
        self.to_score = Conv2d(base_channels, 1, kernel_size=1)

        # Optional class conditioning via global pooled projection
        self.use_class_cond = num_classes is not None and embed_dim > 0
        if self.use_class_cond:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_proj = msr_init(nn.Linear(base_channels, embed_dim, bias=False))
            self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act0(self.from_rgb(x))
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        for layer in self.layers:
            x = layer(x)

        # Patch scores (unconditional spatial signal)
        score = self.to_score(x).squeeze(1)  # [B, H', W']

        if self.use_class_cond and y is not None:
            g = self.global_pool(x).reshape(x.shape[0], -1)
            g = self.global_proj(g)
            e = self.embed(y)
            class_score = torch.sum(g * e, dim=1)  # [B]
            # Add class score to every spatial position
            score = score + class_score[:, None, None]

        return score


class SpectralMatchingLoss(nn.Module):
    """
    Batch-level spectral matching loss using precomputed per-compound power spectra.

    Computes the mean log-magnitude FFT of a batch of generated images and penalises
    deviation from the precomputed target spectrum for each compound.  This gives G a
    stable, non-adversarial signal about missing high frequencies.

    The target spectra should be precomputed once over the full dataset using
    :meth:`precompute_spectrum` and registered via :meth:`register_targets`.

    Usage::

        spec_loss = SpectralMatchingLoss()
        spec_loss.register_targets(precomputed)   # [num_cpd, C, H, W//2+1]
        loss = spec_loss(fake_img, cpd_id)
    """

    def __init__(self) -> None:
        super().__init__()
        # Will hold [num_compounds, C, H, W//2+1] after register_targets
        self.register_buffer("target_spectra", torch.empty(0), persistent=False)

    def register_targets(self, spectra: torch.Tensor) -> None:
        """Register precomputed target spectra [num_compounds, C, H, W//2+1]."""
        self.target_spectra = spectra

    @staticmethod
    def log_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute log-magnitude of 2D rFFT: [B, C, H, W] → [B, C, H, W//2+1]."""
        f = torch.fft.rfft2(x, norm="ortho")
        return torch.log1p(f.abs())

    @staticmethod
    def precompute_spectrum(dataset, num_compounds: int, device: torch.device = torch.device("cpu"),
                            max_samples_per_cpd: int = 2000) -> torch.Tensor:
        """
        Precompute mean log-magnitude spectrum for each compound from dataset.

        Args:
            dataset: iterable yielding (img_ctrl, img_trt, cpd_id) tuples
            num_compounds: total number of compound classes
            device: device for accumulation
            max_samples_per_cpd: cap per compound to limit compute

        Returns:
            Tensor of shape [num_compounds, C, H, W//2+1]
        """
        accum = None
        counts = torch.zeros(num_compounds, device=device)

        for img_ctrl, img_trt, cpd_id in dataset:
            img_trt = img_trt.to(device)
            if img_trt.dim() == 3:
                img_trt = img_trt.unsqueeze(0)
                cpd_id = cpd_id.unsqueeze(0) if cpd_id.dim() == 0 else cpd_id
            cpd_id = cpd_id.to(device)

            spec = SpectralMatchingLoss.log_magnitude(img_trt)  # [B, C, H, W//2+1]

            if accum is None:
                accum = torch.zeros(num_compounds, *spec.shape[1:], device=device)

            for i in range(spec.shape[0]):
                cid = cpd_id[i].item()
                if counts[cid] < max_samples_per_cpd:
                    accum[cid] += spec[i]
                    counts[cid] += 1

        # Average
        counts = counts.clamp(min=1)
        accum = accum / counts[:, None, None, None]
        return accum

    def forward(self, fake: torch.Tensor, cpd_id: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral matching loss for a batch.

        Groups the batch by compound, computes per-group mean spectrum,
        and returns the L1 distance to the precomputed target.
        """
        spec = self.log_magnitude(fake)  # [B, C, H, W//2+1]
        target = self.target_spectra.to(fake.device)

        loss = torch.tensor(0.0, device=fake.device)
        n_groups = 0
        for cid in cpd_id.unique():
            mask = cpd_id == cid
            if mask.sum() < 2:
                continue  # skip singletons — spectrum too noisy
            mean_spec = spec[mask].mean(dim=0)
            loss = loss + F.l1_loss(mean_spec, target[cid.long()])
            n_groups += 1

        return loss / max(n_groups, 1)


class Discriminator(nn.Module):
    """
    Multi-scale residual discriminator with conditional discrimination.

    Features:
    - Progressive downsampling through multiple stages
    - Residual blocks with grouped convolutions
    - Optional class-conditional discrimination via embeddings

    Args:
        widths: Channel counts for each stage
        cardinalities: Group counts for each stage
        blocks_per_stage: Number of residual blocks per stage
        expansion: Channel expansion factor in residual blocks
        num_classes: Number of classes for conditional discrimination (None for unconditional)
        embed_dim: Embedding dimension for class conditioning
        kernel_size: Convolution kernel size
        resample_filter: Filter for interpolative resampling
        in_channels: Number of input channels (default: 2)
    """

    def __init__(
        self,
        widths: list[int],
        cardinalities: list[int],
        blocks_per_stage: list[int],
        expansion: int,
        num_classes: int | None = None,
        embed_dim: int = 0,
        kernel_size: int = 3,
        resample_filter: list[int] = [1, 2, 1],
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        assert len(widths) >= 1
        assert len(widths) == len(cardinalities) == len(blocks_per_stage)

        variance_scale_param = sum(blocks_per_stage)

        main_layers: list[nn.Module] = []
        # 1x1 extraction from input image
        self.from_rgb = Conv2d(in_channels, widths[0], kernel_size=1)
        # Downsampling stages
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
        # Final stage: adaptive pooling to 1x1 via discriminative basis (handles non-square images)
        final_out_dim = 1 if (num_classes is None or embed_dim == 0) else embed_dim
        main_layers.append(
            DiscriminatorStage(
                widths[-1],
                final_out_dim,
                cardinalities[-1],
                blocks_per_stage[-1],
                expansion,
                kernel_size,
                variance_scale_param,
            )
        )
        self.main = nn.ModuleList(main_layers)

        self.embed = None
        if num_classes is not None and embed_dim > 0:
            self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        # x is [B, C, 1, 1]
        x = x.reshape(x.shape[0], -1)
        if self.embed is not None and y is not None:
            e = self.embed(y)
            x = torch.sum(x * e, dim=1, keepdim=False)
        else:
            x = x.reshape(-1)
        return x

    def forward_with_embed(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Forward with pre-computed embedding (for interpolated conditioning)."""
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        return torch.sum(x * e, dim=1, keepdim=False)
