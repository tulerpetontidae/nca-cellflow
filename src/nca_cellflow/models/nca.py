"""
NCA Architecture Variants

- BaseNCA: Simple deterministic NCA with single update rule
- NoiseNCA: NCA with per-step noise injection concatenated to state
- LatentNCA: NCA with single latent z sampled once, injected via FiLM
- GradientSensor: Sobel gradient sensing for spatial feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .impa import ResBlk


class Permute(nn.Module):
    """Simple module to permute tensor dimensions."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class GradientSensor(nn.Module):
    """
    Sobel gradient sensor for NCA.

    Computes spatial gradients using Sobel filters, augmenting each channel
    with its x and y gradients.
    """

    def __init__(self, channel_n: int):
        super().__init__()
        self.channel_n = channel_n

        sobel_x = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=torch.float32)
        sobel_y = sobel_x.T.clone()

        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1)

        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding="same", groups=self.channel_n)
        grad_y = F.conv2d(x, self.sobel_y, padding="same", groups=self.channel_n)
        return torch.cat([x, grad_x, grad_y], dim=1)  # [B, 3C, H, W]


# ============================================================================
# BASELINE NCA: Simple deterministic single-rule NCA
# ============================================================================


class BaseNCA(nn.Module):
    """
    Conditional NCA with FiLM-based compound conditioning.

    Supports two conditioning modes:
    - 'id': compound_id (int) -> Embedding -> FiLM
    - 'fingerprint': fingerprint vector (float) -> Linear -> FiLM

    Args:
        channel_n: Number of state channels
        hidden_dim: Hidden dimension in update MLP
        num_classes: Number of compound classes (used when cond_type='id')
        cond_dim: Dimension of compound embedding
        cond_type: 'id' for learned embedding, 'fingerprint' for projected fingerprint
        fp_dim: Fingerprint vector dimension (used when cond_type='fingerprint')
        fire_rate: Probability of cell update (stochastic masking)
    """

    def __init__(
        self,
        channel_n: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 1,
        cond_dim: int = 64,
        cond_type: str = "id",
        fp_dim: int = 1024,
        fire_rate: float = 1.0,
        step_size: float = 0.1,
        dx_clip: float = 10.0,
        use_tanh: bool = False,
        use_alive_mask: bool = False,
        alive_threshold: float = 0.05,
    ):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_dim = hidden_dim
        self.fire_rate = fire_rate
        self.step_size = step_size
        self.dx_clip = dx_clip
        self.use_tanh = use_tanh
        self.use_alive_mask = use_alive_mask
        self.alive_threshold = alive_threshold
        self.cond_type = cond_type
        self.sensor = GradientSensor(channel_n)

        # Compound conditioning
        if cond_type == "fingerprint":
            self.embed = nn.Linear(fp_dim, cond_dim)
        else:
            self.embed = nn.Embedding(num_classes, cond_dim)

        # FiLM: embedding -> (gamma, beta) for each hidden layer
        self.film1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.film2 = nn.Linear(cond_dim, hidden_dim * 2)

        # Update MLP: [3C] -> hidden -> hidden -> C
        Fin = 3 * channel_n
        self.fc1 = nn.Linear(Fin, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, channel_n)

        # Zero-init final layer for stable residual
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def step(self, x, cond):
        """
        Single NCA step.

        Args:
            x: State [B, C, H, W]
            cond: Compound IDs [B]
        """
        B, C, H, W = x.shape

        # Embed condition and get FiLM params
        emb = self.embed(cond)  # [B, cond_dim]
        film1 = self.film1(emb)
        gamma1, beta1 = film1.chunk(2, dim=1)
        gamma1 = gamma1[:, None, None, :]
        beta1 = beta1[:, None, None, :]
        film2 = self.film2(emb)
        gamma2, beta2 = film2.chunk(2, dim=1)
        gamma2 = gamma2[:, None, None, :]
        beta2 = beta2[:, None, None, :]

        # Gradient features
        features = self.sensor(x)  # [B, 3C, H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, 3C]

        # MLP with FiLM on both layers
        h = F.relu(self.fc1(features))
        h = gamma1 * h + beta1
        h = F.relu(self.fc2(h))
        h = gamma2 * h + beta2
        dx = self.fc3(h).permute(0, 3, 1, 2)  # [B, C, H, W]

        if self.use_tanh:
            dx = torch.tanh(dx)
        else:
            dx = torch.clamp(dx, min=-self.dx_clip, max=self.dx_clip)

        # Fire rate mask
        if self.fire_rate < 1.0:
            mask = (torch.rand(B, 1, H, W, device=x.device) < self.fire_rate).float()
            dx = dx * mask

        new_x = x + dx * self.step_size

        # Alive mask: dead cells get RGB=-1 (black), alive+hidden=0
        if self.use_alive_mask:
            alive = F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > self.alive_threshold
            bg = torch.zeros_like(new_x)
            bg[:, :3] = -1.0
            new_x = torch.where(alive, new_x, bg)

        return new_x

    def forward(self, x, cond, n_steps: int = 1):
        """Apply n_steps NCA updates."""
        for _ in range(n_steps):
            x = self.step(x, cond)
        return x

    def forward_with_intermediate(self, x, cond, n_steps: int, t_intermediate: int):
        """Forward pass returning both final state and state at step t_intermediate."""
        intermediate = None
        for t in range(n_steps):
            x = self.step(x, cond)
            if t == t_intermediate:
                intermediate = x
        return x, intermediate

    def sample(self, x, cond, n_steps: int = 100, output_steps=None):
        """
        Generate trajectory of states.

        Args:
            x: Initial state [B, C, H, W]
            cond: Compound IDs [B]
            n_steps: Number of steps
            output_steps: Which steps to return (None = all)

        Returns:
            List of state tensors
        """
        samples = []
        if output_steps is None or 0 in output_steps:
            samples.append(x.clone())

        for t in range(n_steps):
            x = self.step(x, cond)
            if output_steps is None or (t + 1) in output_steps:
                samples.append(x.clone())

        return samples


# ============================================================================
# NOISE NCA: NCA with per-step noise injection after gradient sensing
# ============================================================================


class NoiseNCA(nn.Module):
    """
    NCA that injects random noise at each step. Gradient sensing runs on
    the state only, then noise channels are concatenated before the MLP.

    Args:
        channel_n: Number of state channels
        noise_channels: Number of noise channels to inject per step
        hidden_dim: Hidden dimension in update MLP
        num_classes: Number of compound classes (used when cond_type='id')
        cond_dim: Dimension of compound embedding
        cond_type: 'id' for learned embedding, 'fingerprint' for projected fingerprint
        fp_dim: Fingerprint vector dimension (used when cond_type='fingerprint')
        fire_rate: Probability of cell update (stochastic masking)
    """

    def __init__(
        self,
        channel_n: int = 3,
        noise_channels: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 1,
        cond_dim: int = 64,
        cond_type: str = "id",
        fp_dim: int = 1024,
        fire_rate: float = 1.0,
        step_size: float = 0.1,
        dx_clip: float = 10.0,
        use_tanh: bool = False,
        use_alive_mask: bool = False,
        alive_threshold: float = 0.05,
    ):
        super().__init__()
        self.channel_n = channel_n
        self.noise_channels = noise_channels
        self.hidden_dim = hidden_dim
        self.fire_rate = fire_rate
        self.step_size = step_size
        self.dx_clip = dx_clip
        self.use_tanh = use_tanh
        self.use_alive_mask = use_alive_mask
        self.alive_threshold = alive_threshold
        self.cond_type = cond_type

        self.sensor = GradientSensor(channel_n)

        if cond_type == "fingerprint":
            self.embed = nn.Linear(fp_dim, cond_dim)
        else:
            self.embed = nn.Embedding(num_classes, cond_dim)
        self.film1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.film2 = nn.Linear(cond_dim, hidden_dim * 2)

        # MLP input: 3C (from sensor) + noise_channels
        Fin = 3 * channel_n + noise_channels
        self.fc1 = nn.Linear(Fin, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, channel_n)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def step(self, x, cond):
        B, C, H, W = x.shape

        emb = self.embed(cond)
        film1 = self.film1(emb)
        gamma1, beta1 = film1.chunk(2, dim=1)
        gamma1 = gamma1[:, None, None, :]
        beta1 = beta1[:, None, None, :]
        film2 = self.film2(emb)
        gamma2, beta2 = film2.chunk(2, dim=1)
        gamma2 = gamma2[:, None, None, :]
        beta2 = beta2[:, None, None, :]

        # Sensor on state only
        features = self.sensor(x)  # [B, 3C, H, W]

        # Concat noise after sensing
        noise = torch.randn(B, self.noise_channels, H, W, device=x.device)
        features = torch.cat([features, noise], dim=1)  # [B, 3C + noise, H, W]
        features = features.permute(0, 2, 3, 1)

        h = F.relu(self.fc1(features))
        h = gamma1 * h + beta1
        h = F.relu(self.fc2(h))
        h = gamma2 * h + beta2
        dx = self.fc3(h).permute(0, 3, 1, 2)

        if self.use_tanh:
            dx = torch.tanh(dx)
        else:
            dx = torch.clamp(dx, min=-self.dx_clip, max=self.dx_clip)

        if self.fire_rate < 1.0:
            mask = (torch.rand(B, 1, H, W, device=x.device) < self.fire_rate).float()
            dx = dx * mask

        new_x = x + dx * self.step_size

        # Alive mask: dead cells get RGB=-1 (black), alive+hidden=0
        if self.use_alive_mask:
            alive = F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > self.alive_threshold
            bg = torch.zeros_like(new_x)
            bg[:, :3] = -1.0
            new_x = torch.where(alive, new_x, bg)

        return new_x

    def forward(self, x, cond, n_steps: int = 1):
        for _ in range(n_steps):
            x = self.step(x, cond)
        return x

    def forward_with_intermediate(self, x, cond, n_steps: int, t_intermediate: int):
        """Forward pass returning both final state and state at step t_intermediate."""
        intermediate = None
        for t in range(n_steps):
            x = self.step(x, cond)
            if t == t_intermediate:
                intermediate = x
        return x, intermediate

    def sample(self, x, cond, n_steps: int = 100, output_steps=None):
        samples = []
        if output_steps is None or 0 in output_steps:
            samples.append(x.clone())
        for t in range(n_steps):
            x = self.step(x, cond)
            if output_steps is None or (t + 1) in output_steps:
                samples.append(x.clone())
        return samples


# ============================================================================
# LATENT NCA: NCA with single latent z injected via FiLM conditioning
# ============================================================================


class LatentNCA(nn.Module):
    """
    NCA with a single latent z sampled once per generation. The compound
    embedding and z are concatenated and fed directly to FiLM layers.
    No mapping network — FiLM layers themselves learn to mix embed and z.

    Conditioning: [embed(compound) || z] -> FiLM (dim = cond_dim + z_dim)

    z occupies dedicated dimensions in the FiLM input, so the network
    structurally cannot ignore it without also ignoring compound conditioning.

    Args:
        channel_n: Number of state channels
        z_dim: Dimension of latent noise vector (default 16)
        hidden_dim: Hidden dimension in update MLP
        num_classes: Number of compound classes (used when cond_type='id')
        cond_dim: Dimension of compound embedding
        cond_type: 'id' for learned embedding, 'fingerprint' for projected fingerprint
        fp_dim: Fingerprint vector dimension (used when cond_type='fingerprint')
        fire_rate: Probability of cell update (stochastic masking)
        step_size: Residual update scaling
    """

    def __init__(
        self,
        channel_n: int = 3,
        z_dim: int = 16,
        hidden_dim: int = 128,
        num_classes: int = 1,
        cond_dim: int = 64,
        cond_type: str = "id",
        fp_dim: int = 1024,
        fire_rate: float = 1.0,
        step_size: float = 0.1,
        dx_clip: float = 10.0,
        use_tanh: bool = False,
        use_alive_mask: bool = False,
        alive_threshold: float = 0.05,
        dose_dim: int = 0,
    ):
        super().__init__()
        self.channel_n = channel_n
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dose_dim = dose_dim
        self.fire_rate = fire_rate
        self.step_size = step_size
        self.dx_clip = dx_clip
        self.use_tanh = use_tanh
        self.use_alive_mask = use_alive_mask
        self.alive_threshold = alive_threshold
        self.cond_type = cond_type

        self.sensor = GradientSensor(channel_n)

        # Compound conditioning
        if cond_type == "fingerprint":
            self.embed = nn.Linear(fp_dim, cond_dim)
        else:
            self.embed = nn.Embedding(num_classes, cond_dim)

        # Optional dose conditioning: log10(dose) -> Linear -> dose_dim
        if dose_dim > 0:
            self.dose_proj = nn.Linear(1, dose_dim)

        # FiLM directly from [embed || dose || z] -> (gamma, beta)
        film_in = cond_dim + dose_dim + z_dim
        self.film1 = nn.Linear(film_in, hidden_dim * 2)
        self.film2 = nn.Linear(film_in, hidden_dim * 2)

        # Update MLP: [3C] -> hidden -> hidden -> C
        Fin = 3 * channel_n
        self.fc1 = nn.Linear(Fin, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, channel_n)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def step(self, x, cond_z):
        """Single NCA step. cond_z: [B, cond_dim + z_dim] concatenated embedding and z."""
        B, C, H, W = x.shape

        film1 = self.film1(cond_z)
        gamma1, beta1 = film1.chunk(2, dim=1)
        gamma1 = gamma1[:, None, None, :]
        beta1 = beta1[:, None, None, :]
        film2 = self.film2(cond_z)
        gamma2, beta2 = film2.chunk(2, dim=1)
        gamma2 = gamma2[:, None, None, :]
        beta2 = beta2[:, None, None, :]

        features = self.sensor(x)  # [B, 3C, H, W]
        features = features.permute(0, 2, 3, 1)

        h = F.relu(self.fc1(features))
        h = gamma1 * h + beta1
        h = F.relu(self.fc2(h))
        h = gamma2 * h + beta2
        dx = self.fc3(h).permute(0, 3, 1, 2)

        if self.use_tanh:
            dx = torch.tanh(dx)
        else:
            dx = torch.clamp(dx, min=-self.dx_clip, max=self.dx_clip)

        if self.fire_rate < 1.0:
            mask = (torch.rand(B, 1, H, W, device=x.device) < self.fire_rate).float()
            dx = dx * mask

        new_x = x + dx * self.step_size

        if self.use_alive_mask:
            alive = F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > self.alive_threshold
            bg = torch.zeros_like(new_x)
            bg[:, :3] = -1.0
            new_x = torch.where(alive, new_x, bg)

        return new_x

    def _prepare_cond(self, cond, z=None, dose=None):
        """Embed compound, optionally encode dose, sample z if needed.

        Args:
            cond: Compound IDs [B] or fingerprints [B, fp_dim].
            z: Optional latent [B, z_dim]. Sampled from N(0,1) if None.
            dose: Optional dose values [B]. Only used when dose_dim > 0.

        Returns:
            (cond_z [B, cond_dim + dose_dim + z_dim], z [B, z_dim])
        """
        emb = self.embed(cond)  # [B, cond_dim]
        if z is None:
            z = torch.randn(emb.shape[0], self.z_dim, device=emb.device)
        parts = [emb]
        if self.dose_dim > 0:
            if dose is not None:
                # log10(dose) with safe handling for dose=0 (DMSO)
                log_dose = torch.log10(dose.clamp(min=1e-4)).unsqueeze(-1)  # [B, 1]
                parts.append(self.dose_proj(log_dose))
            else:
                # No dose info: use zeros
                parts.append(torch.zeros(emb.shape[0], self.dose_dim, device=emb.device))
        parts.append(z)
        cond_z = torch.cat(parts, dim=1)
        return cond_z, z

    def forward(self, x, cond, n_steps: int = 1, z=None, dose=None):
        cond_z, _ = self._prepare_cond(cond, z, dose)
        for _ in range(n_steps):
            x = self.step(x, cond_z)
        return x

    def forward_with_intermediate(self, x, cond, n_steps: int, t_intermediate: int,
                                  z=None, dose=None):
        cond_z, _ = self._prepare_cond(cond, z, dose)
        intermediate = None
        for t in range(n_steps):
            x = self.step(x, cond_z)
            if t == t_intermediate:
                intermediate = x
        return x, intermediate

    def forward_with_style(self, x, cond, n_steps: int = 1, z=None, dose=None):
        """Forward that also returns (cond_z, z) for style reconstruction loss."""
        cond_z, z = self._prepare_cond(cond, z, dose)
        for _ in range(n_steps):
            x = self.step(x, cond_z)
        return x, cond_z, z

    def sample(self, x, cond, n_steps: int = 100, output_steps=None, z=None, dose=None):
        cond_z, _ = self._prepare_cond(cond, z, dose)
        samples = []
        if output_steps is None or 0 in output_steps:
            samples.append(x.clone())
        for t in range(n_steps):
            x = self.step(x, cond_z)
            if output_steps is None or (t + 1) in output_steps:
                samples.append(x.clone())
        return samples


# ============================================================================
# Style Encoder: recovers style from generated image (anti-collapse)
# ============================================================================


class NCAStyleEncoder(nn.Module):
    """
    Predicts [embed || z] from a generated image. Used with LatentNCA
    for style reconstruction loss: |StyleEncoder(fake) - [embed || z]|.

    style_dim should be cond_dim + z_dim (e.g. 32 + 16 = 48) so the
    encoder predicts the full conditioning vector directly.

    If G ignores z, all outputs for a compound look identical, but the
    target [embed || z] varies with z → loss stays high → forces G to use z.

    Simple conv backbone → global average pool → linear → style_dim.
    """

    def __init__(self, in_channels: int = 3, style_dim: int = 32, base_channels: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base_channels * 4, style_dim)

    def forward(self, x):
        h = self.backbone(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)


# ============================================================================
# ResBlk Style Encoder: probe-calibrated larger encoder for style-NCA
# ============================================================================


class ResBlkStyleEncoder(nn.Module):
    """
    ResBlk-based style encoder for self-supervised NCA training.

    Uses the same ResBlk architecture as IMPAStyleEncoder / StyleEncoderClassifier
    (proven in probe experiments to have strong representational capacity).
    Outputs a z_dim style vector that conditions NCA dynamics via FiLM.

    Default configuration (base=128, depth=4) matches the probe-calibrated
    ``probe-s-w128-d4`` architecture.

    Architecture:
        Conv2d(in_channels -> base_channels, 3x3)
        -> ResBlk(downsample=True) x num_downsamples
        -> LeakyReLU -> AdaptiveAvgPool2d(1) -> LeakyReLU
        -> Linear(last_channels -> z_dim)

    Args:
        in_channels: Number of input image channels.
        base_channels: Initial channel width.
        num_downsamples: Number of downsampling ResBlk stages.
        max_channels: Channel cap.
        z_dim: Output style vector dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        num_downsamples: int = 4,
        max_channels: int = 512,
        z_dim: int = 32,
    ):
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
        self.fc = nn.Linear(dim_out, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)
