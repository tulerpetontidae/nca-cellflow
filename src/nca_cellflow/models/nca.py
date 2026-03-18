"""
NCA Architecture Variants

This module provides different NCA architectures for experimentation:
- BaseNCA: Simple deterministic NCA with single update rule
- GradientSensor: Sobel gradient sensing for spatial feature extraction

Future variants could include:
- ModularNCA: Per-channel block-diagonal updates (no mixture)
- MixtureNCA: Full mixture of experts with stochasticity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    compound_id -> Embedding(num_classes, cond_dim) -> FiLM(gamma, beta) -> modulate hidden

    Args:
        channel_n: Number of state channels
        hidden_dim: Hidden dimension in update MLP
        num_classes: Number of compound classes
        cond_dim: Dimension of compound embedding
        fire_rate: Probability of cell update (stochastic masking)
    """

    def __init__(
        self,
        channel_n: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 1,
        cond_dim: int = 64,
        fire_rate: float = 1.0,
    ):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_dim = hidden_dim
        self.fire_rate = fire_rate
        self.sensor = GradientSensor(channel_n)

        # Compound embedding: id -> vector
        self.embed = nn.Embedding(num_classes, cond_dim)

        # FiLM: embedding -> (gamma, beta) for hidden layer
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

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
        film = self.film(emb)  # [B, hidden_dim * 2]
        gamma, beta = film.chunk(2, dim=1)  # [B, hidden_dim] each
        gamma = gamma[:, None, None, :]  # [B, 1, 1, hidden_dim]
        beta = beta[:, None, None, :]

        # Gradient features
        features = self.sensor(x)  # [B, 3C, H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, 3C]

        # MLP with FiLM
        h = F.relu(self.fc1(features))  # [B, H, W, hidden]
        h = gamma * h + beta  # FiLM
        h = F.relu(self.fc2(h))
        dx = self.fc3(h).permute(0, 3, 1, 2)  # [B, C, H, W]

        dx = torch.clamp(dx, min=-10, max=10)

        # Fire rate mask
        if self.fire_rate < 1.0:
            mask = (torch.rand(B, 1, H, W, device=x.device) < self.fire_rate).float()
            dx = dx * mask

        return x + dx * 0.1

    def forward(self, x, cond, n_steps: int = 1):
        """Apply n_steps NCA updates."""
        for _ in range(n_steps):
            x = self.step(x, cond)
        return x

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
