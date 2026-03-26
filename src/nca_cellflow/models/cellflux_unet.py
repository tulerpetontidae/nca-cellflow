"""
CellFlux UNet for flow matching.

Self-contained port of the UNet from https://github.com/yuhui-zh15/CellFlux
(originally from guided-diffusion). No external model dependencies required.

The UNet predicts a velocity field v(x_t, t, cond) for conditional OT flow matching.
Conditioning is via a projected fingerprint embedding added to the timestep embedding.

Original code: Copyright (c) Meta Platforms, Inc. (CC-by-NC license).
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# nn utilities (from CellFlux models/nn.py)
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    return GroupNorm32(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class ConstantEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty((1, out_channels)))
        nn.init.uniform_(self.embedding_table, -(in_channels**0.5), in_channels**0.5)

    def forward(self, emb):
        return self.embedding_table.repeat(emb.shape[0], 1)


class ResBlock(TimestepBlock):
    def __init__(
        self, channels, emb_channels, dropout, out_channels=None,
        use_conv=False, use_scale_shift_norm=False, dims=2,
        use_checkpoint=False, up=False, down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1,
                 use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# UNet model
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class CellFluxUNet(nn.Module):
    """
    UNet for conditional flow matching (CellFlux architecture).

    Predicts velocity field v(x_t, t, cond) where:
    - x_t: noisy image at time t
    - t: scalar time in [0, 1]
    - cond: fingerprint embedding vector

    BBBC021 config: 128 base channels, [2,2,2] mult, 4 res blocks,
    attention at resolution 2, scale-shift norm, 1024-dim conditioning.
    """
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 4
    attention_resolutions: Tuple[int] = (2,)
    dropout: float = 0.3
    channel_mult: Tuple[int] = (2, 2, 2)
    conv_resample: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = True
    resblock_updown: bool = False
    use_new_attention_order: bool = True
    condition_dim: int = 1024

    def __post_init__(self):
        super().__init__()

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        time_embed_dim = self.model_channels * 4

        self.mol_embed_transform = nn.Linear(self.condition_dim, time_embed_dim)
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, dims=self.dims,
                     use_checkpoint=self.use_checkpoint,
                     use_scale_shift_norm=self.use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=self.use_checkpoint,
                           num_heads=self.num_heads,
                           num_head_channels=self.num_head_channels,
                           use_new_attention_order=self.use_new_attention_order),
            ResBlock(ch, time_embed_dim, self.dropout, dims=self.dims,
                     use_checkpoint=self.use_checkpoint,
                     use_scale_shift_norm=self.use_scale_shift_norm),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, t, cond=None):
        """
        Predict velocity field.

        Args:
            x: [B, C, H, W] noisy image at time t
            t: [B] time values in [0, 1]
            cond: [B, condition_dim] fingerprint embedding (None for unconditional)

        Returns:
            [B, C, H, W] predicted velocity
        """
        hs = []
        emb = self.time_embed(timestep_embedding(t, self.model_channels).to(x))

        if cond is not None:
            mol_emb = self.mol_embed_transform(cond)
            emb = emb + mol_emb

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


# ---------------------------------------------------------------------------
# ODE sampling (no external dependency)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ode_sample_heun(model, x_init, time_grid, cond=None, cfg_scale=0.0):
    """
    Heun's method (2nd-order) ODE solver for flow matching.

    Args:
        model: velocity network v(x, t, cond)
        x_init: [B, C, H, W] initial state (ctrl image + noise)
        time_grid: 1-D tensor of time values from 0 to 1
        cond: [B, condition_dim] conditioning (None for unconditional)
        cfg_scale: classifier-free guidance scale (0 = no guidance)

    Returns:
        x: [B, C, H, W] final generated image
    """
    x = x_init
    for i in range(len(time_grid) - 1):
        t_cur = time_grid[i]
        t_next = time_grid[i + 1]
        dt = t_next - t_cur

        t_batch = torch.full((x.shape[0],), t_cur, device=x.device)
        v1 = _eval_velocity(model, x, t_batch, cond, cfg_scale)

        x_euler = x + v1 * dt

        t_batch_next = torch.full((x.shape[0],), t_next, device=x.device)
        v2 = _eval_velocity(model, x_euler, t_batch_next, cond, cfg_scale)

        x = x + 0.5 * dt * (v1 + v2)

    return x


@torch.no_grad()
def ode_sample_euler(model, x_init, time_grid, cond=None, cfg_scale=0.0):
    """Euler method ODE solver."""
    x = x_init
    for i in range(len(time_grid) - 1):
        t_cur = time_grid[i]
        dt = time_grid[i + 1] - t_cur
        t_batch = torch.full((x.shape[0],), t_cur, device=x.device)
        v = _eval_velocity(model, x, t_batch, cond, cfg_scale)
        x = x + v * dt
    return x


@torch.no_grad()
def ode_sample_midpoint(model, x_init, time_grid, cond=None, cfg_scale=0.0):
    """Midpoint method ODE solver."""
    x = x_init
    for i in range(len(time_grid) - 1):
        t_cur = time_grid[i]
        t_next = time_grid[i + 1]
        dt = t_next - t_cur
        t_mid = t_cur + 0.5 * dt

        t_batch = torch.full((x.shape[0],), t_cur, device=x.device)
        v1 = _eval_velocity(model, x, t_batch, cond, cfg_scale)
        x_mid = x + 0.5 * dt * v1

        t_batch_mid = torch.full((x.shape[0],), t_mid, device=x.device)
        v_mid = _eval_velocity(model, x_mid, t_batch_mid, cond, cfg_scale)
        x = x + dt * v_mid

    return x


def _eval_velocity(model, x, t, cond, cfg_scale):
    """Evaluate velocity with optional classifier-free guidance."""
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type):
        v_cond = model(x, t, cond=cond)
    if cfg_scale > 0.0 and cond is not None:
        with torch.amp.autocast(device_type=device_type):
            v_uncond = model(x, t, cond=None)
        v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
    else:
        v = v_cond
    return v.float()


# ---------------------------------------------------------------------------
# EDM time discretization schedule
# ---------------------------------------------------------------------------

def edm_time_grid(nfe, rho=7):
    """EDM-style time discretization (Karras et al. 2022)."""
    step_indices = torch.arange(nfe, dtype=torch.float64)
    sigma_min, sigma_max = 0.002, 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfe - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples.float()


# ---------------------------------------------------------------------------
# Skewed timestep sampling for training
# ---------------------------------------------------------------------------

def skewed_timestep_sample(num_samples, device):
    """Log-normal skewed timestep sampling (EDM-style)."""
    P_mean, P_std = -1.2, 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    return torch.clip(t, min=0.0001, max=1.0)
