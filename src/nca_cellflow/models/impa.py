"""
IMPA model — 1:1 reimplementation of theislab/IMPA (StarGAN-v2 inspired).

Architecture:
  Generator: encoder-decoder with ResBlk (encoder) + AdainResBlk (decoder)
  MappingNetwork: perturbation embedding + noise → style vector
  StyleEncoder: image → style vector (for cycle consistency)
  Discriminator: multi-task head (one output per compound)

Reference: https://github.com/theislab/IMPA
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s, basal=False):
        if not basal:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, basal=False):
        x = self.norm1(x, s, basal)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s, basal)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, basal=False):
        return (self._residual(x, s, basal) + self._shortcut(x)) / math.sqrt(2)


# ---------------------------------------------------------------------------
# Generator (autoencoder)
# ---------------------------------------------------------------------------

class IMPAGenerator(nn.Module):
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512,
                 in_channels=3, dim_in=64):
        super().__init__()
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(in_channels, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, in_channels, 1, 1, 0),
        )

        repeat_num = math.ceil(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s, basal=False):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        z = x.clone()
        for block in self.decode:
            x = block(x, s, basal)
        return z, self.to_rgb(x)

    def encode_single(self, x):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        return x

    def decode_single(self, x, s, basal=False):
        for block in self.decode:
            x = block(x, s, basal)
        return self.to_rgb(x)


# ---------------------------------------------------------------------------
# Mapping Network (perturbation encoder)
# ---------------------------------------------------------------------------

class IMPAMappingNetwork(nn.Module):
    """Maps concatenated [perturbation_embedding, noise_z] → style vector."""
    def __init__(self, latent_dim=1040, style_dim=64, hidden_dim=512, num_layers=1):
        super().__init__()
        in_dim = latent_dim
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else style_dim
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


# ---------------------------------------------------------------------------
# Style Encoder
# ---------------------------------------------------------------------------

class IMPAStyleEncoder(nn.Module):
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512,
                 in_channels=3, dim_in=64):
        super().__init__()
        blocks = [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]
        repeat_num = math.ceil(np.log2(img_size)) - 2
        final_conv_dim = img_size // (2 ** repeat_num)
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, dim_out, final_conv_dim, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*blocks)
        self.linear = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.linear(h)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class IMPADiscriminator(nn.Module):
    """Multi-task discriminator: one scalar output per compound class."""
    def __init__(self, img_size=96, num_domains=6, max_conv_dim=512,
                 in_channels=3, dim_in=64):
        super().__init__()
        blocks = [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]
        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks.append(nn.LeakyReLU(0.2, inplace=True))
        blocks.append(nn.Conv2d(dim_out, dim_out, 3, 1, 0))
        blocks.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(dim_out, num_domains, 1, 1, 0)

    def forward(self, x, mol):
        out = self.conv_blocks(x)
        out = self.head(out)
        out = out.view(out.size(0), -1)
        idx = torch.arange(mol.size(0), device=mol.device)
        return out[idx, mol]


# ---------------------------------------------------------------------------
# Weight init (He / Kaiming)
# ---------------------------------------------------------------------------

def he_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
