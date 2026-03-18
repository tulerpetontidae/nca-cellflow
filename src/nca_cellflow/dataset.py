"""
Minimal IMPA data loader — sampling logic only.

Design:
    - Two independent pools: ctrl (DMSO) and trt (treated).
    - __len__ = number of control images.
    - __getitem__(idx): ctrl is deterministic, trt is uniformly random
      from the ENTIRE treated pool. No plate matching.
    - Returns: ctrl image, trt image, compound index (int).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class IMPADataset(Dataset):

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train", image_size: int = 96):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == split]

        # Two independent pools — no plate constraint between them
        # STATE: 0 = control (DMSO), 1 = treated
        ctrl = df[df["STATE"] == 0]
        trt = df[df["STATE"] == 1]

        self.ctrl_keys = ctrl["SAMPLE_KEY"].values
        self.trt_keys = trt["SAMPLE_KEY"].values
        self.trt_cpd = trt["CPD_NAME"].values

        # compound name → int id
        cpds = sorted(set(self.trt_cpd))
        self.cpd2id = {c: i for i, c in enumerate(cpds)}

        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self):
        # one epoch = one pass over every control image
        return len(self.ctrl_keys)

    def __getitem__(self, idx):
        # ctrl: deterministic by idx
        img_ctrl = self._load(self.ctrl_keys[idx])

        # trt: uniform random from ALL treated images (any plate)
        j = np.random.randint(len(self.trt_keys))
        img_trt = self._load(self.trt_keys[j])
        cpd_id = self.cpd2id[self.trt_cpd[j]]

        return img_ctrl, img_trt, cpd_id

    def _load(self, key: str) -> torch.Tensor:
        parts = key.split("_")
        path = self.image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
        img = np.load(path).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)        # (C, H, W)
        img = (img + torch.rand_like(img)) / 255.0           # dither + [0,1]
        if self.image_size != img.shape[-1]:
            img = F.interpolate(img.unsqueeze(0), size=self.image_size, mode="bilinear", antialias=True).squeeze(0)
        img = img * 2.0 - 1.0                               # [-1, 1]
        if torch.rand(1).item() < 0.5:
            img = img.flip(-1)                               # horizontal flip
        if torch.rand(1).item() < 0.5:
            img = img.flip(-2)                               # vertical flip
        return img
