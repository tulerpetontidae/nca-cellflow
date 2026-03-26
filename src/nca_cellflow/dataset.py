"""
Minimal IMPA data loader — sampling logic only.

Design:
    - Two independent pools: ctrl (DMSO) and trt (treated).
    - __len__ = number of control images (or treated if iter_trt=True).
    - __getitem__(idx): ctrl is deterministic, trt is uniformly random.
    - plate_match=False (default): paired image sampled from entire pool.
    - plate_match=True: paired image sampled from same plate.
    - balanced_cpd=True: sample compound uniformly first, then image from that compound.
    - iter_trt=True: iterate over treated images, randomly sample ctrl (matches CellFlux paper).
    - Returns: ctrl image, trt image, compound index (int).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


def _plate_from_key(key: str) -> str:
    """Extract plate id from SAMPLE_KEY: {BATCH}_{PLATE}_{rest}."""
    return key.split("_")[1]


class IMPADataset(Dataset):

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 96, plate_match: bool = False,
                 balanced_cpd: bool = False, iter_trt: bool = False):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == split]

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
        self.plate_match = plate_match
        self.balanced_cpd = balanced_cpd
        self.iter_trt = iter_trt

        if iter_trt and balanced_cpd:
            raise ValueError("iter_trt and balanced_cpd are mutually exclusive: "
                             "iter_trt gives uniform coverage over treated images, "
                             "balanced_cpd gives uniform coverage over compounds")

        if plate_match or iter_trt:
            # Build per-plate index of treated images
            self._plate_to_trt_idx = {}
            for i, key in enumerate(self.trt_keys):
                plate = _plate_from_key(key)
                self._plate_to_trt_idx.setdefault(plate, []).append(i)
            # Pre-compute plate for each ctrl
            self._ctrl_plates = np.array([_plate_from_key(k) for k in self.ctrl_keys])

        if iter_trt:
            # Pre-compute plate for each trt
            self._trt_plates = np.array([_plate_from_key(k) for k in self.trt_keys])
            # Build per-plate index of ctrl images
            self._plate_to_ctrl_idx = {}
            for i, key in enumerate(self.ctrl_keys):
                plate = _plate_from_key(key)
                self._plate_to_ctrl_idx.setdefault(plate, []).append(i)

        if balanced_cpd:
            # Build per-compound index of treated images
            self._cpd_to_trt_idx = {}
            for i, cpd in enumerate(self.trt_cpd):
                self._cpd_to_trt_idx.setdefault(cpd, []).append(i)
            self._cpd_list = cpds  # sorted compound names

    def __len__(self):
        if self.iter_trt:
            return len(self.trt_keys)
        return len(self.ctrl_keys)

    def __getitem__(self, idx):
        if self.iter_trt:
            return self._getitem_iter_trt(idx)

        if self.balanced_cpd:
            # Sample compound uniformly, then pick a treated image
            cpd = self._cpd_list[np.random.randint(len(self._cpd_list))]
            pool = self._cpd_to_trt_idx[cpd]
            j = pool[np.random.randint(len(pool))]

            if self.plate_match:
                # Pick ctrl from the same plate as the selected treated image
                plate = _plate_from_key(self.trt_keys[j])
                ctrl_pool = np.where(self._ctrl_plates == plate)[0]
                if len(ctrl_pool) > 0:
                    idx = ctrl_pool[np.random.randint(len(ctrl_pool))]
                # else: fall through to original idx

            img_ctrl = self._load(self.ctrl_keys[idx])
        else:
            # ctrl: deterministic by idx
            img_ctrl = self._load(self.ctrl_keys[idx])

        if not self.balanced_cpd:
            if self.plate_match:
                # trt: uniform random from same plate as ctrl
                plate = self._ctrl_plates[idx]
                pool = self._plate_to_trt_idx.get(plate)
                if pool is not None and len(pool) > 0:
                    j = pool[np.random.randint(len(pool))]
                else:
                    # Fallback to global pool if plate has no treated images
                    j = np.random.randint(len(self.trt_keys))
            else:
                # trt: uniform random from ALL treated images (any plate)
                j = np.random.randint(len(self.trt_keys))

        img_trt = self._load(self.trt_keys[j])
        cpd_id = self.cpd2id[self.trt_cpd[j]]

        return img_ctrl, img_trt, cpd_id

    def _getitem_iter_trt(self, idx):
        """Iterate over treated images, randomly sample ctrl (CellFlux paper style)."""
        j = idx  # trt is deterministic
        img_trt = self._load(self.trt_keys[j])
        cpd_id = self.cpd2id[self.trt_cpd[j]]

        if self.plate_match:
            # ctrl from same plate as this treated image
            plate = self._trt_plates[j]
            pool = self._plate_to_ctrl_idx.get(plate)
            if pool is not None and len(pool) > 0:
                ci = pool[np.random.randint(len(pool))]
            else:
                ci = np.random.randint(len(self.ctrl_keys))
        else:
            ci = np.random.randint(len(self.ctrl_keys))

        img_ctrl = self._load(self.ctrl_keys[ci])
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
