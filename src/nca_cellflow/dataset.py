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
from collections import defaultdict


def _plate_from_key(key: str) -> str:
    """Extract plate id from SAMPLE_KEY: {BATCH}_{PLATE}_{rest}."""
    return key.split("_")[1]


def _load_image(image_dir: Path, key: str, image_size: int = 96,
                augment: bool = False) -> torch.Tensor:
    """Load a single .npy image, normalize to [-1, 1]."""
    parts = key.split("_")
    path = image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
    img = np.load(path).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    if augment:
        img = (img + torch.rand_like(img)) / 255.0
    else:
        img = (img + 0.5) / 255.0
    if image_size != img.shape[-1]:
        img = F.interpolate(img.unsqueeze(0), size=image_size,
                            mode="bilinear", antialias=True).squeeze(0)
    img = img * 2.0 - 1.0
    if augment:
        if torch.rand(1).item() < 0.5:
            img = img.flip(-1)
        if torch.rand(1).item() < 0.5:
            img = img.flip(-2)
    return img


class IMPADataset(Dataset):

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 96, plate_match: bool = False,
                 balanced_cpd: bool = False, iter_trt: bool = False,
                 exclude_compounds: list[str] | None = None):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == split]

        # STATE: 0 = control (DMSO), 1 = treated
        ctrl = df[df["STATE"] == 0]
        trt = df[df["STATE"] == 1]

        # OOD split: exclude specified compounds from treated set
        if exclude_compounds:
            n_before = len(trt)
            trt = trt[~trt["CPD_NAME"].isin(exclude_compounds)]
            print(f"[OOD] Excluded {n_before - len(trt)} treated images "
                  f"({len(exclude_compounds)} compounds: {exclude_compounds})")

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

    def _load(self, key: str, augment: bool = True) -> torch.Tensor:
        parts = key.split("_")
        path = self.image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
        img = np.load(path).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)        # (C, H, W)
        if augment:
            img = (img + torch.rand_like(img)) / 255.0       # dither + [0,1]
        else:
            img = (img + 0.5) / 255.0                        # center of bin, no noise
        if self.image_size != img.shape[-1]:
            img = F.interpolate(img.unsqueeze(0), size=self.image_size, mode="bilinear", antialias=True).squeeze(0)
        img = img * 2.0 - 1.0                               # [-1, 1]
        if augment:
            if torch.rand(1).item() < 0.5:
                img = img.flip(-1)                           # horizontal flip
            if torch.rand(1).item() < 0.5:
                img = img.flip(-2)                           # vertical flip
        return img


class ClassificationDataset(Dataset):
    """Treated-only compound classification dataset for capacity probing.

    Returns ``(img, cpd_id)`` for each treated image. Designed for decoupled
    calibration of D/StyleEncoder architectures as plain 34-class classifiers
    — no ctrl, no pair sampling.

    The compound label space is built from the **train split** regardless of
    which split this instance serves, so train and val share identical ids.

    Args:
        metadata_csv: Path to ``bbbc021_df_all.csv``.
        image_dir: Root image directory.
        split: "train" or "test".
        image_size: Target spatial size.
        balanced_cpd: If True, ``__getitem__`` samples a compound uniformly
            and then picks a random treated image from that compound. Used
            for training so rare compounds aren't starved.
        iter_all: If True, ``__getitem__(idx)`` returns the idx-th treated
            image deterministically and ``__len__`` is the number of treated
            images. Used for eval. Mutually exclusive with ``balanced_cpd``.
        augment: If True, apply dither + horizontal/vertical flips (matches
            ``IMPADataset._load``).
    """

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 48, balanced_cpd: bool = True,
                 iter_all: bool = False, augment: bool = True):
        if balanced_cpd and iter_all:
            raise ValueError("balanced_cpd and iter_all are mutually exclusive")

        df = pd.read_csv(metadata_csv, index_col=0)
        df_trt = df[df["STATE"] == 1]

        # Label space is derived from the TRAIN split so train/val align.
        df_trt_train = df_trt[df_trt["SPLIT"] == "train"]
        cpds_sorted = sorted(set(df_trt_train["CPD_NAME"].values))
        self.cpd2id = {c: i for i, c in enumerate(cpds_sorted)}
        self.id2cpd = {i: c for c, i in self.cpd2id.items()}
        self.num_classes = len(self.cpd2id)

        # Now filter to the requested split for actual sampling.
        df_split = df_trt[df_trt["SPLIT"] == split]
        self.trt_keys = df_split["SAMPLE_KEY"].values
        self.trt_cpd = df_split["CPD_NAME"].values
        # Any compound in this split that wasn't in train (shouldn't happen
        # in BBBC021, but guard anyway) is dropped.
        keep = np.array([c in self.cpd2id for c in self.trt_cpd])
        self.trt_keys = self.trt_keys[keep]
        self.trt_cpd = self.trt_cpd[keep]

        if balanced_cpd:
            self._cpd_to_trt_idx: dict[str, list[int]] = {}
            for i, cpd in enumerate(self.trt_cpd):
                self._cpd_to_trt_idx.setdefault(cpd, []).append(i)
            # Only iterate over compounds that actually appear in this split.
            self._cpd_list = sorted(self._cpd_to_trt_idx.keys())

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.balanced_cpd = balanced_cpd
        self.iter_all = iter_all
        self.augment = augment

    def __len__(self):
        return len(self.trt_keys)

    def __getitem__(self, idx):
        if self.balanced_cpd:
            cpd = self._cpd_list[np.random.randint(len(self._cpd_list))]
            pool = self._cpd_to_trt_idx[cpd]
            j = pool[np.random.randint(len(pool))]
        else:
            # iter_all or plain index iteration
            j = idx

        img = self._load(self.trt_keys[j])
        cpd_id = self.cpd2id[self.trt_cpd[j]]
        return img, cpd_id

    def _load(self, key: str) -> torch.Tensor:
        return _load_image(self.image_dir, key, self.image_size, augment=self.augment)


class EvalDataset(Dataset):
    """Deterministic dataset for FID evaluation.

    Always iterates over treated images. Each treated image is paired with a
    deterministic same-plate ctrl (seeded by index). No augmentation.
    """

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "test",
                 image_size: int = 96,
                 exclude_compounds: list[str] | None = None,
                 only_compounds: list[str] | None = None,
                 cpd2id: dict[str, int] | None = None):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == split]

        ctrl = df[df["STATE"] == 0]
        trt = df[df["STATE"] == 1]

        # OOD split: exclude or include specific compounds
        if exclude_compounds:
            trt = trt[~trt["CPD_NAME"].isin(exclude_compounds)]
        if only_compounds:
            trt = trt[trt["CPD_NAME"].isin(only_compounds)]

        self.ctrl_keys = ctrl["SAMPLE_KEY"].values
        self.trt_keys = trt["SAMPLE_KEY"].values
        self.trt_cpd = trt["CPD_NAME"].values
        self.trt_dose = trt["DOSE"].values.astype(np.float32)

        if cpd2id is not None:
            # Use provided base mapping, extend with any new compounds
            self.cpd2id = dict(cpd2id)
            next_id = max(cpd2id.values()) + 1 if cpd2id else 0
            for c in sorted(set(self.trt_cpd)):
                if c not in self.cpd2id:
                    self.cpd2id[c] = next_id
                    next_id += 1
        else:
            cpds = sorted(set(self.trt_cpd))
            self.cpd2id = {c: i for i, c in enumerate(cpds)}

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Build per-plate ctrl index for deterministic pairing
        self._trt_plates = np.array([_plate_from_key(k) for k in self.trt_keys])
        self._plate_to_ctrl_idx = {}
        for i, key in enumerate(self.ctrl_keys):
            plate = _plate_from_key(key)
            self._plate_to_ctrl_idx.setdefault(plate, []).append(i)
        # Sort for determinism
        for plate in self._plate_to_ctrl_idx:
            self._plate_to_ctrl_idx[plate] = sorted(self._plate_to_ctrl_idx[plate])

    def __len__(self):
        return len(self.trt_keys)

    def __getitem__(self, idx):
        img_trt = self._load(self.trt_keys[idx])
        cpd_id = self.cpd2id[self.trt_cpd[idx]]
        dose = self.trt_dose[idx]

        # Deterministic ctrl: pick from same plate using idx as seed
        plate = self._trt_plates[idx]
        pool = self._plate_to_ctrl_idx.get(plate)
        if pool is not None and len(pool) > 0:
            ci = pool[idx % len(pool)]
        else:
            ci = idx % len(self.ctrl_keys)
        img_ctrl = self._load(self.ctrl_keys[ci])

        return img_ctrl, img_trt, cpd_id, dose

    def _load(self, key: str) -> torch.Tensor:
        return _load_image(self.image_dir, key, self.image_size, augment=False)


class CtrlPairDataset(Dataset):
    """Ctrl-only dataset returning pairs of ctrl images for self-supervised NCA.

    Returns (img_input, img_ref) where both are ctrl (DMSO) images.
    img_input serves as NCA input, img_ref as D real sample.
    All images are preloaded into RAM as uint8 arrays and augmented on the fly.

    Args:
        metadata_csv: Path to bbbc021_df_all.csv.
        image_dir: Root image directory.
        split: "train" or "test".
        image_size: Target spatial size.
        augment: Apply dither + flips (True for training, False for eval).
        deterministic_ref: If True, reference is deterministic by index
            (shifted by half the dataset) for reproducible evaluation.
    """

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 96, augment: bool = True,
                 deterministic_ref: bool = False):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[(df["SPLIT"] == split) & (df["STATE"] == 0)]
        self.ctrl_keys = df["SAMPLE_KEY"].values
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        self.deterministic_ref = deterministic_ref

        # Preload all ctrl images into RAM (uint8 to save memory)
        self._cache = {}
        pkl_path = self.image_dir / "images.pkl"
        if pkl_path.exists():
            import pickle
            print(f"[CtrlPairDataset] Loading from {pkl_path}...")
            with open(pkl_path, "rb") as f:
                all_images = pickle.load(f)
            for key in self.ctrl_keys:
                img = all_images[key]
                if image_size != img.shape[0]:
                    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                    t = F.interpolate(t.unsqueeze(0), size=image_size,
                                      mode="bilinear", antialias=True).squeeze(0)
                    self._cache[key] = t
                else:
                    self._cache[key] = img
            del all_images
        else:
            print(f"[CtrlPairDataset] Preloading {len(self.ctrl_keys)} images from disk...")
            for key in self.ctrl_keys:
                parts = key.split("_")
                path = self.image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
                img = np.load(path)
                if image_size != img.shape[0]:
                    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                    t = F.interpolate(t.unsqueeze(0), size=image_size,
                                      mode="bilinear", antialias=True).squeeze(0)
                    self._cache[key] = t
                else:
                    self._cache[key] = img
        print(f"[CtrlPairDataset] {len(self._cache)} images cached ({split})")

    def __len__(self):
        return len(self.ctrl_keys)

    def _load_cached(self, key: str) -> torch.Tensor:
        raw = self._cache[key]
        if isinstance(raw, np.ndarray):
            img = torch.from_numpy(raw.astype(np.float32)).permute(2, 0, 1)
        else:
            img = raw.clone()
        if self.augment:
            img = (img + torch.rand_like(img)) / 255.0
        else:
            img = (img + 0.5) / 255.0
        img = img * 2.0 - 1.0
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = img.flip(-1)
            if torch.rand(1).item() < 0.5:
                img = img.flip(-2)
        return img

    def __getitem__(self, idx):
        img_input = self._load_cached(self.ctrl_keys[idx])
        if self.deterministic_ref:
            ref_idx = (idx + len(self.ctrl_keys) // 2) % len(self.ctrl_keys)
        else:
            ref_idx = np.random.randint(len(self.ctrl_keys))
        img_ref = self._load_cached(self.ctrl_keys[ref_idx])
        return img_input, img_ref


class LabeledImageBank:
    """Plate-aware image sampler indexed by (cpd_id, dose) for petri dish training.

    Provides fast random sampling of real images conditioned on compound, dose,
    and optionally plate. Used by the discriminator to get real targets matching
    the pool state's current conditioning.

    Label 0 = DMSO (control), labels 1..N = compounds (sorted alphabetically,
    matching IMPADataset.cpd2id but shifted by +1 to make room for DMSO).

    Args:
        metadata_csv: Path to bbbc021_df_all.csv.
        image_dir: Root image directory.
        split: "train" or "test".
        image_size: Target spatial size.
    """

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 96, preload: bool = True):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[df["SPLIT"] == split]

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        ctrl = df[df["STATE"] == 0]
        trt = df[df["STATE"] == 1]

        # Compound mapping: 0 = DMSO, 1..N = sorted compound names
        cpds = sorted(set(trt["CPD_NAME"].values))
        self.cpd2id = {c: i + 1 for i, c in enumerate(cpds)}
        self.id2cpd = {i + 1: c for i, c in enumerate(cpds)}
        self.num_compounds = len(cpds)  # excludes DMSO
        self.num_classes = len(cpds) + 1  # includes DMSO

        # Index: cpd_id -> dose -> plate -> [sample_keys]
        # For DMSO (cpd_id=0): dose is always 0.0
        self._index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Add ctrl images under DMSO
        for key in ctrl["SAMPLE_KEY"].values:
            plate = _plate_from_key(key)
            self._index[0][0.0][plate].append(key)

        # Add trt images under their compound + dose
        for _, row in trt.iterrows():
            cpd_id = self.cpd2id[row["CPD_NAME"]]
            dose = float(row["DOSE"])
            plate = _plate_from_key(row["SAMPLE_KEY"])
            self._index[cpd_id][dose][plate].append(row["SAMPLE_KEY"])

        # Build flat index per (cpd_id, dose) for fallback (all plates)
        self._flat_index = defaultdict(list)
        for cpd_id, doses in self._index.items():
            for dose, plates in doses.items():
                for plate, keys in plates.items():
                    self._flat_index[(cpd_id, dose)].extend(keys)

        # Available (cpd_id, dose) pairs for transitions
        self.available_targets = []
        for cpd_id, doses in self._index.items():
            for dose in doses:
                if cpd_id > 0:  # skip DMSO
                    self.available_targets.append((cpd_id, dose))

        # Preload all images into RAM (uint8 to save memory, augment on the fly)
        self._cache = {}
        if preload:
            pkl_path = self.image_dir / "images.pkl"
            all_keys = set(ctrl["SAMPLE_KEY"].values) | set(trt["SAMPLE_KEY"].values)
            if pkl_path.exists():
                import pickle
                print(f"Loading prepacked images from {pkl_path}...")
                with open(pkl_path, "rb") as f:
                    all_images = pickle.load(f)
                for key in all_keys:
                    img = all_images[key]
                    if image_size != img.shape[0]:
                        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                        t = F.interpolate(t.unsqueeze(0), size=image_size,
                                          mode="bilinear", antialias=True).squeeze(0)
                        self._cache[key] = t
                    else:
                        self._cache[key] = img
                del all_images
            else:
                print(f"Preloading {len(all_keys)} images from disk "
                      f"(run scripts/prepack_images.py for 20x faster startup)...")
                for key in all_keys:
                    parts = key.split("_")
                    path = self.image_dir / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
                    img = np.load(path)
                    if image_size != img.shape[0]:
                        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                        t = F.interpolate(t.unsqueeze(0), size=image_size,
                                          mode="bilinear", antialias=True).squeeze(0)
                        self._cache[key] = t
                    else:
                        self._cache[key] = img
            print(f"Preloaded {len(self._cache)} images.")

    def _load_cached(self, key: str, augment: bool = True) -> torch.Tensor:
        """Load image from cache (fast) or disk (fallback)."""
        if key in self._cache:
            raw = self._cache[key]
            if isinstance(raw, np.ndarray):
                img = torch.from_numpy(raw.astype(np.float32)).permute(2, 0, 1)
            else:
                img = raw.clone()  # already float32 [C, H, W]
            if augment:
                img = (img + torch.rand_like(img)) / 255.0
            else:
                img = (img + 0.5) / 255.0
            img = img * 2.0 - 1.0
            if augment:
                if torch.rand(1).item() < 0.5:
                    img = img.flip(-1)
                if torch.rand(1).item() < 0.5:
                    img = img.flip(-2)
            return img
        return _load_image(self.image_dir, key, self.image_size, augment=augment)

    def sample_one(self, cpd_id: int, dose: float = 0.0,
                   plate: str | None = None) -> tuple[torch.Tensor, str]:
        """Sample a single image matching (cpd_id, dose), preferring plate.

        Returns:
            (image_tensor [C, H, W], plate_str)
        """
        # Try plate-matched first
        if plate is not None:
            keys = self._index.get(cpd_id, {}).get(dose, {}).get(plate, [])
            if keys:
                key = keys[np.random.randint(len(keys))]
                return self._load_cached(key), plate

        # Fallback: any plate
        keys = self._flat_index.get((cpd_id, dose), [])
        if not keys:
            # Last resort: any image from this compound (any dose)
            for d, plate_dict in self._index.get(cpd_id, {}).items():
                for p, ks in plate_dict.items():
                    if ks:
                        key = ks[np.random.randint(len(ks))]
                        return self._load_cached(key), p
            raise ValueError(f"No images found for cpd_id={cpd_id}, dose={dose}")

        key = keys[np.random.randint(len(keys))]
        actual_plate = _plate_from_key(key)
        return self._load_cached(key), actual_plate

    def sample_batch(self, cpd_ids: torch.Tensor, doses: torch.Tensor,
                     plates: list[str | None] | None = None) -> torch.Tensor:
        """Sample a batch of real images matching given labels.

        Args:
            cpd_ids: [B] compound IDs.
            doses: [B] dose values.
            plates: Optional list of plate strings for plate-matched sampling.

        Returns:
            [B, C, H, W] tensor of real images.
        """
        B = cpd_ids.shape[0]
        imgs = []
        for i in range(B):
            plate = plates[i] if plates is not None else None
            img, _ = self.sample_one(cpd_ids[i].item(), doses[i].item(), plate)
            imgs.append(img)
        return torch.stack(imgs)
