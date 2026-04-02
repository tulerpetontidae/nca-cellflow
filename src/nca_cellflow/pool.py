"""
Replay pool for petri dish training.

Stores persistent NCA states on GPU that evolve across training iterations.
Each slot tracks its full NCA state, current conditioning (compound, dose),
plate of origin, latent z, and iteration count.
"""

import torch
import numpy as np


class ReplayPool:
    """GPU-resident replay buffer of evolving NCA states.

    Args:
        pool_size: Number of persistent state slots.
        channel_n: Total NCA channels (RGB + optional alive + hidden).
        H, W: Spatial dimensions.
        z_dim: Latent z dimension per slot.
        device: Torch device.
    """

    def __init__(self, pool_size: int, channel_n: int, H: int, W: int,
                 z_dim: int = 16, device: torch.device = torch.device("cpu")):
        self.pool_size = pool_size
        self.channel_n = channel_n
        self.H, self.W = H, W
        self.z_dim = z_dim
        self.device = device

        self.states = torch.zeros((pool_size, channel_n, H, W), device=device)
        self.labels = torch.zeros(pool_size, dtype=torch.long, device=device)  # compound id (0 = DMSO)
        self.doses = torch.zeros(pool_size, dtype=torch.float32, device=device)
        self.plates = np.zeros(pool_size, dtype=object)  # plate strings
        self.iters = torch.zeros(pool_size, dtype=torch.long, device=device)
        self.z = torch.randn(pool_size, z_dim, device=device)

    def populate(self, image_bank, img_channels: int = 3, hidden_channels: int = 0):
        """Initialize all slots with control images from the image bank.

        Args:
            image_bank: LabeledImageBank instance.
            img_channels: Number of visible channels (3 or 4 with alive).
            hidden_channels: Number of hidden channels to zero-pad.
        """
        dmso_id = 0  # DMSO is always class 0 in petri dish
        for i in range(self.pool_size):
            img, plate = image_bank.sample_one(dmso_id, dose=0.0)
            self.states[i, :img_channels] = img.to(self.device)
            if hidden_channels > 0:
                self.states[i, img_channels:] = 0.0
            self.labels[i] = dmso_id
            self.doses[i] = 0.0
            self.plates[i] = plate
        self.iters.zero_()
        self.z.normal_()

    def get_batch(self, batch_size: int):
        """Sample a random batch from the pool.

        Returns:
            indices: [B] pool indices for later update
            states: [B, C, H, W]
            labels: [B]
            doses: [B]
            plates: list of plate strings
            z: [B, z_dim]
        """
        indices = torch.randperm(self.pool_size, device=self.device)[:batch_size]
        return (
            indices,
            self.states[indices].clone(),
            self.labels[indices].clone(),
            self.doses[indices].clone(),
            [self.plates[i.item()] for i in indices],
            self.z[indices].clone(),
        )

    def update(self, indices, states, labels, doses, plates, z, iters_delta=1):
        """Write evolved states back into the pool."""
        self.states[indices] = states.detach()
        self.labels[indices] = labels
        self.doses[indices] = doses
        for i, idx in enumerate(indices):
            self.plates[idx.item()] = plates[i]
        self.z[indices] = z.detach()
        self.iters[indices] += iters_delta

    def recycle(self, threshold: int, image_bank, img_channels: int = 3,
                hidden_channels: int = 0):
        """Replace old states (iter > threshold) with fresh control images.

        Returns number of recycled slots.
        """
        mask = self.iters > threshold
        n = mask.sum().item()
        if n == 0:
            return 0
        idxs = mask.nonzero(as_tuple=True)[0]
        dmso_id = 0
        for idx in idxs:
            i = idx.item()
            img, plate = image_bank.sample_one(dmso_id, dose=0.0)
            self.states[i] = 0.0
            self.states[i, :img_channels] = img.to(self.device)
            self.labels[i] = dmso_id
            self.doses[i] = 0.0
            self.plates[i] = plate
            self.z[i].normal_()
            self.iters[i] = 0
        return n

    def state_dict(self):
        return {
            "states": self.states.cpu(),
            "labels": self.labels.cpu(),
            "doses": self.doses.cpu(),
            "plates": self.plates.copy(),
            "iters": self.iters.cpu(),
            "z": self.z.cpu(),
        }

    def load_state_dict(self, snapshot, device=None):
        dev = device or self.device
        self.states = snapshot["states"].to(dev)
        self.labels = snapshot["labels"].to(dev)
        self.doses = snapshot["doses"].to(dev)
        self.plates = snapshot["plates"]
        self.iters = snapshot["iters"].to(dev)
        self.z = snapshot["z"].to(dev)
