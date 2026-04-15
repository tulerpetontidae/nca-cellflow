"""
MoA (Mode of Action) classifier for evaluating generated cell images.

Frozen Inception-v3 features (2048-dim, same as FID) → 2-layer MLP classifier.
Follows the CellFlux evaluation protocol: train on real treated images,
then evaluate whether generated images preserve correct MoA morphology.
"""

import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance


class MOAClassifier(nn.Module):
    """Inception-v3 feature extractor + MLP classifier for MoA prediction.

    Uses the same Inception-v3 backbone as FID (pool3 features, 2048-dim).
    The backbone is always frozen; only the MLP head is trained.

    Input: [B, 3, H, W] images in [0, 1] float range.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # Extract Inception from FID and discard the FID metric wrapper
        # (the wrapper has float64 buffers that break MPS)
        _fid = FrechetInceptionDistance(normalize=True)
        self.inception = _fid.inception
        del _fid
        for p in self.inception.parameters():
            p.requires_grad = False
        self.inception.eval()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim Inception features from images in [0, 1]."""
        self.inception.eval()
        x_uint8 = (x * 255).clamp(0, 255).byte()
        return self.inception(x_uint8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(x)
        return self.classifier(features)

    def state_dict_head(self):
        """Return only the classifier head state dict (skip Inception)."""
        return self.classifier.state_dict()

    def load_state_dict_head(self, state_dict):
        """Load only the classifier head state dict."""
        self.classifier.load_state_dict(state_dict)
