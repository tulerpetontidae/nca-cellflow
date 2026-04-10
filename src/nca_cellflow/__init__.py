"""
NCA-CellFlow: Neural Cellular Automata for Cellular Morphology Generation

A PyTorch implementation of NCA-GAN for IMPA (Image-based Morphological Profiling Assay)
using BBBC021 dataset.
"""

__version__ = "0.1.0"

from .dataset import IMPADataset, EvalDataset, LabeledImageBank, ClassificationDataset, CtrlPairDataset
from .pool import ReplayPool
from .metrics import compute_texture_stats

__all__ = [
    "IMPADataset",
    "EvalDataset",
    "LabeledImageBank",
    "ClassificationDataset",
    "CtrlPairDataset",
    "ReplayPool",
    "compute_texture_stats",
    "__version__",
]
