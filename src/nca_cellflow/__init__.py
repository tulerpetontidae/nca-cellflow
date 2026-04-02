"""
NCA-CellFlow: Neural Cellular Automata for Cellular Morphology Generation

A PyTorch implementation of NCA-GAN for IMPA (Image-based Morphological Profiling Assay)
using BBBC021 dataset.
"""

__version__ = "0.1.0"

from .dataset import IMPADataset, EvalDataset, LabeledImageBank
from .pool import ReplayPool

__all__ = ["IMPADataset", "EvalDataset", "LabeledImageBank", "ReplayPool", "__version__"]
