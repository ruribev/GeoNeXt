"""
Utility module exports for the reduced GeoNeXt codebase.

Only the essential components are re‑exported here.  The experimental
or orchestration helpers have been removed to simplify the
distribution.  Users can import the following symbols:

``calc_loss``
    Compute a composite segmentation loss (BCE, Dice, Focal).
``dice_loss``
    Compute the boundary‑boosted Dice loss.
``h5_Dataset``
    PyTorch dataset for reading training/validation HDF5 files.
``evaluator``
    Simple segmentation metrics aggregator.
``ModelEMA``
    Exponential moving average wrapper for model parameters.
``Benchmark``
    Comprehensive benchmarking system for evaluating GeoNeXt models.
"""

from .losses import calc_loss, dice_loss
from .dataset import h5_Dataset, h5_CombinedDataset
from .evaluator import evaluator
from .ema import ModelEMA
from .benchmarks import Benchmark

__all__ = [
    "calc_loss",
    "dice_loss",
    "h5_Dataset",
    "h5_CombinedDataset",
    "evaluator",
    "ModelEMA",
    "Benchmark",
]
